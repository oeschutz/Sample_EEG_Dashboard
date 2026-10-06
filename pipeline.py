"""
Galea EEG processing + Holm et al. (2009) cognitive-load index
================================================================================

This file re-implements the steps of the OpenBCI/Galea example notebooks
*verbatim* over the recordings in "EEG Recordings/",
reports data cleanliness at every step, and then computes the Holm et al. (2009)
frontal-theta / parietal-alpha brain-load index for every window of every task
recording.

Every deviation from either the example pipeline or the Holm paper is marked
`DEVIATION` inline at the point where it is made.
Search this file for "DEVIATION" to find them inline.

Outputs (written to ./outputs/, each carrying a `run_meta` block recording argv,
the parameter set and whether the run was filtered):
    qc_steps.csv          per-recording, per-step cleanliness verdicts
    qc_steps.json         same, machine readable
    cognitive_load.json   per task recording: the per-window Holm index for all
                          4 artifact modes x 3 ocular modes, PLUS the four
                          extra measures for 3 ocular modes x 2 references
    baselines.json        per baseline recording: the resting-baseline median of
                          every measure, for every toggle combination
    variants/             the parameter sweep (added 2026-09-01): two .npz per
                          window length (tasks and baselines) holding, for every
                          recording, the
                          per-window measures across all 1800 selectable
                          combinations of interpolation x ocular x reference x
                          window length x spectral estimator -- stored as fewer
                          arrays, since two interpolation modes that replace the
                          same channels share one (see interp_token) -- plus the
                          per-window peak amplitudes
                          and per-channel robust sigma the dashboard thresholds
                          against. index.json names them. Deletable and
                          rebuildable from a re-run.

Usage:
    python pipeline.py                              # run everything
    python pipeline.py --limit 2                    # smoke test on the first 2
    python pipeline.py --only "p01/task_agent"      # substring filter on the key

A filtered run still writes the canonical filenames, but marks
`run_meta.partial_run` and prints a warning; do not read a filtered run as a
complete one.

Scope note: this reproduces the EEG half of the example pipeline. The PPG /
heart-rate analysis is NOT implemented, and nothing here reads the aux file's
EDA, PPG, temperature or battery columns -- but the aux file IS read, for its
IMU, which drives the `Head motion` artifact mode (step 12). A baseline contrast
is computed in step 10: one of three selectable segments of each baseline block,
subtracted from the paired task recording.

Seven processing choices that were fixed constants are now dashboard controls.
Four of them change the spectrum and are precomputed by
step 11; three only threshold quantities already shipped and are evaluated in the
browser as continuous sliders. Every default reproduces the pre-sweep pipeline, and
verify_default_variant() asserts that on every run, for every recording and every
shipped column, against MNE's own spectral estimator -- build_dashboard.py refuses
to build if it fails.

The JSON outputs did not change SHAPE, but qc_steps.json gained a step11_variants
block per recording and all three run_meta blocks gained a `sweep` entry; both are
additive, so existing readers are unaffected.

This copy is the TEMPLATE build, meant to run on another group's recordings.
The study-specific tables -- MANUAL_INTERPOLATION, EXCLUDED_PARTICIPANTS,
EXCLUDED_RECORDINGS -- ship empty for you to fill in, and the per-recording
marker repairs the original dataset needed have been removed. See README.md.
"""

from __future__ import annotations

import argparse
import datetime as dt
import functools
import json
import re
import shutil
import sys
import time
import warnings
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.signal import get_window
from scipy.signal.windows import dpss

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "EEG Recordings"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Parameters transcribed verbatim from example_eeg_processing.ipynb
# and example_baseline_vs_task.ipynb
# ---------------------------------------------------------------------------

EMG_CHANNELS = [1, 2, 3, 4, 7, 8]
EOG_CHANNELS = [5, 6]
EEG_CHANNELS = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

GALEA_SAMPLING_RATE = 250.0
CHANNEL_TYPE_LABELS = ['emg', 'emg', 'emg', 'emg',
                       'eog', 'eog',
                       'emg', 'emg',
                       'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                       'eeg', 'eeg', 'eeg', 'eeg', 'eeg']

LINE_FREQ = 60.0

FREQ_BANDS = {"Delta": [0.5, 4],
              "Theta": [4, 8],
              "Alpha": [8, 13],
              "Beta": [13, 30],
              "Gamma": [30, 50],
              "High Gamma": [50, 100]}

AUX_SAMPLING_RATE = 50.0

obci_color_palette = {'red': '#cd5241',
                      'orange': '#ee8329',
                      'blue': '#419eaf',
                      'yellow': '#eeb45b'}

# Example-notebook filter settings (example_eeg_processing.ipynb cell 24)
LOWCUT = 0.5
HIGHCUT = 100.0

# Example-notebook epoching (example_eeg_processing.ipynb cell 34)
EPOCH_DUR = 2.0
EPOCH_OVERLAP = 0.0

# ---------------------------------------------------------------------------
# Holm et al. (2009) index parameters
#   Holm A, Lukander K, Korpela J, Sallinen M, Muller KMI.
#   "Estimating Brain Load from the EEG." TheScientificWorldJOURNAL 9, 639-651.
#
#   Methods, verbatim: "we transformed the data measured to 4-sec epochs,
#   corrected for eye movement artefacts with the ocular artefact reduction
#   (OAR) utility, excluded epochs containing other artefacts (+-70 uV), and
#   computed spectrograms for the 10-20 system derivations Fz, Cz, and Pz using
#   a Fast Fourier transformation. The sweeps were smoothed using a 2048-sample
#   Hanning window, and absolute spectral power values for theta (4-8 Hz) and
#   alpha (8-12 Hz) were calculated."
#   Index = absolute theta power at Fz / absolute alpha power at Pz.
# ---------------------------------------------------------------------------

HOLM_WINDOW_SEC = 4.0          # paper: "4-sec epochs"
HOLM_THETA = (4.0, 8.0)        # paper: "theta (4-8 Hz)"
HOLM_ALPHA = (8.0, 12.0)       # paper: "alpha (8-12 Hz)"
HOLM_REJECT_UV = 70.0          # paper: "excluded epochs containing other artefacts (+-70 uV)"

# DEVIATION (approved): the Galea montage has no Fz. Frontal theta is taken from
# the TIME-DOMAIN mean of F1 and F2, the two 10-10 sites immediately flanking Fz.
# Sources are cited inline beside each constant.
#
# Revised 2026-08-27 after code review. This previously averaged the two channels'
# POWER, which is dominated by whichever channel is louder: the measured F2/F1
# theta ratio spans 0.11-2.58 across recordings, and on p14/task_ai_personal the
# pooled index was 41.5 against 54.4 (F1 only) and 6.3 (F2 only). Averaging the
# time-domain signals before the FFT is what a midline derivation physically is,
# and is robust to one noisy channel in a way power-averaging is not.
HOLM_FRONTAL_CHANNELS = ["F1", "F2"]
HOLM_PARIETAL_CHANNELS = ["Pz"]

# Plausibility RANGE for the channels feeding the index, evaluated on the
# PRE-interpolation signal. Added 2026-08-27 after code review found that
# detect_bad_channels is purely relative and therefore flags nothing when most of
# a montage is broken: p14/task_ai_personal shipped a full 300-window index from a
# F1 whose plain SD was 22,161 uV (robust SD 375 uV -- the two differ 59x on this
# channel because the plain SD is dominated by transients; every THRESHOLD here is
# in robust SD) with an empty bad-channel list, and was the highest "cognitive load"
# in the dataset. Uses the same run-derived band as step 5.
#
# NO LONGER A GATE (2026-09-02, at the user's request). It was one until now: a
# recording with an implausible Pz got no index at all, and one with a single
# implausible frontal channel had its numerator silently fall back from the F1+F2
# midline mean to whichever channel passed. Both behaviours are withdrawn. The
# bound had no published basis -- Holm gates on nothing of the kind -- and it cut
# both ways: it rejected genuinely-measured channels (clean electrodes on this
# headset do sit above 50 uV), while the fallback quietly changed the DERIVATION
# of the index on 6 of 22 task recordings, so a single "cognitive load" column
# held two different quantities.
#
# The measurement stays and is reported per channel; nothing is withheld for it.
# This mirrors the same removal made browser-side on 2026-09-02 -- signal quality
# is surfaced, not acted on. A consumer that wants the old strict set can still
# form it from `strict_subset` / `by_channel` in step8b_gate.
# The fixed 1-50 uV bound that stood here was REPLACED on 2026-09-03 by a bound
# derived from this dataset's own amplitude distribution. See
# AMPLITUDE_BOUND_K and derive_amplitude_bounds().
#
# Why. The 1-50 uV range had no published source: it was invented for this
# repository by the 2026-08-27 review, hardcoded, and described in a comment as
# "a plausible scalp-EEG range" with nothing behind it. It was not derived from
# electrode data of any kind, and in particular not from DRY electrodes. The
# Galea helmet is 100% dry, which is the user's point: contact impedance,
# drift and movement sensitivity are all higher than a gelled cap, so a bound
# borrowed from wet-electrode intuition is the wrong shape for this hardware.
# The dataset says so directly -- 18.6% of all channel-measurements exceeded
# 50 uV, including 40% of F1 and 39% of F2, the index channels.
#
# What replaces it is RELATIVE and computed from the run itself, so it is
# dry-electrode-appropriate by construction rather than by assertion.
#
# STILL NOT A GATE. Nothing is withheld for falling outside it, here or in the
# dashboard; it drives quality labels only. See the 2026-09-02 removal below,
# which this change does not reverse.

# Distance, in robust log-sigma, beyond which a channel's amplitude is called
# unusual for this dataset. 3.0 is the conventional outlier distance; under a
# lognormal null it would flag ~0.3% of measurements, and it flags ~15% here,
# which is a statement about how heavy this hardware's upper tail really is.
AMPLITUDE_BOUND_K = 3.0
# Fewest channel amplitudes that may be used to fit the band. Below this the run
# reports "not assessed" rather than fitting a band to a handful of points --
# a filtered run of one recording would otherwise derive a bound from 10 numbers
# and label the whole dataset against it.
AMPLITUDE_BOUND_MIN_N = 100

# Robust artifact mode: reject a window if any index channel exceeds
# ROBUST_SIGMA * 1.4826 * MAD for that channel in that recording.
#
# CORRECTION (code review, 2026-08-27; comment fixed 2026-08-28). This used to
# claim "5 sigma against a typical clean-EEG sigma of ~14 uV reproduces Holm's
# 70 uV". That claim was disproved and retracted --
# measured across the 22 task recordings the thresholds span 75.5-1876.0 uV
# (median 148.5) and only 45% fall within 2x of 70 uV. The retraction reached the
# provenance document but not this comment, which went on asserting it. Because
# the threshold rescales with each recording's own noise, `robust` measures
# WITHIN-recording relative cleanliness and is NOT comparable across recordings.
ROBUST_SIGMA = 5.0

# Windowed robust mode (added 2026-09-02, at the user's request; changed the same
# day from fixed blocks to a CENTRED SLIDING window, again at the user's request).
# The same rule as `robust`, but the MAD is recomputed from the samples in a
# window of ROBUST_SLIDE_CHOICES_S seconds CENTRED ON EACH ANALYSIS WINDOW, and
# that window is judged against it.
#
# What this changes. `robust` calibrates on the WHOLE recording, so a recording
# with one bad stretch carries an inflated threshold for its entire length -- the
# clean stretches are then judged too leniently, and the bad stretch, having
# raised the yardstick it is measured against, partly hides itself. Recomputing
# locally makes the criterion local: a drifting impedance or a stretch of
# movement is compared to its own neighbourhood rather than to a recording-wide
# average of good and bad.
#
# WHY SLIDING RATHER THAN TILED BLOCKS. The first version cut the recording into
# fixed 60 s blocks. Two defects, both intrinsic to tiling and neither fixable
# within it:
#   * A window one second before a block boundary and a window one second after
#     it were judged against completely disjoint minutes, so the threshold could
#     jump discontinuously between two adjacent windows that share almost all
#     their neighbourhood.
#   * A window at the edge of a block sat at the edge of its own calibration,
#     with 59 s of context on one side and 1 s on the other, while a window at
#     the block's centre had 30 s each side. Two windows, the same rule, very
#     different amounts of local evidence.
# A centred window gives every analysis window the same amount of context on
# both sides, and moves the threshold smoothly. It also deletes the tiling's
# whole apparatus -- the fold of a short trailing remainder, the block index, the
# clamp, the midpoint assignment rule and the "which block" bookkeeping in the
# browser -- because there is no longer a block to belong to.
#
# The cost is the same one `robust` already has, made worse. `robust` is not
# comparable BETWEEN recordings because the threshold rescales with each
# recording's noise; the windowed mode is additionally not comparable BETWEEN
# MOMENTS of one recording, because the threshold rescales again within it. A
# stretch of solid artifact raises its own threshold and can retain nearly every
# window. Read retention here as "how unusual was this window for its own
# neighbourhood", never as an absolute cleanliness measure.
#
# EDGES. A window whose centred span would run past either end of the analysed
# signal is TRUNCATED to the samples that exist, not shifted inwards to keep the
# length. Truncation keeps the calibration local and honest about it; shifting
# would silently judge the first window against a span centred somewhere else.
# On a signal longer than the window the shortest span is half the length plus
# half an epoch -- at worst 5 s at the 10 s setting, still 1,250 samples for a
# MAD. (A signal SHORTER than the window is the degenerate case: every window
# then sees all of it and the mode is plain Robust. The dashboard says so.)
#
# Lengths offered to the reader. All three are ample for a MAD at 250 Hz (2,500
# to 7,500 samples); the choice is locality against stability, not adequacy.
# The default is the longest, being the most stable and the smallest departure
# from the 60 s blocks this replaced.
ROBUST_SLIDE_CHOICES_S = [10.0, 20.0, 30.0]
ROBUST_SLIDE_DEFAULT_S = 30.0

# The lengths become archive column names, via f"{L:g}", and the browser rebuilds
# the same names with JavaScript's String(Number). The two agree for short
# decimals and diverge for long ones -- "%g" cuts to 6 significant digits and
# switches to exponent form at 1e6, String() does neither -- and a divergence
# would not fail loudly: it would look up a column that does not exist and refuse
# every windowed mask, which reads as a broken control. Checked here so a future
# edit to the list above is caught at import rather than in the browser.
for _L in ROBUST_SLIDE_CHOICES_S:
    _k = f"{_L:g}"
    if "e" in _k or "E" in _k or float(_k) != _L or len(_k.split(".")[-1]) > 6:
        raise SystemExit(
            f"ROBUST_SLIDE_CHOICES_S contains {_L!r}, which does not survive the "
            f"f'{{L:g}}' archive key format used by write_variant_archives "
            f"(it becomes {_k!r}). Use a short decimal.")
if ROBUST_SLIDE_DEFAULT_S not in ROBUST_SLIDE_CHOICES_S:
    raise SystemExit("ROBUST_SLIDE_DEFAULT_S must be one of ROBUST_SLIDE_CHOICES_S; "
                     "the dashboard can only open on a length the sweep stored.")
del _L, _k

# EOG regression degeneracy limits (added 2026-08-28 after review). A Gratton-style
# ocular propagation factor is a physical attenuation, typically 0.1-0.4, and cannot
# exceed ~1. Coefficients of 1e9-1e11 were observed on four recordings where EOG V
# and EOG H are near-collinear and the unregularised solve diverges.
EOG_MAX_COEFFICIENT = 2.0
EOG_MAX_CONDITION = 1e8

ARTIFACT_MODES = ["holm_strict", "holm_imputed", "none", "robust"]

# Ocular correction. `ica` added 2026-09-05 at the user's request as a third
# method to try, alongside doing nothing and the Gratton-style regression.
#
# READ THIS BEFORE TRUSTING AN ICA-CORRECTED NUMBER. ICA separates a recording
# into as many components as it has channels, and this montage has TEN. Ocular
# correction by ICA is normally done on 32-128 channels, where a blink lands in
# one component that is recognisably ocular and removing it costs almost no
# brain signal. With ten channels the same blink is spread across a handful of
# components, each of which also carries cortical activity, so removing "the
# blink component" necessarily removes real EEG with it.
#
# That cost falls hardest exactly where this study looks. A blink is a large
# slow frontal deflection -- and slow and frontal is what frontal midline theta
# is. So on the two frontal-theta measures and on the cognitive-load index whose
# numerator they are, ICA here is the correction most likely to remove signal
# along with artifact. It is offered so the reader can see what it does, not
# because it is the recommended setting; `none` remains the default, and the
# per-recording QC records how many components were removed and how much
# variance went with them.
OCULAR_MODES = ["none", "eog_regression", "ica"]

# Fixed so a re-run reproduces the same decomposition: FastICA starts from a
# random unmixing matrix and converges to a sign- and order-arbitrary solution,
# so without this two runs of the same data would remove different components.
ICA_RANDOM_STATE = 97
ICA_MAX_ITER = 1000
# WHICH COMPONENTS COUNT AS OCULAR.
#
# NOT MNE's default. find_bads_eog defaults to measure="zscore", threshold=3.0 --
# it z-scores the EOG correlations ACROSS COMPONENTS and flags anything past 3
# sigma. That is unusable here, and silently so. MNE's _find_outliers z-scores
# with scipy's default ddof=0, so over n values the largest attainable z-score is
# sqrt(n-1) -- exactly 3.0 at ten components, 2.6458 at eight -- and the
# comparison is a strict `>`. At ten channels the bar is therefore reachable only
# in the exact-tie limit and never actually cleared; at eight it is not even
# approached. Either way the default can NEVER flag anything on this montage, and
# the ICA mode would have shipped as a guaranteed no-op that looked like a working
# correction. Confirmed by construction:
#
#     from mne.preprocessing.bads import _find_outliers
#     _find_outliers([1,0,0,0,0,0,0,0,0,0], threshold=3.0)   # -> []
#
# and empirically on two recordings before this was changed: zero components
# removed. (Corrected 2026-09-08 after review: this comment previously gave the
# bound as (n-1)/sqrt(n) = 2.85, which is the ddof=1 figure. scipy.stats.zscore
# does not use ddof=1. The conclusion is unchanged; the margin is not -- at ten
# channels it fails by an exact tie, not by 0.15.)
#
# So the criterion is an ABSOLUTE correlation with the EOG instead, which has no
# such ceiling and says something a reader can interpret: |r| = 0.5 is a
# component sharing a quarter of its variance with the eye channels.
#
# WHY 0.5, MEASURED. The fit is per PAIR, so the quantity being thresholded is one
# score vector per group, not per recording. Over the 22 groups the largest |r| in
# each runs 0.158 to 0.683, median 0.394. At the candidate thresholds (MNE uses a
# strict >, not >=):
#
#     |r| > 0.3   14 of 22 groups   28 of 44 recordings   20 components removed
#     |r| > 0.4   11 of 22          22 of 44              14 components
#     |r| > 0.5    6 of 22          12 of 44               6 components
#     |r| > 0.6    2 of 22           4 of 44               2 components
#     |r| > 0.7    0 of 22           0 of 44               0 components
#
# CORRECTED 2026-09-08. The table this replaces read 0.159-0.875 with 32/21/14/6/3
# recordings, and it was a survey of the wrong quantity: it was measured on
# INDEPENDENT per-recording fits, before fit_ica_paired made the decomposition a
# property of the pair. A correlation dilutes roughly linearly in sample fraction,
# and the ~10:1 task-weighted join pulls the top of the distribution down hard --
# the old survey's 0.875 does not exist in what ships, and its "3 of 44 at 0.7" is
# now zero. Reproduce with scratch script r_survey.py; the per-component scores are
# NOT written to qc_steps.json, only their maximum, so this table cannot be
# rebuilt from outputs/ alone.
#
# Note what the corrected table says about the top end: above ~0.69 the mode is a
# no-op on this dataset. That is a real hazard for anyone tightening this constant,
# and it is invisible from the shipped QC.
#
# 0.5 is chosen conservatively, because removing a component is expensive on ten
# channels -- each one carries real cortical signal along with the artifact, so a
# loose threshold costs more EEG than it saves. It has NO published basis and is
# and is flagged as such wherever it is used. The per-recording maximum
# correlation is written into the QC either way, so a reader can see how close
# each recording came to the line.
ICA_EOG_R = 0.5

# ---------------------------------------------------------------------------
# Additional dashboard measures (added 2026-08-28)
#
# Four further measures are computed per 4 s window alongside the Holm index and
# shown on their own dashboard tabs. Every parameter below was put to the user and
# chosen by them. Nothing here is inferred.
#
#   fm_theta            frontal midline theta  -- theta power of the time-domain
#                       mean of F1 and F2. Always the full F1+F2 mean, and since
#                       the gate's removal (2026-09-02) identical in derivation
#                       to the index numerator, which it used to differ from
#                       wherever the gate dropped a frontal channel.
#   parietal_alpha      alpha power at Pz
#   parietal_beta_asym  ln(beta power P4) - ln(beta power P3)
#   frontal_alpha_asym  ln(alpha power F2) - ln(alpha power F1)
#
# Band edges: theta 4-8 Hz (identical in Holm and the example pipeline); alpha
# 8-13 Hz and beta 13-30 Hz, both the EXAMPLE PIPELINE's FREQ_BANDS values, chosen
# by the user. Note the consequence: `parietal_alpha` is NOT the Holm index
# denominator, which uses the paper's narrower 8-12 Hz.
# ---------------------------------------------------------------------------

NEW_THETA = (4.0, 8.0)          # user decision: unambiguous, same in both sources
NEW_ALPHA = (8.0, 13.0)         # user decision: example pipeline FREQ_BANDS "Alpha"
NEW_BETA = (13.0, 30.0)         # user decision: example pipeline FREQ_BANDS "Beta"

NEW_MEASURES = ["fm_theta", "parietal_alpha",
                "parietal_beta_asym", "frontal_alpha_asym"]

# Channels whose amplitude decides whether a window is rejected, per measure.
# Only the channels a measure actually consumes are considered.
MEASURE_MASK_CHANNELS = {
    "fm_theta": ["F1", "F2"],
    "parietal_alpha": ["Pz"],
    "parietal_beta_asym": ["P3", "P4"],
    "frontal_alpha_asym": ["F1", "F2"],
}

# User decision: the new tabs expose BOTH references as a toggle.
#   hardware -- the recording's own SRB2/earlobe reference, as the Holm index uses
#   average  -- the example pipeline's average re-reference (cell 30)
REFERENCE_MODES = ["hardware", "average"]

# User decision: the new tabs offer only these two artifact criteria.
NEW_ARTIFACT_MODES = ["none", "robust"]

# ---------------------------------------------------------------------------
# Parameter sweep (added 2026-09-01, at the user's request)
#
# Seven processing choices that were previously fixed constants are now swept by
# the pipeline and exposed as dashboard controls. They divide into two kinds, and
# the division is the whole reason the dashboard can stay a static file:
#
#   PRECOMPUTED (they change the SPECTRUM, so a value has to exist per setting):
#       interpolation (5 modes: off / automatic / manual, the latter two with
#       or without the frontal pair), ocular correction, reference, epoch
#       length, spectral estimator.   5 x 3 x 3 x 10 x 4 = 1800 selectable
#       combinations. FEWER are stored: the archive is keyed by which channels
#       were interpolated rather than by mode name, and on this dataset the five
#       modes collapse to about 2.2 distinct lists per group (interp_token).
#
#   EVALUATED IN THE BROWSER (they only threshold quantities already shipped,
#   so they are continuous sliders rather than discrete toggles):
#       robust-rejection distance, Holm-style absolute cap.
#   The pipeline ships each window's PEAK amplitude per channel, each recording's
#   own robust sigma, and (2026-09-02) that sigma recomputed from a centred
#   sliding window at each of ROBUST_SLIDE_CHOICES_S; every threshold is then a
#   comparison the page can do itself, at any value, without another pipeline run.
#   The sliding sigma is the one that costs archive space rather than nothing:
#   it is per window, per length, so each offered length is a column.
#
#   The amplitude gate was a third browser-side threshold until 2026-09-02, when
#   the gate was removed outright -- see AMPLITUDE_BOUND_K.
#
# The defaults below reproduce the pre-sweep pipeline exactly, and that identity
# is asserted at runtime -- see verify_default_variant().
# ---------------------------------------------------------------------------

EPOCH_CHOICES_S = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
DEFAULT_EPOCH_S = 4.0                      # == HOLM_WINDOW_SEC

# A window is DISCONTINUOUS when its own samples are not adjacent in real time:
# somewhere inside it, consecutive timestamps jump by more than this. Its
# spectrum is then taken across a step in the signal, which is broadband and is
# not EEG -- the filter rings across the join, and every band in that window
# picks up energy that no electrode measured.
#
# Added 2026-09-04 at the user's request. Such a window is excluded from every
# measure at every window length and under every artifact mode INCLUDING `none`,
# because "no artifact rejection" means "reject nothing on the evidence of the
# EEG" -- it does not mean "plot a window that is two recordings stitched
# together". This is a property of the WINDOW GRID, not of any criterion, which
# is why it is applied before the masks rather than as one of them.
#
# A discontinuity reaches this point one way only: the recording itself has a
# gap, where samples either side are not adjacent in real time -- a dropout, or
# a stream that stopped and restarted. Any recording carrying one is named in
# its own step 2 QC, which records the gap count and the largest.
#
# This template no longer SPLICES: the original study cut a long interval out of
# one recording by hand and treated the remainder as continuous, which created a
# discontinuity deliberately. That machinery was removed with the rest of the
# per-recording repairs, so what is left here handles recorded gaps only.
#
# WHY 0.5 s. It is not tuned, and it is not independent: it is exactly the
# threshold step2_segment already uses to call something an internal data gap
# (`max(0.5, 20 * nominal)`, 0.5 s at 250 Hz). Two thresholds for one phenomenon
# is how a recording comes to be described as having a discontinuity by one part
# of the pipeline and analysed as continuous by another, so there is one number.
#
# Nothing on this dataset is sensitive to it. Across all 77,426 windows the gap
# between a window's own first and last sample exceeds its nominal duration by a
# median of -0.002 s and a 99th percentile of +0.044 s; the next value above that
# is 3.879 s, p04's dropout. Every threshold between roughly 0.1 s and 3.8 s
# selects the same windows.
#
# NOT the same rule as MOTION_MAX_SPAN_FACTOR, and deliberately so. That one is
# relative to the window length, which means it catches p04 at 1-3 s and misses
# it at 4 s and above -- the same physical dropout included or excluded depending
# on where the reader has left the window-length slider. This is absolute.
WINDOW_MAX_GAP_SEC = 0.5

# Spectral estimators. All four are scaled identically -- one-sided PSD in
# uV^2/Hz, normalised by sfreq * sum(taper^2) with the non-DC bins doubled -- so
# switching estimator changes the shape of the estimate and nothing else.
#
#   hann        one Hann-tapered FFT over the whole window. Holm's own method
#               ("smoothed using a 2048-sample Hanning window") and the pipeline
#               default. Verified bit-for-bit against the MNE call it replaces.
#   multitaper  MNE's former default: NW = 4, the first 7 DPSS tapers, averaged.
#               Lower variance, but it smooths the spectrum over +-NW/T Hz, where
#               T is the WINDOW LENGTH IN SECONDS. That is +-1 Hz at the 4 s
#               default -- the 25% band-widening documented in PROVENANCE 3.5 --
#               but +-4 Hz at a 1 s window, which is wider than the 4 Hz bands
#               being measured. Corrected 2026-09-01 after review: the +-1 Hz
#               figure was written when the window was a constant and was carried
#               into the sweep unqualified. Measured consequence: at 1 s the
#               median Holm index is ~1.9x the Hann value (up to 5.7x on one
#               recording), essentially all leakage. The half-bandwidth is
#               reported per (window, estimator) in band_bins so the dashboard can
#               state it and warn.
#   welch       Hann-tapered half-length sub-segments at 50% overlap, averaged.
#               Lower variance again, at half the frequency resolution.
#   boxcar      no taper at all. Best frequency resolution, worst leakage; it is
#               here as the null comparison that shows what the taper is doing.
FFT_METHODS = ["hann", "multitaper", "welch", "boxcar"]
DEFAULT_FFT_METHOD = "hann"
MULTITAPER_NW = 4.0                        # time-half-bandwidth product
MULTITAPER_N_TAPERS = 7                    # 2*NW - 1, the low-bias set

# References. `hardware` and `average` are unchanged; `rest` is new.
#
#   hardware  the recording's own SRB2/earlobe reference, as recorded. The Holm
#             index is defined at this reference (PROVENANCE section 2.4).
#   average   the example pipeline's average of the ten EEG channels (cell 30).
#   rest      Reference Electrode Standardization Technique (Yao 2001): the
#             signal is projected back towards a reference at infinity through a
#             lead field. Computed with MNE's own REST implementation over a
#             three-layer spherical head model fitted to this montage. It is an
#             estimate from an assumed head model, not a measurement, and a
#             ten-electrode montage is far sparser than the >=64 channels the
#             technique is usually applied to -- read it as a sensitivity check.
SWEEP_REFERENCE_MODES = ["hardware", "average", "rest"]

# The ten EEG channels this montage carries, in the order step3_to_mne renames
# them. NOT a source of truth: assert_montage_names() checks it against the
# montage actually built from the file's own column headers on every run, so a
# helmet with a different layout fails loudly here instead of silently making
# MANUAL_INTERPOLATION unsatisfiable.
GALEA_EEG_CHANNEL_NAMES = ["F1", "F2", "C3", "C4", "P3", "P4",
                           "O1", "O2", "Cz", "Pz"]

# ---------------------------------------------------------------------------
# Bad-channel interpolation: five modes over two independent choices
# (extended 2026-09-08 at the user's request; was ["on", "off"])
# ---------------------------------------------------------------------------
#
# CHOICE 1 -- where the list of channels to interpolate comes from:
#
#   off      nothing is interpolated. A failed electrode feeds its own measure
#            directly, AND -- under the average or REST reference -- every other
#            channel as well.
#   on       the list detect_bad_channels_paired produced for this task/baseline
#            group. This is step 6 as it has always run, and it remains the
#            pipeline default and the cell verify_default_variant checks.
#   manual   a list the user wrote down per participant and arm, below. It is
#            NOT derived from the detector and does not have to agree with it:
#            on p09 and p14 the detector flags nothing where the user asks for a
#            channel, and on p03, p04, p07, p13 and p14 the user asks for
#            nothing where the detector flags several.
#
# CHOICE 2 -- what to do when BOTH F1 and F2 are on that list:
#
#   (checked, default)   interpolate them like any other channel.
#   (unchecked)          leave F1 and F2 exactly as recorded and interpolate the
#                        rest of the list, if any.
#
# Why choice 2 exists at all. F1 and F2 are the entire numerator of the
# cognitive-load index (their time-domain mean is the frontal-theta signal) and
# the two sides of frontal alpha asymmetry. Spherical-spline interpolation
# rebuilds a channel from the OTHER channels, so when both frontal sites are
# rebuilt at once, the frontal signal is reconstructed entirely from central,
# parietal and occipital electrodes -- and both sites are rebuilt from the same
# donors, which drives frontal alpha asymmetry towards zero by construction. The
# dashboard already calls that case "Fully synthetic"; this control lets a reader
# see the same recording without it, at the cost of reading two electrodes that
# failed detection as recorded.
#
# The rule fires ONLY when both are on the list. One frontal channel rebuilt
# from nine others -- including its partner -- is an ordinary interpolation, so
# toggling this changes nothing there, by design.
INTERPOLATION_SOURCES = ["off", "on", "manual"]
FRONTAL_PAIR = ("F1", "F2")

# The five (source, frontal-pair) combinations, as they are named everywhere
# downstream. `off` takes no frontal variant: there is nothing to withhold.
INTERPOLATION_MODES = ["off",
                       "on", "on_keepfrontal",
                       "manual", "manual_keepfrontal"]
DEFAULT_INTERPOLATION_MODE = "on"

# ---------------------------------------------------------------------------
# The manual interpolation list
# ---------------------------------------------------------------------------
#
# EMPTY IN THIS TEMPLATE. Fill it in with your own hand-picked channel lists, or
# leave it empty and use the `off` and `on` interpolation modes only. An empty
# dict makes the `manual` and `manual_keepfrontal` modes interpolate nothing for
# every recording, which is valid: those modes still build, and still appear on
# the dashboard, they just resolve to the same arrays as `off`.
#
# Keyed by (participant, arm) -- ("p02", "ai") -- where an arm names exactly one
# task/baseline pair for that participant. A participant who ran the same arm
# under both framings needs the finer key, the full condition
# ("none_personal", "none_speedscore"); assert_manual_interpolation() refuses an
# arm key that would match two pairs and names the conditions to use instead.
# recording_arm() and recording_condition() do the mapping.
#
# This list is a JUDGEMENT, not a measurement. Nothing here derives it, it cannot
# be reproduced from the data by any rule in this file, and it is binding on the
# result. A group that is absent gets an EMPTY list, which under `manual` means
# nothing is interpolated for it -- a positive instruction, not a gap. That is
# why assert_manual_interpolation() refuses an entry matching no discovered
# recording rather than ignoring it: a typo would otherwise ship as a result.
MANUAL_INTERPOLATION: dict[tuple[str, str], list[str]] = {}


def recording_arm(rec: dict) -> str:
    """`agent`, `ai` or `none` for one recording, from its folder name.

    `task_agent_personal` -> agent, `baseline_ai_speedscore` -> ai,
    `task_none_speedscore` -> none (the no-AI condition, added 2026-09-16 with
    participant sbx). Raises rather than guessing: an unknown arm, or a rename,
    must not silently resolve to an empty manual list, because an empty list is
    itself a meaningful instruction here (interpolate nothing) and would be
    indistinguishable from a lookup miss.
    """
    m = re.match(r"^(?:task|baseline)_(agent|ai|none)_", rec["recording"])
    if not m:
        # The arm is NOT only a MANUAL_INTERPOLATION key -- it is how the
        # condition is read for every recording, and the dashboard's own
        # condition_label() parses the same tokens out of the same name. The
        # message used to blame MANUAL_INTERPOLATION, which sends a reader to an
        # empty table the template tells them is optional.
        raise ValueError(
            f"cannot read an arm out of recording name {rec['recording']!r} "
            f"(for {rec['key']}). The name must start "
            f"'task_<arm>_<framing>' or 'baseline_<arm>_<framing>', where "
            f"<arm> is one of agent/ai/none -- that is the only place the "
            f"condition is recorded. Rename the recording folder, or change the "
            f"arm vocabulary in recording_arm() here and in condition_label() "
            f"in build_dashboard.py")
    return m.group(1)


def recording_condition(rec: dict) -> str:
    """`<arm>_<framing>` for one recording: `task_none_personal` -> none_personal.

    The finer of the two keys MANUAL_INTERPOLATION accepts. A trailing _1/_2
    file-split suffix is stripped, as in baseline_key_for_task.
    """
    recording_arm(rec)          # same refusal of an unknown arm
    return re.sub(r"_\d+$", "", rec["recording"].split("_", 1)[1])


def manual_interpolation_for(rec: dict) -> list[str]:
    """The user's hardcoded list for this recording's group. [] if not listed.

    A condition key wins over an arm key; assert_manual_interpolation refuses
    a participant carrying both for the same pair, so the order never decides.
    """
    p = rec["participant"]
    for k in ((p, recording_condition(rec)), (p, recording_arm(rec))):
        if k in MANUAL_INTERPOLATION:
            return list(MANUAL_INTERPOLATION[k])
    return []


def drop_frontal_pair(chans) -> list[str]:
    """`chans` minus F1 and F2 -- but ONLY when BOTH are present.

    This is the unchecked position of the frontal-pair checkbox. Written as one
    function used by every mode so the "only when both" rule cannot drift
    between the automatic and the manual branch: with one of the two flagged the
    return value is the input, unchanged, and the toggle is a no-op by
    construction rather than by two separate conditionals agreeing.
    """
    have = [c for c in FRONTAL_PAIR if c in chans]
    if len(have) < len(FRONTAL_PAIR):
        return list(chans)
    return [c for c in chans if c not in FRONTAL_PAIR]


def interp_lists_for(bad_chans, manual_chans) -> dict:
    """{mode: channels} for all five INTERPOLATION_MODES, for one group.

    Every list is sorted into the montage's own channel order, so two modes that
    interpolate the same set produce the same list, the same token, and
    therefore ONE stored signal rather than two identical ones. On this dataset
    that collapses 5 nominal modes to 2.2 distinct signals per group -- see
    interp_token.
    """
    order = {c: i for i, c in enumerate(GALEA_EEG_CHANNEL_NAMES)}
    def norm(chans):
        return sorted({c for c in chans}, key=lambda c: order.get(c, 99))
    auto, man = norm(bad_chans or []), norm(manual_chans or [])
    out = {
        "off": [],
        "on": auto,
        "on_keepfrontal": norm(drop_frontal_pair(auto)),
        "manual": man,
        "manual_keepfrontal": norm(drop_frontal_pair(man)),
    }
    # The mode names are spelled out above AND in INTERPOLATION_MODES. Adding a
    # sixth mode to the constant without adding it here would KeyError in the
    # middle of the sweep, having already done every recording's steps 1-5.
    if set(out) != set(INTERPOLATION_MODES):
        raise SystemExit(
            f"interp_lists_for covers {sorted(out)} but INTERPOLATION_MODES is "
            f"{sorted(INTERPOLATION_MODES)}; the two must agree exactly.")
    return out


def interp_token(chans) -> str:
    """The archive's name for "these channels were interpolated".

    The sweep is keyed by WHAT WAS INTERPOLATED, not by which mode asked for it.
    Two modes that reach the same channel list are the same signal, so they
    share one computation and one stored array; `index.json` carries the
    per-recording mode -> token map that the dashboard resolves through.

    This is not a micro-optimisation. Stored per mode, five interpolation modes
    would take the sweep from 720 cells to 1800 and the deliverable from 365 MB
    to roughly 720 MB. Stored per distinct channel list it is 2.2 signals per
    group instead of 5, and the archive shrinks rather than grows.
    The figures above were measured on the original study's dataset; they will
    differ on yours.
    """
    return "+".join(chans) if chans else "-"


def assert_montage_names(raw) -> None:
    """GALEA_EEG_CHANNEL_NAMES must be what the montage actually holds.

    MANUAL_INTERPOLATION names channels as strings, and a string that matches no
    channel would interpolate nothing while looking like an instruction that had
    been followed. Checked against the real montage once per run rather than
    trusted.
    """
    have = [raw.ch_names[i] for i in
            mne.pick_types(raw.info, eeg=True, eog=False, exclude=[])]
    if sorted(have) != sorted(GALEA_EEG_CHANNEL_NAMES):
        raise SystemExit(
            f"montage mismatch: this recording carries EEG channels {sorted(have)} "
            f"but GALEA_EEG_CHANNEL_NAMES says {sorted(GALEA_EEG_CHANNEL_NAMES)}. "
            f"MANUAL_INTERPOLATION and the frontal-pair rule are written in terms "
            f"of those names and cannot be applied to a different montage.")


def assert_manual_interpolation(recs: list[dict]) -> None:
    """Every MANUAL_INTERPOLATION entry must name a real group and real channels.

    Run once, over the recordings actually discovered, BEFORE any processing.
    A typo here is invisible at runtime -- `("p8", "ai")` or `["Fz"]` would
    simply interpolate nothing under the manual mode, which is a legitimate
    instruction for a group that is genuinely absent from the list -- so the
    typo would ship as a result rather than as an error.
    """
    groups, problems, unreadable = set(), [], []
    # (participant, arm) -> the distinct conditions under it. sbx ran `none`
    # under BOTH framings, so for that participant an arm does not name one
    # task/baseline pair and a manual entry would apply to both at once.
    conditions: dict[tuple, set] = {}
    for r in recs:
        # An EXCLUDED group is not a group any more. Without this, an entry for
        # p02/agent or p04/agent -- both dropped on 2026-09-08, in the same
        # change that added this list -- would validate cleanly and then never
        # be used, which is the exact failure this function exists to catch.
        if exclusion_reason(r):
            continue
        try:
            groups.add((r["participant"], recording_arm(r)))
            groups.add((r["participant"], recording_condition(r)))
            conditions.setdefault((r["participant"], recording_arm(r)), set()).add(
                recording_condition(r))
        except ValueError as exc:
            # REPORTED, not skipped. recording_arm's own docstring says a rename
            # must not silently resolve; swallowing it here let validation pass
            # at second zero and moved the failure into the middle of the sweep,
            # where the driver turns it into a per-recording `error:` row.
            #
            # Kept SEPARATE from `problems`: a misnamed recording is a different
            # fault with a different remedy. It breaks the condition for every
            # purpose, and has nothing to do with whether MANUAL_INTERPOLATION
            # is right -- reporting it under that heading sent the reader to a
            # table this template ships empty.
            unreadable.append(f"{r['key']!r}: {exc}")
    for (p, arm), chans in sorted(MANUAL_INTERPOLATION.items()):
        if (p, arm) not in groups:
            problems.append(
                f"({p!r}, {arm!r}) matches no discovered recording -- the manual "
                f"list for it would silently never be used")
        elif len(conditions.get((p, arm), ())) > 1:
            problems.append(
                f"({p!r}, {arm!r}) matches more than one task/baseline pair "
                f"({sorted(conditions[(p, arm)])}) -- key it by condition instead")
        elif (p, arm) in conditions and any(
                (p, c) in MANUAL_INTERPOLATION for c in conditions[(p, arm)]):
            problems.append(
                f"({p!r}, {arm!r}) and a condition key for the same pair are "
                f"both listed -- only one may say what is interpolated")
        unknown = [c for c in chans if c not in GALEA_EEG_CHANNEL_NAMES]
        if unknown:
            problems.append(f"({p!r}, {arm!r}) names unknown channel(s) {unknown}")
        if len(set(chans)) != len(chans):
            problems.append(f"({p!r}, {arm!r}) repeats a channel: {chans}")
        if len(chans) >= len(GALEA_EEG_CHANNEL_NAMES):
            problems.append(
                f"({p!r}, {arm!r}) would interpolate every channel, which leaves "
                f"the spline nothing to interpolate FROM")
    excluded = {(p, a) for (p, a) in MANUAL_INTERPOLATION
                if p in EXCLUDED_PARTICIPANTS}
    if excluded:
        problems.append(f"entries for excluded participants: {sorted(excluded)}")
    if unreadable:
        raise SystemExit(
            f"{len(unreadable)} recording name(s) do not carry a readable "
            f"condition:\n  " + "\n  ".join(unreadable))
    if problems:
        raise SystemExit("MANUAL_INTERPOLATION is inconsistent with the data:\n  "
                         + "\n  ".join(problems))

# Browser-side thresholds: (minimum, maximum, step, default).
#
# The amplitude-gate slider that stood here was removed 2026-09-02 with the gate
# itself. It is not shipped in index.json any more, and the dashboard no longer
# reads it -- a slider whose only effect was to withhold recordings has nothing
# left to do now that nothing is withheld.
ROBUST_SIGMA_SLIDER = (1.0, 12.0, 0.25, 5.0)      # default == ROBUST_SIGMA
HOLM_CAP_SLIDER_UV = (10.0, 500.0, 5.0, 70.0)     # default == HOLM_REJECT_UV

# The seven numbers stored per window per variant. Raw band powers rather than
# finished measures, so the browser forms each measure itself.
#
# `theta_f1` and `theta_f2` existed to serve the amplitude gate's single-frontal
# fallback, which was removed 2026-09-02: the index numerator is now always the
# F1+F2 midline mean, so nothing on the page reads them. They are still shipped.
# They are two float32 columns, they are the only per-electrode frontal power a
# consumer can get out of the archive, and verify_default_variant checks them --
# an assertion that was added precisely because they had been going unverified.
VARIANT_VALUE_KEYS = [
    "fm_theta",            # mean PSD 4-8 Hz of the F1/F2 time-domain mean
    "parietal_alpha",      # mean PSD 8-13 Hz at Pz   (example-pipeline alpha)
    "parietal_beta_asym",  # ln(beta P4) - ln(beta P3)
    "frontal_alpha_asym",  # ln(alpha F2) - ln(alpha F1)
    "theta_f1",            # mean PSD 4-8 Hz at F1    } per-electrode frontal
    "theta_f2",            # mean PSD 4-8 Hz at F2    } power, no longer consumed
    "alpha_holm_pz",       # mean PSD 8-12 Hz at Pz     index denominator
]

# ---------------------------------------------------------------------------
# Baseline segments (added 2026-08-28; extended to three segments 2026-09-02 at
# the user's request)
#
# Each baseline recording carries two Marker 8.0 entries 6 minutes apart. Within
# that block: 0-2 min mental math, 2-4 min eye movements, 4-6 min rest.
#
# Verified empirically: 23 of 24 baseline recordings have exactly two 8.0 markers
# spanning 359.9-366.4 s. The exception is p04/baseline_ai_speedscore, which has
# one marker; the user's rule for it is encoded in step2_segment.
#
# Until 2026-09-02 only the final two minutes were used and everything else in
# the block was discarded. The user now wants a CHOICE of reference condition, so
# three segments are swept and the dashboard subtracts whichever is selected:
#
#   rest         240-360 s  open-eye resting. The original baseline and still the
#                           default. NOT bit-identical to the pre-2026-09-02 rest
#                           numbers: the old crop was open-ended and let samples
#                           past the 360 s marker into the rest phase's own
#                           calibration. Bounding it is a correction, and it moves
#                           sigma by up to ~3.4% on one channel of one recording.
#                           See crop_to_segment.
#   math         0-120 s    mental arithmetic. An ACTIVE-TASK reference rather
#                           than a resting one: subtracting it asks "more or less
#                           loaded than deliberate mental effort", not "more or
#                           less than rest". Read the sign accordingly.
#   eyes_closed  225-240 s  the last 15 s of the eye-movement phase, which the
#                           user identifies as eyes-closed.
#
# THE EYES-CLOSED SEGMENT IS 15 SECONDS. That is 3 windows at the 4 s default and
# ONE window at 8, 9 or 10 s -- a median over a single window is that window. It
# is also the tail of a phase of deliberate eye movements, so the ocular-artifact
# risk is the highest of the three. Both facts are surfaced per panel and in the
# dashboard's own text rather than left for the reader to deduce; a 15 s baseline
# is offered because it was asked for, not because it is statistically equal to
# the other two.
#
# Bounds are seconds from the block's first marker. Each segment is cropped,
# epoched and calibrated INDEPENDENTLY -- see step10_baseline -- so a segment's
# robust threshold and window grid come from its own samples, exactly as the rest
# phase's did when it was the only segment.
BASELINE_BLOCK_SEC = 360.0
# A task's nominal length. Used to reconstruct a task whose END marker is
# missing (sbx/task_none_speedscore, user decision 2026-09-16) -- the same rule
# as the single-marker baseline above, scaled to a 20-minute block.
TASK_BLOCK_SEC = 1200.0
BASELINE_SEGMENTS = {
    "rest":        (240.0, 360.0),
    "math":        (0.0, 120.0),
    "eyes_closed": (225.0, 240.0),
}
# Display/iteration order, and the default. `rest` is first and default so the
# dashboard opens on the segment every published number was computed against.
BASELINE_SEGMENT_ORDER = ["rest", "math", "eyes_closed"]
DEFAULT_BASELINE_SEGMENT = "rest"

# BASELINE_REST_START_SEC / BASELINE_REST_END_SEC were removed on 2026-09-02 with
# the third baseline segment. They had become aliases for BASELINE_SEGMENTS["rest"]
# whose only remaining use was emitting a `baseline_rest_s` key into run_meta that
# duplicated `baseline_segments_s` -- and a second, older-looking definition of the
# rest bounds sitting beside the authoritative one is exactly the thing a reader
# picks up by mistake. The rest phase is BASELINE_SEGMENTS["rest"].

# The user chose NO amplitude gate on the new measures, so a baseline is never
# dropped for being implausible. It is instead flagged: the robust SD of the
# channels feeding each measure is reported, and the dashboard marks any baseline
# with a channel outside the derived band so a corrupt rest level cannot be
# mistaken for a real one. The bound is the SAME one step 5 uses -- there is now
# exactly one amplitude criterion in the pipeline, computed once per run by
# derive_amplitude_bounds() and applied everywhere by apply_amplitude_labels().

# ---------------------------------------------------------------------------
# Data-set specific decisions (all approved by the researcher)
# ---------------------------------------------------------------------------

TASK_MARKER = 8.0

# ---------------------------------------------------------------------------
# Recordings dropped for data quality
# ---------------------------------------------------------------------------
#
# ALL EMPTY IN THIS TEMPLATE. Every discovered recording is analysed. Fill these
# in with your own exclusions; the dashboard's excluded-recordings table prints
# whatever is here, and prints nothing while they are empty.
#
# Two levels, because exclusions come in two shapes: EXCLUDED_PARTICIPANTS drops
# every recording a participant has, EXCLUDED_RECORDINGS drops named recordings
# and keeps the rest of that participant.
#
# The string below is the default the dashboard prints in the `Reason` column.
# Give a per-entry reason instead where you have one -- the column is the only
# place a reader learns why something is missing.
EXCLUSION_REASON = "excluded for data quality"

EXCLUDED_PARTICIPANTS: list[str] = []
EXCLUDED_PARTICIPANT_REASONS: dict[str, str] = {}

# Both members of a pair must go together -- a task without its baseline cannot
# be baseline-corrected, and a baseline without its task is swept for nothing --
# so name the task AND its baseline explicitly rather than leaving the partner to
# be inferred. assert_excluded_recordings() checks that every key exists in the
# discovered data and that no pair is left half-excluded, because a key matching
# nothing excludes nothing while looking identical to a successful exclusion.
EXCLUDED_RECORDINGS: dict[str, str] = {}


def exclusion_reason(rec: dict) -> str | None:
    """Why this recording is not analysed, or None if it is."""
    if rec["participant"] in EXCLUDED_PARTICIPANTS:
        return EXCLUDED_PARTICIPANT_REASONS.get(rec["participant"],
                                                EXCLUSION_REASON)
    return EXCLUDED_RECORDINGS.get(rec["key"])


def assert_excluded_recordings(recs: list[dict]) -> None:
    """Every EXCLUDED_RECORDINGS key must name a real recording, and no pair may
    be left half-excluded.

    A key that matches nothing excludes nothing, and would look identical to a
    successful exclusion from the console. A pair excluded on one side only is
    worse: the surviving half is swept, plotted and baseline-corrected against
    a partner that no longer exists.
    """
    keys = {r["key"] for r in recs}
    problems = [f"{k!r} matches no discovered recording"
                for k in sorted(EXCLUDED_RECORDINGS) if k not in keys]
    for r in recs:
        if r["kind"] != "task" or r["key"] not in EXCLUDED_RECORDINGS:
            continue
        partner = baseline_key_for_task(r)
        if partner in keys and partner not in EXCLUDED_RECORDINGS:
            problems.append(
                f"{r['key']!r} is excluded but its baseline {partner!r} is not")
    for r in recs:
        if r["kind"] != "baseline" or r["key"] not in EXCLUDED_RECORDINGS:
            continue
        tasks = [t["key"] for t in recs if t["kind"] == "task"
                 and baseline_key_for_task(t) == r["key"]]
        kept = [t for t in tasks if t not in EXCLUDED_RECORDINGS]
        if kept:
            problems.append(
                f"{r['key']!r} is excluded but the task(s) it is the baseline "
                f"for are not: {kept}")
    if problems:
        raise SystemExit("EXCLUDED_RECORDINGS is inconsistent with the data:\n  "
                         + "\n  ".join(problems))

# Bad-channel detection: flag an EEG channel whose log10 standard deviation is
# more than BAD_CHANNEL_Z robust-z from the median of that recording's own
# channels. Replaces the example pipeline's hardcoded ['Cz', 'F2'].
BAD_CHANNEL_Z = 3.29


# ===========================================================================
# Helper functions transcribed verbatim from the example pipeline
# ===========================================================================

def make_galea_mne_montage(eeg_channel_locations, verbose: bool = False) -> mne.channels.DigMontage:
    """
    Creates a montage for the Galea EEG device.

    Verbatim from example_eeg_processing.ipynb (cell 13).
    """
    mont1020 = mne.channels.make_standard_montage('standard_1020')
    kept_channels = eeg_channel_locations.values()
    ind = [i for (i, channel) in enumerate(mont1020.ch_names) if channel in kept_channels]

    mont1020_galea = mont1020.copy()
    mont1020_galea.ch_names = [mont1020.ch_names[x] for x in ind]
    kept_channel_info = [mont1020.dig[x + 3] for x in ind]
    mont1020_galea.dig = mont1020.dig[0:3] + kept_channel_info
    if verbose:
        mont1020_galea.plot()

    return mont1020_galea


def calc_eeg_band_power(epochs: mne.Epochs,
                        f_low: float,
                        f_high: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the power spectral density (PSD) of the EEG data
    in the specified frequency range.

    Verbatim from example_eeg_processing.ipynb (cell 32),
    including the min-max normalization on the final line.
    """
    psds, freqs = epochs.compute_psd(fmin=f_low, fmax=f_high).get_data(return_freqs=True)
    # psds = 10 * np.log10(psds)
    # normalize 0-1 across all epochs
    psds = (psds - psds.min()) / (psds.max() - psds.min())
    return psds, freqs


def calc_eeg_band_power_absolute(epochs: mne.Epochs,
                                 f_low: float,
                                 f_high: float) -> tuple[np.ndarray, np.ndarray]:
    """
    DEVIATION (approved): identical to calc_eeg_band_power above, but WITHOUT the
    min-max normalization line, and using Holm's spectral estimator.

    Estimator revised 2026-08-27 after code review. This used to inherit MNE's
    default for Epochs, which is MULTITAPER. Holm specifies "a Fast Fourier
    transformation ... smoothed using a 2048-sample Hanning window". The default
    multitaper half-bandwidth here is +-1 Hz -- a 25% widening of a 4 Hz band --
    so a pure 8.5 Hz tone carrying zero energy below 8 Hz still produced a
    theta/alpha ratio of 0.34, inflating the index for low-alpha participants.
    Per-window agreement with Holm's estimator was only r = 0.44-0.74, and the
    per-window series is the primary output.

    `method="welch"` with n_per_seg == the full window and a Hann taper is exactly
    one Hann-tapered FFT per epoch, which is Holm's method.

    Holm et al. require the ratio of ABSOLUTE theta and alpha power. The example
    pipeline's min-max rescaling is applied per band, so theta and alpha would
    receive different, arbitrary scale factors and the subtraction of the minimum
    would force the lowest-theta window to exactly zero -- the resulting ratio is
    not the Holm index. The example pipeline's own outputs still use the
    normalized version above; only the index uses this one.

    Note both Holm bands are exactly 4 Hz wide (4-8 and 8-12), so the mean PSD
    over the band and the band-integrated power give an identical ratio.
    """
    n = epochs.get_data(copy=False).shape[-1]
    psds, freqs = epochs.compute_psd(
        method="welch", fmin=f_low, fmax=f_high,
        n_fft=n, n_per_seg=n, n_overlap=0, window="hann",
    ).get_data(return_freqs=True)
    return psds, freqs


def remove_outliers(data, z_thresh=3):
    """
    Verbatim from example_baseline_vs_task.ipynb (cell 12).

    NOT CALLED. The `holm_imputed` mode reproduces this function's interpolation
    idiom inline (see step8b_holm_index) rather than calling it, because it
    operates on a numpy array of window values rather than a pandas column.
    """
    z_scores = np.abs((data - data.mean()) / data.std())
    outlier_mask = z_scores >= z_thresh
    cleaned = data.mask(outlier_mask)
    cleaned = cleaned.interpolate(method='linear', axis=0, limit_direction='both')
    return cleaned


def extract_data_within_marker(data, marker_value):
    """
    Verbatim from example_baseline_vs_task.ipynb (cell 8).

    NOT CALLED. Retained so the example pipeline's own logic is visible for
    comparison. `step2_segment` re-implements it positionally with `.iloc`
    because three recordings need marker repairs this function cannot express
    (it always takes the first two markers). The two were verified behaviourally
    identical for the ordinary two-marker case.
    """
    marker_indices = data.index[data['Marker'] == marker_value].tolist()
    if len(marker_indices) < 2:
        raise ValueError(f"Not enough markers with value {marker_value} found in the data.")
    start_index = marker_indices[0]
    end_index = marker_indices[1]
    mask = (data.index >= start_index) & (data.index <= end_index)
    return data[mask].reset_index(drop=True)


# ===========================================================================
# Recording discovery
# ===========================================================================

def discover_recordings() -> list[dict]:
    """Find every recording folder and its exg/aux files."""
    recs = []
    for exg in sorted(DATA_DIR.glob("*/*/openbci-raw-exg_*.txt")):
        folder = exg.parent
        session = folder.parent.name
        # Any id up to the first hyphen, not just `p<digits>` -- a non-numeric
        # participant id is a participant id too.
        m = re.match(r"galea_session_([^-]+)-", session)
        if not m:
            # This used to be `.group(1)` on the match directly, so a folder
            # named anything else died on `'NoneType' object has no attribute
            # 'group'` without naming the folder or the pattern. It is the first
            # thing a new dataset trips over, so it gets a real message.
            raise SystemExit(
                f"session folder {session!r} does not match the expected name "
                f"'galea_session_<participant>-<anything>'. The participant id "
                f"is read from between 'galea_session_' and the first hyphen, "
                f"so it cannot itself contain one. Rename the folder, or change "
                f"this pattern in discover_recordings().\n"
                f"  full path: {folder.parent}")
        pid = m.group(1)
        stamp = exg.name.replace("openbci-raw-exg_", "").replace(".txt", "")
        recs.append(dict(
            participant=pid,
            session=session,
            recording=folder.name,
            kind="task" if folder.name.startswith("task") else "baseline",
            exg_file=str(exg),
            aux_file=str(folder / f"openbci-raw-aux_{stamp}.txt"),
            packet_loss_file=str(folder / f"openbci-packet-loss_{stamp}.txt"),
            key=f"{pid}/{folder.name}",
        ))
    return recs


# ===========================================================================
# STEP 1 - Read files
#   example_eeg_processing.ipynb cell 11:  pd.read_csv(exg_txt_file, skiprows=4)
# ===========================================================================

def step1_load(rec: dict, qc: dict) -> pd.DataFrame:
    exg_data = pd.read_csv(rec["exg_file"], skiprows=4)

    ts = exg_data["Timestamp"].to_numpy(float)
    eeg_cols = [c for c in exg_data.columns[1:19] if "EEG" in c]
    X = exg_data[eeg_cols].to_numpy(float)

    duration = float(ts[-1] - ts[0])
    n_nan = int(np.isnan(X).sum())
    n_backwards = int((np.diff(ts) <= 0).sum())
    max_gap = float(np.max(np.diff(ts)))

    # packet loss log: header is 7 lines, any further line is a loss event
    try:
        pl_lines = Path(rec["packet_loss_file"]).read_text(errors="ignore").splitlines()
        n_loss_events = max(0, len([l for l in pl_lines[7:] if l.strip()]))
    except OSError:
        n_loss_events = None

    qc["step1_load"] = dict(
        n_samples=int(len(exg_data)),
        duration_s=round(duration, 1),
        effective_sample_rate_hz=round(len(exg_data) / duration, 2),
        n_nan=n_nan,
        n_nonmonotonic_timestamps=n_backwards,
        max_timestamp_gap_s=round(max_gap, 3),
        n_packet_loss_events=n_loss_events,
        clean=bool(n_nan == 0
                   and abs(len(exg_data) / duration - GALEA_SAMPLING_RATE) < 2.5
                   and max_gap < 1.0
                   and (n_loss_events or 0) == 0),
        notes=[],
    )
    q = qc["step1_load"]
    if n_nan:
        q["notes"].append(f"{n_nan} NaN samples in EEG channels")
    if max_gap >= 1.0:
        q["notes"].append(f"{max_gap:.2f}s timestamp gap (data dropout)")
    if (n_loss_events or 0) > 0:
        q["notes"].append(f"{n_loss_events} packet-loss events logged")
    if n_backwards:
        q["notes"].append(f"{n_backwards} non-monotonic timestamps (device/PC clock jitter)")
    return exg_data


# ===========================================================================
# STEP 2 - Segment the task using Marker 8.0
#   User-specified deviation: this study marks the task with two Marker 8.0
#   entries 20 minutes apart, rather than the example pipeline's marker 1
#   (task) / marker 2 (baseline) scheme.
# ===========================================================================

def step2_segment(rec: dict, exg_data: pd.DataFrame, qc: dict):
    marker = exg_data["Marker"].to_numpy(float)
    ts = exg_data["Timestamp"].to_numpy(float)
    mi = np.flatnonzero(marker == TASK_MARKER)

    info = dict(n_markers=int(len(mi)), repair=None, notes=[])

    if len(mi) < 2:
        # A recording with a single marker is still usable IF it runs a full block
        # past that marker -- the END marker is the missing one, and the block is
        # reconstructed as marker -> marker + BASELINE_BLOCK_SEC for a baseline,
        # or + TASK_BLOCK_SEC for a task. Otherwise it is dropped.
        #
        # This is the one marker defect the template still repairs, because it
        # needs no per-recording knowledge: the rule reads the same for every
        # recording. Delete this branch if you would rather a short-marked
        # recording were dropped outright.
        tail = float(ts[-1] - ts[mi[0]]) if len(mi) == 1 else 0.0
        block = BASELINE_BLOCK_SEC if rec["kind"] == "baseline" else TASK_BLOCK_SEC
        if len(mi) == 1 and tail >= block:
            a = int(mi[0])
            b = int(np.searchsorted(ts, ts[a] + block, side="right") - 1)
            seg = exg_data.iloc[a:b + 1].reset_index(drop=True)
            info["repair"] = (
                f"only 1 marker, but {tail:.1f}s of recording follows it; block "
                f"reconstructed as marker -> marker + {block:.0f}s")
            info["notes"].append(
                "block end is inferred from elapsed time, not from a second marker")
        else:
            info.update(clean=False, segment_duration_s=None)
            info["notes"].append(
                f"only {len(mi)} Marker {TASK_MARKER} entries - cannot delimit the segment")
            qc["step2_segment"] = info
            return None
    else:
        # The first two markers delimit the block. A recording carrying MORE than
        # two falls here as well and the extras are ignored -- this template
        # assumes recordings are free of false starts and double presses. If
        # yours are not, this is the branch to change; the excision machinery
        # that used to sit below it has been removed.
        a, b = mi[0], mi[1]
        seg = exg_data.iloc[a:b + 1].reset_index(drop=True)

    seg_dur = len(seg) / GALEA_SAMPLING_RATE

    # Gaps INSIDE the segment. Step 1 checks the whole recording, but the segment
    # was previously built from sample indices alone and never re-checked, so
    # p04/task_agent_personal's 3.91 s dropout sat inside a segment scored clean.
    # A gap means the sample count overstates elapsed time and the window grid is
    # not contiguous; window times are now taken from the real timestamps.
    seg_ts = seg["Timestamp"].to_numpy(float)
    d = np.diff(seg_ts)
    nominal = 1.0 / GALEA_SAMPLING_RATE
    gaps = d[d > max(0.5, 20 * nominal)]
    elapsed = float(seg_ts[-1] - seg_ts[0])

    info.update(
        marker_span_s=round(float(ts[b] - ts[a]), 1),
        segment_duration_s=round(seg_dur, 1),
        segment_elapsed_s=round(elapsed, 1),
        segment_n_samples=int(len(seg)),
        n_internal_gaps=int(len(gaps)),
        max_internal_gap_s=round(float(gaps.max()), 3) if len(gaps) else 0.0,
    )
    if rec["kind"] == "task":
        info["clean"] = bool(abs(seg_dur - 1200.0) < 60.0 and len(gaps) == 0)
        if abs(seg_dur - 1200.0) >= 60.0:
            info["notes"].append(
                f"task segment is {seg_dur / 60:.2f} min, expected ~20 min")
        if len(gaps):
            info["notes"].append(
                f"{len(gaps)} internal data gap(s), largest {gaps.max():.2f}s: "
                f"{elapsed - seg_dur:.1f}s of the task was never recorded, and the "
                f"filter rings across the discontinuity")
    else:
        info["clean"] = bool(len(gaps) == 0)

    qc["step2_segment"] = info
    return seg


# ===========================================================================
# STEP 3 - Convert to MNE
#   example_eeg_processing.ipynb cell 14
# ===========================================================================

def step3_to_mne(seg: pd.DataFrame, qc: dict) -> mne.io.RawArray:
    exg_cols = seg.columns[1:19].to_list()
    eeg_channel_locations = {s: s.split(" - ", 1)[1] for s in exg_cols if "EEG" in s}

    mne_info = mne.create_info(ch_names=exg_cols,
                               sfreq=GALEA_SAMPLING_RATE,
                               ch_types=CHANNEL_TYPE_LABELS)

    exg_data_trimmed = seg[exg_cols]
    galea_mne = mne.io.RawArray(exg_data_trimmed.values.T, mne_info)
    galea_mne.rename_channels(eeg_channel_locations)
    galea_mne = galea_mne.set_montage(make_galea_mne_montage(eeg_channel_locations))

    eeg_names = [galea_mne.ch_names[i] for i in
                 mne.pick_types(galea_mne.info, eeg=True, emg=False, eog=False)]
    X = galea_mne.get_data(picks="eeg")
    flat = [n for n, x in zip(eeg_names, X) if float(np.std(x)) == 0.0]

    qc["step3_to_mne"] = dict(
        n_channels=len(exg_cols),
        eeg_channels=eeg_names,
        montage_positions_found=int(len(make_galea_mne_montage(eeg_channel_locations).ch_names)),
        flat_channels=flat,
        clean=bool(len(exg_cols) == 18 and len(eeg_names) == 10 and not flat),
        notes=([f"flat (zero-variance) channels: {flat}"] if flat else []),
    )
    return galea_mne


# ===========================================================================
# STEP 4 - Notch filter out line noise
#   example_eeg_processing.ipynb cell 22
# ===========================================================================

def _band_power_at(raw, picks, f_lo, f_hi):
    psd = raw.compute_psd(picks=picks, fmin=f_lo, fmax=f_hi)
    return float(np.mean(psd.get_data()))


def step4_notch(galea_mne: mne.io.RawArray, qc: dict) -> mne.io.RawArray:
    eeg_chans = mne.pick_types(galea_mne.info, eeg=True, emg=False, eog=False)

    freqs = []
    freq = LINE_FREQ
    while freq < galea_mne.info['sfreq'] / 2:
        freqs.append(freq)
        freq += freq

    before = _band_power_at(galea_mne, eeg_chans, 59.0, 61.0)
    filtered_galea_mne = galea_mne.copy().notch_filter(freqs=freqs, picks=eeg_chans)
    after = _band_power_at(filtered_galea_mne, eeg_chans, 59.0, 61.0)

    ratio = float(after / before) if before > 0 else np.nan
    qc["step4_notch"] = dict(
        notch_freqs_hz=freqs,
        line_power_before=float(before),
        line_power_after=float(after),
        line_power_remaining_frac=round(ratio, 5),
        clean=bool(np.isfinite(ratio) and ratio < 0.5),
        notes=([] if (np.isfinite(ratio) and ratio < 0.5)
               else ["60 Hz power not meaningfully reduced by the notch filter"]),
    )
    return filtered_galea_mne


# ===========================================================================
# STEP 5 - Bandpass filter
#   example_eeg_processing.ipynb cell 24
# ===========================================================================

def step5_bandpass(filtered_galea_mne: mne.io.RawArray, qc: dict) -> mne.io.RawArray:
    eeg_chans = mne.pick_types(filtered_galea_mne.info, eeg=True, emg=False, eog=False)
    bandpass_data = filtered_galea_mne.copy().filter(l_freq=LOWCUT,
                                                     h_freq=HIGHCUT,
                                                     picks=eeg_chans)

    names = [bandpass_data.ch_names[i] for i in eeg_chans]
    X = bandpass_data.get_data(picks=eeg_chans)

    per_channel = {}
    for n, x in zip(names, X):
        mad = float(np.median(np.abs(x - np.median(x))))
        per_channel[n] = dict(
            sd_uv=round(float(np.std(x)), 2),
            robust_sd_uv=round(1.4826 * mad, 2),
            pct_over_70uv=round(100.0 * float(np.mean(np.abs(x) > HOLM_REJECT_UV)), 2),
            peak_to_peak_uv=round(float(np.ptp(x)), 1),
        )

    # Whether a channel's amplitude is ORDINARY FOR THIS DATASET cannot be decided
    # here: the band is derived from the whole run (derive_amplitude_bounds), which
    # does not exist until every recording is processed. The measurement is made
    # here; apply_amplitude_labels fills in the verdict afterwards. The placeholders
    # below are what a consumer sees if that pass never runs -- deliberately "not
    # assessed", never a silent pass.
    qc["step5_bandpass"] = dict(
        lowcut_hz=LOWCUT, highcut_hz=HIGHCUT,
        per_channel=per_channel,
        amplitude_bound_uv=None,
        n_channels_physiological=None,
        channels_physiological=[],
        clean=False,
        notes=["amplitude labels not yet assigned"],
    )
    return bandpass_data


# ===========================================================================
# STEP 5b - Ocular artifact handling
#   Holm: "corrected for eye movement artefacts with the ocular artefact
#   reduction (OAR) utility". The example pipeline does none.
#   DEVIATION (approved): both are produced, selectable on the dashboard.
# ===========================================================================

def _ocular_prep(bandpass_data: mne.io.RawArray) -> mne.io.RawArray:
    """A copy with the EOG channels made usable and the reference declared.

    Shared by every correction that reads the EOG, so the regression and the ICA
    are fed identical inputs and the toggle compares two methods rather than two
    preparations.
    """
    out = bandpass_data.copy()

    # The example pipeline filters only picks=eeg_chans, so the EOG channels are
    # still raw and carry a DC offset in the tens of thousands of uV. An
    # unfiltered regressor would make the regression meaningless, so the EOG
    # channels get the same notch + bandpass as the EEG before they are used.
    eog_picks = mne.pick_types(out.info, eeg=False, eog=True)
    freqs = []
    freq = LINE_FREQ
    while freq < out.info['sfreq'] / 2:
        freqs.append(freq)
        freq += freq
    out = out.notch_filter(freqs=freqs, picks=eog_picks)
    out = out.filter(l_freq=LOWCUT, h_freq=HIGHCUT, picks=eog_picks)

    # mne's EOGRegression refuses to run unless a reference is declared. This is
    # the example pipeline's own line (example_eeg_processing.ipynb cell 30):
    # it marks the existing SRB2/earlobe reference as applied WITHOUT touching
    # the data, so the hardware reference the index needs is preserved.
    out.set_eeg_reference([])
    return out


def _step5b_ica(bandpass_data: mne.io.RawArray, qc: dict,
                ica_pair: tuple | None = None) -> mne.io.RawArray:
    """Remove the ICA components that correlate with the EOG channels.

    Fitted on the PRE-INTERPOLATION EEG, which is where the toggle sits in the
    chain: interpolating first would feed ICA a rank-deficient signal (a spline
    channel is an exact linear combination of the others), and FastICA on a
    rank-deficient input converges to components that are partly an artefact of
    the interpolation rather than of the recording.

    Ten channels means ten components at most -- see OCULAR_MODES for why that
    matters more here than the method's usual reputation suggests. Everything
    this function can measure about the cost is recorded: how many components
    were removed, which, their EOG correlation, and the share of total variance
    they carried.
    """
    out = _ocular_prep(bandpass_data)
    eeg_names = [out.ch_names[i] for i in
                 mne.pick_types(out.info, eeg=True, eog=False, exclude=[])]
    sd_before = out.get_data(picks="eeg").std(axis=1)
    n_eeg = len(eeg_names)

    _prov0 = (ica_pair[1] if ica_pair else None) or {}
    notes = [f"FastICA on {n_eeg} EEG channels, seeded at {ICA_RANDOM_STATE} so "
             f"the decomposition is reproducible",
             ("fitted ONCE for this recording and the other half of its pair, on "
              "the task joined to the resting phase of its baseline, so both "
              "sides of a baseline-corrected value lose the same components")
             if _prov0.get("paired") else
             (f"NOT fitted on a pair: {_prov0.get('fallback')}"
              if _prov0.get("fallback") else
              "pairing state unknown for this recording"),
             f"a component is removed when its time course correlates with "
             f"either EOG channel at |r| > {ICA_EOG_R}. NOT MNE's z-score "
             f"default, which cannot flag anything on a montage this small -- "
             f"see ICA_EOG_R",
             "the correlation find_bads_eog computes is band-limited to "
             "1-10 Hz, not broadband, so it measures agreement over the range "
             "blinks and saccades occupy",
             f"ONLY {n_eeg} COMPONENTS EXIST. Ocular ICA is normally run on 32 or "
             f"more channels; on ten, a blink is spread across several components "
             f"that also carry cortical activity, so removing it removes real EEG "
             f"with it -- most consequentially the slow frontal activity that IS "
             f"frontal midline theta"]

    # The decomposition is NOT fitted here. fit_ica_paired fits it once on this
    # recording joined to the other half of its pair, so a task and the baseline
    # it is divided by lose the SAME components -- see that function for what
    # independent fits did to the log ratios.
    ica, prov = (ica_pair if ica_pair else (None, dict(
        fallback="no paired decomposition was supplied to this recording")))
    failed = (prov or {}).get("failed")
    if ica is None and not failed:
        failed = (prov or {}).get("fallback") or "no decomposition available"

    excl, scores, converged = [], None, None
    var_pct = mne_var_pct = None
    if ica is not None:
        excl = list(ica.exclude)
        converged = prov.get("converged")
        mx = prov.get("max_abs_eog_correlation")
        scores = np.array([mx], dtype=float) if mx is not None else None

    if failed is not None:
        # A failed decomposition returns the signal untouched rather than a
        # half-corrected one, and says so. Silently handing back the input under
        # a label that claims a correction was applied is the failure mode this
        # branch exists to avoid.
        # Same key set as the success record, so a consumer tabulating these
        # dicts gets a row of Nones rather than a KeyError or a ragged row.
        qc.setdefault("step5b_ocular", {})["ica"] = dict(
            applied=False, failed=True, reason=failed,
            fitted_on=_prov0.get("members"), paired=_prov0.get("paired"),
            fit_fallback=_prov0.get("fallback"),
            n_components_total=None, n_components_removed=0,
            components_removed=[], eog_threshold_r=ICA_EOG_R,
            max_abs_eog_correlation=None, variance_removed_pct=None,
            sd_reduction_by_channel_pct=None,
            converged=None, n_iter=None, mean_sd_reduction_pct=None,
            clean=False,
            notes=notes + [f"ICA FAILED and nothing was removed: {failed}. This "
                           f"recording's `ica` series is identical to `none`."])
        # `out`, not the raw input: _ocular_prep filtered the EOG channels, and
        # handing back the unfiltered ones would give this recording different
        # downstream inputs from every other. The EEG is identical either way.
        return out

    ica.exclude = list(excl)
    var_pct = None
    if excl:
        # MNE's own measure. The hand-rolled version this replaces summed the
        # variance of ica.get_sources(), which MNE WHITENS TO UNIT VARIANCE -- so
        # it reduced identically to n_removed/n_components and reported exactly
        # 10.0 for every recording that lost one of ten components. Measured
        # against the truth it was wrong by up to 1700x: the real EEG-variance
        # share of a single removed component ranges from 0.006% to 30.3% across
        # this dataset. A constant published as a measurement.
        try:
            mne_var_pct = round(100.0 * float(
                ica.get_explained_variance_ratio(
                    out, components=list(excl), ch_type="eeg")["eeg"]), 4)
        except Exception:                                      # pragma: no cover
            mne_var_pct = None
        out = ica.apply(out, verbose=False)

    sd_after = out.get_data(picks="eeg").std(axis=1)
    # MEASURED, not inferred: the total EEG variance on both sides of apply().
    #
    # This number is SIGNED, and the sign is the point. ica.apply is an oblique
    # projection, so removing a component is not guaranteed to remove variance:
    # where one half of a pair carries a broken electrode and the other does not,
    # projecting out the broken half's pattern INJECTS a scaled copy of it into
    # the quiet half. Negative here means exactly that, and it is how the two
    # cases were found -- p02/baseline_agent_personal reads -773% (its total
    # variance goes up 8.7x) and p09/baseline_agent_personal -8.5%. Do not clamp
    # this at zero and do not take abs(): a reader who sees a positive number is
    # entitled to read it as "this much was taken away".
    #
    # MNE's own get_explained_variance_ratio is kept beside it as
    # mne_explained_variance_ratio_pct rather than used, because it disagrees by
    # more than sign -- 41.4% against -0.60% on p14/baseline_agent_speedscore --
    # and it was returning -0.111% on p03/baseline_ai_personal, a recording where
    # the direct measurement says a genuine (tiny) +0.011% was removed. The
    # earlier field was worse than either: it summed the variance of
    # ica.get_sources(), which MNE whitens to unit variance, so it reduced
    # algebraically to n_removed/n_components and printed exactly 10.0 every time.
    if excl:
        v0, v1 = float(np.sum(sd_before ** 2)), float(np.sum(sd_after ** 2))
        var_pct = round(100.0 * (1.0 - v1 / v0), 4) if v0 > 0 else None
    # PER CHANNEL, because the mean is over relative reductions and is dominated
    # by the quietest electrodes. On p03/task_ai_personal the mean reads 19.6%
    # while the actual damage is C3 -63% and P3 -55% with the rest near zero --
    # a single index channel being gutted is invisible behind the average.
    with np.errstate(divide="ignore", invalid="ignore"):
        red = 100.0 * (sd_before - sd_after) / np.where(sd_before > 0,
                                                        sd_before, np.nan)
    sd_by_ch = {n: (round(float(v), 1) if np.isfinite(v) else None)
                for n, v in zip(eeg_names, red)}
    mean_red = float(np.nanmean(red)) if np.any(np.isfinite(red)) else None
    if not excl:
        top = (f"{float(np.max(scores)):.3f}"
               if scores is not None and scores.size else "unknown")
        notes.append(f"no component reached |r| = {ICA_EOG_R} (the strongest was "
                     f"{top}), so nothing was removed and this series is "
                     f"identical to `none` for this recording")
    if converged is False:
        notes.append(f"FastICA did not converge within {ICA_MAX_ITER} iterations; "
                     f"the decomposition is whatever it had reached")

    qc.setdefault("step5b_ocular", {})["ica"] = dict(
        applied=bool(excl), failed=False, reason=None,
        fitted_on=(prov or {}).get("members"),
        paired=(prov or {}).get("paired"),
        fit_fallback=(prov or {}).get("fallback"),
        n_components_total=int(ica.n_components_),
        n_components_removed=len(excl),
        components_removed=[int(i) for i in excl],
        eog_threshold_r=ICA_EOG_R,
        max_abs_eog_correlation=(round(float(np.max(scores)), 4)
                                 if scores is not None and scores.size else None),
        variance_removed_pct=var_pct,
        mne_explained_variance_ratio_pct=mne_var_pct,
        # The headline percentage is a share of TOTAL variance, and on a montage
        # with a broken frontal pair that total is mostly the broken pair: on
        # p03/task_ai_personal F1 and F2 hold 99.9% of it at 2600-3300 uV, so the
        # removal reads as 0.014% while P3 -- which feeds parietal beta asymmetry
        # -- loses 58% of its amplitude. Per channel is the honest view and the
        # only one that shows that.
        sd_reduction_by_channel_pct=sd_by_ch,
        converged=converged,
        n_iter=(int(ica.n_iter_) if getattr(ica, "n_iter_", None) is not None
                else None),
        mean_sd_reduction_pct=(round(mean_red, 2) if mean_red is not None
                               else None),
        # `clean` is about whether the STEP behaved, not whether the result is
        # trustworthy -- the ten-channel caveat above applies either way.
        clean=bool(converged is not False),
        notes=notes,
    )
    return out


def fit_ica_paired(task_preps: list[dict], base_prep: dict | None):
    """ONE ICA decomposition for a task and the baseline it is measured against.

    Added 2026-09-05, on the same reasoning as detect_bad_channels_paired and
    after review found the same defect: fitting separately gave the two sides of
    every log ratio DIFFERENT components to lose. Re-measured over all 22 pairs
    on 2026-09-08 by a separate survey (the first estimate here was "four of
    nine sampled"):

        12 of 22 pairs would have been corrected on ONE SIDE ONLY.

    The damage is concentrated rather than diffuse. Worst asymmetric cases, all
    with the partner losing nothing at all:

        p10/baseline_agent_personal   F2  -85.7%
        p15/baseline_ai_personal      F1  -78.1%   (2 components)
        p14/task_agent_speedscore     P3  -73.7%
        p13/task_agent_speedscore     C3  -68.7%
        p03/baseline_agent_speedscore F1  -67.7%

    F1 and F2 are the frontal-theta numerator; a 68% amplitude cut is a ten-fold
    cut in that channel's power, on one side only, of a ratio whose whole group
    effect is 0.45. Independent fits reproduce, by a different route, exactly the
    artifact the paired bad-channel rule was written to remove.

    So: concatenate the task with the RESTING phase of its baseline, fit one
    decomposition on the join, choose the ocular components once, and hand the
    fitted object to both recordings. Each is prepared by _ocular_prep BEFORE
    the join so the EOG filter never rings across the seam.

    HOW THE JOIN IS WEIGHTED, AND WHAT THAT COSTS. The task brings ~1200 s and
    the rest crop 120 s, so the fit is about 10:1 task-weighted and "paired" is
    in practice "the task's decision, imposed on the baseline". Unlike the SD
    criterion in detect_bad_channels_paired -- where a rest-only fault is
    diluted by a sqrt law and a 10x fault still clears the threshold -- a
    CORRELATION dilutes roughly LINEARLY in the sample fraction, which is
    harsher. Measured over all 22 pairs, solo fits against the shipped paired
    figure:

        |paired - TASK's solo value|      median 0.006, max 0.111
        |paired - BASELINE's solo value|  median 0.220, max 0.506

    So the paired number IS the task's number to three decimal places, except on
    p07/task_agent_speedscore where the join pulls 0.581 down to 0.470 and the
    task loses a correction it would have had alone. Read the other way: 13 of 22
    BASELINES have their own verdict overridden, against 1 of 22 tasks. Examples
    at both extremes --

        p03/baseline_agent_speedscore  solo 0.785 (corrected) -> paired 0.298 (not)
        p10/baseline_agent_personal    solo 0.811 (corrected) -> paired 0.423 (not)
        p13/baseline_agent_speedscore  solo 0.300 (not)       -> paired 0.573 (corrected)

    It is NOT a one-way dilution: 11 pairs land below both solo values and 1 above
    both, because the joint decomposition is a different decomposition and not a
    weighted average of two. What is reliable is that the task dominates it.

    The consequence for a reader is that `max_abs_eog_correlation` on a
    per-recording card is a PAIR-level number. A baseline card reading "the
    strongest was 0.298" is not a statement about that baseline alone -- its own
    signal reaches 0.785.

    This is the price of symmetry, and symmetry is worth more: independent fits
    would put an 86% amplitude cut on one side of a ratio and nothing on the
    other, on 12 of the 22 pairs.

    Returns (ica, provenance). `ica` is None when there is nothing to fit or the
    fit failed, and the provenance says which -- callers then leave the signal
    untouched rather than half-corrected.
    """
    prov = dict(rule="paired", segment=PAIRED_DETECT_BASELINE_SEGMENT,
                fallback=None, members=[], failed=None)

    # crop_to_segment will return 240-360 s of TASK samples labelled as rest if
    # handed a task, so the same guard detect_bad_channels_paired carries.
    if base_prep is not None and base_prep["rec"]["kind"] != "baseline":
        raise ValueError(
            f"fit_ica_paired was given a non-baseline as the baseline of a "
            f"group: {base_prep['rec']['key']}")

    ready = [p for p in task_preps if p.get("ready")]
    base_rest = None
    if base_prep is not None and base_prep.get("ready"):
        base_rest, _ = crop_to_segment(base_prep["bandpassed"],
                                       base_prep["seg_times"],
                                       PAIRED_DETECT_BASELINE_SEGMENT)
    if base_rest is None:
        prov["fallback"] = ("no usable baseline rest segment; ICA was fitted on "
                            "the task(s) alone")
    if not ready and base_rest is not None:
        prov["fallback"] = ("no usable task; ICA was fitted on the baseline's "
                            "rest segment alone")

    parts = ([_ocular_prep(p["bandpassed"]) for p in ready]
             + ([_ocular_prep(base_rest)] if base_rest is not None else []))
    if not parts and base_prep is not None and base_prep.get("ready"):
        # A READY baseline whose rest crop failed, with no usable task. Saying
        # "nothing reached step 5" here would be false, and _step5b_ica promotes
        # that string to `failed`, so the page would report an ICA failure on a
        # recording that was merely uncroppable. Same fallback, and same reason,
        # as detect_bad_channels_paired.
        parts = [_ocular_prep(base_prep["bandpassed"])]
        prov["rule"] = "whole_block"
        prov["segment"] = None
        prov["fallback"] = (
            f"no usable task and the baseline block is too short to contain the "
            f"{PAIRED_DETECT_BASELINE_SEGMENT} segment; ICA was fitted on the "
            f"WHOLE baseline block, which includes the deliberate eye-movement "
            f"phase")
    if not parts:
        prov["fallback"] = "nothing in this group reached step 5"
        return None, prov

    prov["members"] = ([p["rec"]["key"] for p in ready]
                       + ([base_prep["rec"]["key"]] if base_rest is not None
                          else []))
    prov["paired"] = bool(ready) and base_rest is not None

    joined = (parts[0] if len(parts) == 1
              else mne.concatenate_raws([p.copy() for p in parts], verbose=False))
    eeg_names = [joined.ch_names[i] for i in
                 mne.pick_types(joined.info, eeg=True, eog=False, exclude=[])]

    ica = mne.preprocessing.ICA(n_components=len(eeg_names), method="fastica",
                                random_state=ICA_RANDOM_STATE,
                                max_iter=ICA_MAX_ITER, verbose=False)
    try:
        ica.fit(joined, picks="eeg", verbose=False)
        excl, sc = ica.find_bads_eog(joined, threshold=ICA_EOG_R,
                                     measure="correlation", verbose=False)
    except Exception as exc:                                   # pragma: no cover
        prov["failed"] = f"{type(exc).__name__}: {exc}"
        return None, prov

    scores = np.abs(np.asarray(sc, dtype=float))
    if scores.ndim > 1:
        scores = scores.max(axis=0)
    ica.exclude = list(excl)
    prov.update(
        n_samples_joined=int(joined.n_times),
        n_components_total=int(ica.n_components_),
        components_removed=[int(i) for i in excl],
        n_components_removed=len(excl),
        eog_threshold_r=ICA_EOG_R,
        max_abs_eog_correlation=(round(float(np.max(scores)), 4)
                                 if scores.size else None),
        converged=bool(getattr(ica, "n_iter_", ICA_MAX_ITER) < ICA_MAX_ITER),
        n_iter=(int(ica.n_iter_) if getattr(ica, "n_iter_", None) is not None
                else None),
    )
    return ica, prov


def step5b_ocular(bandpass_data: mne.io.RawArray, mode: str, qc: dict,
                  ica_pair: tuple | None = None) -> mne.io.RawArray:
    if mode == "none":
        qc.setdefault("step5b_ocular", {})[mode] = dict(
            applied=False, clean=True,
            notes=["example pipeline applies no ocular correction; frontal theta "
                   "retains blink/eye-movement contamination"])
        return bandpass_data

    if mode == "ica":
        return _step5b_ica(bandpass_data, qc, ica_pair)

    out = _ocular_prep(bandpass_data)

    sd_before = out.get_data(picks="eeg").std(axis=1)

    # Condition of the EOG regressor pair, checked BEFORE fitting. Added
    # 2026-08-28 after review: on four recordings the unregularised least-squares
    # solution diverged, with max|coefficient| reaching 1.4e11 on
    # p02/baseline_ai_speedscore against a physically plausible 0.1-0.4, and a
    # near-zero SD reduction alongside it -- the signature of two enormous terms
    # almost cancelling. A Gratton-style propagation factor is an attenuation and
    # cannot exceed ~1; anything larger means EOG V and EOG H are near-collinear
    # and the fit is numerically meaningless.
    E = out.get_data(picks="eog")
    cond = float(np.linalg.cond(E @ E.T)) if E.shape[0] > 1 else float("inf")

    model = mne.preprocessing.EOGRegression(picks="eeg", picks_artifact="eog").fit(out)
    out = model.apply(out)
    sd_after = out.get_data(picks="eeg").std(axis=1)
    coefs = np.asarray(model.coef_, dtype=float)
    max_coef = float(np.max(np.abs(coefs)))
    degenerate = bool(not np.isfinite(cond) or cond > EOG_MAX_CONDITION
                      or max_coef > EOG_MAX_COEFFICIENT)

    notes = ["EOG V/H regressed out of the EEG channels "
             "(Gratton-style; closest deterministic analogue to Holm's OAR)",
             "EOG channels were notch+bandpass filtered first so they are "
             "valid regressors; the example pipeline filters EEG picks only"]
    if degenerate:
        notes.append(
            f"REGRESSION IS NUMERICALLY DEGENERATE: max|coefficient| = {max_coef:.3g} "
            f"(a physical ocular propagation factor is well below 1) and the EOG "
            f"regressor pair has condition number {cond:.3g}. The EOG V and EOG H "
            f"channels are near-collinear, so this correction is not trustworthy "
            f"for this recording.")

    qc.setdefault("step5b_ocular", {})[mode] = dict(
        applied=True,
        max_abs_coefficient=round(max_coef, 4),
        eog_condition_number=(round(cond, 1) if np.isfinite(cond) else None),
        degenerate=degenerate,
        mean_sd_reduction_pct=round(
            float(100.0 * np.mean((sd_before - sd_after) / sd_before)), 2),
        clean=bool(not degenerate),
        notes=notes,
    )
    return out


# ===========================================================================
# STEP 6 - Interpolate bad channels
#   example_eeg_processing.ipynb cell 27, which hardcodes bad_chans=['Cz','F2'].
#   DEVIATION (approved): the bad-channel list is derived per recording instead.
#
#   DEVIATION (approved 2026-09-04): for everything except the `example` branch
#   the list is derived PER PAIR, from the task concatenated with the RESTING
#   part of its own baseline, and the one resulting list is interpolated in both
#   recordings. See detect_bad_channels_paired for why.
# ===========================================================================

def detect_bad_channels(*raws: mne.io.RawArray):
    """
    Robust-z of log10(SD) against the median of the recording's own EEG channels.

    Accepts MORE THAN ONE raw since 2026-09-04, in which case the channels are
    concatenated along time before the SD is taken and ONE list is returned for
    all of them. Passing a single raw is the original per-recording behaviour and
    is still what the `example` branch uses.

    Concatenation, not averaging of two SDs: the criterion is defined on a
    channel's dispersion over the samples it is judged on, so joining the samples
    is what "judge these recordings together" means. Every raw must carry the same
    EEG channels in the same order -- asserted below rather than assumed, because
    a silent mismatch would z-score one electrode against another.
    """
    if not raws:
        raise ValueError("detect_bad_channels needs at least one recording")

    names = None
    blocks = []
    for raw in raws:
        # exclude=[] explicitly: pick_types defaults to exclude='bads', and a raw
        # arriving here with info['bads'] already set would silently be scored on
        # nine channels, with the excluded electrode unable to be flagged at all.
        # Nothing sets bads on a bandpassed raw today; this keeps it that way by
        # construction rather than by audit.
        picks = mne.pick_types(raw.info, eeg=True, emg=False, eog=False,
                               exclude=[])
        these = [raw.ch_names[i] for i in picks]
        if names is None:
            names = these
        elif these != names:
            raise ValueError(
                f"cannot pool bad-channel detection across recordings with "
                f"different EEG channels: {names} vs {these}")
        blocks.append(raw.get_data(picks=picks))

    X = np.concatenate(blocks, axis=1) if len(blocks) > 1 else blocks[0]

    sd = np.array([np.std(x) for x in X], dtype=float)
    logsd = np.log10(np.where(sd > 0, sd, np.nan))
    med = np.nanmedian(logsd)
    mad = np.nanmedian(np.abs(logsd - med))
    sigma = 1.4826 * mad if mad > 0 else np.nan

    if not np.isfinite(sigma) or sigma == 0:
        z = np.zeros_like(logsd)
    else:
        z = (logsd - med) / sigma

    bad = [n for n, zi, s in zip(names, z, sd)
           if (not np.isfinite(zi)) or abs(zi) > BAD_CHANNEL_Z or s == 0.0]
    detail = {n: (round(float(zi), 2) if np.isfinite(zi) else None)
              for n, zi in zip(names, z)}
    return bad, detail


# The baseline segment whose samples join the task in the paired detection.
# `rest` by the user's decision (2026-09-04): a baseline block is 2 min mental
# arithmetic + 2 min deliberate eye movements + 2 min rest, and the middle phase
# is deliberate ocular artifact. Feeding it to a criterion that measures channel
# dispersion would flag frontal electrodes for doing exactly what the protocol
# asked the participant to do.
#
# NOT a free choice of segment: the decision is made once per pair and applies to
# the WHOLE baseline block, so selecting `math` or `eyes_closed` on the dashboard
# still gets channels chosen on `rest`. That is deliberate -- one pair, one
# interpolation decision, stable across a control the reader can move -- but it
# does mean the two non-default segments are interpolated on evidence drawn from
# outside themselves. See PROVENANCE.
PAIRED_DETECT_BASELINE_SEGMENT = "rest"


def detect_bad_channels_paired(task_preps: list[dict], base_prep: dict | None):
    """
    ONE bad-channel list for a task and the baseline recorded in the same session
    under the same condition, decided on the two of them TOGETHER.

    Added 2026-09-04. Until then the criterion ran independently on each
    recording, and since it is purely RELATIVE -- robust-z against the median of
    that recording's own ten channels -- the two runs could disagree about the
    same electrode. They did, in 14 of 22 pairs across the ten EEG channels and
    in 11 of 22 on the three that feed the index, and the disagreement produced
    the three most negative cognitive-load log ratios in the dataset. Measured on
    the 2026-09-03 run, with what this rule gives instead in the last column:

        p02/task_ai_speedscore   F1,F2 flagged on the task (z 7.63/7.56) and NOT
                                 on its baseline (z 2.55/2.58) despite 292/341 uV
                                 robust SD there. Task numerator = spline
                                 reconstruction, baseline numerator = the
                                 artifact itself. Baseline fm_theta 875 against a
                                 task 25; ln ratio -3.76, 99.7% of task windows
                                 "below baseline".   -> now F1,F2 on BOTH,
                                 ln +0.06, 48.0% below.
        p08/task_ai_speedscore   the same, larger: baseline fm_theta 1847 against
                                 a task 36, ln -4.56, 92.4% below.
                                 -> now F1,F2 on both, ln +0.59, 31.8% below.
        p07/task_ai_personal     the mirror image, on the denominator. Pz NOT
                                 flagged on the task (z 1.46, 109 uV) and flagged
                                 on the baseline (z 4.64), so the task alpha is
                                 real and the baseline alpha is a spline: 42.7
                                 against 1.78, ln -3.44, of which -3.18 is the
                                 denominator alone.   -> pooled, Pz scores z 1.66
                                 and is flagged on NEITHER, so both sides keep the
                                 raw electrode: ln -0.18, 56.3% below.

    And one in the other direction, which matters just as much because it was the
    dataset's headline number: p14/task_ai_personal had an empty task list and a
    baseline list of F1,F2,O1,O2, so its baseline was smoothed and its task was
    not. It was the highest "cognitive load" in the run at ln +2.01; pooled, its
    list is empty on both sides and it falls to ln +0.01. Nothing here is a
    cognitive effect in either direction.

    Group median ln(task/rest) moves only +0.404 -> +0.453 across the 22
    recordings, which is the point: this removes an artifact, it does not
    manufacture a result. Asymmetric pairs go 14/22 -> 0/22 by construction.

    None of those is a cognitive effect. Each is a contrast between interpolated
    and non-interpolated data, and a ratio whose two sides come from different
    physical signals is not a baseline correction of the same quantity.

    THE RULE (user's decision, 2026-09-04): concatenate the task with the RESTING
    part of its baseline and run the identical criterion once over the join.
    Alternatives considered and rejected:

      * union of the two per-recording lists -- fixes the three above, but lets
        the weakest evidence win: p04/task_agent_personal and p14/task_ai_personal
        have empty task lists and 4-channel baseline lists, so a clean 20-minute
        task would have had 4 of its 10 electrodes replaced by splines on the
        say-so of a 6-minute recording.
      * intersection -- symmetric, but leaves p02, p07 and p08 exactly as they
        are, i.e. does not fix the thing it was written for.

    WHY ONLY THE RESTING PART. A baseline block is 2 min mental arithmetic + 2 min
    deliberate eye movements + 2 min rest. The middle phase is instructed ocular
    artifact, and feeding it to a criterion that measures channel dispersion would
    flag frontal electrodes for doing exactly what the protocol asked of the
    participant. Only PAIRED_DETECT_BASELINE_SEGMENT joins the task.

    WHAT CONCATENATION WEIGHTS BY. Its own sample count, which is the honest
    weighting for "how did this electrode behave in this session": the task brings
    ~20 min and rest ~2 min, so rest cannot by itself condemn a channel the task
    found quiet, while a channel loud in BOTH still clears the threshold easily.
    HOW MUCH A REST-ONLY FAULT IS DILUTED. Less than the 10:1 sample ratio
    suggests, because that ratio weights the VARIANCE and the criterion is on
    log10(SD). With R = SD_rest / SD_task on some channel, and rest at 1/11 of the
    joined samples, SD_pooled / SD_task = sqrt(0.909 + 0.0909 R^2), so in decades:

        R =  3   -> +0.12   (z +0.5 at the median across-channel sigma of 0.245)
        R = 10   -> +0.50   (z +2.0)
        R = 30   -> +0.96   (z +3.9)

    So only a rest-only fault below roughly 3-5x survives unflagged; a 10x one is
    still very likely caught. That residue is the accepted cost of this rule --
    where it does happen both sides keep the raw channel, which is at least the
    SAME quantity on both sides of the ratio. That is what this buys: a clean
    baseline correction, not a clean electrode.

    WHY THE SEAM DOES NOT MATTER. Not because the recordings are high-passed and
    join smoothly -- np.std is order-invariant, and nothing downstream of the join
    is filtered or spectrally estimated, so a discontinuity at the seam cannot
    reach the statistic at all. The only way joining can distort an SD is the
    between-group term: pooled variance picks up w * delta^2, where delta is the
    difference of the two per-recording channel means and w = n1 n2 / (n1+n2)^2
    = 0.083 at 20 min : 2 min. What bounds that term is delta RELATIVE to the
    channel's own SD, not delta in microvolts -- 30 uV on a 300 uV channel is
    +0.08% variance and moves z by under 0.01, while the same 30 uV on a quiet
    12 uV channel is +50% and can move z by several. So the provenance records
    `max_abs_channel_mean_over_sd`, the ratio that actually bounds it, alongside
    the raw microvolts. Below ~0.05 the term moves z by under 0.01 even in the
    tightest recording in this dataset (across-channel sigma 0.018), so that is
    the number to check the field against. Measured across all 88 sources of the
    2026-09-04 run, the largest is 0.046 (p10/baseline_ai_speedscore::rest) -- so
    the term is negligible everywhere here, but it is recorded per source rather
    than asserted once, because that is a property of this data and not of the
    rule.

    A GROUP, not strictly a pair, because a baseline can serve more than one task
    file: a task split across two files -- `task_ai_speedscore_1` and `_2` -- has
    both halves resolving to one `baseline_ai_speedscore`. Deciding per pair
    would give that baseline two different lists and reinstate the asymmetry for
    whichever task lost. Every task in the group joins the concatenation and
    every member gets the one list.

    NOTE this path is UNTESTED in the template. The original study's only split
    recording belonged to a participant it excluded, so the code ran but the
    branch never did. If your data has a split recording, check it.
    p05 is excluded, so nothing exercises that path today -- it is here so a
    future split recording is not silently mishandled.

    Falls back, and SAYS SO in the provenance, rather than silently reverting to
    the per-recording behaviour this replaces: to tasks-only where there is no
    baseline or its rest segment is too short to crop, and to the baseline's own
    rest segment where there is no usable task.
    """
    prov = dict(rule="concatenated", segment=PAIRED_DETECT_BASELINE_SEGMENT,
                fallback=None)

    # crop_to_segment reads BASELINE_SEGMENTS bounds, which lie inside a ~20 min
    # task as happily as inside a 6 min baseline block: handed a task it would
    # succeed silently and return 240-360 s of task samples labelled as rest.
    # discover_recordings calls anything not starting with "task" a baseline, so
    # a stray folder could reach here. Cheap guard against a silent mislabel.
    if base_prep is not None and base_prep["rec"]["kind"] != "baseline":
        raise ValueError(
            f"detect_bad_channels_paired was given a non-baseline as the "
            f"baseline of a group: {base_prep['rec']['key']}")

    base_rest = None
    if base_prep is None:
        prov["fallback"] = ("no paired baseline recording; detection used the "
                            "task(s) alone")
    elif not base_prep.get("ready"):
        prov["fallback"] = ("the paired baseline reached no usable segment; "
                            "detection used the task(s) alone")
    else:
        base_rest, _ = crop_to_segment(base_prep["bandpassed"],
                                       base_prep["seg_times"],
                                       PAIRED_DETECT_BASELINE_SEGMENT)
        if base_rest is None:
            prov["fallback"] = (
                f"the baseline block is too short to contain the "
                f"{PAIRED_DETECT_BASELINE_SEGMENT} segment; detection used the "
                f"task(s) alone")

    ready_tasks = [p for p in task_preps if p.get("ready")]
    if not ready_tasks and base_rest is not None:
        prov["fallback"] = (f"no usable task in this group; detection used the "
                            f"baseline's {PAIRED_DETECT_BASELINE_SEGMENT} "
                            f"segment alone")

    raws = ([p["bandpassed"] for p in ready_tasks]
            + ([base_rest] if base_rest is not None else []))
    whole_block = None
    if not raws and base_prep is not None and base_prep.get("ready"):
        # A READY baseline whose rest segment could not be cropped, with no usable
        # task. Returning an empty list here would not mean "nothing to do": the
        # baseline still runs, and because [] is not None step6_interpolate would
        # NOT re-detect, so it would be processed end to end with zero channels
        # interpolated and a `clean: True` verdict that was never computed. Fall
        # back to the whole block -- the pre-2026-09-04 behaviour for a lone
        # recording -- rather than to nothing.
        whole_block = base_prep["bandpassed"]
        raws = [whole_block]
        prov["rule"] = "whole_block"
        prov["segment"] = None
        prov["fallback"] = (
            f"no usable task in this group and the baseline block is too short "
            f"to contain the {PAIRED_DETECT_BASELINE_SEGMENT} segment; detection "
            f"used the WHOLE baseline block, which includes the deliberate "
            f"eye-movement phase")
    if not raws:
        # Now genuinely nothing: every member carries a terminal status already,
        # and finish_recording passes a not-ready prep straight through, so this
        # empty list is never interpolated into anything.
        prov["rule"] = None
        prov["segment"] = None
        prov["fallback"] = "nothing in this group reached step 5"
        prov["paired"] = False
        prov["sources"] = []
        prov["n_samples_total"] = 0
        prov["members"] = []
        return [], {}, prov

    bad, detail = detect_bad_channels(*raws)

    def _describe(raw, label):
        picks = mne.pick_types(raw.info, eeg=True, emg=False, eog=False,
                               exclude=[])
        X = raw.get_data(picks=picks)
        mu = np.abs(X.mean(axis=1))
        sd = X.std(axis=1)
        # The ratio, not the microvolts, is what bounds the between-group variance
        # term -- see the docstring. Both are recorded; only this one is checkable.
        ratio = float(np.max(np.where(sd > 0, mu / np.where(sd > 0, sd, 1.0), 0.0)))
        return dict(role=label, n_samples=int(X.shape[1]),
                    duration_s=round(X.shape[1] / float(raw.info["sfreq"]), 1),
                    max_abs_channel_mean_uv=round(float(np.max(mu)), 4),
                    max_abs_channel_mean_over_sd=round(ratio, 5))

    base_label = None
    if base_rest is not None:
        base_label = (f"baseline:{base_prep['rec']['key']}"
                      f"::{PAIRED_DETECT_BASELINE_SEGMENT}")

    prov["sources"] = (
        [_describe(p["bandpassed"], f"task:{p['rec']['key']}")
         for p in ready_tasks]
        + ([_describe(base_rest, base_label)] if base_rest is not None else [])
        + ([_describe(whole_block, f"baseline:{base_prep['rec']['key']}::whole")]
           if whole_block is not None else []))
    prov["n_samples_total"] = sum(d["n_samples"] for d in prov["sources"])
    prov["members"] = ([p["rec"]["key"] for p in ready_tasks]
                       + ([base_prep["rec"]["key"]]
                          if base_rest is not None or whole_block is not None
                          else []))
    prov["paired"] = bool(ready_tasks) and base_rest is not None
    return bad, detail, prov


def step6_interpolate(bandpass_data: mne.io.RawArray, qc: dict, tag: str,
                      bad_chans: list[str] | None = None,
                      z_detail: dict | None = None,
                      pairing: dict | None = None) -> mne.io.RawArray:
    # `bad_chans` is passed in so the SAME list is used for both ocular modes.
    # Detecting per branch (the original behaviour) meant the ocular toggle also
    # changed which channels were interpolated in 3 of 22 recordings -- for
    # p08/task_agent_personal it silently flipped whether Pz, the entire
    # denominator, was real or synthetic. The toggle is now a clean A/B.
    #
    # `pairing` records WHERE the list came from -- task+baseline-rest since
    # 2026-09-04, or task-only for the `example` branch and for a fallback. It is
    # written into qc verbatim so a reader of qc_steps.json can tell a paired
    # decision from a per-recording one without diffing the code. None means the
    # question was never asked, which is not the same as "per-recording"; the
    # callers all pass it.
    if bad_chans is None:
        bad_chans, z_detail = detect_bad_channels(bandpass_data)

    work = bandpass_data.copy()
    if len(bad_chans) > 0:
        work.info['bads'].extend(bad_chans)

    bad_chan_interp_data = work.copy().interpolate_bads(reset_bads=True)

    index_chans = HOLM_FRONTAL_CHANNELS + HOLM_PARIETAL_CHANNELS
    index_interp = [c for c in index_chans if c in bad_chans]

    qc.setdefault("step6_interpolate", {})[tag] = dict(
        example_pipeline_hardcoded=['Cz', 'F2'],
        bad_channels_detected=bad_chans,
        robust_z_by_channel=z_detail,
        index_channels_interpolated=index_interp,
        detection=pairing,
        clean=bool(len(bad_chans) <= 2 and not index_interp),
        notes=(
            ([f"{len(bad_chans)}/10 EEG channels interpolated: {bad_chans}"] if bad_chans else [])
            + ([f"index channel(s) {index_interp} were interpolated - the Holm "
                f"index for this recording is partly synthetic"] if index_interp else [])
            # Worded FROM the provenance. It used to assert "chosen on this
            # recording ALONE", which is wrong in every fallback that matters:
            # where the rest crop fails but the task is fine, the list is chosen
            # on the TASK and then applied to the baseline, and in a multi-task
            # group "alone" was never true either.
            + ([f"bad channels chosen on "
                f"{', '.join((pairing or {}).get('members') or ['no recording'])}"
                f", not on the full task/baseline pair: "
                f"{(pairing or {}).get('fallback')}"]
               if pairing and pairing.get("fallback") else [])
        ),
    )
    return bad_chan_interp_data


# ===========================================================================
# STEP 7 - Re-reference (example pipeline branch only)
#   example_eeg_processing.ipynb cell 30
#   DEVIATION (approved): the Holm index is computed BEFORE this step, keeping
#   the hardware SRB2/earlobe reference, which is the closest analogue to Holm's
#   right-mastoid reference. The example pipeline's own outputs still use the
#   average reference below.
# ===========================================================================

def step7_reref(bad_chan_interp_data: mne.io.RawArray, qc: dict) -> mne.io.RawArray:
    reref = bad_chan_interp_data.set_eeg_reference([])
    avg_ref_data = reref.set_eeg_reference(ref_channels='average', projection=True)
    avg_ref_data.apply_proj()

    qc["step7_reref"] = dict(
        reference="average (example pipeline)",
        index_reference="hardware SRB2/earlobe (Holm branch, pre-average)",
        clean=True,
        notes=["Holm index is computed before this step to preserve absolute "
               "power at the recording reference"],
    )
    return avg_ref_data


# ===========================================================================
# STEP 8a - Example pipeline feature extraction
#   example_eeg_processing.ipynb cell 34
# ===========================================================================

def step8a_example_bandpower(avg_ref_data: mne.io.RawArray, qc: dict) -> dict:
    cleaned_eeg_data = avg_ref_data.copy().pick(picks="eeg")

    mne_eeg_epochs = mne.make_fixed_length_epochs(cleaned_eeg_data,
                                                  duration=EPOCH_DUR,
                                                  overlap=EPOCH_OVERLAP)

    power_info = {}
    for band, (f_low, f_high) in FREQ_BANDS.items():
        psds, freqs = calc_eeg_band_power(mne_eeg_epochs, f_low, f_high)
        power_info[band] = [psds, freqs]

    n_epochs = len(mne_eeg_epochs)
    ch_names = mne_eeg_epochs.ch_names
    summary = {band: dict(
        shape=list(psds.shape),
        mean_norm_power_by_channel={c: round(float(v), 5) for c, v in
                                    zip(ch_names, psds.mean(axis=(0, 2)))},
    ) for band, (psds, freqs) in power_info.items()}

    qc["step8a_example_bandpower"] = dict(
        epoch_duration_s=EPOCH_DUR,
        epoch_overlap_s=EPOCH_OVERLAP,
        n_epochs=int(n_epochs),
        bands=summary,
        normalization="min-max 0-1 per band (example pipeline, verbatim)",
        clean=bool(n_epochs > 0),
        notes=[],
    )
    return power_info


# ===========================================================================
# STEP 8b - Holm et al. (2009) cognitive-load index
# ===========================================================================

def index_channel_gate(bandpass_data: mne.io.RawArray, qc: dict) -> dict:
    """
    Plausibility REPORT on F1, F2 and Pz, evaluated on the PRE-interpolation signal.
    Added 2026-08-27 as a gate; stopped gating 2026-09-02 at the user's request.

    detect_bad_channels is a purely RELATIVE criterion -- robust-z against the median
    of that recording's own ten channels -- so when most of a montage is broken the
    median moves with it and nothing is flagged. p14/task_ai_personal passed with an
    empty bad-channel list while carrying an F1 of 375 uV robust SD (22,161 uV
    plain SD -- quote the robust figure when comparing against any threshold
    here, they are the same quantity), and produced the highest
    "cognitive load" in the dataset. That is what this function exists to make
    visible, and it still does.

    What changed is only the consequence. It used to refuse an index outright when a
    channel feeding it was implausible, and to drop a failing frontal channel from
    the numerator. Now every task recording gets an index, always from the full
    F1+F2 midline mean, and the amplitudes are published beside it. See the
    INDEX_CH_*_ROBUST_SD_UV comment for why.

    The function keeps its name and its qc key (`step8b_gate`), which are a data
    contract with cognitive_load.json, qc_steps.csv and the walkthrough; the fields
    that used to gate now report:

        index_computed        always True. RENAMED from `index_computable`, and
                              the rename is the point: an analysis that did
                              `df[df.index_computable]` selected 12 of 22
                              recordings before this change and would select all
                              22 after it, silently, with no error and no change
                              of dtype -- including the recording the
                              gate was written to catch. A KeyError is the honest
                              outcome. The old gate's rule was
                              `parietal_plausible AND frontal_plausible_n >= 1`
                              (Pz in band AND at least one frontal in band).
                              NOTE: the band those counts were taken against was
                              the fixed 1-50 uV one, replaced 2026-09-03 by a band
                              derived from the run, so any count quoted from
                              before that date reproduces only under the old
                              band. Recompute from the shipped labels rather than
                              trusting a number written here.
        frontal_channels_used always HOLM_FRONTAL_CHANNELS
        gate_applied          False -- the flag that says so out loud
        parietal_plausible    Pz within the band. NOT on its own the old gate --
                              it passes more recordings than the gate did, because
                              the gate also required a frontal channel.
        frontal_plausible     which of F1/F2 are within range. Shipped so the old
                              gate is reconstructible in full from labels, without
                              anyone having to re-derive it from by_channel.
        strict_subset         both frontal channels plausible. Still computed, but
                              now a LABEL a consumer may filter on, not a subset
                              this pipeline treats differently.
    """
    picks = mne.pick_types(bandpass_data.info, eeg=True, emg=False, eog=False)
    names = [bandpass_data.ch_names[i] for i in picks]
    X = bandpass_data.get_data(picks=picks)

    detail = {}          # `plausible` per channel is filled in later
    for c in HOLM_FRONTAL_CHANNELS + HOLM_PARIETAL_CHANNELS:
        x = X[names.index(c)]
        rsd = float(1.4826 * np.median(np.abs(x - np.median(x))))
        # `plausible` is decided later, against the run-derived band. None here
        # means "not assessed yet" and is overwritten by apply_amplitude_labels.
        detail[c] = dict(robust_sd_uv=round(rsd, 2), plausible=None)

    # The index is now computed for every recording, from the full F1+F2 midline
    # mean, whatever these amplitudes say. `plausible` per channel and the derived
    # `strict_subset` are the labels that carry the old policy's information: a
    # consumer wanting the pre-2026-09-02 strict set can select on strict_subset,
    # and one wanting the old permissive gate can select on parietal_plausible.
    parietal_ok = None
    frontal_ok = []
    frontal_used = list(HOLM_FRONTAL_CHANNELS)

    notes = []

    info = dict(
        range_uv=None,          # filled in by apply_amplitude_labels
        by_channel=detail,
        failed_channels=[],     # rewritten by apply_amplitude_labels
        frontal_channels_used=frontal_used,
        # Renamed from `index_computable` so a consumer still filtering on the
        # old name fails loudly instead of silently widening 12 -> 22. See the
        # docstring.
        index_computed=True,
        gate_applied=False,
        parietal_plausible=bool(parietal_ok),
        frontal_plausible=frontal_ok,
        strict_subset=bool(parietal_ok and len(frontal_ok) == 2),
        clean=None,             # rewritten by apply_amplitude_labels
        notes=notes,
    )
    qc["step8b_gate"] = info
    return info


def step8b_holm_index(interp_data: mne.io.RawArray,
                      pre_interp_data: mne.io.RawArray,
                      gate: dict,
                      seg_times: np.ndarray,
                      qc: dict,
                      tag: str) -> dict:
    """
    Per-window Holm index, plus the four artifact masks.

    Two data arguments where there used to be one, both from the 2026-08-27 review:

    `interp_data` (post bad-channel interpolation) supplies the SPECTRA.
    `pre_interp_data` (the same data before interpolation) supplies the ARTIFACT
    MASKS. Holm's +-70 uV criterion used to be evaluated after interpolation, and
    because a spline-interpolated channel is a smooth blend of its neighbours it
    passed the threshold far more readily than a real electrode: mean retention was
    14.5% where an index channel had been interpolated against 2.5% where none had,
    so "% kept" was an inverse proxy for how much of the index was synthetic. The
    criterion is now applied to the measured signal.

    `seg_times` are the segment's real unix timestamps, so window times reflect
    elapsed task time rather than assuming strict 4 s contiguity -- p04's 3.91 s
    dropout previously shifted every window after index 172 by ~3.9 s.
    """
    # The withholding branch that used to stand here was removed 2026-09-02 with
    # the amplitude gate: no recording is refused an index for its amplitudes, and
    # `gate` now supplies one thing only -- the frontal set, which is always both
    # channels. The argument is kept so the derivation still comes from a single
    # place rather than being re-derived here.
    eeg_only = interp_data.copy().pick(picks="eeg")
    pre_only = pre_interp_data.copy().pick(picks="eeg")

    epochs = mne.make_fixed_length_epochs(eeg_only, duration=HOLM_WINDOW_SEC,
                                          overlap=0.0, preload=True)
    ch_names = epochs.ch_names
    n_win = len(epochs)
    if n_win == 0:
        qc.setdefault("step8b_holm", {})[tag] = dict(clean=False, n_windows=0,
                                                     notes=["no complete 4 s windows"])
        return dict(n_windows=0, index_computed=True)

    # ---- build the midline-frontal derivation in the TIME domain ----------
    # (F1 + F2) / 2 sample-by-sample, then one spectrum -- which is what a midline
    # electrode physically is. Averaging the two channels' POWER instead let the
    # louder channel dominate the numerator (up to 7x on p14/task_ai_personal).
    data = epochs.get_data(copy=True)                       # (n_win, n_ch, n_samp)
    frontal_used = gate["frontal_channels_used"]
    f_idx = [ch_names.index(c) for c in frontal_used]
    p_idx = [ch_names.index(c) for c in HOLM_PARIETAL_CHANNELS]
    frontal_ts = data[:, f_idx, :].mean(axis=1, keepdims=True)
    parietal_ts = data[:, p_idx, :].mean(axis=1, keepdims=True)

    derived = mne.EpochsArray(
        np.concatenate([frontal_ts, parietal_ts], axis=1),
        mne.create_info(["Fz_est", "Pz_est"], epochs.info["sfreq"], ["eeg", "eeg"]),
        verbose=False)

    theta_psds, theta_freqs = calc_eeg_band_power_absolute(derived, *HOLM_THETA)
    alpha_psds, alpha_freqs = calc_eeg_band_power_absolute(derived, *HOLM_ALPHA)

    theta_frontal = theta_psds[:, 0, :].mean(axis=1)
    alpha_parietal = alpha_psds[:, 1, :].mean(axis=1)
    index = theta_frontal / alpha_parietal

    # ---- artifact masks, evaluated on the PRE-interpolation signal ---------
    pre_epochs = mne.make_fixed_length_epochs(pre_only, duration=HOLM_WINDOW_SEC,
                                              overlap=0.0, preload=True)
    pre_names = pre_epochs.ch_names
    index_chs = frontal_used + HOLM_PARIETAL_CHANNELS
    idx_ch = [pre_names.index(c) for c in index_chs]
    pre_data = pre_epochs.get_data(copy=False)

    n_win = min(n_win, pre_data.shape[0])
    peak = np.abs(pre_data[:n_win, idx_ch, :]).max(axis=2)   # (n_win, n_index_ch)
    index = index[:n_win]
    theta_frontal = theta_frontal[:n_win]
    alpha_parietal = alpha_parietal[:n_win]

    keep_strict = (peak <= HOLM_REJECT_UV).all(axis=1)

    full = pre_only.get_data(picks=idx_ch)
    robust_thr = np.array([
        ROBUST_SIGMA * 1.4826 * np.median(np.abs(x - np.median(x))) for x in full
    ], dtype=float)
    robust_thr = np.where(robust_thr > 0, robust_thr, np.inf)
    keep_robust = (peak <= robust_thr[None, :]).all(axis=1)

    # A window straddling a discontinuity is dropped from EVERY mode, `none`
    # included: "no artifact rejection" is a statement about not judging the EEG,
    # not a licence to plot a spectrum taken across a splice. ANDed into the
    # masks rather than applied to `index` so each mode's retention figure
    # continues to describe the windows that were actually eligible.
    n_samp_holm = int(round(HOLM_WINDOW_SEC * float(eeg_only.info["sfreq"])))
    disc = discontinuous_windows(seg_times, n_samp_holm, n_win)
    contiguous = ~disc

    masks = {
        "holm_strict": keep_strict & contiguous,
        "holm_imputed": keep_strict & contiguous,
        "none": contiguous,
        "robust": keep_robust & contiguous,
    }

    # The numerator and denominator series shipped beside the index in
    # cognitive_load.json get the same treatment. Without this the file holds two
    # mutually inconsistent per-window records: `series` nulls the discontinuous
    # windows and `theta_frontal`/`alpha_parietal` still carry their
    # across-the-splice values at full precision. Nothing reads them today, which
    # is exactly why it would go unnoticed.
    theta_frontal = np.where(disc, np.nan, theta_frontal)
    alpha_parietal = np.where(disc, np.nan, alpha_parietal)

    series, fabricated = {}, {}
    for mode, keep in masks.items():
        vals = np.where(keep, index, np.nan)
        if mode == "holm_imputed":
            # NOTE: limit_direction='both' does NOT extrapolate -- it back/forward
            # fills the nearest value, so leading and trailing gaps become flat runs.
            # p09/task_agent_personal opened with 31 identical windows. The fabricated
            # fraction is reported so this cannot be mistaken for measurement.
            #
            # RUN BY RUN, so no fill reaches across a discontinuity. Interpolating
            # the whole series at once would bridge the hole left by an excluded
            # window from the values on either side of it -- and worse, would fill
            # a genuinely REJECTED window next to the splice from the far side of
            # a 449 s cut. limit_direction='both' makes that reach unbounded: a
            # single valid window on the far side is enough to flat-fill
            # everything up to it. Splitting first bounds every fill to samples
            # that are actually adjacent in time. The excluded windows themselves
            # are never in a run, so they stay NaN without needing to be re-voided.
            filled = np.asarray(vals, dtype=float).copy()
            _s0, _n = 0, len(filled)
            while _s0 < _n:
                if disc[_s0]:
                    _s0 += 1
                    continue
                _e0 = _s0
                while _e0 < _n and not disc[_e0]:
                    _e0 += 1
                filled[_s0:_e0] = (pd.Series(vals[_s0:_e0])
                                   .interpolate(method='linear',
                                                limit_direction='both')
                                   .to_numpy())
                _s0 = _e0
            n_real = int(np.isfinite(vals).sum())
            n_out = int(np.isfinite(filled).sum())
            fabricated[mode] = dict(
                measured=n_real, delivered=n_out, fabricated=n_out - n_real,
                fabricated_pct=(round(100.0 * (n_out - n_real) / n_out, 1)
                                if n_out else None))
            vals = filled
        else:
            n_real = int(np.isfinite(vals).sum())
            fabricated[mode] = dict(measured=n_real, delivered=n_real,
                                    fabricated=0, fabricated_pct=0.0)
        series[mode] = [None if not np.isfinite(v) else round(float(v), 6) for v in vals]

    # real elapsed time at each window start, not an assumed 4 s grid
    step = int(HOLM_WINDOW_SEC * GALEA_SAMPLING_RATE)
    times = []
    for w in range(n_win):
        s = w * step
        times.append(round(float(seg_times[s] - seg_times[0]), 3)
                     if s < len(seg_times) else None)

    retention = {m: round(100.0 * float(np.mean(k)), 1) for m, k in masks.items()}
    qc.setdefault("step8b_holm", {})[tag] = dict(
        window_s=HOLM_WINDOW_SEC,
        n_windows=int(n_win),
        theta_band_hz=list(HOLM_THETA), alpha_band_hz=list(HOLM_ALPHA),
        # Always the midline mean since 2026-09-02. The "<channel> alone" form
        # this used to take when the gate dropped a frontal channel is gone with
        # the gate -- kept as a sentence here because it is the per-recording
        # record of WHICH derivation produced these numbers, and that is the
        # field to check when comparing against a pre-2026-09-02 output.
        frontal_derivation="mean(%s) in the time domain" % ",".join(frontal_used),
        frontal_channels_used=frontal_used,
        # `strict_subset` USED to be copied here. It is not any more: the label is
        # decided against a band derived from the whole run, which does not exist
        # when this dict is built, so the copy was frozen at its pre-band value --
        # False on every recording, contradicting the corrected copy in
        # step8b_gate two fields away in the same file. One authoritative field,
        # in step8b_gate; consumers read it there.
        parietal_derivation="Pz",
        estimator="welch / single full-window Hann taper (Holm's FFT)",
        rejection_evaluated_on="pre-interpolation signal",
        retention_pct=retention,
        # How many windows never reached the criterion at all. Without it,
        # retention_pct["none"] silently stops being 100.0 -- it was 100.0 by
        # construction until 2026-09-04 -- with nothing in the file saying why.
        # `compute_measure_series` reports the same thing under the same name.
        n_discontinuous=int(disc.sum()),
        # CONTRACT NOTE: the per-window mask strings published for the other
        # modes still read "1" for a discontinuous window, because they record
        # what the AMPLITUDE criterion decided and it decided nothing here. A
        # consumer recomputing retention as sum(mask)/len(mask) will therefore
        # disagree with retention_pct by exactly n_discontinuous. Use the
        # published percentages, or intersect with the excluded set.
        fabricated=fabricated,
        # Judged on the mask the SERIES actually used, not on the pre-contiguity
        # one. A recording whose only strict-surviving window was discontinuous
        # would otherwise be reported clean beside an empty series.
        clean=bool(masks["holm_strict"].mean() > 0.0),
        notes=([] if masks["holm_strict"].mean() > 0
               else ["no window survives Holm's +-70 uV criterion"]
                    + (["every window that did survive it spans a discontinuity "
                        "and is excluded"] if keep_strict.mean() > 0 else [])),
    )

    return dict(
        n_windows=int(n_win),
        index_computed=True,
        frontal_channels_used=frontal_used,
        # strict_subset lives in qc step8b_gate only -- see above.
        times_s=times,
        # None rather than a bare NaN for an excluded window: `series` above
        # already uses None for the same thing, and json.dump would otherwise
        # write the literal NaN, which is not valid JSON and which no consumer
        # outside Python can read.
        theta_frontal=[None if not np.isfinite(v) else round(float(v), 8)
                       for v in theta_frontal],
        alpha_parietal=[None if not np.isfinite(v) else round(float(v), 8)
                        for v in alpha_parietal],
        series=series,
        retention_pct=retention,
        n_discontinuous=int(disc.sum()),
        fabricated=fabricated,
    )


# ===========================================================================
# STEP 9 - The four additional dashboard measures (added 2026-08-28)
#
# These are NOT Holm measures. Every parameter was chosen by the user; see
# Three decisions are load-bearing and are repeated here
# because they are easy to misread from the numbers alone:
#
#   1. NO AMPLITUDE GATE. The user chose to compute these for every task
#      recording. A recording whose electrode read 375 uV robust SD will therefore
#      produce a value on these tabs. Signal quality is reported per recording
#      and shown on the dashboard, but nothing is withheld. Since 2026-09-02 the
#      Holm index is ungated too, so this is no longer a difference between them
#      -- it was, and these measures were the ones that never gated.
#   2. `fm_theta` is ALWAYS the full F1+F2 time-domain mean, and so, since the
#      gate's removal, is the Holm index numerator. The two are now the same
#      derivation on every recording; before 2026-09-02 the index fell back to a
#      single frontal channel where the gate dropped one, and they diverged.
#   3. Both asymmetries need both of their channels by construction -- a
#      difference has no single-channel substitute.
# ===========================================================================

MEASURE_SOURCE_CHANNELS = ["F1", "F2", "P3", "P4", "Pz"]


def average_reference(raw: mne.io.RawArray) -> mne.io.RawArray:
    """
    The example pipeline's average re-reference (cell 30), applied to a COPY.

    DEVIATION (approved 2026-08-28): the user asked for the reference to be a
    dashboard toggle on the new tabs rather than a fixed choice. The Holm index
    keeps its fixed hardware SRB2/earlobe reference (section 2.4) and is not
    affected by that toggle.
    """
    out = raw.copy()
    out.set_eeg_reference([])
    out = out.set_eeg_reference(ref_channels='average', projection=True)
    out.apply_proj()
    return out


def _measure_epochs(raw: mne.io.RawArray):
    """4 s epochs of the EEG channels, or None if the segment is too short."""
    eeg = raw.copy().pick(picks="eeg")
    ep = mne.make_fixed_length_epochs(eeg, duration=HOLM_WINDOW_SEC,
                                      overlap=0.0, preload=True)
    return ep if len(ep) else None


def _derived_epochs(ep: mne.Epochs) -> mne.EpochsArray:
    """
    F1, F2, P3, P4, Pz as recorded, plus FM = the time-domain mean of F1 and F2.

    FM is built sample-by-sample BEFORE the FFT for the same reason the Holm
    numerator is (section 3.1): that is what a midline derivation physically is,
    and it is robust to one noisy channel in a way power-averaging is not.
    """
    names = ep.ch_names
    data = ep.get_data(copy=True)
    fm = data[:, [names.index("F1"), names.index("F2")], :].mean(axis=1, keepdims=True)
    rest = data[:, [names.index(c) for c in MEASURE_SOURCE_CHANNELS], :]
    info = mne.create_info(["FM"] + MEASURE_SOURCE_CHANNELS,
                           ep.info["sfreq"], ["eeg"] * 6)
    return mne.EpochsArray(np.concatenate([fm, rest], axis=1), info, verbose=False)


def _peaks_and_robust_thresholds(mask_raw: mne.io.RawArray, n_win: int,
                                 calib: tuple[int, int] | None = None):
    """
    Per-window peak |amplitude| for each source channel, and this recording's own
    5 x MAD robust threshold for each.

    Evaluated on `mask_raw`, the PRE-interpolation signal in the HARDWARE
    reference. Pre-interpolation for the reason given in step8b_holm_index: a
    spline-interpolated channel is a smooth blend of its neighbours and passes an
    amplitude threshold far more readily than a real electrode, which made
    retention an inverse proxy for how much of the value was synthetic.

    ALWAYS hardware reference, whichever reference is being displayed. Fixed
    2026-08-28 after review. Previously the mask was rebuilt in the displayed
    reference, which had two consequences:

      * Under the average reference the mask was computed on data that still
        contained the broken channels (`oc`) while the VALUES came from data where
        they had been interpolated away (`interp`). Average-referencing spreads one
        broken electrode across all ten channels at -1/10 weight, so the mask
        stopped being channel-specific: on p02/task_agent_personal the `robust`
        masks for parietal_alpha (mask channel Pz) and parietal_beta_asym (mask
        channels P3, P4) came out byte-identical.
      * Switching reference changed which windows were retained -- mean
        |delta retention| 7.4 pp, max 33.0 pp -- so the toggle was not a clean A/B.
        This is the same defect review found in the ocular toggle and which
        was fixed on 2026-08-27 by detecting bad channels once; the fix is extended
        to the reference toggle here.

    Whether a window's measured electrode was contaminated is a property of the
    measurement, not of a later linear re-referencing choice.

    `calib` optionally restricts the sample range the MAD threshold is calibrated
    over, without restricting which windows get a peak. Baselines pass their REST
    range: a baseline block is, by the protocol, 2 min of mental arithmetic + 2 min
    of deliberate eye movements + 2 min of rest, so calibrating over the whole block
    inflated the threshold applied to the rest windows by a median 1.15x and up to
    2.01x, making the criterion more permissive on the baseline than the same rule
    is on the task it is subtracted from.
    """
    pre = mask_raw.copy().pick(picks="eeg")
    ep = mne.make_fixed_length_epochs(pre, duration=HOLM_WINDOW_SEC,
                                      overlap=0.0, preload=True)
    names = ep.ch_names
    X = ep.get_data(copy=False)
    n = min(n_win, X.shape[0])

    peaks, thr = {}, {}
    full = pre.get_data(picks=[names.index(c) for c in MEASURE_SOURCE_CHANNELS])
    for c, x in zip(MEASURE_SOURCE_CHANNELS, full):
        peaks[c] = np.abs(X[:n, names.index(c), :]).max(axis=1)
        xc = x[calib[0]:calib[1]] if calib else x
        if xc.size == 0:
            xc = x
        t = ROBUST_SIGMA * 1.4826 * float(np.median(np.abs(xc - np.median(xc))))
        thr[c] = t if t > 0 else np.inf
    return n, peaks, thr


def channel_amplitudes(raw: mne.io.RawArray,
                       sl: tuple[int, int] | None = None) -> dict:
    """
    Robust SD of every channel a dashboard measure can read, on `raw` as given.

    Added 2026-08-28 after review. The dashboard's "is this value physiological"
    badge was computed from step 5's pre-interpolation, HARDWARE-referenced
    amplitudes and then shown next to a trace that might be average-referenced.
    Where a broken channel escaped bad-channel detection, average-referencing
    smeared it across all ten channels and 14 panels were badged "Measured" while
    the data actually plotted sat 2-14x outside the physiological range --
    p15/task_agent_speedscore has a 1355.6 uV Pz, and its F1/F2/P3/P4 go from
    ~20 uV to ~142 uV once the average reference folds Pz in.

    Amplitudes are therefore now computed on the data ACTUALLY PLOTTED, per
    reference, and stored alongside the pre-interpolation figures.
    """
    picks = mne.pick_types(raw.info, eeg=True, emg=False, eog=False)
    names = [raw.ch_names[i] for i in picks]
    X = raw.get_data(picks=picks)
    out = {}
    for c in MEASURE_SOURCE_CHANNELS:
        x = X[names.index(c)]
        if sl:
            x = x[sl[0]:sl[1]]
        rsd = float(1.4826 * np.median(np.abs(x - np.median(x)))) if x.size else float("nan")
        out[c] = round(rsd, 2) if np.isfinite(rsd) else None
    return out


def compute_measure_series(meas_raw: mne.io.RawArray,
                           mask_raw: mne.io.RawArray,
                           bad_chans: list[str] | None = None,
                           calib: tuple[int, int] | None = None,
                           seg_times: np.ndarray | None = None) -> dict | None:
    """
    Per-window value and robust-rejection mask for all four new measures.

    `meas_raw` supplies the spectra (post bad-channel interpolation);
    `mask_raw` supplies the rejection masks (pre-interpolation). Both are in the
    same reference. Spectra use calc_eeg_band_power_absolute -- Holm's single
    full-window Hann-tapered FFT -- chosen by the user so the new tabs use the
    same estimator as the index tab and avoid the multitaper band-widening
    documented in section 3.5.
    """
    ep = _measure_epochs(meas_raw)
    if ep is None:
        return None
    derived = _derived_epochs(ep)

    theta, _ = calc_eeg_band_power_absolute(derived, *NEW_THETA)
    alpha, _ = calc_eeg_band_power_absolute(derived, *NEW_ALPHA)
    beta, _ = calc_eeg_band_power_absolute(derived, *NEW_BETA)
    dn = derived.ch_names

    def band(arr, ch):
        return arr[:, dn.index(ch), :].mean(axis=1)

    def ln(x):
        # PSD is strictly positive for real data, but an interpolated-then-flat
        # channel can produce a zero. Emit NaN rather than -inf so the window is
        # dropped explicitly instead of dominating the axis.
        return np.log(np.where(x > 0, x, np.nan))

    values = {
        "fm_theta": band(theta, "FM"),
        "parietal_alpha": band(alpha, "Pz"),
        "parietal_beta_asym": ln(band(beta, "P4")) - ln(band(beta, "P3")),
        "frontal_alpha_asym": ln(band(alpha, "F2")) - ln(band(alpha, "F1")),
    }

    n_win, peaks, thr = _peaks_and_robust_thresholds(mask_raw, len(ep), calib)
    bad_chans = bad_chans or []

    # Windows taken across a discontinuity, excluded from every measure and every
    # mode -- see WINDOW_MAX_GAP_SEC. Applied to the VALUES rather than to the
    # robust mask, because `none` here is not a mask at all but the finiteness of
    # the value, so masking would have left the window in the unrejected series.
    # The cost is that `values` now carries null for two distinguishable reasons,
    # a non-positive PSD and this; `n_discontinuous` below says how many are the
    # second so the two are never confused.
    # Same window length and same signal _measure_epochs gridded, so the
    # mask lines up with `values` row for row.
    n_samp_meas = int(round(HOLM_WINDOW_SEC * float(meas_raw.info["sfreq"])))
    disc = (discontinuous_windows(seg_times, n_samp_meas, n_win)
            if seg_times is not None else np.zeros(n_win, dtype=bool))

    out = {}
    for m in NEW_MEASURES:
        chans = MEASURE_MASK_CHANNELS[m]
        v = values[m][:n_win].astype(float, copy=True)
        v[disc[:n_win]] = np.nan
        keep = np.ones(n_win, dtype=bool)
        for c in chans:
            keep &= peaks[c][:n_win] <= thr[c]
        finite = np.isfinite(v)

        # Which of THIS measure's channels are spline reconstructions rather than
        # measurements. Added 2026-08-28 after review: where F1 and F2 are both
        # interpolated they are rebuilt from the same donor electrodes, so
        # frontal_alpha_asym becomes a function of montage geometry carrying no
        # participant information -- measured across the 22 task recordings its
        # magnitude and within-recording SD both collapse ~4x when both frontal
        # channels are synthetic. Reported per measure so a consumer can separate
        # measurement from interpolant. Nothing is withheld (no gate, per user
        # decision); this is the label, not a filter.
        interp = [c for c in chans if c in bad_chans]

        out[m] = dict(
            mask_channels=chans,
            source_channels_interpolated=interp,
            fully_synthetic=bool(interp and len(interp) == len(chans)),
            robust_threshold_uv={c: (round(thr[c], 1) if np.isfinite(thr[c]) else None)
                                 for c in chans},
            values=[None if not np.isfinite(x) else round(float(x), 6) for x in v],
            robust="".join("1" if k else "0" for k in keep),
            # `none` was hardcoded to 100.0, which would have been a false claim for
            # any window whose PSD was non-positive and therefore emitted as null.
            retention_pct=dict(
                none=round(100.0 * float(finite.mean()), 1) if n_win else 0.0,
                robust=round(100.0 * float((keep & finite).mean()), 1) if n_win else 0.0),
        )
    return dict(n_windows=int(n_win), measures=out,
                n_discontinuous=int(disc[:n_win].sum()))


def holm_index_values(meas_raw: mne.io.RawArray, frontal: list[str]) -> np.ndarray | None:
    """
    The Holm index per window for an explicit frontal channel set.

    Mirrors step8b_holm_index's derivation exactly (time-domain frontal mean,
    Holm bands, Hann-tapered FFT). It exists so a BASELINE recording's index can
    be computed with the SAME frontal set as the task it is subtracted from --
    subtracting a baseline built from a different derivation would not be a
    baseline correction of the same quantity.

    Since the amplitude gate's removal (2026-09-02) the task side always uses
    F1+F2, so `frontal` is F1+F2 for every subtraction the pipeline actually
    performs. It stays a parameter because _baseline_medians still tabulates all
    three sets -- see its comment for why they are kept.
    """
    ep = _measure_epochs(meas_raw)
    if ep is None:
        return None
    names = ep.ch_names
    data = ep.get_data(copy=True)
    f = data[:, [names.index(c) for c in frontal], :].mean(axis=1, keepdims=True)
    p = data[:, [names.index(c) for c in HOLM_PARIETAL_CHANNELS], :].mean(axis=1, keepdims=True)
    derived = mne.EpochsArray(
        np.concatenate([f, p], axis=1),
        mne.create_info(["Fz_est", "Pz_est"], ep.info["sfreq"], ["eeg", "eeg"]),
        verbose=False)
    theta, _ = calc_eeg_band_power_absolute(derived, *HOLM_THETA)
    alpha, _ = calc_eeg_band_power_absolute(derived, *HOLM_ALPHA)
    return theta[:, 0, :].mean(axis=1) / alpha[:, 1, :].mean(axis=1)


def step9_new_measures(oc: mne.io.RawArray,
                       interp: mne.io.RawArray,
                       ocular: str,
                       bad_chans: list[str],
                       qc: dict,
                       seg_times: np.ndarray | None = None) -> dict:
    """
    One entry per reference mode for a single ocular mode.

    The reference changes the VALUES only. The artifact mask is built once, from
    the pre-interpolation hardware-referenced signal, so the toggle is a clean A/B
    over a fixed set of retained windows -- see _peaks_and_robust_thresholds for
    what went wrong when it was not.
    """
    out = {}
    for reference in REFERENCE_MODES:
        meas_raw = average_reference(interp) if reference == "average" else interp
        ser = compute_measure_series(meas_raw, oc, bad_chans,
                                     seg_times=seg_times)
        if ser is not None:
            # Amplitude of the data ACTUALLY PLOTTED in this reference, so the
            # dashboard can badge what it is showing rather than what step 5 saw.
            ser["amplitude_displayed_uv"] = channel_amplitudes(meas_raw)
        out[reference] = ser

    qc.setdefault("step9_new_measures", {})[ocular] = dict(
        measures=NEW_MEASURES,
        theta_band_hz=list(NEW_THETA),
        alpha_band_hz=list(NEW_ALPHA),
        beta_band_hz=list(NEW_BETA),
        window_s=HOLM_WINDOW_SEC,
        estimator="welch / single full-window Hann taper",
        references=REFERENCE_MODES,
        artifact_modes=NEW_ARTIFACT_MODES,
        gated=False,
        mask_reference="hardware (fixed; the reference toggle changes values only)",
        source_channels_interpolated={
            m: out["hardware"]["measures"][m]["source_channels_interpolated"]
            for m in NEW_MEASURES} if out.get("hardware") else {},
        clean=bool(out.get("hardware") is not None),
        notes=["no amplitude gate (user decision): these measures are computed "
               "for every task recording",
               "fm_theta is always the full F1+F2 time-domain mean. Since the "
               "index's own amplitude gate was removed on 2026-09-02 the index "
               "numerator is too, so the two are now the same derivation on "
               "every recording; before that date the index fell back to a "
               "single frontal channel where the gate dropped one, and on those "
               "six recordings fm_theta was NOT the index numerator"],
    )
    return out


# ===========================================================================
# STEP 11 - Parameter-sweep engine (added 2026-09-01)
#
# Recomputes every measure across the grid of processing choices defined at the
# top of this file, and ships the two quantities the dashboard needs to apply
# the three THRESHOLD choices itself:
#
#     peaks   per window, per channel, the largest |amplitude| in that window,
#             measured on the pre-interpolation signal in the hardware
#             reference. Feeds both the Holm-style absolute cap and the robust
#             per-recording criterion.
#     sigma   per channel, that recording's own 1.4826 x MAD. The robust
#             threshold at any distance k is exactly k * sigma, so the slider
#             needs no further pipeline output.
#
# Everything here is deliberately independent of MNE's spectral machinery: the
# grid is 720 spectra per recording and MNE's per-call overhead dominates at
# that count. psd_epochs() is asserted equal to the MNE call it replaces in
# verify_default_variant(), which runs on every pipeline invocation.
# ===========================================================================

DERIVED_CHANNELS = ["FM"] + MEASURE_SOURCE_CHANNELS   # FM, F1, F2, P3, P4, Pz


def _taper_bank(n_samp: int, method: str) -> tuple[np.ndarray, int]:
    """
    (tapers, segment_length) for one estimator.

    `hann`, `boxcar` and `multitaper` taper the whole window, so the segment
    length is the window length. `welch` splits the window into half-length
    segments at 50% overlap, which is where its variance reduction comes from
    and why its frequency resolution is half that of the others.
    """
    if method == "hann":
        return get_window("hann", n_samp, fftbins=True)[None, :], n_samp
    if method == "boxcar":
        return np.ones((1, n_samp)), n_samp
    if method == "multitaper":
        k = min(MULTITAPER_N_TAPERS, max(1, n_samp - 1))
        return np.asarray(dpss(n_samp, MULTITAPER_NW, k)), n_samp
    if method == "welch":
        # Half-length segments. For an odd window length the step is m//2 rather
        # than exactly m/2, so the overlap is one sample over half and the final
        # sample of the window is not covered -- verified to move the estimate by
        # 6e-5 relative against scipy.signal.welch, which is below the storage
        # precision.
        m = max(int(n_samp // 2), 8)
        return get_window("hann", m, fftbins=True)[None, :], m
    raise ValueError(f"unknown spectral estimator {method!r}")


def psd_epochs(X: np.ndarray, sfreq: float, method: str) -> tuple[np.ndarray, np.ndarray]:
    """
    One-sided PSD of every epoch, in the input's units squared per Hz.

    X is (n_epochs, n_channels, n_samples); the return is
    ((n_epochs, n_channels, n_freqs), freqs).

    The normalisation -- |rfft(x * w)|^2 / (sfreq * sum(w^2)), with every bin but
    DC and Nyquist doubled -- is scipy's and MNE's. For method="hann" this
    reproduces `epochs.compute_psd(method="welch", n_fft=n, n_per_seg=n,
    n_overlap=0, window="hann")` to floating-point identity, which is what
    calc_eeg_band_power_absolute has computed since 2026-08-27. Estimators are
    averaged over tapers and then over segments, so all four are on one scale.
    """
    n = X.shape[-1]
    tapers, m = _taper_bank(n, method)
    starts = list(range(0, n - m + 1, max(m // 2, 1))) if m < n else [0]

    acc, count = None, 0
    for s0 in starts:
        seg = X[..., s0:s0 + m]
        for w in tapers:
            spec = np.fft.rfft(seg * w, n=m, axis=-1)
            p = (np.abs(spec) ** 2) / (sfreq * float((w ** 2).sum()))
            p[..., 1:] *= 2.0
            if m % 2 == 0:
                p[..., -1] /= 2.0
            acc = p if acc is None else acc + p
            count += 1
    return acc / count, np.fft.rfftfreq(m, 1.0 / sfreq)


def band_mean(psd: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """
    Mean PSD over the bins inside [lo, hi], matching MNE's inclusive fmin/fmax.

    Short epochs and the Welch estimator both coarsen the frequency grid -- a 1 s
    Welch window resolves 2 Hz -- so a band can end up with very few bins. If it
    would have none at all the single nearest bin to the band centre is used, and
    the caller records the bin count so the dashboard can say how thin the
    estimate is rather than presenting it as equivalent.
    """
    sel = (freqs >= lo) & (freqs <= hi)
    if not sel.any():
        sel = np.zeros(freqs.shape, dtype=bool)
        sel[int(np.argmin(np.abs(freqs - 0.5 * (lo + hi))))] = True
    return psd[..., sel].mean(axis=-1)


def epoch_stack(sig: np.ndarray, n_samp: int) -> np.ndarray:
    """
    (n_channels, n_total) -> (n_epochs, n_channels, n_samp), dropping the tail.

    Identical to mne.make_fixed_length_epochs(duration=n_samp/sfreq, overlap=0):
    contiguous non-overlapping windows from sample 0, and a partial final window
    is discarded rather than zero-padded.
    """
    n_ep = sig.shape[-1] // n_samp
    if n_ep == 0:
        return np.zeros((0, sig.shape[0], n_samp), dtype=sig.dtype)
    return sig[:, :n_ep * n_samp].reshape(sig.shape[0], n_ep, n_samp).transpose(1, 0, 2)


def derived_six(eeg_raw: mne.io.RawArray) -> np.ndarray:
    """
    The six signals every measure reads: FM, then F1, F2, P3, P4, Pz.

    FM is the sample-by-sample mean of F1 and F2, built BEFORE any spectrum for
    the reason given in _derived_epochs: that is what a midline derivation
    physically is, and it is robust to one noisy channel in a way averaging the
    two channels' power is not.
    """
    names = eeg_raw.ch_names
    X = eeg_raw.get_data(picks=[names.index(c) for c in MEASURE_SOURCE_CHANNELS])
    fm = 0.5 * (X[MEASURE_SOURCE_CHANNELS.index("F1")]
                + X[MEASURE_SOURCE_CHANNELS.index("F2")])
    return np.vstack([fm[None, :], X])


def measure_matrix(sig6: np.ndarray, sfreq: float, n_samp: int,
                   method: str) -> tuple[np.ndarray, dict]:
    """
    (n_windows, 7) of VARIANT_VALUE_KEYS, plus the bin count behind each band.

    Raw band powers, not finished measures -- the two asymmetries excepted,
    because those are differences of logs and cannot be reconstructed from a
    rounded ratio. Everything else the dashboard forms itself, so the index is a
    division the page does at draw time rather than a column shipped per frontal
    set. (Until 2026-09-02 that was also what let the amplitude-gate slider pick
    a different frontal derivation without another pipeline run; the gate is gone
    and the numerator is now always F1+F2.)
    """
    X = epoch_stack(sig6, n_samp)
    if X.shape[0] == 0:
        return np.zeros((0, len(VARIANT_VALUE_KEYS)), np.float32), {}

    psd, freqs = psd_epochs(X, sfreq, method)
    i = {c: k for k, c in enumerate(DERIVED_CHANNELS)}

    theta = band_mean(psd, freqs, *NEW_THETA)        # NEW_THETA == HOLM_THETA
    alpha13 = band_mean(psd, freqs, *NEW_ALPHA)
    alpha12 = band_mean(psd, freqs, *HOLM_ALPHA)
    beta = band_mean(psd, freqs, *NEW_BETA)

    def ln(x):
        # A flat (interpolated-then-constant) channel can give exactly zero
        # power. Emit NaN so the window drops out explicitly rather than
        # becoming -inf and dominating the axis.
        return np.log(np.where(x > 0, x, np.nan))

    out = np.stack([
        theta[:, i["FM"]],
        alpha13[:, i["Pz"]],
        ln(beta[:, i["P4"]]) - ln(beta[:, i["P3"]]),
        ln(alpha13[:, i["F2"]]) - ln(alpha13[:, i["F1"]]),
        theta[:, i["F1"]],
        theta[:, i["F2"]],
        alpha12[:, i["Pz"]],
    ], axis=1)

    nbins = {name: int(((freqs >= lo) & (freqs <= hi)).sum())
             for name, (lo, hi) in (("theta", NEW_THETA), ("alpha", NEW_ALPHA),
                                    ("alpha_holm", HOLM_ALPHA), ("beta", NEW_BETA))}
    nbins["freq_resolution_hz"] = (round(float(freqs[1] - freqs[0]), 4)
                                   if len(freqs) > 1 else None)
    # The bin count is IDENTICAL for hann and multitaper, so on its own it tells a
    # consumer nothing about the multitaper's extra smoothing. Ship the smoothing
    # half-width explicitly: NW divided by the window length in seconds.
    nbins["smoothing_half_bandwidth_hz"] = (
        round(MULTITAPER_NW * sfreq / n_samp, 4) if method == "multitaper" else 0.0)
    return out.astype(np.float32), nbins


def peak_matrix(mask_eeg: mne.io.RawArray, n_samp: int) -> np.ndarray:
    """
    (n_windows, 5) largest |amplitude| per window for F1, F2, P3, P4, Pz.

    Always taken from the PRE-INTERPOLATION signal in the HARDWARE reference,
    whichever variant is on screen, for the two reasons
    _peaks_and_robust_thresholds gives: a spline is smooth by construction and
    passes an amplitude threshold far more readily than a real electrode, and
    re-referencing smears one broken channel across all ten so the mask stops
    being channel-specific. Whether a window's electrode was contaminated is a
    property of the measurement, not of a later processing choice.
    """
    names = mask_eeg.ch_names
    X = mask_eeg.get_data(picks=[names.index(c) for c in MEASURE_SOURCE_CHANNELS])
    E = epoch_stack(X, n_samp)
    if E.shape[0] == 0:
        return np.zeros((0, len(MEASURE_SOURCE_CHANNELS)), np.float32)
    return np.abs(E).max(axis=2).astype(np.float32)


def robust_sigma_uv(mask_eeg: mne.io.RawArray) -> dict:
    """
    Per channel, 1.4826 x MAD over the whole signal -- the unit the robust
    rejection distance is counted in.

    The pipeline's fixed criterion was ROBUST_SIGMA * 1.4826 * MAD; shipping the
    1.4826 * MAD factor alone makes the threshold at any distance k exactly
    k * sigma, so the browser reproduces the pipeline's own rule at k = 5 and
    generalises it continuously either side. A channel with zero dispersion gets
    None, which the dashboard reads as "never rejected" exactly as the pipeline's
    np.inf did.
    """
    names = mask_eeg.ch_names
    X = mask_eeg.get_data(picks=[names.index(c) for c in MEASURE_SOURCE_CHANNELS])
    out = {}
    for c, x in zip(MEASURE_SOURCE_CHANNELS, X):
        s = float(1.4826 * np.median(np.abs(x - np.median(x)))) if x.size else 0.0
        out[c] = round(s, 6) if s > 0 else None
    return out


def robust_sigma_sliding_uv(mask_eeg: mne.io.RawArray, n_samp: int,
                            win_sec: float) -> np.ndarray:
    """
    (n_windows, 5) of 1.4826 x MAD from the `win_sec` seconds CENTRED on each
    analysis window -- the unit the windowed robust rejection distance is counted
    in. Replaces the tiled-block version on 2026-09-02; see ROBUST_SLIDE_CHOICES_S
    for why sliding rather than tiled.

    This has to be computed here rather than in the browser: sigma is the MAD of
    the raw samples, and the browser only ever sees each window's peak. It is the
    one part of the sweep's browser-side thresholding that needs a pipeline run,
    and the reason a window-length toggle costs an archive column rather than a
    line of JavaScript.

    Window w spans samples [w*n_samp, (w+1)*n_samp) and is judged against the
    span centred on w's own centre sample. Rows line up 1:1 with peak_matrix's,
    so the browser indexes both with the same window number and there is no
    block, no clamp and no assignment rule to get wrong.

    A span that would run past either end is TRUNCATED, never shifted: the first
    window is calibrated on the half-span that exists rather than on a span
    centred somewhere it is not.

    A channel with zero dispersion in its span gets 0.0, and 0.0 travels all the
    way to the browser as 0.0 -- nothing converts it to a null, unlike the
    whole-recording sigma, which is JSON and uses None. The dashboard's threshold
    is `sg > 0 ? k*sg : Infinity`, so a 0.0 (and a NaN) becomes "never rejected"
    there instead. Same outcome by a different route; stated because the earlier
    wording claimed a conversion that does not happen.
    """
    names = mask_eeg.ch_names
    X = mask_eeg.get_data(picks=[names.index(c) for c in MEASURE_SOURCE_CHANNELS])
    n_ch = len(MEASURE_SOURCE_CHANNELS)
    n_tot = X.shape[1] if X.ndim == 2 else 0
    n_win = n_tot // n_samp if n_samp > 0 else 0
    if n_win == 0:
        return np.zeros((0, n_ch), np.float32)

    half = int(round(win_sec * float(mask_eeg.info["sfreq"]) / 2.0))
    out = np.zeros((n_win, n_ch), np.float32)
    for w in range(n_win):
        c = w * n_samp + n_samp // 2
        a, b = max(0, c - half), min(n_tot, c + half)
        seg = X[:, a:b]
        if seg.shape[1] == 0:
            continue
        med = np.median(seg, axis=1, keepdims=True)
        out[w] = 1.4826 * np.median(np.abs(seg - med), axis=1)
    return out




# MNE's own documented REST recipe (mne.set_eeg_reference's example), adopted
# 2026-09-01 after review found the previous pos=10.0 with no exclusion radius was
# an undocumented deviation. `exclude` keeps sources away from the sphere centre,
# where a dipole is not anatomically meaningful; the previous setting admitted
# them. Measured effect of the change: <= 1% of signal SD.
REST_SOURCE_SPACING_MM = 15.0
REST_SOURCE_EXCLUDE_MM = 30.0

_REST_FORWARD_CACHE: dict = {}


def rest_forward(info):
    """
    Lead field for the REST reference, over a three-layer spherical head model
    fitted to this montage. Cached: the Galea montage is identical in every
    recording, so this is built once per process.

    REST (Yao 2001) needs a forward model because it reconstructs what the
    montage would have measured against a reference at infinity. The sphere
    model is MNE's own documented recipe for this; it is an ASSUMPTION about
    head geometry, not a measurement, and with ten electrodes the problem it
    inverts is very poorly determined. That is why `rest` is offered as a
    sensitivity check beside `hardware` rather than as a better reference.
    """
    key = tuple(info["ch_names"])
    if key not in _REST_FORWARD_CACHE:
        sphere = mne.make_sphere_model("auto", "auto", info)
        src = mne.setup_volume_source_space(sphere=sphere,
                                            pos=REST_SOURCE_SPACING_MM,
                                            exclude=REST_SOURCE_EXCLUDE_MM,
                                            sphere_units="mm")
        _REST_FORWARD_CACHE[key] = mne.make_forward_solution(
            info, trans=None, src=src, bem=sphere)
    return _REST_FORWARD_CACHE[key]


def apply_sweep_reference(eeg_raw: mne.io.RawArray, reference: str) -> mne.io.RawArray:
    """One EEG-only Raw re-referenced as `reference` says. `hardware` is a no-op."""
    if reference == "hardware":
        return eeg_raw
    out = eeg_raw.copy()
    if reference == "average":
        out.set_eeg_reference([])
        out = out.set_eeg_reference(ref_channels="average", projection=True)
        out.apply_proj()
        return out
    if reference == "rest":
        return out.set_eeg_reference("REST", forward=rest_forward(out.info))
    raise ValueError(f"unknown reference {reference!r}")


def window_times(seg_times: np.ndarray, epoch_s: float, n_win: int,
                 sfreq: float = GALEA_SAMPLING_RATE) -> list:
    """
    Elapsed time at each window start, from the RECORDED timestamps.

    Not an assumed grid. Any gap in the recording -- a dropout, or a stream that
    stopped and restarted -- puts every later window at the wrong time if the
    window index is simply multiplied by the epoch length. A few seconds lost
    early in a 20-minute task is enough to misplace the end of it.
    """
    step = int(round(epoch_s * sfreq))
    return [round(float(seg_times[w * step] - seg_times[0]), 3)
            if w * step < len(seg_times) else None
            for w in range(n_win)]


def variant_key(interp: str, ocular: str, reference: str, fft: str) -> str:
    """The archive's name for one swept cell.

    `interp` is an interpolation TOKEN -- the channels that were interpolated,
    per interp_token -- and not the name of an interpolation mode. Since
    2026-09-08 there are five modes and they collapse onto far fewer distinct
    channel lists, so the archive is keyed by the list and `index.json` carries
    the per-recording mode -> token map. Callers that hold a mode name must
    resolve it through that map first.
    """
    return f"{interp}|{ocular}|{reference}|{fft}"


@functools.lru_cache(maxsize=None)
def bandpass_fir_half_width_s(sfreq: float = GALEA_SAMPLING_RATE) -> float:
    """
    Half the length, in seconds, of the FIR that step5_bandpass designs.

    DERIVED from MNE with step 5's own LOWCUT/HIGHCUT rather than written down,
    so it cannot drift away from the filter it describes: widen the passband and
    this shrinks with it. At LOWCUT 0.5 Hz and 250 Hz it is 1651 taps, i.e.
    3.3 s either side.

    This is how far a step in the signal reaches. The filter is zero-phase
    (filtfilt-style, applied forwards and backwards), so the ringing is
    SYMMETRIC about the discontinuity -- it runs backwards in time as well as
    forwards, which is why the window before a splice is as corrupted as the one
    after it and neither is visible as a "later" artifact.
    """
    h = mne.filter.create_filter(None, sfreq, l_freq=LOWCUT, h_freq=HIGHCUT,
                                 method="fir", verbose=False)
    return (len(h) - 1) / 2.0 / float(sfreq)


def discontinuous_windows(seg_times: np.ndarray, n_samp: int,
                          n_win: int,
                          sfreq: float = GALEA_SAMPLING_RATE) -> np.ndarray:
    """
    Boolean mask, one per window: is this window's spectrum taken across a
    discontinuity, or close enough to one that the bandpass has smeared it in?

    A window is flagged when a jump of more than WINDOW_MAX_GAP_SEC between
    consecutive samples falls anywhere within `bandpass_fir_half_width_s` of it.
    That reach is the point, and the first version of this function did not have
    it: it flagged only the window that CONTAINED the jump.

    WHY CONTAINMENT IS NOT ENOUGH. The gap is already in the segment when
    step5_bandpass runs, with a 1651-tap zero-phase FIR at the default cutoffs.
    So the step at the join is smeared 3.3 s in BOTH directions before any window
    is cut, and a window merely adjacent to the join carries the ringing without
    containing the join. Roughly three windows are corrupted at the 4 s default
    rather than one, and more at shorter window lengths, where the same 3.3 s of
    ringing covers more of them -- under the containment rule all but one of
    those entered mode `none` at full weight. This closes that.

    The dilation is in SAMPLES around the offending step, then mapped to windows,
    rather than window-by-window: a window is in when any part of it lies within
    the filter's reach of the join.

    Returns an all-False mask when there is nothing to judge, so a caller with no
    timestamps behaves exactly as it did before this existed rather than
    excluding everything.

    Raises when `seg_times` is too SHORT to cover the grid it is being asked
    about, which is always wrong: the mask would describe windows whose samples
    it does not hold. The other direction is NOT checked, because it is
    legitimate -- every caller may pass an `n_win` trimmed below
    floor(len/n_samp) (step 8b takes a min against the pre-interpolation epoch
    count, the sweep takes a min across every matrix in the epoch), and those
    windows are a prefix of the grid, so a longer `seg_times` is expected rather
    than suspicious. A caller handing over timestamps from a DIFFERENT recording
    of at least the same length is therefore still undetectable here; only the
    caller knows both, which is why each call site passes the timestamps of the
    very signal it just epoched.
    """
    out = np.zeros(int(max(n_win, 0)), dtype=bool)
    if n_win <= 0 or seg_times is None or len(seg_times) < 2 or n_samp <= 0:
        return out

    n_needed = int(n_win) * int(n_samp)
    if len(seg_times) < n_needed:
        raise ValueError(
            f"seg_times has {len(seg_times)} samples but {n_win} windows of "
            f"{n_samp} samples need {n_needed}. The mask would describe windows "
            f"whose samples it does not hold.")

    d = np.diff(np.asarray(seg_times, dtype=float))
    reach = int(round(bandpass_fir_half_width_s(float(sfreq)) * float(sfreq)))
    for i in np.flatnonzero(d > WINDOW_MAX_GAP_SEC):
        # The jump sits between samples i and i+1, so the corrupted span is
        # [i - reach, i + 1 + reach] in samples; // n_samp maps to windows.
        lo = max(0, (int(i) - reach) // n_samp)
        hi = min(int(n_win) - 1, (int(i) + 1 + reach) // n_samp)
        if hi >= lo:
            out[lo:hi + 1] = True
    return out


def compute_variant_bundle(oc_by_mode: dict, interp_lists: dict,
                           sfreq: float, qc: dict | None = None,
                           tag: str = "task", imu: dict | None = None,
                           seg_times: np.ndarray | None = None) -> dict:
    """
    The whole grid for one recording.

    `oc_by_mode` maps each ocular mode to that mode's PRE-INTERPOLATION,
    hardware-referenced, EEG-only Raw -- the same object step 8b and step 9 use
    for their masks. Bad channels are passed in rather than re-detected, so the
    interpolation toggle is a clean A/B over one fixed list. That is the defect
    review found in the ocular toggle, generalised to every toggle here.

    `interp_lists` maps each of the five INTERPOLATION_MODES to the channel list
    that mode interpolates for THIS group -- interp_lists_for() builds it. Since
    2026-09-08 the signals are built and stored per DISTINCT LIST rather than per
    mode: two modes that interpolate the same channels are the same signal, and
    storing them twice would have taken the sweep from 720 cells to 1800 and the
    deliverable past 700 MB for no new information. On this dataset the five
    modes collapse to 2.2 distinct lists per group. The mode -> token map goes
    into index.json so the dashboard can resolve a control position to an array.

    Returns per-epoch peaks and per-variant value matrices, plus the two
    per-recording scalars the browser thresholds against.
    """
    # Mode -> token, and the distinct tokens actually built. Sorted so the npz
    # member order is stable between runs.
    interp_tokens = {m: interp_token(interp_lists[m]) for m in INTERPOLATION_MODES}
    tokens = {}
    for mode in INTERPOLATION_MODES:
        tokens.setdefault(interp_tokens[mode], interp_lists[mode])

    bundle = dict(
        interp_lists={m: list(v) for m, v in interp_lists.items()},
        interp_tokens=dict(interp_tokens),
        sigma={oc: robust_sigma_uv(raw) for oc, raw in oc_by_mode.items()},
        # The per-block sigma that stood here went with the tiled windowed-robust
        # mode on 2026-09-02. Its replacement is per WINDOW, so it varies with the
        # epoch length and lives in `epochs[e]["sigma_slide"]` beside the peaks it
        # is indexed alongside, not here.
        amplitude_displayed={},
        band_bins={},
        epochs={},
    )

    # Signal variants: distinct interpolation list x ocular x reference. Built
    # once each and reused across all 40 (epoch, estimator) cells.
    # A failure has to name the MODES a reader can select, not the storage
    # token they happen to share -- and the token for "nothing interpolated" is
    # "-", which is unreadable as a failure label.
    def _modes_for(tok):
        ms = [m for m in INTERPOLATION_MODES if interp_tokens[m] == tok]
        return f"{tok} ({'/'.join(ms)})"

    signals, refs_failed = {}, []
    for ocular, oc in oc_by_mode.items():
        for token, chans in tokens.items():
            base = oc.copy()
            present = [c for c in chans if c in base.ch_names]
            if present:
                base.info["bads"] = present
                base = base.interpolate_bads(reset_bads=True)
            elif chans:
                # A named channel that this recording does not carry. Cannot
                # happen while assert_montage_names passes, and is recorded
                # rather than ignored if it ever does: silently interpolating
                # nothing under a mode that asked for something is exactly the
                # failure MANUAL_INTERPOLATION's validation exists to prevent.
                refs_failed.append(
                    f"{_modes_for(token)}|{ocular}: none of {chans} is a channel "
                    f"of this recording, so nothing was interpolated")
            for reference in SWEEP_REFERENCE_MODES:
                try:
                    ref_raw = apply_sweep_reference(base, reference)
                except Exception as exc:                       # pragma: no cover
                    refs_failed.append(
                        f"{_modes_for(token)}|{ocular}|{reference}: {exc}")
                    continue
                signals[(token, ocular, reference)] = derived_six(ref_raw)
                bundle["amplitude_displayed"][f"{token}|{ocular}|{reference}"] = \
                    channel_amplitudes(ref_raw)

    for epoch_s in EPOCH_CHOICES_S:
        n_samp = int(round(epoch_s * sfreq))
        entry = dict(
            peaks={oc: peak_matrix(raw, n_samp) for oc, raw in oc_by_mode.items()},
            # Per-window sigma for the windowed robust mode, one matrix per
            # (ocular mode, sliding length). Same shape and row order as `peaks`,
            # so the browser indexes them with the same window number.
            sigma_slide={f"{oc}|{L:g}": robust_sigma_sliding_uv(raw, n_samp, L)
                         for oc, raw in oc_by_mode.items()
                         for L in ROBUST_SLIDE_CHOICES_S},
            values={},
        )
        entry["n_windows"] = int(min([p.shape[0] for p in entry["peaks"].values()]
                                     or [0]))
        # Head motion per window, from the helmet's IMU. One matrix per epoch --
        # it depends on the window grid but on none of the other toggles, since
        # the IMU is a property of the recording rather than of any processing
        # choice. Computed after n_windows so its rows line up 1:1 with the
        # peaks. See the step 12 block above.
        entry["motion"] = motion_matrix(
            imu, seg_times if seg_times is not None else np.empty(0),
            n_samp, entry["n_windows"])
        # Windows whose own samples are not contiguous in real time. Like
        # `motion` this depends on the window grid and on nothing else, so it is
        # computed once per epoch and lines up 1:1 with the peaks. Unlike motion
        # it is not a criterion the reader can turn off: see WINDOW_MAX_GAP_SEC.
        entry["discontinuous"] = discontinuous_windows(
            seg_times if seg_times is not None else np.empty(0),
            n_samp, entry["n_windows"])
        for (token, ocular, reference), sig6 in signals.items():
            for fft in FFT_METHODS:
                v, nbins = measure_matrix(sig6, sfreq, n_samp, fft)
                entry["values"][variant_key(token, ocular, reference, fft)] = v
                bundle["band_bins"].setdefault(f"{epoch_s:g}|{fft}", nbins)
                entry["n_windows"] = min(entry["n_windows"], v.shape[0])
        bundle["epochs"][f"{epoch_s:g}"] = entry

    # Trim every matrix in an epoch to that epoch's common window count, so the
    # browser never has to reason about ragged lengths between the values it
    # plots and the peaks it thresholds them with.
    for entry in bundle["epochs"].values():
        n = entry["n_windows"]
        entry["peaks"] = {k: v[:n] for k, v in entry["peaks"].items()}
        entry["values"] = {k: v[:n] for k, v in entry["values"].items()}
        # sigma_slide too, so "every matrix" above stays true. It is built from
        # the same n_tot // n_samp as the peaks and so is never actually ragged,
        # but a matrix left out of a trim that claims to cover everything is how
        # the next ragged-length bug gets in.
        entry["sigma_slide"] = {k: v[:n] for k, v in entry["sigma_slide"].items()}
        entry["motion"] = entry["motion"][:n]
        entry["discontinuous"] = entry["discontinuous"][:n]

    if qc is not None:
        qc.setdefault("step11_variants", {})[tag] = dict(
            epochs_s=EPOCH_CHOICES_S,
            fft_methods=FFT_METHODS,
            references=SWEEP_REFERENCE_MODES,
            interpolation_modes=INTERPOLATION_MODES,
            # What each mode actually interpolates here, and which of them are
            # the same signal. A reader who moves the interpolation control and
            # sees nothing change can look this up rather than guess whether the
            # control is broken.
            interpolation_lists={m: list(v) for m, v in interp_lists.items()},
            interpolation_tokens=dict(interp_tokens),
            n_distinct_interpolations=len(tokens),
            ocular_modes=list(oc_by_mode),
            # The cells a reader can SELECT ...
            n_combinations=(len(EPOCH_CHOICES_S) * len(FFT_METHODS)
                            * len(SWEEP_REFERENCE_MODES) * len(INTERPOLATION_MODES)
                            * len(oc_by_mode)),
            # ... and the cells actually STORED, which is fewer whenever two
            # interpolation modes reach the same channel list.
            n_combinations_stored=(len(EPOCH_CHOICES_S) * len(FFT_METHODS)
                                   * len(SWEEP_REFERENCE_MODES) * len(tokens)
                                   * len(oc_by_mode)),
            n_signal_variants=len(signals),
            references_failed=refs_failed,
            # The order the three value-changing toggles are composed in, and
            # what it costs. _step5b_ica is deliberately FITTED before
            # interpolation, but the sweep APPLIES interpolation after it, so the
            # cell (interpolation=on, ocular=ica) stacks two rank reductions on a
            # ten-channel montage and then re-references on top. Nothing raises:
            # spline interpolation and both the average and REST operators are
            # functions of head geometry alone, never of the data covariance, so
            # they return well-defined arithmetic on a signal that no longer
            # spans the space they assume. Recorded because it is invisible
            # otherwise -- references_failed catches only thrown exceptions, and
            # the default cell that verify_default_variant checks is ocular=none,
            # so every ica cell is unverified.
            signal_chain="ocular -> interpolation -> reference",
            rank_cost=dict(
                n_eeg_channels=len(MEASURE_SOURCE_CHANNELS) + 5,
                components_removed_by_ocular_mode={
                    oc: ((qc.get("step5b_ocular", {}).get(oc) or {})
                         .get("n_components_removed") or 0)
                    for oc in oc_by_mode},
                # Per MODE since 2026-09-08: there is no single "channels
                # interpolated" any more, and reporting the automatic count
                # while the reader is holding the manual list would understate
                # the rank cost on exactly the recordings where the two differ.
                channels_interpolated_by_mode={m: len(v)
                                               for m, v in interp_lists.items()},
                note="each removed component and each interpolated channel costs "
                     "one degree of freedom, and an average reference costs one "
                     "more; on ten channels those add up quickly"),
            mask_signal="pre-interpolation, hardware reference, per ocular mode "
                        "(fixed; the other toggles change values only)",
            band_bins=bundle["band_bins"],
            clean=bool(not refs_failed),
            notes=(refs_failed or []),
        )
    return bundle


# Agreement tolerance for the default-cell check below, as an absolute plus a
# relative term, because the two paths differ for two unrelated reasons:
#
#   VERIFY_ATOL   MNE's own path is float64 while the sweep stores float32; and
#                 where the reference values come from JSON they were rounded to
#                 6 decimals, which is up to 5e-7 of absolute error. On the
#                 asymmetry measures, whose values sit near zero, that alone is
#                 2e-4 RELATIVE -- which is why a purely relative tolerance is
#                 the wrong test here.
#   VERIFY_RTOL   float32 storage is ~1.2e-7 relative; 1e-5 leaves two orders of
#                 headroom for accumulation while still failing any real change
#                 of estimator, band edge, epoch grid or derivation. Verified to
#                 catch a 0.01% perturbation on every column.
VERIFY_ATOL = 5e-7
VERIFY_RTOL = 1e-5


def reference_columns_via_mne(eeg_interp: mne.io.RawArray) -> np.ndarray | None:
    """
    All seven shipped columns, recomputed through MNE's own spectral machinery.

    This is the independent second opinion the default-cell check compares
    against. It deliberately goes through _measure_epochs / _derived_epochs /
    calc_eeg_band_power_absolute -- the pre-sweep code path -- rather than
    reusing anything in step 11.

    Rewritten 2026-09-01 after review. The previous check compared the sweep
    against step 9's four measures and step 8b's index, which left THREE of the
    seven shipped columns untested: `theta_f1` and `theta_f2` were exercised only
    when the amplitude gate happened to select a single frontal channel (3 of 22
    recordings each), and `alpha_holm_pz` only where an index was computable
    (12 of 22). A mutation test showed a 50% error injected into either
    single-frontal column passing silently.

    That is now the ONLY thing checking those two columns. The gate was removed
    on 2026-09-02, so nothing downstream consumes `theta_f1` or `theta_f2` any
    more -- which makes this assertion the sole guard on two of the seven columns
    the archive ships, rather than a backstop behind a live consumer. Reason
    enough to keep it, and to keep the columns verified rather than dropped.
    """
    ep = _measure_epochs(eeg_interp)
    if ep is None:
        return None
    derived = _derived_epochs(ep)
    theta, _ = calc_eeg_band_power_absolute(derived, *NEW_THETA)
    alpha13, _ = calc_eeg_band_power_absolute(derived, *NEW_ALPHA)
    alpha12, _ = calc_eeg_band_power_absolute(derived, *HOLM_ALPHA)
    beta, _ = calc_eeg_band_power_absolute(derived, *NEW_BETA)
    dn = derived.ch_names

    def band(arr, ch):
        return arr[:, dn.index(ch), :].mean(axis=1)

    def ln(x):
        return np.log(np.where(x > 0, x, np.nan))

    return np.stack([
        band(theta, "FM"),
        band(alpha13, "Pz"),
        ln(band(beta, "P4")) - ln(band(beta, "P3")),
        ln(band(alpha13, "F2")) - ln(band(alpha13, "F1")),
        band(theta, "F1"),
        band(theta, "F2"),
        band(alpha12, "Pz"),
    ], axis=1)


def verify_default_variant(bundle: dict, reference: np.ndarray | None) -> dict:
    """
    Assert that the sweep's default cell reproduces MNE's own answer.

    The default cell is (interpolation on, ocular none, hardware reference, 4 s
    epochs, Hann taper) -- the configuration every published number in
    the published figures were computed under. This runs on every invocation, for
    TASKS AND BASELINES, and its result is written into
    outputs/variants/index.json; build_dashboard.py refuses to build if it fails.

    Returns a STATUS, not a bare list. It previously returned `[]` in five
    distinct "nothing to compare" situations -- no measures, a missing hardware
    series, an ungated recording, all-NaN values, and a window-count mismatch --
    each of which was then written out as `reproduces_default: true` and printed
    as a clean bill of health. `verified` is now false unless real numbers were
    compared, and a length mismatch is an explicit problem rather than something
    `min(len(old), len(new))` quietly trimmed away.
    """
    status = dict(verified=False, n_windows=0, n_columns=0, problems=[])
    # The default cell is the MODE "on"; the archive is keyed by the channel
    # LIST that mode resolves to for this recording, so the token is read from
    # the bundle rather than assumed. Hardcoding "on" here would have looked
    # right and compared nothing -- variant_key would build a key no recording
    # has, and the "missing from the sweep" branch below would fire on all 44.
    token = (bundle.get("interp_tokens") or {}).get(DEFAULT_INTERPOLATION_MODE)
    if token is None:
        status["problems"].append(
            f"bundle carries no interpolation token for the default mode "
            f"{DEFAULT_INTERPOLATION_MODE!r}")
        return status
    key = variant_key(token, "none", "hardware", DEFAULT_FFT_METHOD)
    ep = bundle["epochs"].get(f"{DEFAULT_EPOCH_S:g}")
    if ep is None or key not in ep["values"]:
        status["problems"].append(f"default variant {key} missing from the sweep")
        return status
    if reference is None:
        status["problems"].append(
            "no MNE reference could be computed for this recording, so the "
            "default cell was not verified")
        return status

    V = np.asarray(ep["values"][key], dtype=float)
    R = np.asarray(reference, dtype=float)
    if V.shape[1] != R.shape[1]:
        status["problems"].append(
            f"column count differs: sweep {V.shape[1]}, MNE {R.shape[1]}")
        return status
    if V.shape[0] != R.shape[0]:
        # Trimming to the overlap would hide a systematic off-by-N in the epoch
        # grid, which is precisely the kind of error this check exists for.
        status["problems"].append(
            f"window count differs: sweep {V.shape[0]}, MNE {R.shape[0]}")
    n = min(V.shape[0], R.shape[0])
    if n == 0:
        status["problems"].append("no windows to compare")
        return status

    compared = 0
    for c, name in enumerate(VARIANT_VALUE_KEYS):
        a, b = R[:n, c], V[:n, c]
        finite_a, finite_b = np.isfinite(a), np.isfinite(b)
        if (finite_a != finite_b).any():
            status["problems"].append(
                f"{name}: {int((finite_a != finite_b).sum())} window(s) finite in "
                f"one path and not the other")
        both = finite_a & finite_b
        if not both.any():
            status["problems"].append(f"{name}: no finite value in either path")
            continue
        compared += 1
        allowed = VERIFY_ATOL + VERIFY_RTOL * np.abs(a[both])
        excess = np.abs(a[both] - b[both]) / allowed
        if excess.max() > 1.0:
            k = int(np.argmax(excess))
            status["problems"].append(
                f"{name}: worst window exceeds tolerance by {excess.max():.2f}x "
                f"(MNE {a[both][k]:.8g}, sweep {b[both][k]:.8g})")

    status["n_windows"] = int(n)
    status["n_columns"] = compared
    status["verified"] = bool(compared == len(VARIANT_VALUE_KEYS)
                              and not status["problems"])
    return status


# ===========================================================================
# STEP 10 - Resting baseline (added 2026-08-28)
#
# Structure supplied by the user: two Marker 8.0 entries 6 minutes apart, inside
# which 0-2 min is a mental-math task, 2-4 min eye movements, and 4-6 min rest.
# Only the final two minutes are used.
# ===========================================================================

def segment_start_sample(seg_times: np.ndarray, t_sec: float) -> int:
    """
    Earliest sample at or after `t_sec` seconds from the block's first marker.

    Each segment's windows are gridded from ITS OWN start, not from sample 0 of
    the block. Established 2026-08-28 after review, for the rest phase; kept for
    all three segments in the 2026-09-02 extension, and it matters more now, not
    less. Gridding from the block start put the window boundaries wherever the 4 s
    grid happened to land relative to the phase boundary: the measured effective
    rate is 250.11-250.33 Hz, so a 1000-sample window spans ~3.996 s and the
    window straddling 240 s was rejected, leaving windows that began at 243.7 s --
    116.0 s of the rest phase rather than 120, starting ~3.7 s late. Anchoring at
    the boundary uses the whole phase and yields 30 windows instead of 29.

    On a 120 s segment that is a 3% difference. On the 15 s eyes-closed segment,
    which holds 3 windows at the 4 s default, losing one to a straddled boundary
    would be a THIRD of the segment -- which is why each segment is cropped and
    gridded separately rather than indexed out of one pass over the block.
    """
    t = seg_times - seg_times[0]
    return int(np.searchsorted(t, t_sec, side="left"))


# Shortest segment the sweep can handle. Every epoch length in EPOCH_CHOICES_S
# must yield at least one window, and step 9's fixed 4 s epoching must too --
# mne.make_fixed_length_epochs RAISES on a raw shorter than one epoch rather than
# returning an empty Epochs, and a segment that produced no windows at the 10 s
# epoch would ship an empty band_bins entry, which build_dashboard.py treats as
# "the sweep was not run at one sampling rate" and refuses to build on.
MIN_SEGMENT_SEC = max(max(EPOCH_CHOICES_S), HOLM_WINDOW_SEC)


def segment_sample_bounds(seg_times: np.ndarray, segment: str) -> tuple[int, int]:
    """
    Half-open sample range [s, e) of one segment within a baseline block.

    Single source of truth for the bounds, so the crop and the timestamps that
    describe it cannot drift apart.
    """
    lo, hi = BASELINE_SEGMENTS[segment]
    n = len(seg_times)
    s = segment_start_sample(seg_times, lo)
    e = min(segment_start_sample(seg_times, hi), n)
    return s, e


def crop_to_segment(raw: mne.io.RawArray, seg_times: np.ndarray,
                    segment: str):
    """
    Restrict a baseline block to one named segment, returning (raw, times).

    Cropping happens AFTER filtering, so it introduces no new edge effects, and it
    means everything downstream -- the spectra, the per-window peaks, the robust
    MAD threshold and the reported amplitudes -- is computed on that segment
    alone. That is the point: the robust threshold was once calibrated over the
    whole 6-minute block, which by the protocol (2 min mental arithmetic, 2 min
    deliberate eye movements, 2 min rest) inflated it by a median 1.15x and up to
    2.01x, making the criterion more permissive on the baseline than the identical
    rule is on the task it is subtracted from. One crop per segment keeps that fix
    for all three, and keeps the eye-movement phase out of the calibration of the
    two segments that do not lie inside it.

    BOUNDED AT BOTH ENDS, which the rest phase's crop was not (2026-09-02; the
    consequence was found on review). `crop_to_rest` set tmin only and ran to the
    end of the recording, so samples PAST the 360 s marker -- up to 2.4 s of them,
    since blocks span 359.9-362.4 s -- fed the rest phase's sigma, its channel
    amplitudes and its MAD threshold, even though `rest_window_indices` already
    excluded those windows from the median. That is the same defect the
    2026-08-28 fix addressed one level up (calibrating on the block rather than
    the phase), left behind at the phase's own trailing edge.

    It is therefore a deliberate correction, and it MOVES PUBLISHED NUMBERS: the
    rest sigma shifts by up to ~3.4% per channel, one robust-mask window flips on
    p08/baseline_ai_speedscore, and the rest window count drops 122->120 at the
    1 s epoch. Measured, recorded in PROVENANCE, and not to be described as
    reproducing the pre-2026-09-02 rest numbers, because it does not.

    Returns (None, empty) where the block is too short to contain the segment --
    p04/baseline_ai_speedscore is reconstructed from a single marker, and a short
    block would otherwise raise inside MNE's epoching. See MIN_SEGMENT_SEC.
    """
    s, e = segment_sample_bounds(seg_times, segment)
    sf = float(raw.info["sfreq"])
    if e - s < int(round(MIN_SEGMENT_SEC * sf)):
        return None, seg_times[0:0]
    # tmax is inclusive in MNE, so e-1 is the last sample kept.
    out = raw.copy().crop(tmin=s / sf, tmax=(e - 1) / sf)
    return out, seg_times[s:e]


def segment_window_indices(times: np.ndarray, block_t0: float, n_win: int,
                           segment: str,
                           epoch_s: float = HOLM_WINDOW_SEC) -> list[int]:
    """
    Which windows of a SEGMENT-CROPPED signal lie entirely inside that segment's
    bounds in the original block.

    Window edges are tested against the recorded timestamps rather than an assumed
    contiguous grid, for the same reason window times are (step 2).

    IN PRACTICE THIS IS NOW THE IDENTITY LIST. Since the crop became bounded at
    both ends (2026-09-02) every sample already lies inside [lo, hi), so no window
    can be rejected -- verified across 3 segments x 10 epoch lengths. Under the
    old open-ended rest crop it did real work, rejecting windows that ran past
    360 s. Nor does it catch a dropout: a gap inside the segment changes what a
    window CONTAINS, but its timestamps still fall in range.

    Kept, deliberately, as the assertion that the crop and the bounds agree. If
    either ever changes -- a segment defined by markers rather than offsets, an
    unbounded crop reintroduced -- this is what stops out-of-range windows
    reaching a median, and it costs one pass over an index list to keep.
    """
    lo, hi = BASELINE_SEGMENTS[segment]
    step = int(round(epoch_s * GALEA_SAMPLING_RATE))
    keep = []
    for w in range(n_win):
        s, e = w * step, (w + 1) * step - 1
        if e >= len(times):
            break
        t0 = float(times[s] - block_t0)
        t1 = float(times[e] - block_t0)
        if t0 >= lo and t1 <= hi:
            keep.append(w)
    return keep


def _median_over(values, mask_str, idx):
    """Median of `values` over rest windows `idx` that the mask retains."""
    v = [values[i] for i in idx
         if i < len(values) and values[i] is not None
         and (mask_str is None or mask_str[i] == "1")]
    if not v:
        return None, 0
    return round(float(np.median(v)), 6), len(v)


def _median_dispersion(values, mask_str, idx):
    """
    Robust standard error of the baseline median, so a median resting on 29 windows
    that happen to be filthy cannot be mistaken for a stable one.

    Added 2026-08-28 after review, which showed the existing "fewer than 10 windows"
    flag misses the real failure: p02/baseline_ai_speedscore has a full 29 windows
    and a bootstrap SE of 1042 on a median of 791, and p08/baseline_ai_speedscore an
    SE of 256 on 1673. SE(median) ~ 1.2533 * sigma / sqrt(n), with sigma taken as the
    MAD-based robust SD so one wild window does not set it.
    """
    v = [values[i] for i in idx
         if i < len(values) and values[i] is not None
         and (mask_str is None or mask_str[i] == "1")]
    if len(v) < 2:
        return None
    a = np.asarray(v, dtype=float)
    sigma = 1.4826 * float(np.median(np.abs(a - np.median(a))))
    return round(float(1.2533 * sigma / np.sqrt(len(a))), 6)


def step10_baseline(rec: dict, seg: pd.DataFrame,
                    bandpassed: mne.io.RawArray,
                    bad_chans: list[str], z_detail: dict, qc: dict,
                    imu: dict | None = None,
                    pairing: dict | None = None,
                    ica_pair: tuple | None = None) -> dict | None:
    """
    Every baseline segment, swept independently. See BASELINE_SEGMENTS.

    Extended 2026-09-02 at the user's request from one segment (rest) to three
    (rest, math, eyes_closed), so the dashboard can choose which condition a task
    is referenced against. `rest` remains the default, and each segment is
    cropped, gridded and calibrated as the rest phase alone used to be -- with one
    deliberate difference that DOES move rest's published numbers: the crop is now
    bounded at both ends. See crop_to_segment for the measured effect.

    The ocular correction and the bad-channel interpolation run ONCE on the whole
    block and are cropped afterwards. Doing it the other way -- correcting each
    crop separately -- would fit the Gratton regression on 15 s of eyes-closed
    data for that segment, so the same electrode would get a different correction
    depending on which baseline the reader selected.

    `bad_chans` is the PAIR's list since 2026-09-04, not this recording's own, and
    it was decided on the task joined to this block's REST segment. Two
    consequences worth stating because neither is visible from here:

      * the `math` and `eyes_closed` segments are interpolated according to
        evidence drawn from outside themselves. The same is true of the ICA
        decomposition since 2026-09-05: it is fitted on the task joined to this
        block's REST crop and then applied to the whole block, so choosing either
        of the other two segments gets components chosen without seeing them. That is the deliberate trade --
        one interpolation decision per pair, stable across a control the reader
        can move, rather than three baselines that differ from each other in
        which electrodes are real.
      * a channel this block alone would have flagged may now stay raw, if the
        ~20-minute task it is judged with found it quiet. Where that happens it
        stays raw on BOTH sides of the subtraction, which is the property the
        change was made to buy. The channel amplitudes are published either way
        and _baseline_qc still flags anything outside the run's own band.
    """
    seg_times = seg["Timestamp"].to_numpy(float)
    block_t0 = float(seg_times[0])

    out = dict(
        block_duration_s=round(float(seg_times[-1] - seg_times[0]), 1),
        interpolated=[c for c in MEASURE_SOURCE_CHANNELS if c in bad_chans],
        segment_bounds_s={k: list(v) for k, v in BASELINE_SEGMENTS.items()},
        segment_order=list(BASELINE_SEGMENT_ORDER),
        default_segment=DEFAULT_BASELINE_SEGMENT,
        segments={},
    )

    # One ocular correction and one interpolation for the whole block, reused by
    # every segment.
    full_by_mode = {}
    for ocular in OCULAR_MODES:
        oc_full = step5b_ocular(bandpassed, ocular, qc, ica_pair=ica_pair)
        interp_full = step6_interpolate(oc_full, qc, f"baseline_{ocular}",
                                        bad_chans, z_detail, pairing)
        full_by_mode[ocular] = (oc_full, interp_full)

    # One set of interpolation lists for the whole block, and the same set the
    # task will use: interp_lists_for is a pure function of the group's bad
    # channels and the group's manual entry, and both are group-level.
    interp_lists = interp_lists_for(bad_chans, manual_interpolation_for(rec))

    for segment in BASELINE_SEGMENT_ORDER:
        s_out = _baseline_segment(segment, full_by_mode, seg_times, block_t0,
                                  bandpassed, bad_chans, z_detail, qc, imu,
                                  pairing=pairing, interp_lists=interp_lists)
        if s_out is not None:
            out["segments"][segment] = s_out

    _baseline_qc(out, seg_times, qc)
    return out


def _baseline_segment(segment: str, full_by_mode: dict, seg_times: np.ndarray,
                      block_t0: float, bandpassed: mne.io.RawArray,
                      bad_chans: list[str], z_detail: dict,
                      qc: dict, imu: dict | None = None,
                      pairing: dict | None = None,
                      interp_lists: dict | None = None) -> dict | None:
    """
    One baseline segment: medians per measure per toggle, plus its own sweep.

    Two decisions from the researcher, both
    unchanged by the 2026-09-02 extension:

    * The artifact criterion selected for the TASK is applied to the baseline's
      windows too, so the two are measured the same way. The median is taken
      over the windows that criterion RETAINS. `holm_imputed` therefore shares
      `holm_strict`'s median -- the two differ only in whether rejected windows
      are refilled to keep a time course continuous, and a median has no gaps to
      fill. Refilling would insert duplicated neighbour values and bias it.
    * There is no amplitude gate, so a baseline is never dropped for being
      implausible. Its robust SD is reported per channel instead, and the
      dashboard flags any baseline with a channel outside the physiological
      range so a corrupt rest level cannot be read as a real one.

    Everything below operates on the SEGMENT-CROPPED signal, so no other phase of
    the block influences this segment's values, robust threshold or amplitudes.
    """
    out = dict(
        segment=segment,
        bounds_s=list(BASELINE_SEGMENTS[segment]),
        duration_s=round(BASELINE_SEGMENTS[segment][1]
                         - BASELINE_SEGMENTS[segment][0], 1),
        ocular={},
    )
    amp, implausible = None, []
    oc_by_mode = {}

    for ocular in OCULAR_MODES:
        oc_full, interp_full = full_by_mode[ocular]

        # This segment only, from here down.
        oc, seg_ts = crop_to_segment(oc_full, seg_times, segment)
        interp, _ = crop_to_segment(interp_full, seg_times, segment)
        if oc is None or interp is None:
            return None
        oc_by_mode[ocular] = oc.copy().pick(picks="eeg")

        if ocular == "none":
            # Pre-interpolation, hardware-referenced amplitude of the rest samples
            # actually used. Taken from the UNCORRECTED branch so it describes the
            # electrode rather than a correction applied to it. This used to test
            # `amp is None`, i.e. "whichever mode comes first", which was the same
            # thing only because "none" happens to head OCULAR_MODES -- reordering
            # the list would have published a post-correction amplitude under an
            # "as recorded" label with nothing failing.
            a = channel_amplitudes(oc)
            # `plausible` is filled in by apply_amplitude_labels against the
            # run-derived band; None means not assessed yet.
            amp = {c: dict(robust_sd_uv=a[c], plausible=None)
                   for c in MEASURE_SOURCE_CHANNELS}
            implausible = []

        entry = dict(reference={}, index={})

        # ---- the four new measures, both references ------------------------
        # The mask is built once from the hardware-referenced pre-interpolation rest
        # signal; the reference changes the values only.
        hardware_ser = None
        for reference in REFERENCE_MODES:
            meas_raw = average_reference(interp) if reference == "average" else interp
            ser = compute_measure_series(meas_raw, oc, bad_chans,
                                         seg_times=seg_ts)
            if reference == "hardware":
                hardware_ser = ser
            if ser is None:
                entry["reference"][reference] = None
                continue
            idx = segment_window_indices(seg_ts, block_t0, ser["n_windows"], segment)
            per = {}
            for m in NEW_MEASURES:
                d = ser["measures"][m]
                per[m] = {}
                for mode in NEW_ARTIFACT_MODES:
                    mask = None if mode == "none" else d["robust"]
                    med, n = _median_over(d["values"], mask, idx)
                    per[m][mode] = dict(
                        median=med, n_windows=n,
                        se=_median_dispersion(d["values"], mask, idx))
            entry["reference"][reference] = dict(
                n_windows_in_segment=len(idx),
                amplitude_displayed_uv=ser.get("amplitude_displayed_uv")
                or channel_amplitudes(meas_raw),
                source_channels_interpolated={
                    m: ser["measures"][m]["source_channels_interpolated"]
                    for m in NEW_MEASURES},
                measures=per)

        # ---- the Holm index, hardware reference only ------------------------
        # All three frontal sets are tabulated so the baseline uses the SAME
        # derivation as the task it is subtracted from. Since the gate's removal
        # (2026-09-02) the task side is always F1+F2, so "F1" and "F2" are no
        # longer reachable by any subtraction; they are kept because they are the
        # only per-electrode frontal baseline level published anywhere, and
        # dropping them would silently change baselines.json's shape.
        n_all = hardware_ser["n_windows"] if hardware_ser else 0
        idx = (segment_window_indices(seg_ts, block_t0, n_all, segment)
               if n_all else [])
        _, peaks, thr = (_peaks_and_robust_thresholds(oc, n_all)
                         if n_all else (0, {}, {}))
        for frontal in (["F1"], ["F2"], ["F1", "F2"]):
            key = "+".join(frontal)
            vals = holm_index_values(interp, frontal) if n_all else None
            if vals is None:
                entry["index"][key] = None
                continue
            n = min(len(vals), n_all)
            chans = frontal + HOLM_PARIETAL_CHANNELS
            peak_ok_strict = np.ones(n, dtype=bool)
            peak_ok_robust = np.ones(n, dtype=bool)
            for c in chans:
                peak_ok_strict &= peaks[c][:n] <= HOLM_REJECT_UV
                peak_ok_robust &= peaks[c][:n] <= thr[c]
            v = [None if not np.isfinite(x) else round(float(x), 6) for x in vals[:n]]
            # The contiguity rule applies here too. No baseline segment in this
            # dataset contains a gap -- both discontinuities are inside task
            # recordings -- so nothing moves today. It is here because a baseline
            # median must be computed by the same rule as the series it is
            # subtracted from: leaving it out would make `baselines.json` the one
            # output where `none` still means "every window", and would put the
            # two sides of the subtraction back under different rules, which is
            # the defect this whole change set exists to remove.
            b_disc = discontinuous_windows(
                seg_ts, int(round(HOLM_WINDOW_SEC * float(interp.info["sfreq"]))), n)
            b_ok = ~b_disc
            masks = {
                "holm_strict": "".join("1" if k else "0"
                                       for k in (peak_ok_strict & b_ok)),
                "holm_imputed": "".join("1" if k else "0"
                                        for k in (peak_ok_strict & b_ok)),
                "none": "".join("1" if k else "0" for k in b_ok),
                "robust": "".join("1" if k else "0"
                                  for k in (peak_ok_robust & b_ok)),
            }
            entry["index"][key] = {}
            for mode, mk in masks.items():
                med, nw = _median_over(v, mk, idx)
                entry["index"][key][mode] = dict(
                    median=med, n_windows=nw,
                    se=_median_dispersion(v, mk, idx),
                    interpolated=[c for c in chans if c in bad_chans])

        out["ocular"][ocular] = entry

    # ---- step 11: the same sweep, over this segment ----------------------
    # Peaks, the robust sigma and every value are computed on the segment-cropped
    # signal, so the criterion the dashboard applies to the baseline is measured
    # exactly as the one it applies to the task -- and every slider position moves
    # both sides of the subtraction together.
    tag = f"baseline_{segment}"
    # This segment's own timestamps, so motion is matched to the samples the
    # segment actually holds rather than to the whole six-minute block.
    _s0, _e0 = segment_sample_bounds(seg_times, segment)
    if interp_lists is None:                                   # pragma: no cover
        raise ValueError(
            f"_baseline_segment({segment!r}) was given no interpolation lists. "
            f"They are decided once per task/baseline group in step10_baseline "
            f"and must not be re-derived here: a baseline that swept a different "
            f"list from its task would put a spline on one side of every ratio "
            f"and the raw electrode on the other.")
    bundle = compute_variant_bundle(oc_by_mode, interp_lists,
                                    float(bandpassed.info["sfreq"]), qc, tag,
                                    imu=imu, seg_times=seg_times[_s0:_e0])
    # Recomputed here rather than reused from the loop variable: crop_to_segment
    # returns the same slice for both ocular modes, but relying on which one the
    # loop happened to leave behind is the kind of thing that breaks silently.
    _s, _e = segment_sample_bounds(seg_times, segment)
    seg_times_all = seg_times[_s:_e]
    bundle["window_index"] = {
        e: segment_window_indices(seg_times_all, block_t0,
                                  bundle["epochs"][e]["n_windows"], segment,
                                  float(e))
        for e in bundle["epochs"]}
    out["variants"] = bundle

    # The same check on the baseline side. It previously ran on tasks only, so
    # every resting baseline -- one whole side of every baseline subtraction --
    # was unverified while the console reported success on "every recording".
    # It now runs on all THREE segments, for the same reason: a segment the
    # dashboard can subtract is a segment that has to be verified.
    interp_seg = step6_interpolate(oc_by_mode["none"].copy(), {},
                                   f"verify_{tag}", bad_chans, z_detail, pairing)
    check = verify_default_variant(bundle, reference_columns_via_mne(interp_seg))
    qc["step11_variants"][tag]["reproduces_default"] = check["verified"]
    qc["step11_variants"][tag]["default_check"] = check
    if not check["verified"]:
        qc["step11_variants"][tag]["clean"] = False
        qc["step11_variants"][tag]["notes"] = (
            (qc["step11_variants"][tag].get("notes") or [])
            + (check["problems"] or ["nothing was compared"]))
    out["variant_default_check"] = check

    # "none" by name rather than by position, for the same reason as the
    # amplitude above. (This one is insensitive either way -- a window COUNT
    # cannot change with the ocular mode -- but the two should not disagree
    # about how they name the uncorrected branch.)
    hw = out["ocular"]["none"]["reference"]["hardware"]
    out["n_windows"] = hw["n_windows_in_segment"] if hw else 0
    out["amplitude"] = amp
    out["implausible_channels"] = implausible
    return out


def _baseline_qc(out: dict, seg_times: np.ndarray, qc: dict) -> None:
    """
    Block-level QC for the baseline, across every segment.

    The window-count warning is per segment and thresholded per segment, because
    "fewer than 10 windows" is a defect on a 120 s segment and the DESIGN on the
    15 s one -- eyes_closed holds 3 windows at the 4 s default and 1 at 8 s or
    longer. Warning on it every time would train a reader to ignore the warning
    that matters.
    """
    span = float(seg_times[-1] - seg_times[0])
    span_ok = bool(abs(span - BASELINE_BLOCK_SEC) < 5.0)

    segs = out["segments"]
    # Prefer the default segment, but fall back to whichever survived: sourcing
    # these from `rest` alone meant that a block missing its rest phase reported
    # no amplitudes and suppressed the implausible-channel note, even where the
    # segments it DID produce had implausible channels.
    src = next((segs[s] for s in [DEFAULT_BASELINE_SEGMENT] + BASELINE_SEGMENT_ORDER
                if s in segs), {})
    amp = src.get("amplitude")
    implausible = src.get("implausible_channels") or []
    n_by_seg = {k: (v.get("n_windows") or 0) for k, v in segs.items()}

    notes = []
    missing = [s for s in BASELINE_SEGMENT_ORDER if s not in segs]
    if missing:
        notes.append(
            f"segment(s) {missing} could not be cut from this block -- it spans "
            f"{span:.1f}s and the segment needs samples beyond that; the dashboard "
            f"reports them as unavailable for this recording")
    if implausible:
        notes.append(
            f"baseline channel(s) {implausible} are outside "
            f"this run's derived amplitude band; "
            f"the median is still used (no gate, per user decision) but the "
            f"dashboard flags it")
    if out["interpolated"]:
        notes.append(
            f"baseline channel(s) {out['interpolated']} are spline reconstructions, "
            f"not measurements; any measure reading them is partly synthetic")
    if not span_ok:
        notes.append(f"block spans {span:.1f}s, not the expected "
                     f"{BASELINE_BLOCK_SEC:.0f}s")
    # Only the two 120 s segments are held to the 10-window bar. eyes_closed is
    # 15 s BY DESIGN and can never clear it; its smallness is reported as a fact
    # about the segment (n_windows, below, and prominently in the dashboard)
    # rather than as a per-recording defect.
    for s in ("rest", "math"):
        if s in segs and n_by_seg[s] < 10:
            notes.append(f"only {n_by_seg[s]} {s} windows available before "
                         f"any rejection")

    qc["step10_baseline"] = dict(
        segments_s={k: list(v) for k, v in BASELINE_SEGMENTS.items()},
        default_segment=DEFAULT_BASELINE_SEGMENT,
        block_structure="0-2 min mental math, 2-4 min eye movements, 4-6 min rest "
                        "(user-supplied). Since 2026-09-02 three segments are "
                        "swept and the dashboard chooses between them: rest "
                        "(240-360 s, the default), math (0-120 s) and eyes_closed "
                        "(225-240 s, the last 15 s of the eye-movement phase)",
        block_span_s=round(span, 1),
        n_windows_by_segment=n_by_seg,
        # Kept under its old name and still the REST count, so a consumer reading
        # it gets the number it has always meant rather than a silently redefined
        # one. n_windows_by_segment is where the other two live.
        n_rest_windows=n_by_seg.get("rest", 0),
        amplitude=amp,
        implausible_channels=implausible,
        interpolated_channels=out["interpolated"],
        clean=bool(not implausible and not out["interpolated"] and span_ok
                   and not missing and n_by_seg.get("rest", 0) >= 10),
        notes=notes,
    )


# ===========================================================================
# STEP 12 - Head motion from the helmet's IMU (added 2026-09-03)
#
# Suggested by a professional who works with this helmet: motion is a common
# source of EEG artifact, and the Galea's aux stream carries an IMU, so the IMU
# can be used as a rejection criterion in its own right.
#
# WHAT THE AUX FILE ACTUALLY CONTAINS. Checked, not assumed. Every one of the 49
# recordings has an `openbci-raw-aux_*.txt` beside its exg file, all parse, all
# at an effective 50.69-50.85 Hz against a nominal 50. Twenty-one columns, of
# which three groups are inertial: Accelerometer X/Y/Z, Gyroscope X/Y/Z and
# Magnetometer X/Y/Z.
#
#   * The ACCELEROMETER is dominated by gravity: on the recording checked in
#     detail its vector magnitude sits at 0.506 with a 1st-99th percentile spread
#     of 0.491-0.519. It therefore measures ORIENTATION, and the informative
#     quantity is how fast that vector moves, not how big it is -- hence the
#     jerk, the norm of its sample-to-sample difference.
#   * The GYROSCOPE is the dynamic channel: near zero at rest, with excursions
#     past 22 per axis. It is the direct head-rotation signal.
#   * The MAGNETOMETER is not offered, and the honest reason is that it was not
#     evaluated rather than that it was tested and rejected. It is heavily
#     quantised (32/38/46 distinct values per axis on the checked recording) and
#     it measures heading against an external field, which a nearby ferrous
#     object or the hardware itself can move without the head moving. An earlier
#     version of this comment claimed it updates at only ~1.2 Hz and was
#     therefore too slow for a 4 s window; that was wrong -- it was computed as
#     distinct-values-over-duration, and its vector actually changes on 8,132
#     sample transitions in 875.9 s, about 9.3 Hz. Corrected 2026-09-03.
#
# UNITS ARE NOT KNOWN AND ARE NOT ASSERTED. The accelerometer implies a scale on
# which 1 g is about 0.5 units, so it is not in g; the gyroscope is plausibly
# deg/s but the file header does not say and no vendor specification is in this
# repository. THIS IS WHY THE THRESHOLD IS RELATIVE. An absolute cut in units
# that cannot be named is exactly the unsourced-threshold problem that the
# 1-50 uV amplitude range turned out to be. A per-recording criterion needs no
# unit at all.
#
# ALIGNMENT is by absolute Timestamp, the same clock the exg file uses: across
# all 49 recordings the two streams start a median 4.0 ms apart (max 16.1) and
# end a median 4.2 ms apart (max 16.1). The aux file carries no markers (its
# Marker column is all zeros on all 49), so timestamps are the only alignment
# available -- which is what the rest of the pipeline already uses for window
# times.
#
# WHAT THE DATA SAYS THIS BUYS, over the 22 task recordings at the 4 s default.
# These figures were CORRECTED on 2026-09-03 after review; the first version of
# this block correlated motion against the RAW UNFILTERED segment, whose peaks
# are dominated by DC offset and drift, and drew the opposite conclusion from
# the one the data supports. Against the bandpassed peaks the pipeline actually
# thresholds on:
#
#   * Motion is a STRONG and CONSISTENT predictor of per-window peak amplitude.
#     Spearman rho between the accelerometer jerk and the index channels' peak
#     is +0.67 median, at or above 0.3 on 21 of the 21 evaluable recordings, and
#     negative on none.
#   * It tracks BAND POWER, which is what the measures read. In the top decile of
#     motion, median frontal theta is 5.19x and parietal alpha 2.70x their level
#     in the remaining windows, with no recording going the other way.
#   * The amplitude rule ALREADY REJECTS almost all of it. Of the top-decile
#     motion windows, a mean of 1.0% survive `robust` at k=5 (median 0%, max
#     6.7%). Motion and the amplitude rules are largely selecting the SAME
#     windows.
#   * As a criterion in its own right it is selective. At k=3 it keeps a mean
#     94.7% of the assessed windows, and what it removes carries a median 7.2x
#     the frontal theta and 3.3x the parietal alpha of what it keeps.
#
#     Both ratios are MEDIANS OF PER-RECORDING RATIOS and they move with the set
#     of recordings included: the frontal-theta figure ranges 5.7x to 7.9x
#     depending on whether recordings with a dead IMU or few assessed windows are
#     counted, and its per-recording spread runs 1.2x to 2818x. Quote it as "much
#     larger", not as a point estimate. Earlier drafts of this block said 2.7%
#     and 7.89x; both were computed before the dead-sensor guard existed, and the
#     first was inflated by recordings whose IMU never moved (a constant stream
#     makes the decile boundary meaningless). Corrected 2026-09-03.
#
# READ THAT HONESTLY. This mode is not finding artifacts the amplitude rules
# miss -- it is largely finding the same ones. Its value is that it reaches that
# verdict WITHOUT LOOKING AT THE EEG: the amplitude rules threshold the signal
# whose spectrum is then reported, so a window is judged by the same data it
# contributes; the IMU is independent evidence about the same moment. Use it as
# a check on the amplitude rules, or where a physical cause matters more than a
# statistical one. Do not present it as broader coverage.
MOTION_SOURCES = ["accel_jerk", "gyro"]
DEFAULT_MOTION_SOURCE = "accel_jerk"

# Distance, in robust sigma of a recording's OWN per-window motion, past which a
# window is called a motion outlier. 3.0 is the conventional outlier distance and
# is the value used for the amplitude band (AMPLITUDE_BOUND_K), so the two
# relative criteria in this pipeline count in the same units. It is a convention,
# not an empirical claim: see the retention/selectivity table above for what it
# does on this dataset at other values.
MOTION_K_SLIDER = (1.0, 8.0, 0.25, 3.0)

# Fewest aux samples a window needs before its motion may be judged. At 50 Hz a
# 4 s window holds about 200 and a 1 s window about 50; this floor exists for the
# short end of EPOCH_CHOICES_S and for windows straddling a gap in the aux
# stream. A window below it is marked NOT ASSESSED and is never silently kept --
# the dashboard reports it rather than treating unmeasured as clean.
MOTION_MIN_SAMPLES = 10

# How much longer than its nominal duration a window's wall-clock span may run
# before its motion is refused. A window that straddles a gap spans the gap too,
# so its aux samples cover wall-clock time the EEG does not. 2.0 is deliberately
# loose -- it has to clear ordinary jitter and sub-second dropouts, and only
# needs to catch a straddle, which overshoots by orders of magnitude: a gap of
# minutes inside a 4 s window is not a borderline call.
MOTION_MAX_SPAN_FACTOR = 2.0

# Columns read from the aux file. Named explicitly so a file with a different
# layout fails loudly on the read rather than silently producing a wrong column.
AUX_ACCEL_COLS = ["Accelerometer X", "Accelerometer Y", "Accelerometer Z"]
AUX_GYRO_COLS = ["Gyroscope X", "Gyroscope Y", "Gyroscope Z"]
AUX_TIME_COL = "Timestamp"


def load_aux_imu(rec: dict, qc: dict | None = None) -> dict | None:
    """
    The IMU streams for one recording, on the exg file's own clock.

    Returns None when the file is missing or unreadable, which the caller turns
    into "motion not available for this recording" rather than into an error:
    the aux stream is an addition to the pipeline, and a recording without one
    must still produce every other measure.
    """
    path = Path(rec.get("aux_file", ""))
    info = dict(available=False, path=path.name, reason=None)
    try:
        aux = pd.read_csv(path, skiprows=4,
                          usecols=[AUX_TIME_COL] + AUX_ACCEL_COLS + AUX_GYRO_COLS)
    except Exception as exc:                                   # pragma: no cover
        info["reason"] = f"{type(exc).__name__}: {exc}"
        if qc is not None:
            qc["step12_motion"] = info
        return None

    ts = aux[AUX_TIME_COL].to_numpy(float)
    A = aux[AUX_ACCEL_COLS].to_numpy(float)
    G = aux[AUX_GYRO_COLS].to_numpy(float)
    if len(ts) < 2 or not np.all(np.isfinite(ts)):
        info["reason"] = "aux timestamps missing or too few rows"
        if qc is not None:
            qc["step12_motion"] = info
        return None

    # SORTED BY TIME before anything else. 46 of the 49 aux files contain
    # non-increasing timestamp steps (up to 397 per file, largest backward step
    # 0.20 s), and motion_matrix locates a window's samples with
    # np.searchsorted, whose precondition is a sorted array -- on unsorted input
    # it does not error, it returns the wrong bounds. Measured before this sort
    # was added: 215 aux samples double-counted and 53 skipped across the 6,608
    # task windows at 4 s. Small, but wrong for no reason. Sorting first also
    # makes `max_gap_s` a gap rather than an artefact of the ordering, and makes
    # the jerk a difference between temporal neighbours.
    n_unsorted = int((np.diff(ts) < 0).sum())
    if n_unsorted:
        order = np.argsort(ts, kind="stable")
        ts, A, G = ts[order], A[order], G[order]

    # Jerk: how fast the acceleration vector is moving, sample to sample. The
    # first sample has no predecessor and is set to 0 rather than dropped, so
    # the array stays aligned with `ts`.
    jerk = np.concatenate([[0.0], np.linalg.norm(np.diff(A, axis=0), axis=1)])
    gyro = np.linalg.norm(G, axis=1)

    # A SENSOR THAT NEVER MOVED is not a recording without motion, it is a
    # recording without a working IMU -- and the two must not look alike. On
    # p09/task_ai_speedscore both the accelerometer and the gyroscope are
    # identically zero for the whole file. Left unchecked it reports
    # `available: True`, every window's motion is 0, the MAD is 0, and the
    # dashboard retains 100% of windows while appearing to have applied a
    # criterion. `available` therefore means "the IMU produced varying data",
    # not "the file parsed".
    if not (np.ptp(A) > 0 or np.ptp(G) > 0):
        info["reason"] = ("the IMU reported no variation at all in this recording "
                          "(accelerometer and gyroscope are both constant); the "
                          "sensor is treated as unavailable rather than as a "
                          "perfectly still head")
        if qc is not None:
            qc["step12_motion"] = info
        return None

    dur = float(ts[-1] - ts[0])
    info.update(available=True,
                n_unsorted_steps=n_unsorted,
                n_samples=int(len(ts)),
                duration_s=round(dur, 1),
                effective_rate_hz=round(len(ts) / dur, 2) if dur > 0 else None,
                max_gap_s=round(float(np.max(np.diff(ts))), 3),
                nominal_rate_hz=AUX_SAMPLING_RATE)
    if qc is not None:
        qc["step12_motion"] = info
    return dict(t=ts, accel_jerk=jerk, gyro=gyro, info=info)


def motion_matrix(imu: dict | None, seg_times: np.ndarray, n_samp: int,
                  n_win: int) -> np.ndarray:
    """
    (n_windows, 3) of [accel_jerk, gyro, n_aux_samples] per analysis window.

    Column order is MOTION_SOURCES + the sample count. Each window's value is the
    MEAN of that stream over the aux samples whose timestamp falls inside the
    window. The mean rather than the max because a single sample of a 50 Hz
    stream is as likely to be sensor noise as head movement, and because the
    window statistic is being compared against a robust spread computed from the
    same statistic -- a max would put the criterion's own tail into its
    threshold. The comparison between mean and max was made on the RAW segment
    and is not reliable; treat the choice as a design judgement, not a measured
    result.

    A window the sensor did not properly cover gets NaN -- too few aux samples, or
    a wall-clock span too long for its own duration (see MOTION_MAX_SPAN_FACTOR).
    The browser keeps such a window, because there is no evidence against it, but
    counts it as NOT ASSESSED and says so rather than folding it into the
    surviving count. `n_aux_samples` ships
    beside the values so the reason is visible rather than inferred.

    Rows line up 1:1 with peak_matrix's for the same epoch, so one window index
    addresses both.
    """
    out = np.zeros((n_win, 3), np.float32)
    if n_win == 0:
        return out
    if imu is None or len(seg_times) == 0:
        out[:] = np.nan
        out[:, 2] = 0.0
        return out

    t = imu["t"]
    # Windows are contiguous in SAMPLES, so their edges come from seg_times,
    # which is the recorded clock rather than an assumed grid -- the same
    # principle as window_times(). searchsorted keeps this O(n log n) rather
    # than scanning the aux stream once per window.
    for w in range(n_win):
        s = w * n_samp
        e = min((w + 1) * n_samp - 1, len(seg_times) - 1)
        if s >= len(seg_times):
            out[w] = (np.nan, np.nan, 0.0)
            continue
        lo = int(np.searchsorted(t, seg_times[s], side="left"))
        hi = int(np.searchsorted(t, seg_times[e], side="right"))
        n_in = hi - lo
        out[w, 2] = n_in
        # A window that STRADDLES AN EXCISION spans far more wall-clock than its
        # own duration, and the aux samples between its first and last EEG sample
        # cover the gap as well. A window straddling a gap of minutes collects
        # tens of thousands of aux samples instead of a couple of hundred, and
        # its "motion" would be the mean over all the wall-clock time the EEG
        # skipped -- not NaN, not flagged, and almost certainly rejected.
        #
        # Guarded on the window's own time span rather than on the sample count,
        # because the span is what identifies the straddle; the count is only its
        # symptom. Both ends matter: too few samples means the sensor did not
        # cover the window, too many means the window is not what it claims.
        span = float(seg_times[e] - seg_times[s])
        nominal = n_samp / GALEA_SAMPLING_RATE
        if n_in < MOTION_MIN_SAMPLES or span > MOTION_MAX_SPAN_FACTOR * nominal:
            out[w, 0] = np.nan
            out[w, 1] = np.nan
        else:
            out[w, 0] = float(imu["accel_jerk"][lo:hi].mean())
            out[w, 1] = float(imu["gyro"][lo:hi].mean())
    return out


# ===========================================================================
# Variant archives (added 2026-09-01)
#
# The sweep is far too large for JSON -- roughly 25 million float32 values across
# the dataset -- so the numeric side goes to one uncompressed .npz per epoch
# length, and a small JSON index carries the metadata that names it. Both are
# read by build_dashboard.py, which is their only consumer; nothing in the JSON
# outputs changed shape, so every existing reader still works.
#
# outputs/variants/ can be deleted at any time and rebuilt by re-running this
# file. It is a cache of a deterministic computation, not a primary record.
# ===========================================================================

VARIANT_DIR = OUT_DIR / "variants"

# A filtered run writes here instead, so a 30-second smoke test cannot destroy a
# 40-minute complete sweep. Found on review 2026-09-01: write_variant_archives
# unlinked every archive before writing and was called unconditionally, so
# a filter naming only excluded recordings emptied the directory and wrote an
# index of zero recordings -- indistinguishable by inspection from a complete
# run.
VARIANT_DIR_PARTIAL = OUT_DIR / "variants_partial"


def _remove_with_retry(path: Path, attempts: int = 6) -> bool:
    """
    Delete a file or directory, retrying briefly. True if it is gone.

    OneDrive, Windows Search and antivirus all take transient handles on files in
    a synced folder, and a delete that touches one raises PermissionError. A
    second later it usually succeeds.
    """
    for i in range(attempts):
        try:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
            return True
        except (PermissionError, OSError):
            if i == attempts - 1:
                return False
            time.sleep(0.5 * (i + 1))
    return False


def swap_directory_contents(staging: Path, target: Path) -> list[str]:
    """
    Replace `target`'s contents with `staging`'s, then remove `staging`.

    Deliberately moves FILES rather than renaming directories. The first version
    built into a staging directory and did `shutil.rmtree(target)` followed by
    `staging.rename(target)` -- which is the standard POSIX idiom and which failed
    on this machine after a 35-minute sweep: rmtree deleted all 21 files and then
    raised PermissionError on the now-empty directory itself, because OneDrive was
    holding it. The data survived in staging, but the run reported failure and the
    target was left empty.

    A per-file move never has to delete or rename a directory another process is
    watching, and the target is not emptied until every replacement file exists.
    Returns a list of warnings rather than raising: a leftover file is untidy, but
    losing a completed sweep to a sync client is not acceptable.
    """
    warnings_out = []
    target.mkdir(parents=True, exist_ok=True)
    for stale in list(target.iterdir()):
        if not _remove_with_retry(stale):
            warnings_out.append(f"could not remove stale {stale.name}")
    for f in sorted(staging.iterdir()):
        dest = target / f.name
        _remove_with_retry(dest)
        shutil.move(str(f), str(dest))
    if not _remove_with_retry(staging):
        warnings_out.append(f"could not remove the staging directory {staging.name}")
    return warnings_out



def json_default(o):
    """
    Strict encoder shared by every JSON output.

    Was local to main(). index.json used `default=float` instead, which silently
    coerced any unhandled type through float() -- the opposite of the 2026-08-28
    decision recorded on the local copy, and on the one file build_dashboard.py
    reads first.
    """
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"unserializable {type(o).__name__} in pipeline output: {o!r}")


def dump_json(path: Path, obj) -> None:
    """
    allow_nan=False: json.dumps otherwise emits the bare token `NaN`, which is not
    valid JSON and which JSON.parse rejects, so the dashboard would fail to load
    rather than show a wrong number.
    """
    path.write_text(json.dumps(obj, indent=1, default=json_default, allow_nan=False),
                    encoding="utf-8")



def derive_amplitude_bounds(results: list[dict], partial_run: bool = False) -> dict:
    """
    The amplitude band that counts as ordinary FOR THIS DATASET, from this run.

    Replaces the fixed 1-50 uV range on 2026-09-03. That range had no published
    source and was not derived from electrode data of any kind; this helmet is
    100% dry, and dry contact runs noisier and driftier than the gelled-cap
    intuition the old number carried. Rather than substitute one asserted number
    for another, the band is estimated from the run's own channel amplitudes.

    Method, and why each part of it:

    * Pooled ACROSS CHANNELS, not per channel. A per-channel band would take
      F1's chronic ocular contamination as F1's normal and stop flagging it --
      on this dataset a per-channel fit by THIS method puts F1's upper bound at
      202 uV, against 62.7 pooled. (An earlier draft of this comment said 283 uV.
      That figure is real but came from a per-channel fit using the SYMMETRIC MAD
      spread, not the lower-half spread used below, so it argued for this design
      choice with a number the shipped method does not produce. Corrected
      2026-09-03; the argument is unchanged, since 202 uV is no more usable than
      283 for deciding whether F1 is behaving.) The question a quality label
      answers is "is this electrode unusual for this montage", so the montage is
      the reference.
    * In LOG space. Amplitudes are ratio-scaled and the distribution is heavy
      right-tailed (median 21 uV, p95 266); a symmetric band in linear space
      would put the lower bound below zero.
    * Spread estimated from the LOWER HALF ONLY, as (median - p25) / 0.6745.
      The upper tail is exactly what is being detected, so letting it set the
      scale would widen the band until it accepted anything -- the failure the
      per-channel fit shows.

      This buys robustness against the HIGH tail and none against the LOW one.
      A dead or disconnected electrode reads a small robust SD, sits below p25,
      drags the quartile down, inflates sigma and so WIDENS the upper bound --
      the opposite of what is wanted. Measured: 40 injected flat channels at
      0.5 uV move the band from 7.32-62.73 to 6.12-68.40. Only exactly-zero
      channels are excluded (the `a > 0` filter below); near-zero ones are not.
      A dataset with many dead electrodes will therefore be judged too
      leniently at the top, and nothing here detects that.
    * AMPLITUDE_BOUND_K robust sigmas either side.

    WHAT THIS IS NOT. It is a within-dataset RELATIVE criterion. It says
    "unusual for this helmet on these recordings", not "physiologically
    implausible" -- no absolute claim is available without a hardware noise spec
    or a published dry-electrode reference, and inventing one is how the number
    it replaces came to exist. It is a label, not a gate, and that limitation is
    exactly why it stays a label.

    It breaks when the BAD mode is the majority, not only when it is everything:
    the median itself moves into the bad mode once more than half the
    measurements come from it, and the band is then fitted to the artifact.
    "If every electrode were bad" understates how early that happens.

    It also pools TASK and BASELINE recordings, which are different regimes --
    a baseline block deliberately contains two minutes of eye movements, and its
    step 5 amplitudes cover the whole six minutes. Fitted separately the bands
    are 7.76-80.15 (tasks) and 7.09-52.38 (baselines); pooling makes the task
    criterion tighter and the baseline criterion looser than either alone. That
    is a real choice, made for one criterion over two, and it is not obviously
    the right one -- it is recorded here so it can be revisited.

    Returns the band plus the basis it was computed from, so a consumer can see
    what it rests on.
    """
    amps = []
    for r in results:
        pc = ((r.get("qc") or {}).get("step5_bandpass") or {}).get("per_channel") or {}
        for v in pc.values():
            a = v.get("robust_sd_uv")
            if a is not None and a > 0:
                amps.append(float(a))

    if len(amps) < AMPLITUDE_BOUND_MIN_N:
        # Too little to fit. Report that rather than fall back to a made-up band:
        # every label downstream then reads "not assessed", which is honest and
        # visibly different from "assessed and passed".
        return dict(available=False, n_measurements=len(amps),
                    partial_run=bool(partial_run),
                    reason=f"only {len(amps)} channel amplitudes in this run; "
                           f"{AMPLITUDE_BOUND_MIN_N} are needed to fit a band",
                    k=AMPLITUDE_BOUND_K)

    a = np.log10(np.asarray(amps, dtype=float))
    med = float(np.median(a))
    p25 = float(np.percentile(a, 25))
    # 0.6745 is the standard normal's 75th percentile; (median - p25)/0.6745 is
    # the usual robust sigma, taken one-sided here.
    sigma = (med - p25) / 0.6744897501960817
    if not (sigma > 0):
        return dict(available=False, n_measurements=len(amps),
                    partial_run=bool(partial_run),
                    reason="the lower half of the amplitude distribution has zero "
                           "spread, so no scale could be estimated",
                    k=AMPLITUDE_BOUND_K)

    lo = float(10.0 ** (med - AMPLITUDE_BOUND_K * sigma))
    hi = float(10.0 ** (med + AMPLITUDE_BOUND_K * sigma))
    arr = np.asarray(amps, dtype=float)
    return dict(
        available=True,
        low_uv=round(lo, 2), high_uv=round(hi, 2),
        # Stamped INSIDE the band, not only in run_meta beside it: a filtered run
        # fits this from whatever subset was filtered and then writes it into the
        # canonical qc/cognitive_load/baselines files under their canonical names.
        # The flag travels with the number so a consumer holding only the band
        # can still tell what it was fitted on.
        partial_run=bool(partial_run),
        k=AMPLITUDE_BOUND_K,
        median_uv=round(float(10.0 ** med), 2),
        sigma_log10=round(float(sigma), 6),
        n_measurements=len(amps),
        n_recordings=sum(1 for r in results
                         if ((r.get("qc") or {}).get("step5_bandpass") or {})
                         .get("per_channel")),
        pct_outside=round(100.0 * float(((arr < lo) | (arr > hi)).mean()), 1),
        method="pooled across channels; log10; spread from (median - p25)/0.6745 "
               "so the upper tail cannot set its own threshold; +/- k sigma",
        basis="this run's own channel amplitudes (relative, not absolute)",
    )


def _amp_ok(value, bounds: dict):
    """True/False against the derived band; None when no band could be fitted."""
    if not bounds.get("available"):
        return None
    if value is None:
        return False
    return bool(bounds["low_uv"] <= value <= bounds["high_uv"])


def apply_amplitude_labels(results: list[dict], bounds: dict) -> None:
    """
    Stamp every amplitude LABEL in the run with the derived band.

    Runs after all recordings are processed, because the band needs the whole
    run to exist before any label can be assigned. That ordering is the one
    structural cost of a relative criterion over a hardcoded one, and the reason
    these labels are written here rather than where they are computed.

    Rewrites, in place: step 5's physiological-channel list, step 8b's index
    channel labels (parietal_plausible / frontal_plausible / strict_subset), and
    each baseline segment's per-channel plausibility and implausible list.
    NOTHING here withholds data; every one of these is a label.
    """
    lo = bounds.get("low_uv")
    hi = bounds.get("high_uv")
    rng = [lo, hi] if bounds.get("available") else None

    for r in results:
        qc = r.get("qc") or {}

        # ---- step 5: which channels look ordinary for this montage ----------
        s5 = qc.get("step5_bandpass")
        if s5 and s5.get("per_channel"):
            ok = [n for n, v in s5["per_channel"].items()
                  if _amp_ok(v.get("robust_sd_uv"), bounds)]
            s5["amplitude_bound_uv"] = rng
            s5["n_channels_physiological"] = len(ok) if rng else None
            s5["channels_physiological"] = ok if rng else []
            # None, not False: "not assessed" must not be indistinguishable from
            # "assessed and failed" in the CSV's boolean column.
            s5["clean"] = bool(len(ok) >= 8) if rng else None
            if rng and len(ok) >= 8:
                s5["notes"] = []
            elif rng:
                s5["notes"] = [f"only {len(ok)}/{len(s5['per_channel'])} EEG channels "
                               f"sit inside this run's derived amplitude band "
                               f"({lo}-{hi} uV robust SD)"]
            else:
                s5["notes"] = ["no amplitude band could be derived for this run; "
                               "channel amplitudes are reported but not labelled"]

        # ---- step 8b: the index channels, as labels only --------------------
        g = qc.get("step8b_gate")
        if g and g.get("by_channel"):
            for c, d in g["by_channel"].items():
                d["plausible"] = _amp_ok(d.get("robust_sd_uv"), bounds)
            g["range_uv"] = rng
            if not rng:
                # NOT ASSESSED is not the same as FAILED. `plausible` is None
                # here, and None is falsy, so a bare `if not plausible` filter
                # would report every index electrode as having failed on a run
                # too small to fit a band -- "F1, F2, Pz all failed" when the
                # truth is that nothing was checked.
                g["parietal_plausible"] = None
                g["frontal_plausible"] = None
                g["strict_subset"] = None
                g["failed_channels"] = []
                g["clean"] = None
                g["notes"] = ["no amplitude band could be derived for this run; "
                              "index channel amplitudes are published but not "
                              "labelled"]
            else:
                par = [c for c in HOLM_PARIETAL_CHANNELS
                       if g["by_channel"].get(c, {}).get("plausible")]
                fro = [c for c in HOLM_FRONTAL_CHANNELS
                       if g["by_channel"].get(c, {}).get("plausible")]
                failed = [c for c, d in g["by_channel"].items()
                          if d.get("plausible") is False]
                g["parietal_plausible"] = bool(len(par) == len(HOLM_PARIETAL_CHANNELS))
                g["frontal_plausible"] = fro
                g["strict_subset"] = bool(g["parietal_plausible"]
                                          and len(fro) == len(HOLM_FRONTAL_CHANNELS))
                g["failed_channels"] = failed
                # `clean` and `notes` are rewritten HERE, not left at their
                # write-time values. They were computed before the band existed,
                # so they said "clean, no notes" beside a populated
                # failed_channels list on 13 of 22 recordings -- and the
                # FLAGGED-NOT-WITHHELD disclosure, which is the whole basis of
                # surfacing quality instead of acting on it, vanished from the
                # published record.
                g["clean"] = bool(not failed)
                g["notes"] = ([] if not failed else [
                    f"index channel(s) {failed} are outside this run's derived "
                    f"amplitude band ({lo}-{hi} uV robust SD, see "
                    f"run_meta.parameters.amplitude_bound). FLAGGED, NOT WITHHELD: "
                    f"the index is still computed for this recording, from the "
                    f"full {'+'.join(HOLM_FRONTAL_CHANNELS)} midline mean, and "
                    f"these amplitudes are published with it. The band is derived "
                    f"from this dataset, so this means 'unusual here', not "
                    f"'physiologically impossible'. Read them before trusting "
                    f"the value"])

        # ---- baselines: per segment, per channel ----------------------------
        b = r.get("baseline") or {}
        for seg in (b.get("segments") or {}).values():
            amp = seg.get("amplitude") or {}
            for c, v in amp.items():
                v["plausible"] = _amp_ok(v.get("robust_sd_uv"), bounds)
            # `is False`, not `not ...`: an unassessed channel (None) must not be
            # reported as implausible. Same trap as step 8b above.
            seg["implausible_channels"] = ([] if not rng else
                                           [c for c, v in amp.items()
                                            if v.get("plausible") is False])
        qcb = qc.get("step10_baseline")
        if qcb is not None:
            src = None
            for name in [DEFAULT_BASELINE_SEGMENT] + BASELINE_SEGMENT_ORDER:
                if name in (b.get("segments") or {}):
                    src = b["segments"][name]
                    break
            impl = (src or {}).get("implausible_channels") or []
            qcb["amplitude_bound_uv"] = rng
            qcb["amplitude"] = (src or {}).get("amplitude")
            qcb["implausible_channels"] = impl
            # Rewritten here for the same reason as step 8b's: `_baseline_qc`
            # computed these before the band existed, so its implausible branch
            # was unreachable -- the "flagged, not gated" note was never emitted
            # and `clean` stopped reflecting amplitude at all. Whatever else
            # `_baseline_qc` found (short block, interpolated channels, too few
            # windows) is preserved; only the amplitude terms are added back.
            notes = [n for n in (qcb.get("notes") or [])
                     if "amplitude band" not in n and "robust SD" not in n]
            if impl:
                notes.insert(0, f"baseline channel(s) {impl} are outside this run's "
                                f"derived amplitude band ({lo}-{hi} uV robust SD); "
                                f"the median is still used (no gate, per user "
                                f"decision) but the dashboard flags it")
            elif not rng:
                notes.insert(0, "no amplitude band could be derived for this run; "
                                "baseline channel amplitudes are published but not "
                                "labelled")
            qcb["notes"] = notes
            if qcb.get("clean") is not None:
                qcb["clean"] = bool(qcb["clean"] and not impl) if rng else None


def partial_out_dir(run_meta: dict) -> Path:
    """Where this run's JSON/CSV outputs belong.

    A FULL run writes outputs/. A filtered one (--only / --limit) writes
    outputs/partial/, for exactly the reason write_variant_archives redirects
    the sweep archive: a diagnostic run must not be able to destroy the result
    of a real one.

    Until 2026-09-08 only the archive was protected, and a four-recording smoke
    test duly overwrote qc_steps.json, cognitive_load.json and baselines.json
    with its subset -- under a console message saying the archive had been left
    untouched, which read as though everything had been. The paired
    bad-channel decisions in that qc_steps.json were the only copy outside the
    git history.
    """
    if not run_meta.get("partial_run"):
        return OUT_DIR
    d = OUT_DIR / "partial"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_variant_archives(results: list[dict], run_meta: dict) -> dict:
    """
    Write outputs/variants/{tasks,baselines}_eNN.npz plus index.json.

    Returns the index so main() can report what it wrote. Arrays are keyed
    "<recording>::v::<variant>" for values, "<recording>::p::<ocular>" for peaks
    and "<recording>::s::<ocular>|<slide length>" for the windowed mode's
    per-window sigma. The third was added 2026-09-02 and is the reason a sliding
    length is a toggle with three fixed positions rather than a slider: each
    position is a stored column, not something the page can derive. In every case
    "/" is swapped for "~" because the key becomes a zip member name.
    """
    # A filtered run writes to variants_partial/ and leaves the complete archive
    # alone; a complete run builds into a staging directory and swaps it in only
    # once every file is written, so a failure part-way through cannot leave the
    # directory half-empty either.
    target = VARIANT_DIR_PARTIAL if run_meta.get("partial_run") else VARIANT_DIR
    staging = target.with_name(target.name + ".tmp")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    index = dict(
        run_meta=run_meta,
        value_keys=VARIANT_VALUE_KEYS,
        peak_channels=MEASURE_SOURCE_CHANNELS,
        epochs_s=EPOCH_CHOICES_S,
        fft_methods=FFT_METHODS,
        # Needed by the dashboard to state the multitaper smoothing width, which
        # is NW divided by the window length in seconds and therefore moves with
        # the window-length slider.
        multitaper_nw=MULTITAPER_NW,
        multitaper_n_tapers=MULTITAPER_N_TAPERS,
        # Sliding-window lengths the per-window `sigma_slide` matrices were
        # computed at, and which one the dashboard opens on. The page offers
        # exactly these; it cannot invent a length, because each one is an
        # archive column.
        robust_slide_choices_s=list(ROBUST_SLIDE_CHOICES_S),
        robust_slide_default_s=ROBUST_SLIDE_DEFAULT_S,
        # Head-motion rejection. `motion_columns` names the "::m::imu" matrix's
        # columns so the page reads them by name rather than by position.
        motion_sources=list(MOTION_SOURCES),
        default_motion_source=DEFAULT_MOTION_SOURCE,
        motion_columns=list(MOTION_SOURCES) + ["n_aux_samples"],
        motion_min_samples=MOTION_MIN_SAMPLES,
        motion_k=dict(min=MOTION_K_SLIDER[0], max=MOTION_K_SLIDER[1],
                      step=MOTION_K_SLIDER[2], default=MOTION_K_SLIDER[3]),
        # The baseline segments the dashboard can subtract. A baseline appears in
        # `recordings` once per segment, keyed "<baseline key>::<segment>".
        # The run-derived amplitude band, so the dashboard's quality badges use
        # the same criterion the pipeline labelled with rather than a second copy.
        amplitude_bound=run_meta.get("parameters", {}).get("amplitude_bound"),
        window_max_gap_s=WINDOW_MAX_GAP_SEC,
        baseline_segments={k: list(v) for k, v in BASELINE_SEGMENTS.items()},
        baseline_segment_order=list(BASELINE_SEGMENT_ORDER),
        default_baseline_segment=DEFAULT_BASELINE_SEGMENT,
        references=SWEEP_REFERENCE_MODES,
        interpolation_modes=INTERPOLATION_MODES,
        # The two choices the five modes are the cross product of, named
        # separately so the page can render a 3-way selector plus a checkbox
        # rather than a five-button row, and so a reader of index.json can see
        # that `on_keepfrontal` is not a third source of channels.
        interpolation_sources=INTERPOLATION_SOURCES,
        interpolation_frontal_pair=list(FRONTAL_PAIR),
        default_interpolation_mode=DEFAULT_INTERPOLATION_MODE,
        # The manual list exactly as written down, keyed "<participant>|<arm>"
        # because JSON has no tuple keys. Shipped so the dashboard can show it
        # and so a reader can check the archive against the decision.
        manual_interpolation={f"{p}|{a}": list(v)
                              for (p, a), v in sorted(MANUAL_INTERPOLATION.items())},
        ocular_modes=OCULAR_MODES,
        default=dict(epoch_s=DEFAULT_EPOCH_S, fft=DEFAULT_FFT_METHOD,
                     reference="hardware",
                     interpolation=DEFAULT_INTERPOLATION_MODE, ocular="none"),
        sliders=dict(
            robust_sigma=dict(min=ROBUST_SIGMA_SLIDER[0], max=ROBUST_SIGMA_SLIDER[1],
                              step=ROBUST_SIGMA_SLIDER[2],
                              default=ROBUST_SIGMA_SLIDER[3]),
            holm_cap_uv=dict(min=HOLM_CAP_SLIDER_UV[0], max=HOLM_CAP_SLIDER_UV[1],
                             step=HOLM_CAP_SLIDER_UV[2],
                             default=HOLM_CAP_SLIDER_UV[3]),
        ),
        recordings={},
    )

    buckets: dict = {}
    for r in results:
        kind = r.get("kind")
        safe = r["key"].replace("/", "~")
        qc11_all = ((r.get("qc") or {}).get("step11_variants") or {})

        # One archive entry per swept unit. A task is one unit; a baseline is one
        # per SEGMENT (2026-09-02), because each segment is cropped, epoched and
        # calibrated independently and therefore has its own peaks, sigma, window
        # grid and verification verdict. Keying them as separate entries lets the
        # whole downstream -- npz members, shard builder, dashboard lookup -- treat
        # a segment exactly as it already treats a recording, instead of growing a
        # segment dimension through every one of those layers.
        if kind == "task":
            units = [(r["key"], safe, None, r.get("variants"),
                      r.get("variant_default_check"), qc11_all.get("task", {}))]
        else:
            base = r.get("baseline") or {}
            units = [(f"{r['key']}::{s}", f"{safe}::{s}", s,
                      (sv or {}).get("variants"),
                      (sv or {}).get("variant_default_check"),
                      qc11_all.get(f"baseline_{s}", {}))
                     for s in BASELINE_SEGMENT_ORDER
                     for sv in [(base.get("segments") or {}).get(s)]]

        for entry_key, prefix, segment, bundle, chk, qc11 in units:
            if not bundle:
                continue
            index["recordings"][entry_key] = dict(
                kind=kind,
                array_prefix=prefix,
                # Which baseline segment this entry is, and its bounds in the
                # block. None for a task.
                segment=segment,
                segment_bounds_s=(list(BASELINE_SEGMENTS[segment])
                                  if segment else None),
                # Which channels were replaced by a spline under interpolation="on".
                # Empty means the interpolation toggle CANNOT change anything for
                # this recording, and without this the reader moves the control,
                # sees nothing happen, and cannot tell whether that is the science
                # or a broken control.
                #
                # The baseline side read `baseline["interpolated"]` until
                # 2026-09-04. That field is filtered to MEASURE_SOURCE_CHANNELS
                # (5 of the 10 EEG channels) because it exists to flag the
                # channels feeding the five measures -- but the baseline SWEEP
                # interpolates the full ten-channel list, so anything flagged on
                # C3/C4/Cz/O1/O2 was missing here and the "empty means nothing can
                # change" contract above was false for 9 of 22 baselines
                # (p09/baseline_ai_speedscore claimed [] while the sweep splined
                # C3). A pre-existing defect, found now because the paired rule
                # makes a task entry and its baseline entry comparable and this
                # filter was the only thing left making them differ.
                bad_channels=(r.get("qc", {}).get("step6_interpolate", {})
                              .get("none" if kind == "task" else "baseline_none",
                                   {}).get("bad_channels_detected", [])),
                # THE MODE -> ARRAY MAP. The archive is keyed by the channel list
                # that was interpolated, not by the mode that asked for it, so a
                # consumer holding "manual_keepfrontal" cannot name an array
                # without this. `interpolation_lists` is the same information in
                # the form a reader wants: what does this control actually do to
                # THIS recording.
                #
                # Both are per recording because both genuinely vary: the manual
                # list is per participant and arm, the automatic list per
                # task/baseline group.
                interpolation_lists=bundle.get("interp_lists") or {},
                interpolation_tokens=bundle.get("interp_tokens") or {},
                # Any (interpolation, ocular, reference) combination the sweep could
                # not build. Previously this lived only in qc_steps.json and nothing
                # printed it, so a systematically failing REST would have produced an
                # archive missing a third of its variants under a success message.
                references_failed=qc11.get("references_failed", []),
                sigma=bundle["sigma"],
                # False where the sensor cannot be used: the page then says
                # motion is unavailable for this recording instead of offering a
                # rejection mode that silently keeps everything.
                motion_available=bool(
                    ((r.get("qc") or {}).get("step12_motion") or {}).get("available")),
                # WHY it is unavailable, shipped alongside since 2026-09-04.
                # The boolean alone made the dashboard say "no IMU file at all",
                # which on this dataset is false: p09's aux file is present and
                # readable and was rejected because the IMU reported no variation
                # at all in either the accelerometer or the gyroscope. A missing
                # file and a flatlined sensor are different facts for anyone
                # deciding whether a recording is salvageable, and a boolean
                # cannot tell them apart. None when the sensor IS available.
                #
                # It lives here rather than being read out of qc_steps.json by
                # build_dashboard.py, which is where it went first: that made the
                # page depend on a second file from the same run and needed a
                # generated_utc guard to be sure the two matched. One file, one
                # guard, no way for the reason to be attached to the wrong
                # recording.
                motion_unavailable_reason=(
                    ((r.get("qc") or {}).get("step12_motion") or {}).get("reason")
                    if not ((r.get("qc") or {}).get("step12_motion") or {})
                    .get("available") else None),
                # Windows excluded outright, per epoch length -- their samples are
                # not contiguous in real time, so their spectra are taken across a
                # step in the signal. NOT a rejection criterion: they are dropped
                # under every artifact mode including `none`, because "no
                # rejection" is a statement about not judging the EEG rather than
                # a licence to plot a window that is two recordings spliced
                # together. See WINDOW_MAX_GAP_SEC.
                excluded_windows={
                    e: [int(w) for w in np.flatnonzero(v["discontinuous"])]
                    for e, v in bundle["epochs"].items()},
                amplitude_displayed=bundle["amplitude_displayed"],
                band_bins=bundle["band_bins"],
                n_windows={e: v["n_windows"] for e, v in bundle["epochs"].items()},
                times_s=(r.get("variant_times_s") if kind == "task" else None),
                # Renamed from `rest_window_index` with the 2026-09-02 extension:
                # it is no longer always the rest phase, and a name that says
                # "rest" while holding a math or eyes-closed index would be the
                # kind of thing a reader trusts and should not.
                window_index=bundle.get("window_index"),
                # Now recorded for BASELINES too, and per segment. It previously ran
                # on tasks only, so 22 of the 44 swept recordings -- every resting
                # baseline, i.e. one whole side of every baseline subtraction --
                # were verified against nothing while the console reported success.
                default_check=(chk or {}).get("problems", []),
                default_verified=bool((chk or {}).get("verified")),
                default_check_detail={k: v for k, v in (chk or {}).items()
                                      if k in ("n_windows", "n_columns")},
            )
            for e, ep_entry in bundle["epochs"].items():
                b = buckets.setdefault((kind, e), {})
                for oc, arr in ep_entry["peaks"].items():
                    b[f"{prefix}::p::{oc}"] = arr
                for sk, arr in ep_entry["sigma_slide"].items():
                    b[f"{prefix}::s::{sk}"] = arr
                if ep_entry.get("motion") is not None:
                    b[f"{prefix}::m::imu"] = ep_entry["motion"]
                for vk, arr in ep_entry["values"].items():
                    b[f"{prefix}::v::{vk}"] = arr

    total = 0
    for (kind, e), arrays in sorted(buckets.items()):
        stem = "tasks" if kind == "task" else "baselines"
        path = staging / f"{stem}_e{float(e):04.1f}.npz"
        np.savez(path, **arrays)
        total += path.stat().st_size
    index["bytes_on_disk"] = total
    index["directory"] = target.name
    dump_json(staging / "index.json", index)

    index["swap_warnings"] = swap_directory_contents(staging, target)
    return index


# ===========================================================================
# Driver
# ===========================================================================

def baseline_key_for_task(rec: dict) -> str:
    """
    The baseline recorded in the SAME session under the SAME condition as `rec`:
    task_agent_personal <-> baseline_agent_personal. Pairing is by folder name,
    which is 1:1 for every participant analysed.

    A trailing _1/_2 suffix is stripped: a task split across two files has halves
    named `task_ai_speedscore_1` and `_2`, and without stripping, each would
    resolve to a baseline that does not exist. Like the group handling in
    detect_bad_channels_paired, this is UNTESTED -- the only split recording in
    the original study belonged to an excluded participant.

    Module-level since 2026-09-04. It used to be a closure inside main() used only
    to label cognitive_load.json; the bad-channel decision is now made per pair
    too, and two independent spellings of "which baseline goes with this task"
    is exactly the kind of drift that produced the defect this change fixes.
    """
    stem = re.sub(r"_\d+$", "", rec["recording"][len("task_"):])
    return f"{rec['participant']}/baseline_{stem}"


def prepare_recording(rec: dict) -> dict:
    """
    Steps 1-5 for one recording: everything needed BEFORE a bad-channel decision.

    Split out of process_recording on 2026-09-04, when that decision stopped being
    a property of one recording and became a property of a task/baseline pair --
    see detect_bad_channels_paired. Both members of a pair must reach step 5
    before either can go further, so the driver prepares them together and
    finishes them together.

    Returns the prepared state. `ready` False means a terminal status is already
    on `result` ("skipped_no_segment") and finish_recording passes it straight
    through. An exception here is caught by the driver exactly as before.
    """
    qc: dict = {}
    result = dict(rec)
    result["qc"] = qc

    exg_data = step1_load(rec, qc)
    seg = step2_segment(rec, exg_data, qc)
    if seg is None:
        result["status"] = "skipped_no_segment"
        return dict(rec=rec, result=result, qc=qc, ready=False)

    # The helmet's IMU, on the same clock as the exg stream. Loaded once per
    # recording and reused for every epoch and every baseline segment. None when
    # the aux file is missing or unreadable, which downstream reports as
    # "motion not available" rather than failing the recording.
    imu = load_aux_imu(rec, qc)

    galea_mne = step3_to_mne(seg, qc)
    # MANUAL_INTERPOLATION and the frontal-pair rule name channels as strings.
    # Check those names against the montage this file actually produced before
    # anything downstream tries to match them.
    assert_montage_names(galea_mne)
    filtered = step4_notch(galea_mne, qc)
    bandpassed = step5_bandpass(filtered, qc)

    return dict(rec=rec, result=result, qc=qc, ready=True,
                seg=seg, imu=imu, bandpassed=bandpassed,
                seg_times=seg["Timestamp"].to_numpy(float))


def finish_recording(prep: dict, bad_chans: list[str], z_detail: dict,
                     pairing: dict, ica_pair: tuple | None = None) -> dict:
    """
    Steps 6-11 for one recording, given the bad-channel list decided for its PAIR.

    `bad_chans` / `z_detail` / `pairing` come from detect_bad_channels_paired and
    are IDENTICAL for the task and the baseline of a pair -- that identity is the
    whole point of the 2026-09-04 change. The `example` branch below is the one
    deliberate exception and re-detects on this recording alone.
    """
    if not prep.get("ready"):
        return prep["result"]

    rec, result, qc = prep["rec"], prep["result"], prep["qc"]
    seg, imu, bandpassed = prep["seg"], prep["imu"], prep["bandpassed"]

    # ---- example pipeline branch (no ocular correction, average reference) --
    # PER-RECORDING detection, deliberately (user's decision 2026-09-04). This
    # branch exists to reproduce example_eeg_processing.ipynb over this dataset,
    # and that notebook processes one recording at a time and knows nothing of a
    # paired baseline. Pairing is OUR analysis decision; applying it here would
    # make step8a's output something the reference notebook could not produce and
    # cost the traceability claim the branch exists to support.
    #
    # Consequence, stated because it looks like a bug otherwise: qc
    # step6_interpolate["example"] and step6_interpolate["none"] can now carry
    # DIFFERENT bad-channel lists for the same recording. `detection` on each says
    # which rule produced it.
    bad_example, z_example = detect_bad_channels(bandpassed)
    interp_example = step6_interpolate(bandpassed, qc, "example",
                                       bad_example, z_example,
                                       dict(rule="per_recording", paired=False,
                                            segment=None, fallback=None,
                                            members=[rec["key"]], sources=[],
                                            n_samples_total=None,
                                            reason="reproduces the example "
                                                   "pipeline, which sees one "
                                                   "recording at a time"))
    avg_ref_data = step7_reref(interp_example, qc)
    step8a_example_bandpower(avg_ref_data, qc)

    seg_times = prep["seg_times"]

    # ---- Holm index + the four new measures ---------------------------------
    if rec["kind"] == "task":
        gate = index_channel_gate(bandpassed, qc)
        holm, measures, oc_by_mode = {}, {}, {}
        for ocular in OCULAR_MODES:
            oc = step5b_ocular(bandpassed, ocular, qc, ica_pair=ica_pair)
            interp = step6_interpolate(oc, qc, ocular, bad_chans, z_detail,
                                       pairing)
            holm[ocular] = step8b_holm_index(interp, oc, gate, seg_times, qc, ocular)
            # The new measures are NOT gated (user decision), so they are computed
            # here whether or not the index was.
            measures[ocular] = step9_new_measures(oc, interp, ocular, bad_chans, qc,
                                                 seg_times=seg_times)
            # Pre-interpolation, hardware-referenced, EEG only: the signal the
            # sweep re-references, re-interpolates and re-transforms, and the
            # signal every artifact mask is measured on.
            oc_by_mode[ocular] = oc.copy().pick(picks="eeg")
        result["holm"] = holm
        result["measures"] = measures

        # ---- step 11: the parameter sweep --------------------------------
        bundle = compute_variant_bundle(oc_by_mode,
                                        interp_lists_for(bad_chans,
                                                         manual_interpolation_for(rec)),
                                        float(bandpassed.info["sfreq"]), qc, "task",
                                        imu=imu, seg_times=seg_times)
        result["variants"] = bundle
        result["variant_times_s"] = {
            e: window_times(seg_times, float(e), bundle["epochs"][e]["n_windows"])
            for e in bundle["epochs"]}
        # Verified against MNE's own estimator on the default cell, for all seven
        # shipped columns, on every recording. (Before 2026-09-02 that "every"
        # was the notable part: the sweep was verified even where the amplitude
        # gate refused an index.) `interp_default` is the interpolation="on",
        # ocular="none", hardware-referenced signal -- the default cell exactly.
        interp_default = step6_interpolate(
            oc_by_mode["none"].copy(), {}, "verify", bad_chans, z_detail, pairing)
        check = verify_default_variant(bundle,
                                       reference_columns_via_mne(interp_default))
        qc["step11_variants"]["task"]["reproduces_default"] = check["verified"]
        qc["step11_variants"]["task"]["default_check"] = check
        if not check["verified"]:
            qc["step11_variants"]["task"]["clean"] = False
            qc["step11_variants"]["task"]["notes"] = (
                (qc["step11_variants"]["task"].get("notes") or [])
                + (check["problems"] or ["nothing was compared"]))
        result["variant_default_check"] = check

        # Window start times, needed by the new tabs where holm[...]["times_s"]
        # does not exist -- which since the gate's removal means only a recording
        # with no complete 4 s window.
        n_win = 0
        for oc_entry in measures.values():
            for ser in oc_entry.values():
                if ser:
                    n_win = max(n_win, ser["n_windows"])
        result["measure_times_s"] = window_times(seg_times, HOLM_WINDOW_SEC, n_win)

    # ---- resting baseline ----------------------------------------------------
    else:
        result["baseline"] = step10_baseline(rec, seg, bandpassed,
                                             bad_chans, z_detail, qc, imu,
                                             pairing=pairing, ica_pair=ica_pair)

    result["status"] = "ok"
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", type=str, default=None,
                    help="substring filter on the recording key, e.g. p09/task")
    args = ap.parse_args()

    run_t0 = time.time()

    recs = discover_recordings()
    n_discovered = len(recs)

    # Both lists are checked against the recordings that actually exist, BEFORE
    # anything is processed. A typo in either is invisible at runtime: an
    # unmatched MANUAL_INTERPOLATION key interpolates nothing, which is a
    # legitimate instruction for a group genuinely absent from the list, and an
    # unmatched EXCLUDED_RECORDINGS key excludes nothing while the console shows
    # the same output either way. Checked on the FULL discovery, before --only
    # or --limit narrows it, so a filtered run still validates the whole
    # decision rather than the slice it happens to touch.
    assert_manual_interpolation(recs)
    assert_excluded_recordings(recs)

    # Refuse a run that would analyse nothing, BEFORE anything is written. The
    # end-of-run guard below catches this too, but only after
    # write_variant_archives() has already emptied outputs/variants/ -- so by the
    # time it fires the previous good archive is gone. Excluding everything is an
    # ordinary slip now that EXCLUDED_* are empty knobs for the reader to fill,
    # not a hypothetical.
    kept = [r for r in recs if not exclusion_reason(r)]
    if recs and not kept:
        raise SystemExit(
            f"every one of the {len(recs)} discovered recordings is excluded, so "
            f"there is nothing to analyse. Check EXCLUDED_PARTICIPANTS "
            f"({EXCLUDED_PARTICIPANTS or 'empty'}) and EXCLUDED_RECORDINGS "
            f"({sorted(EXCLUDED_RECORDINGS) or 'empty'}). Nothing was written; "
            f"outputs/ is untouched.")

    if args.only:
        recs = [r for r in recs if args.only in r["key"]]
    if args.limit:
        recs = recs[:args.limit]

    # ---- group each task with its own baseline ----------------------------
    # Since 2026-09-04 the bad-channel decision belongs to the PAIR, not to one
    # recording (detect_bad_channels_paired), so both members must reach step 5
    # before either can go past step 6. Recordings are therefore prepared and
    # finished a group at a time instead of one at a time.
    #
    # The group key is the baseline's key, so a baseline serving two task files
    # collects all of them -- see detect_bad_channels_paired.
    groups, group_order = {}, []
    for rec in recs:
        gkey = (baseline_key_for_task(rec) if rec["kind"] == "task"
                else rec["key"])
        if gkey not in groups:
            groups[gkey] = dict(key=gkey, tasks=[], baseline=None)
            group_order.append(gkey)
        if rec["kind"] == "task":
            groups[gkey]["tasks"].append(rec)
        else:
            groups[gkey]["baseline"] = rec

    # A filter can now change the NUMBERS, not merely the coverage: --only
    # p09/task leaves that task without the baseline its bad-channel list is
    # supposed to be decided against, so it silently falls back to task-only
    # detection. run_meta already marks the run partial and build_dashboard.py
    # already refuses to build from one, but the old failure mode was "fewer
    # recordings" and this one is "different values", which deserves saying out
    # loud at the top of the run rather than being inferred from a JSON flag.
    if args.only or args.limit:
        # BOTH directions. A filter can strand a task without its baseline or a
        # baseline without its task, and each falls back to a different rule --
        # neither of them the rule a full run would use. The group key is the
        # BASELINE's key, so naming it would name the absent recording in the
        # first case; name the members that are actually present.
        split = [g for g in groups.values()
                 if bool(g["tasks"]) != bool(g["baseline"])]
        if split:
            def _members(g):
                return ", ".join([r["key"] for r in g["tasks"]]
                                 or [g["baseline"]["key"]])
            print(f"\n*** {len(split)} group(s) were filtered away from the "
                  f"other side of their pair: "
                  f"{'; '.join(_members(g) for g in split[:5])}"
                  f"{' ...' if len(split) > 5 else ''}. Bad-channel detection "
                  f"falls back for these, so their values will NOT match a full "
                  f"run. ***\n", flush=True)

    results, rows = [], []
    n_done = 0
    for gkey in group_order:
        g = groups[gkey]
        members = g["tasks"] + ([g["baseline"]] if g["baseline"] else [])

        # ---- steps 1-5 for every member of the group ----------------------
        prepped = {}
        for rec in members:
            n_done += 1
            why = exclusion_reason(rec)
            tag = " [EXCLUDED]" if why else ""
            print(f"[{n_done}/{len(recs)}] {rec['key']}{tag}", flush=True)
            if why:
                # The status token stays stable -- downstream branches on it --
                # and the human sentence rides beside it, because that sentence
                # is what the dashboard prints in its Reason column. Before
                # 2026-09-08 the column showed the token itself, which told a
                # reader that a recording was "excluded_by_user_decision" and
                # nothing whatever about why.
                print(f"    {why}", flush=True)
                results.append(dict(rec, status="excluded_by_user_decision",
                                    exclusion_reason=why, qc={}, holm=None))
                continue
            try:
                prepped[rec["key"]] = prepare_recording(rec)
            except Exception as exc:
                print(f"    FAILED: {exc}", flush=True)
                results.append(dict(rec, status=f"error: {exc}", qc={}))

        if not prepped:
            continue

        # ---- one bad-channel decision for the group -----------------------
        task_preps = [prepped[r["key"]] for r in g["tasks"]
                      if r["key"] in prepped]
        base_prep = (prepped.get(g["baseline"]["key"]) if g["baseline"]
                     else None)
        try:
            bad_chans, z_detail, pairing = detect_bad_channels_paired(
                task_preps, base_prep)
        except Exception as exc:
            # One failed decision fails the whole group, and saying so per member
            # keeps every recording accounted for in the results list.
            #
            # Two things this must NOT do, both found in review:
            #   * discard prep["result"], which already carries steps 1-5 QC. The
            #     run-wide amplitude band is fitted on step5_bandpass per-channel
            #     amplitudes pooled across every recording, so throwing away a
            #     sibling's measurements here would move the labels on recordings
            #     that had nothing to do with the failure. Under the old
            #     one-at-a-time driver a failure could not reach a sibling at all;
            #     grouping introduced that coupling, and this is where it is cut.
            #   * overwrite a member that never got as far as the decision. A
            #     not-ready prep already holds its own terminal status
            #     ("skipped_no_segment") and is not an error; relabelling it would
            #     also feed the "EVERY recording failed" and ">50% failed" guards
            #     a count that is not failures.
            print(f"    FAILED (bad-channel detection for {gkey}): {exc}",
                  flush=True)
            for rec in members:
                p = prepped.get(rec["key"])
                if p is None:
                    continue
                results.append(p["result"] if not p.get("ready")
                               else dict(p["result"], status=f"error: {exc}"))
            continue
        if pairing.get("fallback"):
            print(f"    NOTE: {pairing['fallback']}", flush=True)

        # One ICA for the group, on the same join and for the same reason as the
        # bad-channel list above: a task and the baseline it is divided by must
        # lose the same components, or the ratio compares two differently
        # cleaned signals. Fitted here rather than inside step5b_ocular because
        # only the driver holds both members of the pair.
        try:
            ica_obj, ica_prov = fit_ica_paired(task_preps, base_prep)
        except Exception as exc:
            print(f"    NOTE: ICA fit failed for {gkey}: {exc}", flush=True)
            ica_obj, ica_prov = None, dict(failed=f"{type(exc).__name__}: {exc}")
        ica_pair = (ica_obj, ica_prov)
        if ica_prov.get("failed"):
            print(f"    NOTE: ICA unavailable ({ica_prov['failed']}); the `ica` "
                  f"mode will equal `none` for this group", flush=True)
        elif ica_prov.get("fallback"):
            print(f"    NOTE: ICA {ica_prov['fallback']}", flush=True)

        # ---- steps 6-11 for every member, on that one list -----------------
        for rec in members:
            prep = prepped.get(rec["key"])
            if prep is None:
                continue
            try:
                results.append(finish_recording(prep, bad_chans, z_detail,
                                                pairing, ica_pair=ica_pair))
            except Exception as exc:
                print(f"    FAILED: {rec['key']}: {exc}", flush=True)
                results.append(dict(rec, status=f"error: {exc}", qc={}))

    # The last group's prepared raws (~190 MB of MNE objects) would otherwise
    # stay reachable through derive_amplitude_bounds, the JSON dumps and
    # write_variant_archives -- which is exactly where the run's memory peaks.
    prepped = None

    # Back into discovery order. Grouping reorders the run, and every consumer
    # below is order-sensitive in some visible way -- the CSV's row order, the
    # insertion order of the JSON dicts, the order failures are reported in.
    # None of that is correctness, but a reordered qc_steps.csv makes the diff
    # against the previous run unreadable for no reason.
    _discovery = {r["key"]: i for i, r in enumerate(recs)}
    results.sort(key=lambda r: _discovery.get(r["key"], len(_discovery)))

    # ---- amplitude labels -------------------------------------------------
    # These need the WHOLE run: the band is derived from this dataset's own
    # channel amplitudes (derive_amplitude_bounds), so no recording can be
    # labelled until every recording has been measured. Everything above records
    # amplitudes and leaves `plausible` as None; this is where the verdicts are
    # written. Nothing here withholds data -- see apply_amplitude_labels.
    amplitude_bounds = derive_amplitude_bounds(
        results, partial_run=bool(args.only or args.limit))
    apply_amplitude_labels(results, amplitude_bounds)
    if amplitude_bounds.get("available"):
        print(f"\n  amplitude band derived from this run: "
              f"{amplitude_bounds['low_uv']}-{amplitude_bounds['high_uv']} uV robust SD "
              f"(median {amplitude_bounds['median_uv']}, k={amplitude_bounds['k']}, "
              f"{amplitude_bounds['n_measurements']} channel measurements, "
              f"{amplitude_bounds['pct_outside']}% outside). Labels only; nothing "
              f"is withheld for it.", flush=True)
    else:
        print(f"\n  NOTE: no amplitude band could be derived "
              f"({amplitude_bounds.get('reason')}). Channel amplitudes are "
              f"published but not labelled.", flush=True)

    # The CSV rows read those labels, so they are built here rather than in the
    # loop above, where the labels did not exist yet.
    for res in results:
        if res.get("status") == "excluded_by_user_decision":
            continue
        rec = res
        q = res.get("qc", {})
        holm_qc = q.get("step8b_holm", {}).get("none", {}) or {}
        # Which step6 entry describes the ANALYSIS. Tasks carry "none", baselines
        # "baseline_none"; "example" is a different branch with its own list.
        s6 = q.get("step6_interpolate", {})
        s6_analysis = s6.get("none") or s6.get("baseline_none") or {}
        s6_example = s6.get("example", {}) or {}
        rows.append(dict(
            key=rec["key"], participant=rec["participant"], recording=rec["recording"],
            kind=rec["kind"], status=res.get("status"),
            step1_clean=q.get("step1_load", {}).get("clean"),
            step2_clean=q.get("step2_segment", {}).get("clean"),
            step3_clean=q.get("step3_to_mne", {}).get("clean"),
            step4_clean=q.get("step4_notch", {}).get("clean"),
            step5_clean=q.get("step5_bandpass", {}).get("clean"),
            step6_clean=s6_analysis.get("clean"),
            step8_clean=q.get("step8a_example_bandpower", {}).get("clean"),
            duration_s=q.get("step2_segment", {}).get("segment_duration_s"),
            bad_channels=", ".join(s6_analysis.get("bad_channels_detected", [])),
            # The `example` branch keeps its own per-recording detection
            # (finish_recording), so since 2026-09-04 these two columns can
            # legitimately differ. Both are printed: reporting only the example
            # list, as this CSV did until the columns above were repointed, would
            # have described a branch that never feeds the index while every
            # other column in the row describes the analysis.
            bad_channels_example=", ".join(
                s6_example.get("bad_channels_detected", [])),
            holm_retention_strict=holm_qc.get("retention_pct", {}).get("holm_strict"),
            holm_retention_robust=holm_qc.get("retention_pct", {}).get("robust"),
            # `index_computable` was renamed to `index_computed` on 2026-09-02 so
            # a stale filter breaks rather than silently widening 12 -> 22. The
            # label columns beside it reconstruct what the old gate selected:
            #   old 12  parietal_plausible & frontal_plausible_n >= 1
            #   old  6  strict_subset
            # parietal_plausible ALONE is 16, not 12 -- the old rule needed a
            # frontal channel too.
            index_computed=q.get("step8b_gate", {}).get("index_computed"),
            parietal_plausible=q.get("step8b_gate", {}).get("parietal_plausible"),
            # `or []` on all three: step8b_gate writes an explicit None for
            # these when no amplitude band could be derived, and `.get(k, [])`
            # returns that None rather than the default. A full run always
            # derives a band, so this only ever fired on a FILTERED run -- which
            # is exactly when somebody is smoke-testing a change and least wants
            # a TypeError from the CSV writer after the whole sweep has finished.
            # Pre-existing; found 2026-09-08 running `pipeline.py --only p09`.
            frontal_plausible_n=len(q.get("step8b_gate", {})
                                    .get("frontal_plausible") or []),
            strict_subset=q.get("step8b_gate", {}).get("strict_subset"),
            frontal_used=", ".join(
                q.get("step8b_gate", {}).get("frontal_channels_used") or []),
            gate_failed_channels=", ".join(
                q.get("step8b_gate", {}).get("failed_channels") or []),
        ))

    # Everything a filtered run writes goes to outputs/partial/ instead. Derived
    # from `args`, not from run_meta -- run_meta is not built until after the CSV
    # is written, and reading it here raised UnboundLocalError. Same expression
    # run_meta["partial_run"] uses, so the two cannot disagree.
    out_dir = partial_out_dir({"partial_run": bool(args.only or args.limit)})
    pd.DataFrame(rows).to_csv(out_dir / "qc_steps.csv", index=False)

    # `_clean` / `_dump` were local to main(); they are now module level as
    # json_default / dump_json so index.json is written by the same strict
    # encoder. Aliased here so the call sites below read unchanged.
    _dump = dump_json

    # Provenance of the run itself. Added 2026-08-28 after review: a filtered run
    # (--only / --limit) wrote the canonical filenames and was indistinguishable by
    # inspection from a complete run that had silently failed.
    run_meta = dict(
        generated_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        argv=sys.argv[1:],
        partial_run=bool(args.only or args.limit),
        n_discovered=n_discovered,
        n_processed=len(results),
        excluded_participants=EXCLUDED_PARTICIPANTS,
        excluded_recordings=sorted(EXCLUDED_RECORDINGS),
        parameters=dict(
            holm_window_s=HOLM_WINDOW_SEC,
            holm_theta_hz=list(HOLM_THETA), holm_alpha_hz=list(HOLM_ALPHA),
            holm_reject_uv=HOLM_REJECT_UV, robust_sigma=ROBUST_SIGMA,
            new_theta_hz=list(NEW_THETA), new_alpha_hz=list(NEW_ALPHA),
            new_beta_hz=list(NEW_BETA),
            # Reported range, not a gate, since 2026-09-02: no recording is
            # withheld for falling outside it. `index_gate_applied` says so in the
            # provenance record itself, so a reader of an old and a new
            # cognitive_load.json can tell them apart without diffing the code.
            # The fixed 1-50 uV range that stood here was replaced 2026-09-03 by
            # a band derived from this run's own amplitudes. Published whole --
            # bounds, k, the sample it was fitted on and the method -- because a
            # relative criterion is only reproducible if its basis travels with it.
            amplitude_bound=amplitude_bounds,
            index_gate_applied_note="no amplitude gate; labels only",
            index_gate_applied=False,
            robust_slide_choices_s=list(ROBUST_SLIDE_CHOICES_S),
            robust_slide_default_s=ROBUST_SLIDE_DEFAULT_S,
            bad_channel_z=BAD_CHANNEL_Z,
            bad_channel_detection="paired",
            bad_channel_detection_note=(
                "since 2026-09-04 the bad-channel list is decided ONCE per "
                "task/baseline pair, on the task concatenated with the "
                f"{PAIRED_DETECT_BASELINE_SEGMENT} segment of its baseline, and "
                "the same channels are interpolated in both recordings. The "
                "`example` branch alone still detects per recording, to stay "
                "faithful to the reference notebook. Per-recording detection "
                "before that date; values from the two rules are not comparable."),
            bad_channel_detection_segment=PAIRED_DETECT_BASELINE_SEGMENT,
            window_max_gap_s=WINDOW_MAX_GAP_SEC,
            window_max_gap_note=(
                "a window whose own samples jump by more than this in real time "
                "is excluded from every measure, at every window length, under "
                "every artifact mode including `none` -- its spectrum would be "
                "taken across a discontinuity. Which recordings this affects "
                "depends on the run; qc_steps.json records the gaps found in "
                "each."),
            bandpass_hz=[LOWCUT, HIGHCUT], line_freq_hz=LINE_FREQ,
            baseline_block_s=BASELINE_BLOCK_SEC,
            baseline_segments_s={k: list(v) for k, v in BASELINE_SEGMENTS.items()},
            baseline_default_segment=DEFAULT_BASELINE_SEGMENT,

            mask_reference="hardware (fixed for every reference setting)",
            sweep=dict(
                epochs_s=EPOCH_CHOICES_S,
                fft_methods=FFT_METHODS,
                references=SWEEP_REFERENCE_MODES,
                interpolation_modes=INTERPOLATION_MODES,
                ocular_modes=OCULAR_MODES,
                # Cells a reader can SELECT. Fewer are STORED: two interpolation
                # modes that reach the same channel list are one signal and one
                # array (see interp_token), which on this dataset is about 2.2
                # distinct lists per group rather than 5.
                # `n_combinations_stored` per recording in qc_steps.json has the
                # actual figure.
                precomputed_cells=(len(EPOCH_CHOICES_S) * len(FFT_METHODS)
                                   * len(SWEEP_REFERENCE_MODES)
                                   * len(INTERPOLATION_MODES) * len(OCULAR_MODES)),
                interpolation_sources=INTERPOLATION_SOURCES,
                frontal_pair=list(FRONTAL_PAIR),
                browser_thresholds=dict(
                    robust_sigma=list(ROBUST_SIGMA_SLIDER),
                    robust_slide_choices_s=list(ROBUST_SLIDE_CHOICES_S),
                    motion_k=list(MOTION_K_SLIDER),
                    holm_cap_uv=list(HOLM_CAP_SLIDER_UV)),
                default_cell=dict(epoch_s=DEFAULT_EPOCH_S, fft=DEFAULT_FFT_METHOD,
                                  reference="hardware",
                                  interpolation=DEFAULT_INTERPOLATION_MODE,
                                  ocular="none"),
            ),
        ),
    )

    bulky = {"holm", "measures", "measure_times_s", "baseline",
             "variants", "variant_times_s", "variant_default_check"}
    _dump(out_dir / "qc_steps.json",
          dict(run_meta=run_meta,
               recordings=[{k: v for k, v in r.items() if k not in bulky}
                           for r in results]))

    # Each task recording is paired with the baseline recorded under the SAME
    # condition in the same session -- see baseline_key_for_task, which is also
    # what grouped the recordings for the bad-channel decision above.
    load = {r["key"]: dict(participant=r["participant"], recording=r["recording"],
                           status=r["status"],
                           # Why this recording is not analysed, in a sentence
                           # meant for the dashboard's Reason column. None for
                           # every recording that IS analysed.
                           exclusion_reason=r.get("exclusion_reason"),
                           holm=r.get("holm"),
                           measures=r.get("measures"),
                           measure_times_s=r.get("measure_times_s"),
                           baseline_key=baseline_key_for_task(r),
                           qc_holm=r.get("qc", {}).get("step8b_holm"),
                           qc_interp=r.get("qc", {}).get("step6_interpolate"),
                           qc_segment=r.get("qc", {}).get("step2_segment"),
                           qc_bandpass=r.get("qc", {}).get("step5_bandpass"),
                           qc_ocular=r.get("qc", {}).get("step5b_ocular"),
                           qc_gate=r.get("qc", {}).get("step8b_gate"))
            for r in results if r.get("kind") == "task"}
    _dump(out_dir / "cognitive_load.json", dict(run_meta=run_meta, tasks=load))

    def _baseline_json(r):
        # `variants` holds the sweep matrices, which live in the npz archives;
        # everything else about the baseline stays in this file.
        #
        # Stripped PER SEGMENT since 2026-09-02: the bundles moved inside
        # `segments` when baselines gained three of them, and a top-level-only
        # strip would have dumped three sweeps of float32 matrices into this
        # JSON. It would not even crash: json_default coerces ndarray via
        # tolist(), so the failure mode is a silently enormous file.
        b = r.get("baseline")
        if not b:
            return b
        out = {k: v for k, v in b.items() if k not in ("variants", "segments")}
        out["segments"] = {
            s: {k: v for k, v in (sv or {}).items() if k != "variants"}
            for s, sv in (b.get("segments") or {}).items()}
        return out

    base = {r["key"]: dict(participant=r["participant"], recording=r["recording"],
                           status=r["status"],
                           # Same field as cognitive_load.json carries for tasks:
                           # the dashboard's excluded table reads both since
                           # 2026-09-08, so an excluded baseline can say why.
                           exclusion_reason=r.get("exclusion_reason"),
                           baseline=_baseline_json(r),
                           qc_segment=r.get("qc", {}).get("step2_segment"),
                           qc_ocular=r.get("qc", {}).get("step5b_ocular"),
                           qc_baseline=r.get("qc", {}).get("step10_baseline"))
            for r in results if r.get("kind") == "baseline"}
    _dump(out_dir / "baselines.json", dict(run_meta=run_meta, baselines=base))

    vindex = write_variant_archives(results, run_meta)

    # Timing is set HERE, before the two wholesale-failure guards below, not at
    # the end of the report. A run that mostly failed is exactly the run whose
    # duration somebody wants to know, and those guards raise SystemExit. The
    # archive swap is already done, so this still covers the slowest non-compute
    # step. `run_meta` is the same object every output serialized, so only the
    # files written AFTER this point carry it -- cognitive_load.json is re-dumped
    # at the end of main for that reason, and is the file to read the figure
    # from.
    elapsed = time.time() - run_t0
    run_meta["elapsed_s"] = round(elapsed, 1)
    run_meta["elapsed_min"] = round(elapsed / 60.0, 2)
    run_meta["elapsed_note"] = (
        "wall clock from the first line of main() to just after the variant "
        "archive was written; the JSON files written before that point carry a "
        "run_meta WITHOUT this field")

    # A run in which everything failed still reaches here: per-recording errors
    # are caught and recorded, so the summary would print "0 recordings" over
    # freshly emptied outputs and exit 0. That happened on 2026-09-02, when an
    # edit deleted four functions the sweep calls and every recording died with a
    # NameError; the archive was clobbered under a success message. Failures are
    # normal and individually tolerable -- a WHOLESALE failure is a broken build,
    # and it should stop rather than overwrite good outputs with nothing.
    n_ok = sum(1 for r in results if r.get("status") == "ok")
    failed = [r for r in results if str(r.get("status", "")).startswith("error")]
    n_excluded = sum(1 for r in results
                     if r.get("status") == "excluded_by_user_decision")
    if results and n_ok == 0:
        # "Nothing succeeded" has two quite different causes and they used to
        # print the same sentence. An all-excluded run has no error to quote, so
        # the old message reported `The first error was: unknown` for a run in
        # which nothing had gone wrong at all.
        why = (f"EVERY recording failed ({len(failed)} of {len(results)}). The "
               f"first error was: {failed[0].get('status')}."
               if failed else
               f"NO recording was analysed: {n_excluded} of {len(results)} are "
               f"excluded by EXCLUDED_PARTICIPANTS or EXCLUDED_RECORDINGS, and "
               f"the rest produced no result.")
        raise SystemExit(
            f"{why} outputs/ has been left as this run wrote it, which may be "
            f"empty -- fix the fault and re-run to rebuild it.")
    if failed and not run_meta["partial_run"] and len(failed) > 0.5 * len(results):
        raise SystemExit(
            f"{len(failed)} of {len(results)} recordings failed, which is more "
            f"than half. Refusing to treat this as a complete run. First error: "
            f"{failed[0].get('status')}")

    n_bad = [k for k, v in vindex["recordings"].items() if v.get("default_check")]
    n_unverified = [k for k, v in vindex["recordings"].items()
                    if not v.get("default_verified")]
    ref_failed = {k: v["references_failed"] for k, v in vindex["recordings"].items()
                  if v.get("references_failed")}

    if run_meta["partial_run"]:
        print("\n*** PARTIAL RUN — outputs cover only the filtered subset. ***")
    print(f"\nWrote {out_dir/'qc_steps.csv'}")
    print(f"Wrote {out_dir/'qc_steps.json'}")
    print(f"Wrote {out_dir/'cognitive_load.json'}")
    print(f"Wrote {out_dir/'baselines.json'}")
    n_rec = len(vindex["recordings"])
    mb = vindex["bytes_on_disk"] / 1024 / 1024
    where = OUT_DIR / vindex["directory"]
    print(f"Wrote {where} ({n_rec} recordings, {mb:.0f} MB across "
          f"{len(EPOCH_CHOICES_S)} epoch lengths)")
    if run_meta["partial_run"]:
        print(f"  filtered run: written to {where.name}/, the complete "
              f"{VARIANT_DIR.name}/ archive was left untouched")
    for w in vindex.get("swap_warnings") or []:
        print(f"  NOTE: {w} (the archive itself is complete)")
    if ref_failed:
        print(f"  *** {len(ref_failed)} recording(s) where a reference could not be "
              f"applied; those variants are ABSENT from the archive:")
        for k, v in list(ref_failed.items())[:5]:
            print(f"      {k}: {'; '.join(v)}")
    if n_bad:
        print(f"  *** {len(n_bad)} recording(s) where the swept default cell does "
              f"NOT reproduce MNE's own answer: {', '.join(n_bad)}")
    elif n_unverified:
        print(f"  *** {len(n_unverified)} recording(s) where the default cell could "
              f"not be verified at all (nothing to compare): "
              f"{', '.join(n_unverified[:5])}")
    else:
        print(f"  default sweep cell (interp {DEFAULT_INTERPOLATION_MODE} / ocular "
              f"none / hardware / {DEFAULT_EPOCH_S:g} s / {DEFAULT_FFT_METHOD}) "
              f"reproduces MNE's own estimator on all {n_rec} recordings, all "
              f"{len(VARIANT_VALUE_KEYS)} columns")

    # ---- how long this took, measured rather than estimated ---------------
    # Asked for 2026-09-08: the run is long enough that "how long should I
    # expect?" is a real question, and long enough that guessing wastes an
    # afternoon. `elapsed` was captured above, before the failure guards.
    n_excl = sum(1 for r in results
                 if r.get("status") == "excluded_by_user_decision")
    total = time.time() - run_t0
    print(f"\n  RUN TIME {total / 60:.1f} min ({total:.0f} s) wall clock, "
          f"for {n_ok} analysed + {n_excl} excluded = {len(results)} recordings")
    if n_ok:
        print(f"    {total / n_ok:.1f} s per analysed recording. Scale a future "
              f"run by that: the sweep dominates, and it is linear in "
              f"recordings x epochs x estimators x references x ocular modes x "
              f"DISTINCT interpolation lists.")
    # Re-dumped so the file carries the timing: run_meta is one shared object,
    # and cognitive_load.json was serialized before `elapsed` existed.
    run_meta["elapsed_s"] = round(total, 1)
    run_meta["elapsed_min"] = round(total / 60.0, 2)
    _dump(out_dir / "cognitive_load.json", dict(run_meta=run_meta, tasks=load))


if __name__ == "__main__":
    main()

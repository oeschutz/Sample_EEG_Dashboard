"""
Writes a complete synthetic `outputs/` tree that build_dashboard.py accepts.

Why this exists
---------------
`build_dashboard.py` does not read raw recordings. It reads three JSON files and
a directory of .npz arrays that `pipeline.py` leaves in `outputs/`. That is the
whole contract, and it is documented field by field in
`dashboard_template.json` beside this script.

This generator produces that contract from nothing, so that:

  * another group can see a working dashboard without any of our data;
  * a group whose EEG comes from a different pipeline can check their own
    exporter against a known-good tree rather than against prose;
  * the format can be tested without a 40-minute pipeline run.

Nothing here is real. Every number is drawn from a seeded RNG, so two runs of
this script produce identical output apart from the `generated_utc` stamp, and
no value came off an electrode.

    python generate_template_data.py                 # -> ./template_run/
    python generate_template_data.py --out DIR       # -> DIR/outputs/...

Then, from a directory holding a copy of build_dashboard.py and that outputs/:

    python build_dashboard.py

DROPPING IN A REAL RECORDING
----------------------------
See REAL_RECORDING_SLOT near the bottom. One consented recording processed by
`pipeline.py` can replace one synthetic unit without touching anything else:
the tree is assembled per recording, so a real entry and a synthetic entry sit
side by side as long as both satisfy the contract. The band_bins table and the
sampling rate must match, and the amplitude band is refitted over whatever the
tree ends up holding.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Rig constants. These mirror pipeline.py; a different rig changes them here.
# ---------------------------------------------------------------------------

SFREQ = 250.0
EEG_CHANNELS = ["F1", "F2", "C3", "C4", "P3", "P4", "O1", "O2", "Cz", "Pz"]

# The five channels any dashboard measure reads. Order is load-bearing: it is
# the column order of every `::p::` and `::s::` array.
PEAK_CHANNELS = ["F1", "F2", "P3", "P4", "Pz"]

# Column order of every `::v::` array. Load-bearing in the same way.
VALUE_KEYS = ["fm_theta", "parietal_alpha", "parietal_beta_asym",
              "frontal_alpha_asym", "theta_f1", "theta_f2", "alpha_holm_pz"]

# Column order of every `::m::imu` array.
MOTION_COLUMNS = ["accel_jerk", "gyro", "n_aux_samples"]

EPOCHS_S = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
FFT_METHODS = ["hann", "multitaper", "welch", "boxcar"]
REFERENCES = ["hardware", "average", "rest"]
OCULAR_MODES = ["none", "eog_regression", "ica"]
INTERPOLATION_MODES = ["off", "on", "on_keepfrontal", "manual", "manual_keepfrontal"]
INTERPOLATION_SOURCES = ["off", "on", "manual"]
FRONTAL_PAIR = ["F1", "F2"]
ROBUST_SLIDE_CHOICES_S = [10.0, 20.0, 30.0]
MOTION_SOURCES = ["accel_jerk", "gyro"]

BASELINE_SEGMENTS = {"rest": [240.0, 360.0],
                     "math": [0.0, 120.0],
                     "eyes_closed": [225.0, 240.0]}
BASELINE_SEGMENT_ORDER = ["rest", "math", "eyes_closed"]
DEFAULT_BASELINE_SEGMENT = "rest"

MULTITAPER_NW = 4.0
MULTITAPER_N_TAPERS = 7

# Bands, in Hz. `alpha_holm` is Holm's 8-12; `alpha` is the example pipeline's
# 8-13. They are deliberately different quantities -- see PROVENANCE.md 7.2.
BANDS = {"theta": (4.0, 8.0), "alpha": (8.0, 13.0),
         "alpha_holm": (8.0, 12.0), "beta": (13.0, 30.0)}

AMPLITUDE_BOUND_K = 3.0
AMPLITUDE_BOUND_MIN_N = 100

# ---------------------------------------------------------------------------
# The synthetic study.
#
# Four participants. Three are analysed and one is excluded upstream, so the
# dashboard's excluded-recordings table has something to print. Each analysed
# participant contributes two cells; each cell is a task plus the baseline
# recorded in the same session.
#
# `bad` is the channel list bad-channel detection returns for that PAIR (the
# task and its baseline share one list -- PROVENANCE.md 2.3). `manual` is the
# hand-picked list, which may disagree with `bad` in either direction; that
# disagreement is the point of shipping both.
# ---------------------------------------------------------------------------

TASK_DURATION_S = 600.0
BASELINE_BLOCK_S = 360.0

CELLS = [
    # (participant, arm, framing, detected bad channels, manual list, notes)
    dict(participant="t01", arm="agent", framing="personal",
         bad=["F1", "F2"], manual=["F1", "F2"],
         note="both frontal channels flagged -- the frontal-pair checkbox bites here"),
    dict(participant="t01", arm="ai", framing="speedscore",
         bad=[], manual=[],
         note="clean pair: every interpolation mode resolves to the same array"),
    dict(participant="t02", arm="agent", framing="speedscore",
         bad=["Pz"], manual=["Pz", "O2"],
         note="the manual list asks for a channel detection never flagged"),
    dict(participant="t02", arm="ai", framing="personal",
         bad=["P4", "O2"], manual=["P4"],
         note="the manual list omits a channel detection did flag"),
    dict(participant="t03", arm="none", framing="personal",
         bad=["C3"], manual=[],
         note="no-AI control arm; empty manual list means interpolate nothing"),
    dict(participant="t03", arm="none", framing="speedscore",
         bad=["F1"], manual=["F1"],
         note="one frontal channel only -- the frontal-pair checkbox is a no-op",
         motion=False,
         motion_why=("the IMU stream was identically zero for the whole file; "
                     "a flat sensor is treated as NO sensor, not as a "
                     "perfectly still head")),
]

# Recordings discovered but not analysed. They appear in the JSON with a
# non-ok status so the page can say what dropped out and why.
EXCLUDED = [
    dict(participant="t04", arm="ai", framing="personal",
         status="excluded_by_user_decision",
         reason=("excluded for data quality: nearly every channel flat in the "
                 "task while the paired baseline looked normal (synthetic "
                 "example of an unrecoverable recording)")),
]

# Channels whose synthetic amplitude is pushed outside the derived band, so the
# quality badge has something to flag. Keyed by recording key.
LOUD_CHANNELS = {
    "t01/task_agent_personal": {"F1": 180.0, "F2": 210.0},
    "t01/baseline_agent_personal": {"F1": 165.0, "F2": 198.0},
    "t02/task_ai_personal": {"P4": 96.0},
    "t02/baseline_ai_personal": {"P4": 88.0},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def interp_token(channels) -> str:
    """The archive's key for a channel list. Empty list -> '-'.

    The sweep is keyed by WHICH CHANNELS were interpolated, not by mode name,
    so two modes that reach the same list share one stored array.
    """
    if not channels:
        return "-"
    order = {c: i for i, c in enumerate(EEG_CHANNELS)}
    return "+".join(sorted(channels, key=lambda c: order.get(c, 99)))


def drop_frontal_pair(channels) -> list:
    """`channels` minus F1 and F2 -- but only when BOTH are present."""
    if all(c in channels for c in FRONTAL_PAIR):
        return [c for c in channels if c not in FRONTAL_PAIR]
    return list(channels)


def interpolation_lists(detected, manual) -> dict:
    """The channel list each of the five modes replaces, for one pair."""
    return {
        "off": [],
        "on": list(detected),
        "on_keepfrontal": drop_frontal_pair(detected),
        "manual": list(manual),
        "manual_keepfrontal": drop_frontal_pair(manual),
    }


def band_bin_table() -> dict:
    """Bins per band per (window length, estimator).

    A pure function of the sampling rate and the window length, which is why
    build_dashboard.py refuses to build if two recordings report different
    tables -- that would mean something upstream resampled.
    """
    out = {}
    for epoch in EPOCHS_S:
        for fft in FFT_METHODS:
            # Welch halves the segment, so its bin spacing is twice as coarse.
            res = (2.0 if fft == "welch" else 1.0) / epoch
            entry = {"freq_resolution_hz": round(res, 10),
                     "smoothing_half_bandwidth_hz":
                         round(MULTITAPER_NW / epoch, 10) if fft == "multitaper" else 0.0}
            for band, (lo, hi) in BANDS.items():
                entry[band] = int(math.floor(hi / res + 1e-9)
                                  - math.ceil(lo / res - 1e-9) + 1)
            out[f"{epoch:g}|{fft}"] = entry
    return out


def n_windows_for(duration_s: float) -> dict:
    """Whole windows of each offered length that fit in `duration_s`."""
    return {f"{e:g}": max(1, int(duration_s // e)) for e in EPOCHS_S}


def derive_amplitude_bound(amplitudes) -> dict:
    """The band outside which a channel is UNUSUAL FOR THIS RUN.

    Pooled across channels, fitted in log10, with the spread taken from the
    LOWER HALF only so a noisy tail cannot set its own threshold. Relative by
    construction: it says "unusual for this helmet on these recordings", never
    "physiologically implausible".
    """
    vals = np.asarray([v for v in amplitudes if v and v > 0], dtype=float)
    if vals.size < AMPLITUDE_BOUND_MIN_N:
        return {"available": False,
                "reason": f"only {vals.size} channel measurements; "
                          f"{AMPLITUDE_BOUND_MIN_N} are needed to fit a band"}
    logs = np.log10(vals)
    median = float(np.median(logs))
    p25 = float(np.percentile(logs, 25))
    sigma = (median - p25) / 0.6745
    lo = 10.0 ** (median - AMPLITUDE_BOUND_K * sigma)
    hi = 10.0 ** (median + AMPLITUDE_BOUND_K * sigma)
    outside = float(np.mean((vals < lo) | (vals > hi)) * 100.0)
    return {
        "available": True,
        "low_uv": round(lo, 2),
        "high_uv": round(hi, 2),
        "partial_run": False,
        "k": AMPLITUDE_BOUND_K,
        "median_uv": round(float(10.0 ** median), 2),
        "sigma_log10": round(sigma, 5),
        "n_measurements": int(vals.size),
        "pct_outside": round(outside, 1),
        "method": ("pooled across channels; log10; spread from "
                   "(median - p25)/0.6745 so the upper tail cannot set its own "
                   "threshold; +/- k sigma"),
        "basis": "this run's own channel amplitudes (relative, not absolute)",
    }


# ---------------------------------------------------------------------------
# Synthetic signal
# ---------------------------------------------------------------------------

def stable_seed(*parts) -> int:
    """A deterministic 32-bit seed from any tuple of values.

    NOT `hash()`: Python randomises string hashing per process unless
    PYTHONHASHSEED is set, so `hash()` here would make two runs of this script
    produce different numbers. The whole point of the generator is that it does
    not.
    """
    payload = "|".join(repr(x) for x in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


class Unit:
    """One swept unit: a task recording, or one segment of one baseline."""

    def __init__(self, key, kind, participant, recording, duration_s,
                 detected, manual, seed, segment=None, bounds=None,
                 motion=True, motion_why=None, loud=None):
        self.key = key
        self.kind = kind
        self.participant = participant
        self.recording = recording
        self.duration_s = duration_s
        self.detected = list(detected)
        self.manual = list(manual)
        self.segment = segment
        self.bounds = bounds
        self.motion = motion
        self.motion_why = motion_why
        self.loud = dict(loud or {})
        self.rng = np.random.default_rng(seed)
        self.lists = interpolation_lists(detected, manual)
        self.tokens = {m: interp_token(v) for m, v in self.lists.items()}
        self.n_windows = n_windows_for(duration_s)
        # Channel amplitude as recorded: robust SD in microvolts, hardware
        # reference, before interpolation. This is what the quality badge reads.
        self.amp_hardware = {}
        for ch in PEAK_CHANNELS:
            if ch in self.loud:
                self.amp_hardware[ch] = self.loud[ch]
            else:
                self.amp_hardware[ch] = float(
                    np.round(10.0 ** self.rng.normal(1.32, 0.12), 2))

    @property
    def array_prefix(self) -> str:
        return self.key.replace("/", "~")

    def distinct_tokens(self) -> list:
        seen, out = set(), []
        for mode in INTERPOLATION_MODES:
            tok = self.tokens[mode]
            if tok not in seen:
                seen.add(tok)
                out.append(tok)
        return out

    # -- amplitudes -------------------------------------------------------

    def amplitude_displayed(self) -> dict:
        """Amplitude of what is actually PLOTTED, per token|ocular|reference.

        Interpolating a channel replaces it with a spline, which is smooth by
        construction and much quieter. Re-referencing mixes every channel, so
        one loud electrode is smeared across all of them -- which is exactly
        why the quality badge over-flags under Average and REST.
        """
        out = {}
        for token in self.distinct_tokens():
            replaced = set() if token == "-" else set(token.split("+"))
            for oc in OCULAR_MODES:
                oc_gain = {"none": 1.0, "eog_regression": 0.94, "ica": 0.97}[oc]
                base = {}
                for ch in PEAK_CHANNELS:
                    v = self.amp_hardware[ch] * oc_gain
                    if ch in replaced:
                        # A spline built from the surviving channels.
                        donors = [self.amp_hardware[c] for c in PEAK_CHANNELS
                                  if c not in replaced]
                        v = (float(np.mean(donors)) if donors else v) * 0.85
                    base[ch] = v
                mean_amp = float(np.mean(list(base.values())))
                for ref in REFERENCES:
                    if ref == "hardware":
                        vals = base
                    elif ref == "average":
                        # Subtracting the common average pulls loud channels
                        # down and pushes quiet ones up.
                        vals = {c: abs(v - 0.55 * mean_amp) + 0.45 * mean_amp
                                for c, v in base.items()}
                    else:
                        vals = {c: abs(v - 0.30 * mean_amp) + 0.80 * mean_amp
                                for c, v in base.items()}
                    out[f"{token}|{oc}|{ref}"] = {
                        c: round(float(v), 2) for c, v in vals.items()}
        return out

    def sigma(self) -> dict:
        """Whole-recording robust SD per channel, per ocular mode."""
        return {oc: {c: round(self.amp_hardware[c]
                              * {"none": 1.0, "eog_regression": 0.94,
                                 "ica": 0.97}[oc], 6)
                     for c in PEAK_CHANNELS}
                for oc in OCULAR_MODES}

    # -- arrays -----------------------------------------------------------

    def values(self, n, token, oc, ref, fft) -> np.ndarray:
        """(n, 7) float32 of per-window measures, in VALUE_KEYS order."""
        rng = np.random.default_rng(
            stable_seed(self.key, token, oc, ref, fft, n))
        replaced = set() if token == "-" else set(token.split("+"))

        # A slow drift plus per-window noise, so the traces have shape rather
        # than looking like white noise.
        drift = np.sin(np.linspace(0, 2.4 * np.pi, n)) * 0.18
        gain = {"hann": 1.0, "multitaper": 1.04, "welch": 0.97, "boxcar": 1.09}[fft]
        gain *= {"hardware": 1.0, "average": 0.72, "rest": 0.81}[ref]
        gain *= {"none": 1.0, "eog_regression": 0.88, "ica": 0.93}[oc]

        # Interpolated frontal channels are rebuilt from posterior donors, so
        # their theta drops towards the posterior level.
        f_scale = 0.55 if replaced & set(FRONTAL_PAIR) else 1.0

        theta_f1 = np.exp(rng.normal(1.05, 0.45, n) + drift) * gain * f_scale
        theta_f2 = np.exp(rng.normal(1.02, 0.45, n) + drift) * gain * f_scale
        alpha_pz = np.exp(rng.normal(1.28, 0.40, n) - drift) * gain
        fm_theta = 0.5 * (theta_f1 + theta_f2)
        par_alpha = alpha_pz * np.exp(rng.normal(0.04, 0.10, n))

        # Asymmetries are differences of logs: already signed, centred near
        # zero, and driven towards zero when both sides are rebuilt from the
        # same donors.
        asym_scale = 0.25 if replaced & set(FRONTAL_PAIR) else 1.0
        frontal_asym = rng.normal(0.0, 0.42, n) * asym_scale
        par_beta_asym = rng.normal(0.05, 0.38, n)

        out = np.empty((n, len(VALUE_KEYS)), dtype="<f4")
        for i, name in enumerate(VALUE_KEYS):
            out[:, i] = {"fm_theta": fm_theta, "parietal_alpha": par_alpha,
                         "parietal_beta_asym": par_beta_asym,
                         "frontal_alpha_asym": frontal_asym,
                         "theta_f1": theta_f1, "theta_f2": theta_f2,
                         "alpha_holm_pz": alpha_pz}[name]
        return out

    def peaks(self, n, oc) -> np.ndarray:
        """(n, 5) float32 per-window peak amplitude, microvolts.

        Measured on the PRE-INTERPOLATION signal in the hardware reference,
        whatever is displayed -- a spline passes an amplitude threshold far
        more readily than a real electrode.
        """
        rng = np.random.default_rng(
            stable_seed(self.key, "p", oc, n))
        out = np.empty((n, len(PEAK_CHANNELS)), dtype="<f4")
        oc_gain = {"none": 1.0, "eog_regression": 0.9, "ica": 0.95}[oc]
        for i, ch in enumerate(PEAK_CHANNELS):
            scale = self.amp_hardware[ch] * 2.6 * oc_gain
            col = scale * np.exp(rng.normal(0.0, 0.35, n))
            # A handful of transients, so the rejection sliders do something.
            hits = rng.choice(n, size=max(1, n // 40), replace=False)
            col[hits] *= rng.uniform(3.0, 9.0, hits.size)
            out[:, i] = col
        return out

    def sigma_slide(self, n, oc, length_s) -> np.ndarray:
        """(n, 5) float32 per-window robust sigma for the Windowed Robust mode.

        Recomputed from the `length_s` seconds CENTRED on each window, so a
        shorter calibration window tracks the signal more closely and is more
        volatile. Not derivable in the browser, which is why it is stored per
        offered length rather than offered as a slider.
        """
        rng = np.random.default_rng(
            stable_seed(self.key, "s", oc, length_s, n))
        volatility = {10.0: 0.42, 20.0: 0.28, 30.0: 0.20}[length_s]
        out = np.empty((n, len(PEAK_CHANNELS)), dtype="<f4")
        oc_gain = {"none": 1.0, "eog_regression": 0.9, "ica": 0.95}[oc]
        for i, ch in enumerate(PEAK_CHANNELS):
            base = self.amp_hardware[ch] * oc_gain
            out[:, i] = base * np.exp(rng.normal(0.0, volatility, n))
        return out

    def motion_matrix(self, n, epoch_s) -> np.ndarray:
        """(n, 3) float32: accelerometer jerk, gyroscope, aux sample count.

        One matrix per recording per window length -- no ocular, reference or
        estimator dimension, because the IMU is a property of the recording
        rather than of a processing choice.
        """
        rng = np.random.default_rng(
            stable_seed(self.key, "m", n))
        accel = np.exp(rng.normal(-5.4, 0.7, n))
        # The two sensors correlate but disagree on which windows are extreme.
        gyro = np.exp(rng.normal(-0.6, 0.8, n) + 0.55 * (np.log(accel) + 5.4))
        hits = rng.choice(n, size=max(1, n // 30), replace=False)
        accel[hits] *= rng.uniform(4.0, 12.0, hits.size)
        gyro[rng.choice(n, size=max(1, n // 30), replace=False)] *= 6.0
        counts = np.full(n, round(epoch_s * 50.0), dtype=float)
        out = np.empty((n, 3), dtype="<f4")
        out[:, 0], out[:, 1], out[:, 2] = accel, gyro, counts
        return out

    # -- index entry ------------------------------------------------------

    def index_entry(self, band_bins) -> dict:
        n4 = self.n_windows["4"]
        entry = {
            "kind": self.kind,
            "array_prefix": self.array_prefix,
            "segment": self.segment,
            "segment_bounds_s": list(self.bounds) if self.bounds else None,
            "bad_channels": list(self.detected),
            "interpolation_lists": self.lists,
            "interpolation_tokens": self.tokens,
            "references_failed": [],
            "sigma": self.sigma(),
            "motion_available": bool(self.motion),
            "motion_unavailable_reason": self.motion_why,
            "excluded_windows": {f"{e:g}": [] for e in EPOCHS_S},
            "amplitude_displayed": self.amplitude_displayed(),
            "band_bins": band_bins,
            "n_windows": self.n_windows,
            # A task carries RECORDED window times; a baseline segment ships no
            # per-window times and the page assumes a contiguous grid.
            "times_s": ({f"{e:g}": [round(i * e, 3)
                                    for i in range(self.n_windows[f"{e:g}"])]
                         for e in EPOCHS_S} if self.kind == "task" else None),
            "window_index": (None if self.kind == "task" else
                             {f"{e:g}": list(range(self.n_windows[f"{e:g}"]))
                              for e in EPOCHS_S}),
            # An empty `default_check` with `default_verified` true means the
            # default cell was compared against MNE's own estimator and agreed.
            # build_dashboard.py refuses to build without both.
            "default_check": [],
            "default_verified": True,
            "default_check_detail": {"n_windows": n4, "n_columns": len(VALUE_KEYS)},
        }
        return entry


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def build_units() -> tuple:
    """Every swept unit, plus the task/baseline bookkeeping around them."""
    tasks, baselines, units = {}, {}, []
    seed = 1000

    for cell in CELLS:
        p, arm, framing = cell["participant"], cell["arm"], cell["framing"]
        tname = f"task_{arm}_{framing}"
        bname = f"baseline_{arm}_{framing}"
        tkey, bkey = f"{p}/{tname}", f"{p}/{bname}"
        seed += 1

        task = Unit(tkey, "task", p, tname, TASK_DURATION_S,
                    cell["bad"], cell["manual"], seed,
                    motion=cell.get("motion", True),
                    motion_why=cell.get("motion_why"),
                    loud=LOUD_CHANNELS.get(tkey))
        units.append(task)

        segs = {}
        for si, seg in enumerate(BASELINE_SEGMENT_ORDER):
            lo, hi = BASELINE_SEGMENTS[seg]
            u = Unit(f"{bkey}::{seg}", "baseline", p, bname, hi - lo,
                     cell["bad"], cell["manual"], seed + 10 + si,
                     segment=seg, bounds=(lo, hi),
                     motion=cell.get("motion", True),
                     motion_why=cell.get("motion_why"),
                     loud=LOUD_CHANNELS.get(bkey))
            units.append(u)
            segs[seg] = u

        tasks[tkey] = (task, bkey, cell)
        baselines[bkey] = (segs, cell)

    return tasks, baselines, units


def write_json_outputs(out_dir: Path, tasks, baselines, bound, stamp):
    """outputs/cognitive_load.json and outputs/baselines.json.

    build_dashboard.py reads a narrow slice of these -- see
    dashboard_template.json for exactly which fields are required. The rest are
    pipeline products kept here because a real tree has them.
    """
    low, high = (bound["low_uv"], bound["high_uv"]) if bound["available"] else (None, None)

    def plausible(v):
        return None if not bound["available"] else bool(low <= v <= high)

    run_meta = {
        "generated_utc": stamp,
        "argv": [],
        "partial_run": False,
        "n_discovered": 2 * (len(CELLS) + len(EXCLUDED)),
        "n_processed": 2 * (len(CELLS) + len(EXCLUDED)),
        "excluded_participants": [],
        "excluded_recordings": sorted(
            f"{e['participant']}/{k}_{e['arm']}_{e['framing']}"
            for e in EXCLUDED for k in ("baseline", "task")),
        "parameters": {
            "holm_window_s": 4.0,
            "holm_theta_hz": list(BANDS["theta"]),
            "holm_alpha_hz": list(BANDS["alpha_holm"]),
            "new_theta_hz": list(BANDS["theta"]),
            "new_alpha_hz": list(BANDS["alpha"]),
            "new_beta_hz": list(BANDS["beta"]),
            "robust_sigma": 5.0,
            "amplitude_bound": bound,
            "index_gate_applied": False,
            "index_gate_applied_note": "no amplitude gate; labels only",
            # build_dashboard.py REFUSES to build unless this reads "paired".
            # It is the record that one bad-channel list was decided per
            # task/baseline pair rather than per recording.
            "bad_channel_detection": "paired",
            "sfreq_hz": SFREQ,
            "synthetic": True,
        },
        "note": ("SYNTHETIC. Every value in this tree was generated by "
                 "template/generate_template_data.py. No electrode was "
                 "involved and no participant exists."),
    }

    task_doc = {}
    for tkey, (task, bkey, cell) in tasks.items():
        task_doc[tkey] = {
            "participant": task.participant,
            "recording": task.recording,
            "status": "ok",
            "exclusion_reason": None,
            "baseline_key": bkey,
            # Not read by build_dashboard.py; a real tree carries the
            # per-window measures here.
            "holm": None,
            "measures": None,
            "measure_times_s": None,
            "qc_holm": None,
            "qc_gate": None,
            "qc_segment": {
                "n_markers": 2,
                "repair": None,
                "notes": [cell["note"]],
                "marker_span_s": TASK_DURATION_S,
                "segment_duration_s": TASK_DURATION_S,
                "segment_elapsed_s": TASK_DURATION_S,
                "segment_n_samples": int(TASK_DURATION_S * SFREQ),
                "n_internal_gaps": 0,
                "max_internal_gap_s": 0.0,
                "clean": True,
            },
            "qc_bandpass": {
                "lowcut_hz": 0.5,
                "highcut_hz": 100.0,
                "per_channel": {
                    c: {"robust_sd_uv": task.amp_hardware[c],
                        "sd_uv": round(task.amp_hardware[c] * 1.8, 2),
                        "pct_over_70uv": round(
                            min(99.0, task.amp_hardware[c] / 1.6), 2),
                        "peak_to_peak_uv": round(task.amp_hardware[c] * 14.0, 1)}
                    for c in PEAK_CHANNELS},
                "amplitude_bound_uv": [low, high],
                "clean": True,
                "notes": [],
            },
            "qc_interp": {
                # `none` is the PAIRED list -- the one that feeds every
                # published number. `example` is the reference notebook's own
                # per-recording detection, kept only so step 8a stays
                # reproducible.
                "none": {"bad_channels_detected": list(task.detected),
                         "index_channels_interpolated": list(task.detected),
                         "detection": {"rule": "robust z of log10 SD > 3.29",
                                       "scope": "task + rest segment of its baseline"}},
                "example": {"example_pipeline_hardcoded": ["Cz", "F2"]},
            },
            "qc_ocular": {
                "none": {"paired": True},
                "eog_regression": {"paired": True, "degenerate": False,
                                   "max_abs_coefficient": 0.41,
                                   "sd_reduction_by_channel_pct":
                                       {c: 6.0 for c in PEAK_CHANNELS}},
                "ica": {"paired": True, "converged": True,
                        "n_components_removed": 1,
                        "variance_removed_pct": 8.4,
                        "max_abs_eog_correlation": 0.62},
            },
        }

    for e in EXCLUDED:
        key = f"{e['participant']}/task_{e['arm']}_{e['framing']}"
        task_doc[key] = {
            "participant": e["participant"],
            "recording": f"task_{e['arm']}_{e['framing']}",
            "status": e["status"],
            "exclusion_reason": e["reason"],
            "baseline_key": f"{e['participant']}/baseline_{e['arm']}_{e['framing']}",
            "holm": None, "measures": None, "measure_times_s": None,
            "qc_holm": None, "qc_interp": None, "qc_segment": None,
            "qc_bandpass": None, "qc_ocular": None, "qc_gate": None,
        }

    base_doc = {}
    for bkey, (segs, cell) in baselines.items():
        any_seg = segs[DEFAULT_BASELINE_SEGMENT]
        base_doc[bkey] = {
            "participant": any_seg.participant,
            "recording": any_seg.recording,
            "status": "ok",
            "exclusion_reason": None,
            "baseline": {
                "block_duration_s": BASELINE_BLOCK_S,
                # The channels actually interpolated in this baseline -- the
                # SAME list as its task, decided once per pair.
                "interpolated": list(cell["bad"]),
                "segment_bounds_s": BASELINE_SEGMENTS,
                "segment_order": BASELINE_SEGMENT_ORDER,
                "default_segment": DEFAULT_BASELINE_SEGMENT,
                "segments": {
                    seg: {
                        "segment": seg,
                        "bounds_s": list(BASELINE_SEGMENTS[seg]),
                        "duration_s": u.duration_s,
                        "n_windows": u.n_windows["4"],
                        # Each segment is cropped and calibrated on its OWN
                        # samples, so it carries its own amplitudes.
                        "amplitude": {c: {"robust_sd_uv": u.amp_hardware[c],
                                          "plausible": plausible(u.amp_hardware[c])}
                                      for c in PEAK_CHANNELS},
                        "implausible_channels": [
                            c for c in PEAK_CHANNELS
                            if bound["available"] and not plausible(u.amp_hardware[c])],
                        "ocular": {"mode_order": OCULAR_MODES},
                        "variant_default_check": [],
                    }
                    for seg, u in segs.items()},
            },
            "qc_segment": {"n_markers": 2, "repair": None, "notes": [],
                           "segment_duration_s": BASELINE_BLOCK_S, "clean": True},
            "qc_ocular": {
                "none": {"paired": True},
                "eog_regression": {"paired": True, "degenerate": False,
                                   "max_abs_coefficient": 0.38},
                "ica": {"paired": True, "converged": True,
                        "n_components_removed": 1, "variance_removed_pct": 7.1,
                        "max_abs_eog_correlation": 0.58},
            },
            "qc_baseline": {"clean": True},
        }

    for e in EXCLUDED:
        key = f"{e['participant']}/baseline_{e['arm']}_{e['framing']}"
        base_doc[key] = {
            "participant": e["participant"],
            "recording": f"baseline_{e['arm']}_{e['framing']}",
            "status": e["status"],
            "exclusion_reason": e["reason"],
            "baseline": None, "qc_segment": None, "qc_ocular": None,
            "qc_baseline": None,
        }

    (out_dir / "cognitive_load.json").write_text(
        json.dumps({"run_meta": run_meta, "tasks": task_doc}, indent=1),
        encoding="utf-8")
    (out_dir / "baselines.json").write_text(
        json.dumps({"run_meta": run_meta, "baselines": base_doc}, indent=1),
        encoding="utf-8")
    return run_meta


def write_arrays(variants: Path, units, epoch: float) -> None:
    """The two .npz files for one window length.

    Array naming is the whole contract:

        <prefix>::v::<token>|<ocular>|<reference>|<estimator>   (n, 7)
        <prefix>::p::<ocular>                                   (n, 5)
        <prefix>::s::<ocular>|<calibration length>              (n, 5)
        <prefix>::m::imu                                        (n, 3)

    `<prefix>` is the archive key with '/' replaced by '~'.
    """
    ekey = f"{epoch:g}"
    for kind, fname in (("task", "tasks"), ("baseline", "baselines")):
        arrays = {}
        for u in units:
            if u.kind != kind:
                continue
            n = u.n_windows[ekey]
            pre = u.array_prefix
            for token in u.distinct_tokens():
                for oc in OCULAR_MODES:
                    for ref in REFERENCES:
                        for fft in FFT_METHODS:
                            arrays[f"{pre}::v::{token}|{oc}|{ref}|{fft}"] = \
                                u.values(n, token, oc, ref, fft)
            for oc in OCULAR_MODES:
                arrays[f"{pre}::p::{oc}"] = u.peaks(n, oc)
                for L in ROBUST_SLIDE_CHOICES_S:
                    arrays[f"{pre}::s::{oc}|{L:g}"] = u.sigma_slide(n, oc, L)
            # A unit with no usable IMU ships NO motion matrix. The page then
            # counts its windows as unassessed rather than as clean.
            if u.motion:
                arrays[f"{pre}::m::imu"] = u.motion_matrix(n, epoch)
        np.savez_compressed(variants / f"{fname}_e{epoch:04.1f}.npz", **arrays)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", default="template_run",
                    help="directory to write outputs/ into (default: template_run)")
    args = ap.parse_args()

    root = Path(args.out).resolve()
    out_dir = root / "outputs"
    variants = out_dir / "variants"
    variants.mkdir(parents=True, exist_ok=True)

    stamp = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    tasks, baselines, units = build_units()

    # The band is fitted over every channel measurement in the tree, exactly as
    # pipeline.py fits it over a real run.
    bound = derive_amplitude_bound(
        [u.amp_hardware[c] for u in units for c in PEAK_CHANNELS])
    bound["n_recordings"] = len(units)
    if not bound["available"]:
        raise SystemExit(
            f"only {len(units) * len(PEAK_CHANNELS)} channel measurements; "
            f"{AMPLITUDE_BOUND_MIN_N} are needed. Add cells to CELLS.")

    run_meta = write_json_outputs(out_dir, tasks, baselines, bound, stamp)

    band_bins = band_bin_table()
    index = {
        "run_meta": run_meta,
        "value_keys": VALUE_KEYS,
        "peak_channels": PEAK_CHANNELS,
        "epochs_s": EPOCHS_S,
        "fft_methods": FFT_METHODS,
        "multitaper_nw": MULTITAPER_NW,
        "multitaper_n_tapers": MULTITAPER_N_TAPERS,
        "robust_slide_choices_s": ROBUST_SLIDE_CHOICES_S,
        "robust_slide_default_s": 30.0,
        "motion_sources": MOTION_SOURCES,
        "default_motion_source": "accel_jerk",
        "motion_columns": MOTION_COLUMNS,
        "motion_min_samples": 10,
        "motion_k": {"min": 1.0, "max": 8.0, "step": 0.25, "default": 3.0},
        "amplitude_bound": bound,
        "window_max_gap_s": 0.5,
        "baseline_segments": BASELINE_SEGMENTS,
        "baseline_segment_order": BASELINE_SEGMENT_ORDER,
        "default_baseline_segment": DEFAULT_BASELINE_SEGMENT,
        "references": REFERENCES,
        "interpolation_modes": INTERPOLATION_MODES,
        "interpolation_sources": INTERPOLATION_SOURCES,
        "interpolation_frontal_pair": FRONTAL_PAIR,
        "default_interpolation_mode": "on",
        "manual_interpolation": {
            f"{c['participant']}|{c['arm']}": list(c["manual"])
            for c in CELLS if c["manual"]},
        "ocular_modes": OCULAR_MODES,
        "default": {"epoch_s": 4.0, "fft": "hann", "reference": "hardware",
                    "interpolation": "on", "ocular": "none"},
        "sliders": {
            "robust_sigma": {"min": 1.0, "max": 12.0, "step": 0.25, "default": 5.0},
            "holm_cap_uv": {"min": 10.0, "max": 500.0, "step": 5.0, "default": 70.0},
        },
        "recordings": {u.key: u.index_entry(band_bins) for u in units},
        "directory": "variants",
    }

    for epoch in EPOCHS_S:
        write_arrays(variants, units, epoch)

    index["bytes_on_disk"] = sum(p.stat().st_size for p in variants.glob("*.npz"))
    (variants / "index.json").write_text(json.dumps(index, indent=1),
                                         encoding="utf-8")

    n_task = sum(1 for u in units if u.kind == "task")
    mb = index["bytes_on_disk"] / 1024 / 1024
    print(f"Wrote {out_dir}")
    print(f"  {n_task} task recordings, {len(units) - n_task} baseline segments, "
          f"{len(units)} swept units")
    print(f"  {len(EXCLUDED)} recording(s) excluded upstream")
    print(f"  amplitude band {bound['low_uv']}-{bound['high_uv']} uV "
          f"from {bound['n_measurements']} measurements "
          f"({bound['pct_outside']}% outside)")
    print(f"  variants/ {mb:.1f} MB across {len(EPOCHS_S)} window lengths")
    print(f"\nNow copy build_dashboard.py next to {root} and run it there.")


# ---------------------------------------------------------------------------
# REAL_RECORDING_SLOT
# ---------------------------------------------------------------------------
# To publish this template with one real consented recording instead of a
# synthetic one:
#
#   1. Run `pipeline.py --only <participant>/<recording>` on that session. It
#      writes outputs/partial/ and outputs/variants_partial/ rather than
#      clobbering a complete archive.
#   2. Take that recording's entry out of variants_partial/index.json under
#      "recordings", and its arrays out of the partial .npz files. The arrays
#      are named by `array_prefix`, so a whole recording moves as a name
#      prefix.
#   3. Drop both into the tree this script writes, and remove the synthetic
#      cell it replaces from CELLS.
#   4. Re-run this script with that cell removed, then merge. The amplitude
#      band refits over whatever the tree holds, so it must be recomputed
#      after the merge, not before -- `derive_amplitude_bound` above is the
#      same fit pipeline.py uses.
#
# What must match for a mixed tree to build: the sampling rate (the band_bins
# table is a function of it), the five PEAK_CHANNELS, the seven VALUE_KEYS,
# and the mode lists. build_dashboard.py checks the mode lists and the
# band_bins table explicitly and refuses rather than building something wrong.
#
# IRB: a real recording here is identifiable human data. That is a consent
# question, not a technical one, and it is deliberately left unanswered in
# this file.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()

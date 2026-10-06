"""
Builds dashboard.html (plus dashboard_data/) from outputs/cognitive_load.json,
outputs/baselines.json and outputs/variants/.

Run pipeline.py first. Analysis decisions are recorded inline there, beside the
reviews that prompted the 2026-08-27, 2026-08-28 and 2026-09-01 revisions.

2026-08-28: five tabs instead of one. The Holm cognitive-load index keeps the
original tab; four further measures were added on tabs of their own, and every
tab gained a resting-baseline subtraction.

2026-09-01: seven processing choices that were fixed constants became controls.
Five of them change the values and are precomputed by the pipeline across a
1800-cell grid -- bad-channel interpolation (three sources x the frontal-pair
option, 2026-09-08), ocular correction, reference, window length and spectral
estimator. Fewer arrays than cells are stored: two interpolation modes that
replace the same channels are one signal and share one array. The rest only threshold quantities the pipeline
already ships, so they are evaluated in the browser and are continuous sliders:
the robust rejection distance and the absolute cap. (The amplitude gate was a
third such slider until 2026-09-02, when the gate was removed outright.)

2026-09-02: a "Windowed Robust" rejection mode, which applies the robust rule
against a sigma recomputed from a centred sliding window (10, 20 or 30 s,
toggled) rather than over the whole recording. It is the one browser-side threshold that is NOT derivable from
what was already shipped -- sigma is the MAD of the raw samples, and the page
only ever receives per-window peaks -- so the pipeline computes it per window at
each offered length, and this script refuses to build against an archive that
predates it.

The grid is far too large to embed whole -- about 25 million float32 values -- so
this script writes the 4 s window length inline (every other combination of the
remaining four controls, which is what the page opens on) and the other nine
window lengths into dashboard_data/ beside it, loaded on demand as plain
<script> tags so it works from a file:// URL. Copy the folder with the file if
you want the window-length slider to move; the file alone still opens and works
at 4 s.
"""

from __future__ import annotations

import base64
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
VARIANTS = OUT / "variants"
SHARD_DIR_NAME = "dashboard_data"

# Channels any measure on the dashboard reads.
CH5 = ["F1", "F2", "P3", "P4", "Pz"]

# Baseline segments, filled from the sweep index in main() so the page and the
# pipeline cannot disagree about what "eyes_closed" means.
SEGMENTS: dict = {}
SEG_ORDER: list = []

# Robust-SD band outside which a channel's amplitude is UNUSUAL FOR THIS DATASET.
# Filled in from the sweep index in main(); the pipeline derives it per run from
# its own channel amplitudes (pipeline.derive_amplitude_bounds).
#
# It replaced a hardcoded (1.0, 50.0) on 2026-09-03. That pair had no published
# source, was never derived from electrode data, and in particular was not a
# dry-electrode figure -- this helmet is 100% dry, where contact impedance and
# drift run higher, and 18.6% of all channel-measurements here exceeded 50 uV.
#
# It is what the quality badge judges against, and since the gate's removal
# (2026-09-02, browser-side and pipeline-side) that is ALL it does: no recording
# is withheld for falling outside it, on this page or upstream. Being relative,
# it says "unusual for this helmet on these recordings", never "physiologically
# implausible" -- the page says so too rather than implying an absolute standard.
PLAUSIBLE_UV = (None, None)

# The ocular axis this page is written for, in order. Compared against the
# archive's own list at build time -- ordered equality, because position is
# load-bearing: pipeline.py takes the FIRST entry as the uncorrected branch when
# it records a baseline segment's "as recorded" amplitude. Kept as a literal
# rather than imported from pipeline.py, matching how SEGMENTS and PLAUSIBLE_UV
# are handled; the build refuses rather than drifting if the two disagree.
EXPECTED_INTERPOLATION_MODES = ["off", "on", "on_keepfrontal",
                                "manual", "manual_keepfrontal"]
EXPECTED_OCULAR_MODES = ["none", "eog_regression", "ica"]


def _remove_with_retry(path: Path, attempts: int = 6) -> bool:
    """
    Delete a file or directory, retrying briefly. True if it is gone.

    OneDrive, Windows Search and antivirus all take transient handles on files in
    a synced folder, and a delete that touches one raises PermissionError.
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

    Moves FILES rather than renaming directories. `shutil.rmtree(target)` followed
    by `staging.rename(target)` is the standard POSIX idiom and it failed here on
    both this file and pipeline.py: rmtree removed every file and then raised
    PermissionError on the now-empty directory, because OneDrive was holding it.
    A per-file move never has to delete or rename a directory another process is
    watching, and the target is not emptied until every replacement exists.

    The twin of pipeline.py's function of the same name. Duplicated rather than
    imported so that building the dashboard does not have to import MNE.
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



def sig(v, n: int = 5):
    """
    Round for the payload, or None.

    Decimal places for values of order 1 and above; significant figures only
    below that, where band power legitimately runs small and fixed decimals
    would flatten it to zero. Used for the scalar metadata only -- the per-window
    series are shipped as raw float32 and never pass through here.
    """
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    if abs(f) >= 1.0:
        return round(f, 5)
    return float(f"{f:.{n}g}")


def condition_label(recording: str) -> tuple[str, str]:
    """('Agent'|'AI'|'No AI', 'Personal'|'Speedscore') from the folder name."""
    arm = ("Agent" if "_agent_" in recording
           else "No AI" if "_none_" in recording
           else "AI")
    framing = "Speedscore" if "speedscore" in recording else "Personal"
    return arm, framing


def b64(arr: np.ndarray) -> str:
    """
    One recording's (n_windows x n_columns) matrix as base64 little-endian
    float32 -- the same four-byte floats the pipeline wrote, so nothing is
    rounded on the way to the browser.

    Little-endian is asserted rather than negotiated: every platform a browser
    ships on is little-endian, and `new Float32Array(buffer)` on the other side
    reads in platform order. Writing '<f4' explicitly means the file does not
    depend on the byte order of whichever machine built it.
    """
    return base64.b64encode(np.ascontiguousarray(arr, dtype="<f4").tobytes()).decode("ascii")


def load_index() -> dict:
    p = VARIANTS / "index.json"
    if not p.exists():
        raise SystemExit(
            f"{p} is missing. Run `python pipeline.py` first -- the sweep it "
            f"writes into outputs/variants/ is what the window-length, "
            f"estimator, reference and interpolation controls read.")
    return json.loads(p.read_text(encoding="utf-8"))


def npz_name(kind: str, epoch: str) -> str:
    return f"{'tasks' if kind == 'task' else 'baselines'}_e{float(epoch):04.1f}.npz"


def build_epoch_shards(epoch: str, vindex: dict,
                       task_keys: list[str], base_keys: list[str]) -> dict:
    """
    Every estimator's shard for one window length: {fft: shard}.

    All four estimators come out of one pass over the two .npz files. Opening
    them once per estimator instead read the 120 MB sweep four times per epoch.

    Each shard is keyed the way the page asks for it: `v`/`bv` by
    "interpolation|ocular|reference" and `p`/`bp` by ocular mode alone, because
    the artifact masks are measured once on the pre-interpolation
    hardware-referenced signal and do not move with the other toggles.

    `missing` records every array the sweep did not produce. Without it the page
    cannot tell "the pipeline never built this variant" from "this recording has
    no windows at this length", and its excluded-recordings table asserted the
    latter for both -- so a REST reference that failed on one recording, or a
    half-written archive, was reported to the reader as a consequence of the
    slider they had just moved.
    """
    shards = {f: {"v": {}, "p": {}, "bv": {}, "bp": {},
                  "s": {}, "bs": {}, "mo": {}, "bmo": {}, "missing": []}
              for f in vindex["fft_methods"]}
    rec_meta = vindex["recordings"]

    for kind, vslot, pslot, sslot, mslot, keys in (
            ("task", "v", "p", "s", "mo", task_keys),
            ("baseline", "bv", "bp", "bs", "bmo", base_keys)):
        path = VARIANTS / npz_name(kind, epoch)
        if not path.exists():
            for f in shards:
                shards[f]["missing"].append(
                    {"kind": kind, "why": f"{path.name} was not written by the pipeline"})
            continue
        with np.load(path) as z:
            for key in keys:
                pre = rec_meta[key]["array_prefix"]
                # Distinct tokens for THIS recording, in mode order, de-duped.
                # A recording whose five modes collapse to two channel lists
                # contributes two entries here; the page resolves mode -> token
                # through `interpTokens` before looking anything up.
                by_mode = rec_meta[key].get("interpolation_tokens") or {}
                tokens, seen = [], set()
                for mode in vindex["interpolation_modes"]:
                    token = by_mode.get(mode)
                    if token is None:
                        for f in vindex["fft_methods"]:
                            # `mode`, not `variant`: there IS no variant key --
                            # that is the whole problem being recorded -- and the
                            # page matches on this field when its own lookup
                            # returns null.
                            shards[f]["missing"].append(
                                {"kind": kind, "key": key, "variant": None,
                                 "mode": mode,
                                 "why": "index.json carries no interpolation token "
                                        "for this mode, so no array can be named "
                                        "for it"})
                        continue
                    if token not in seen:
                        seen.add(token)
                        tokens.append(token)
                for token in tokens:
                    for oc in vindex["ocular_modes"]:
                        for ref in vindex["references"]:
                            vk = f"{token}|{oc}|{ref}"
                            for f in vindex["fft_methods"]:
                                name = f"{pre}::v::{vk}|{f}"
                                if name in z:
                                    shards[f][vslot].setdefault(vk, {})[key] = b64(z[name])
                                else:
                                    shards[f]["missing"].append(
                                        {"kind": kind, "key": key, "variant": vk,
                                         "why": "the pipeline produced no values for "
                                                "this processing combination"})
                for oc in vindex["ocular_modes"]:
                    name = f"{pre}::p::{oc}"
                    if name in z:
                        for f in vindex["fft_methods"]:
                            shards[f][pslot].setdefault(oc, {})[key] = b64(z[name])
                    else:
                        for f in vindex["fft_methods"]:
                            shards[f]["missing"].append(
                                {"kind": kind, "key": key, "variant": f"peaks|{oc}",
                                 "why": "the pipeline produced no peak amplitudes"})
                    # Per-window sigma for the windowed robust mode, one matrix
                    # per sliding length. Keyed "<ocular>|<length>" exactly as the
                    # pipeline wrote it, so the page's lookup is the toggle value.
                    for L in vindex["robust_slide_choices_s"]:
                        sk = f"{oc}|{L:g}"
                        sname = f"{pre}::s::{sk}"
                        if sname in z:
                            for f in vindex["fft_methods"]:
                                shards[f][sslot].setdefault(sk, {})[key] = b64(z[sname])
                        else:
                            for f in vindex["fft_methods"]:
                                shards[f]["missing"].append(
                                    {"kind": kind, "key": key,
                                     "variant": f"sigma_slide|{sk}",
                                     "why": "the pipeline produced no sliding-window "
                                            "sigma for this combination"})
                # Head motion: ONE matrix per recording per epoch, with no
                # ocular, reference or estimator dimension -- the IMU is a
                # property of the recording, not of a processing choice -- so it
                # is keyed flat by recording rather than nested like the rest.
                mname = f"{pre}::m::imu"
                if mname in z:
                    for f in vindex["fft_methods"]:
                        shards[f][mslot][key] = b64(z[mname])
                else:
                    for f in vindex["fft_methods"]:
                        shards[f]["missing"].append(
                            {"kind": kind, "key": key, "variant": "motion|imu",
                             "why": "the pipeline produced no head-motion matrix "
                                    "(the recording's aux/IMU file was missing or "
                                    "unreadable)"})
    return shards


def main() -> None:
    load_doc = json.loads((OUT / "cognitive_load.json").read_text(encoding="utf-8"))
    base_doc = json.loads((OUT / "baselines.json").read_text(encoding="utf-8"))
    vindex = load_index()
    load = load_doc["tasks"]
    bases = base_doc["baselines"]
    run_meta = load_doc.get("run_meta", {})

    if run_meta.get("partial_run"):
        raise SystemExit(
            "outputs/ came from a PARTIAL pipeline run "
            f"({run_meta.get('n_processed')} of {run_meta.get('n_discovered')} "
            f"recordings, argv={run_meta.get('argv')}). Re-run `python pipeline.py` "
            "with no filter before building the dashboard.")

    vrec = vindex["recordings"]
    global PLAUSIBLE_UV
    _ab = vindex.get("amplitude_bound") or {}
    if not _ab.get("available"):
        raise SystemExit(
            "outputs/variants/index.json carries no derived amplitude band "
            f"({_ab.get('reason', 'field absent')}). It predates the 2026-09-03 "
            "change from the hardcoded 1-50 uV range. Re-run `python pipeline.py`.")
    PLAUSIBLE_UV = (float(_ab["low_uv"]), float(_ab["high_uv"]))
    # Baseline segments, read from the sweep index rather than hardcoded, so the
    # page's bounds and ordering follow pipeline.BASELINE_SEGMENTS.
    global SEGMENTS, SEG_ORDER
    SEGMENTS = {k: tuple(v) for k, v in (vindex.get("baseline_segments") or {}).items()}
    SEG_ORDER = list(vindex.get("baseline_segment_order") or [])
    if not SEGMENTS or not SEG_ORDER:
        raise SystemExit(
            "outputs/variants/index.json names no baseline segments. It predates "
            "the 2026-09-02 three-segment baseline. Re-run `python pipeline.py`.")
    # `default_check` absent means an index.json older than the check itself. The
    # whole point of the refusal below is that a dashboard is not published unless
    # its defaults are the published numbers, so an index that cannot answer the
    # question is a refusal, not a pass.
    unchecked = [k for k, v in vrec.items() if "default_verified" not in v]
    if unchecked:
        raise SystemExit(
            f"outputs/variants/index.json carries no default-cell verdict for "
            f"{len(unchecked)} recording(s) (e.g. {unchecked[0]}). It predates the "
            f"2026-09-01 verify_default_variant(). Re-run `python pipeline.py`.")
    failed_default = [k for k, v in vrec.items() if v.get("default_check")]
    if failed_default:
        raise SystemExit(
            "the sweep's default cell does not reproduce MNE's own estimator on "
            f"{len(failed_default)} recording(s): {', '.join(failed_default)}. "
            "outputs/variants/index.json has the per-column deviations. Refusing to "
            "build a dashboard whose defaults are not the published numbers.")
    # A recording that reported no PROBLEMS but was never actually compared is not
    # a pass. The check used to return an empty list in five distinct
    # nothing-to-compare situations, each of which was written out as success.
    unverified = [k for k, v in vrec.items() if not v.get("default_verified")]
    if unverified:
        # Entry keys are "<recording>::<segment>" for baselines, so name the
        # segments explicitly: since 2026-09-02 the commonest way to land here is
        # one short baseline segment, and "re-run the pipeline" is useless advice
        # for a segment that is 15 s long by design. Say which, so the fix
        # (shorten the segment, or drop it) is visible from the message.
        segs = sorted({k.split("::", 1)[1] for k in unverified if "::" in k})
        hint = (f" All of them are baseline segment(s) {', '.join(segs)} -- check "
                f"whether that segment is long enough to yield windows at every "
                f"epoch length in EPOCH_CHOICES_S."
                if segs and all("::" in k for k in unverified) else
                " Re-run `python pipeline.py`.")
        raise SystemExit(
            f"the default cell could not be VERIFIED on {len(unverified)} "
            f"swept unit(s): {', '.join(unverified[:5])}"
            f"{' ...' if len(unverified) > 5 else ''}. That is not the same as "
            f"passing -- nothing was compared. See default_check in "
            f"outputs/variants/index.json.{hint}")
    # The windowed robust mode thresholds against a per-WINDOW sigma, which is the
    # MAD of the raw samples -- the one browser-side threshold the page cannot
    # derive from what it already has. An index without it would build a dashboard
    # offering the mode and then refusing every mask, which reads as a broken
    # control rather than as stale data.
    if not vindex.get("robust_slide_choices_s"):
        raise SystemExit(
            "outputs/variants/index.json names no sliding-window lengths for the "
            "windowed robust mode. It predates the 2026-09-02 change from tiled "
            "60 s blocks to a centred sliding window. Re-run `python pipeline.py`."
        )
    # The ocular axis grew a third mode on 2026-09-05. An archive swept with
    # two builds perfectly happily -- every other guard passes, the control
    # renders two buttons, and the page then states 720 cells in its own prose
    # while computing 480 from the archive it is holding. Ordered equality, not
    # length or membership: position matters, because the first entry is taken
    # as the uncorrected branch (see pipeline's _baseline_segment).
    _oc = list(vindex.get("ocular_modes") or [])
    if _oc != list(EXPECTED_OCULAR_MODES):
        raise SystemExit(
            f"outputs/variants/index.json was swept with ocular modes {_oc}, but "
            f"this page offers {list(EXPECTED_OCULAR_MODES)}. It predates the "
            f"2026-09-05 addition of the `ica` mode. Re-run `python pipeline.py`.")

    # The same guard for the interpolation control, and for the same reason: an
    # archive swept with the old ["on", "off"] would build a page offering a
    # manual mode and a frontal-pair checkbox that address arrays nobody wrote.
    # Every panel would fall through to "no data in this combination", which
    # reads like a data problem rather than a stale archive.
    _int = list(vindex.get("interpolation_modes") or [])
    if _int != list(EXPECTED_INTERPOLATION_MODES):
        raise SystemExit(
            f"outputs/variants/index.json was swept with interpolation modes "
            f"{_int}, but this page offers {list(EXPECTED_INTERPOLATION_MODES)}. "
            f"It predates the 2026-09-08 addition of the manual list and the "
            f"frontal-pair option. Re-run `python pipeline.py`.")

    # And the map every value lookup goes through. Without it the page cannot
    # turn a control position into an array name at all, so it would render
    # empty rather than wrong -- but it would render, and the reader would be
    # told the sweep was incomplete.
    _no_tokens = sorted(k for k, v in vrec.items()
                        if not (v or {}).get("interpolation_tokens"))
    if _no_tokens:
        raise SystemExit(
            f"{len(_no_tokens)} archive entries carry no `interpolation_tokens` "
            f"(e.g. {_no_tokens[:3]}). The sweep is keyed by which channels were "
            f"interpolated, not by mode name, so without that map no value can "
            f"be addressed. Re-run `python pipeline.py`.")

    # `interpolation_lists` needs the SAME guard. Tokens decide which numbers are
    # drawn; lists decide every sentence the page writes about what is synthetic.
    # With the lists missing and the tokens present the page plots the correct
    # interpolated series while the badge reads "F1, F2 failed detection and is
    # being read as recorded" -- the exact inverse of the truth, and no louder
    # than any other badge.
    _bad_lists = sorted(
        k for k, v in vrec.items()
        if sorted(((v or {}).get("interpolation_lists") or {}).keys())
        != sorted(EXPECTED_INTERPOLATION_MODES))
    if _bad_lists:
        raise SystemExit(
            f"{len(_bad_lists)} archive entries do not carry an "
            f"`interpolation_lists` entry for every mode in "
            f"{EXPECTED_INTERPOLATION_MODES} (e.g. {_bad_lists[:3]}). The page "
            f"describes what each mode replaced from that map. "
            f"Re-run `python pipeline.py`.")

    # Windows spanning a discontinuity became structurally excluded on
    # 2026-09-04. An archive without the field carries no way to know which
    # windows those are, so the page would silently plot spectra taken across a
    # 449 s splice under every artifact mode -- including the one whose label
    # promises no rejection, where a reader is least likely to suspect it.
    if not all("excluded_windows" in (v or {}) for v in vrec.values()):
        raise SystemExit(
            "outputs/variants/index.json names no excluded_windows. It predates "
            "the 2026-09-04 change that drops windows whose samples are not "
            "contiguous in real time. Re-run `python pipeline.py`.")

    # Bad-channel detection became a per-PAIR decision on 2026-09-04. An archive
    # written before that carries a task and its baseline interpolated on
    # independently-decided channel lists, which is the defect the change fixed:
    # a log ratio whose numerator or denominator is a spline on one side and raw
    # signal on the other. The values are not comparable across the two rules, so
    # this is a refusal and not a warning. run_meta travels inside index.json for
    # exactly this kind of question.
    _det = ((vindex.get("run_meta") or {}).get("parameters") or {}).get(
        "bad_channel_detection")
    if _det != "paired":
        raise SystemExit(
            f"outputs/variants/index.json was written with bad-channel detection "
            f"= {_det!r}, not 'paired'. It predates the 2026-09-04 change that "
            f"decides one bad-channel list per task/baseline pair and "
            f"interpolates the same channels in both. Re-run `python "
            f"pipeline.py`.")

    # A reference that failed leaves whole variants absent from the archive; the
    # page reports them as missing, but the build should say so out loud.
    ref_failed = {k: v["references_failed"] for k, v in vrec.items()
                  if v.get("references_failed")}

    recordings, excluded_upstream = [], []

    for key, r in load.items():
        if r["status"] != "ok":
            # `exclusion_reason` since 2026-09-08. The column used to print the
            # status TOKEN, so a reader was told a recording was
            # "excluded_by_user_decision" and nothing whatever about why.
            excluded_upstream.append(dict(
                key=key, participant=r["participant"], recording=r["recording"],
                reason=r.get("exclusion_reason") or r["status"],
                status=r["status"], kind="upstream"))
            continue
        if key not in vrec:
            excluded_upstream.append(dict(
                key=key, participant=r["participant"], recording=r["recording"],
                reason="no swept variants were produced for this recording",
                kind="upstream"))
            continue

        arm, framing = condition_label(r["recording"])
        seg = r["qc_segment"]
        per_ch = r["qc_bandpass"]["per_channel"]
        bad = r["qc_interp"]["none"]["bad_channels_detected"]
        vi = vrec[key]

        # Robust SD of each channel AS RECORDED: after filtering, before
        # interpolation, in the hardware reference. Deliberately not the
        # amplitude of whatever is currently plotted -- the page shows both, one
        # labelled "as recorded" and one "as plotted", because re-referencing can
        # smear a broken electrode across every channel. Fed the amplitude gate
        # until 2026-09-02; a badge input only now.
        # Through sig() like ampDisp and times: a NaN here would reach the page
        # as a bare NaN literal and leave the export as a silent null.
        amp = {c: sig(per_ch[c]["robust_sd_uv"]) for c in CH5}

        entry = dict(
            key=key, participant=r["participant"], recording=r["recording"],
            arm=arm, framing=framing,
            duration_s=seg["segment_duration_s"],
            elapsed_s=seg.get("segment_elapsed_s"),
            n_gaps=seg.get("n_internal_gaps", 0),
            max_gap_s=seg.get("max_internal_gap_s", 0.0),
            repair=seg.get("repair"),
            notes=seg.get("notes", []),
            amp=amp,
            bad=bad,
            # The DETECTED list. Empty means the AUTOMATIC position replaces
            # nothing here, which is true of 12 of the 40 swept recordings -- but
            # since 2026-09-08 that no longer means the control is inert, because
            # the manual position can replace channels detection never flagged.
            # `interpLists` below is what actually answers "does moving this
            # control change anything", and only 3 of the 20 groups are inert
            # under every mode.
            interpChans=vi.get("bad_channels", bad),
            # Which channels each of the five interpolation modes replaces here,
            # and which stored array each mode resolves to. `interpChans` above
            # is the DETECTED list and stays what it was -- it is what the
            # bad-channel badge means -- while these say what the control the
            # reader is holding actually does. They differ: under `manual` a
            # recording can interpolate a channel detection never flagged, and
            # under `manual` most recordings interpolate nothing at all.
            interpLists=vi.get("interpolation_lists") or {},
            interpTokens=vi.get("interpolation_tokens") or {},
            # Per-mode ocular QC, shipped to the page since 2026-09-05.
            # cognitive_load.json has carried it all along and nothing read it,
            # which stopped mattering the day EOG regression became the OPENING
            # default: the regression is numerically degenerate on 9 of the 22
            # task recordings, and the page said nothing about it in the one
            # mode a reader now arrives in. `ica` carries what it removed.
            # The baseline's own ocular QC, carried on the TASK. A correction
            # that injects artifact into the baseline corrupts the denominator of
            # every value on this panel, and the reader is looking at the task.
            baseOcQc={m: {k: v for k, v in (d or {}).items()
                          if k in ("degenerate", "max_abs_coefficient",
                                   "n_components_removed", "variance_removed_pct",
                                   "max_abs_eog_correlation", "failed",
                                   "converged", "paired",
                                   "sd_reduction_by_channel_pct")}
                       for m, d in ((bases.get(r["baseline_key"]) or {})
                                    .get("qc_ocular") or {}).items()},
            ocQc={m: {k: v for k, v in (d or {}).items()
                      if k in ("degenerate", "max_abs_coefficient",
                               "n_components_removed", "variance_removed_pct",
                               "max_abs_eog_correlation", "failed", "converged",
                               "paired", "sd_reduction_by_channel_pct")}
                  for m, d in (r.get("qc_ocular") or {}).items()},
            motionOk=bool(vi.get("motion_available")),
            # Why the sensor is unusable, straight from the sweep index since
            # 2026-09-04. It came out of qc_steps.json for one build before that,
            # which meant the page depended on a second file from the same run and
            # needed a generated_utc guard to be sure they matched. One file now.
            motionWhy=vi.get("motion_unavailable_reason"),
            # Windows this recording can never plot, per epoch length: their
            # samples are not contiguous in real time. Not a criterion -- see
            # `excluded` in buildFrame.
            excl=vi.get("excluded_windows") or {},
            sigma={oc: {c: sig(v) for c, v in d.items()}
                   for oc, d in vi["sigma"].items()},
            ampDisp={vk: {c: sig(v) for c, v in d.items()}
                     for vk, d in vi["amplitude_displayed"].items()},
            times={e: [sig(t, 6) for t in (ts or [])]
                   for e, ts in (vi["times_s"] or {}).items()},
            base=None,
        )

        # ---- the paired baseline, one entry per segment -----------------------
        # Each segment is its own archive entry keyed "<baseline key>::<segment>",
        # with its own peaks, sigma, window grid and amplitudes, because the
        # pipeline crops and calibrates each one independently. The page therefore
        # switches segment by switching which key it reads, and nothing else about
        # the baseline machinery has to know that segments exist.
        bkey = r["baseline_key"]
        brec = bases.get(bkey)
        bsegs = {s: vrec.get(f"{bkey}::{s}") for s in SEG_ORDER}
        have = [s for s, v in bsegs.items() if v]
        if brec and brec.get("status") == "ok" and brec.get("baseline") and have:
            b = brec["baseline"]
            bseg_doc = b.get("segments") or {}

            def _seg_payload(s):
                bvi, sd = bsegs[s], (bseg_doc.get(s) or {})
                amp_doc = sd.get("amplitude") or {}
                return dict(
                    key=f"{bkey}::{s}",
                    bounds=list(SEGMENTS[s]),
                    # Per-segment amplitude: the eyes-closed slice can be a very
                    # different electrode picture from the rest phase, so a single
                    # block-level figure would misdescribe two of the three.
                    amp={c: sig(v.get("robust_sd_uv")) for c, v in amp_doc.items()},
                    implausible=sd.get("implausible_channels") or [],
                    nWindows=sd.get("n_windows"),
                    sigma={oc: {c: sig(v) for c, v in d.items()}
                           for oc, d in bvi["sigma"].items()},
                    # The windowed mode's per-window sigma is loaded from the
                    # shards by segment-qualified key, like the peaks it is
                    # indexed alongside; nothing about it is inline here.
                    idx=bvi["window_index"] or {},
                    excl=bvi.get("excluded_windows") or {},
                )

            _btok = [(s, tuple(sorted(((bsegs[s] or {})
                                       .get("interpolation_tokens") or {}).items())))
                     for s in have]
            if len({t for _s, t in _btok}) > 1:
                raise SystemExit(
                    f"{bkey}: the three baseline segments were swept with "
                    f"DIFFERENT interpolation lists {_btok}. They are decided "
                    f"once per group in step10_baseline, so this means the "
                    f"archive is inconsistent; re-run `python pipeline.py`.")
            _bvi0 = bsegs[have[0]] or {}
            # And the TASK must agree with its baseline. The segments agreeing
            # with each other is the smaller half of the property: valsOf reads
            # the task's token and bValsOf the baseline's, so a per-mode
            # disagreement puts the numerator and denominator of every ratio on
            # different channel lists, with nothing on screen to show it. This is
            # the last place both maps are in one scope.
            _ttok = vi.get("interpolation_tokens") or {}
            _btok0 = _bvi0.get("interpolation_tokens") or {}
            if _ttok != _btok0:
                raise SystemExit(
                    f"{key} and its baseline {bkey} resolve interpolation modes "
                    f"to DIFFERENT stored arrays:\n  task     {_ttok}\n  "
                    f"baseline {_btok0}\nThe two sides of every ratio would be "
                    f"interpolated differently. They are decided once per group "
                    f"in the pipeline, so this means the archive is inconsistent; "
                    f"re-run `python pipeline.py`.")
            entry["base"] = dict(
                key=bkey,
                interpolated=b["interpolated"],
                interpLists=_bvi0.get("interpolation_lists") or {},
                interpTokens=_bvi0.get("interpolation_tokens") or {},
                block_s=b["block_duration_s"],
                repair=(brec.get("qc_segment") or {}).get("repair"),
                segs={s: _seg_payload(s) for s in have},
            )
        else:
            reason = ("no baseline recording found" if not brec else
                      "no swept variants were produced for the baseline" if not have else
                      f"baseline unusable: "
                      f"{(brec.get('qc_segment') or {}).get('notes') or brec.get('status')}")
            entry["base"] = dict(key=bkey, missing=True, reason=reason)

        recordings.append(entry)

    recordings.sort(key=lambda e: (e["participant"], e["recording"]))
    # Excluded BASELINES, from baselines.json. The excluded table was built from
    # tasks alone, so a reader saw p02/task_agent_personal disappear and was told
    # nothing about p02/baseline_agent_personal going with it -- while the table's
    # own hint promises that what dropped out and why "is part of the result, so
    # nothing is omitted silently".
    for bkey, brec in sorted(bases.items()):
        if brec.get("status") == "ok":
            continue
        excluded_upstream.append(dict(
            key=bkey, participant=brec.get("participant", bkey.split("/")[0]),
            recording=brec.get("recording", bkey.split("/")[-1]),
            reason=brec.get("exclusion_reason") or brec.get("status"),
            status=brec.get("status"), kind="upstream"))

    excluded_upstream.sort(key=lambda e: (e["participant"], e["recording"]))

    task_keys = [e["key"] for e in recordings]
    # Segment-qualified: the shard builder loads arrays per archive entry, and
    # since 2026-09-02 a baseline is one entry PER SEGMENT.
    base_keys = sorted({sp["key"] for e in recordings
                        if not e["base"].get("missing")
                        for sp in e["base"]["segs"].values()})

    default_epoch = f"{float(vindex['default']['epoch_s']):g}"

    # Frequency-bin counts per (window length, estimator). They are a function of
    # the sampling rate alone, so every recording must report the same table; if
    # one does not, something upstream resampled and the page would quote bin
    # counts that are wrong for most of what it shows.
    band_bins, bins_src = {}, None
    for k, v in vrec.items():
        if not v.get("band_bins"):
            continue
        if bins_src is None:
            band_bins, bins_src = v["band_bins"], k
        elif v["band_bins"] != band_bins:
            raise SystemExit(
                f"frequency-bin counts differ between recordings ({bins_src} vs "
                f"{k}). They depend only on the sampling rate, so this means the "
                f"sweep was not run at one rate. Refusing to build.")

    payload = dict(
        recordings=recordings,
        excluded_upstream=excluded_upstream,
        plausible_uv=list(PLAUSIBLE_UV),
        # The band's provenance travels with it, so the page can say what the
        # badge is measured against instead of asserting a standard.
        amplitude_bound=_ab,
        n_participants=len({r["participant"] for r in recordings}),
        value_keys=vindex["value_keys"],
        peak_channels=vindex["peak_channels"],
        sliders=vindex["sliders"],
        epochs_s=[int(e) for e in vindex["epochs_s"]],
        fft_methods=vindex["fft_methods"],
        references=vindex["references"],
        default_epoch=int(float(default_epoch)),
        # Cells a reader can SELECT. Fewer are stored, because two interpolation
        # modes that replace the same channels are one array -- `n_stored_cells`
        # counts what is actually in the shards.
        n_precomputed_cells=(len(vindex["epochs_s"]) * len(vindex["fft_methods"])
                             * len(vindex["references"])
                             * len(vindex["interpolation_modes"])
                             * len(EXPECTED_OCULAR_MODES)),
        # Every value array actually written, summed over the archive: each
        # entry contributes its OWN number of distinct interpolation lists, which
        # is between 1 and 5.
        n_stored_arrays=(len(vindex["epochs_s"]) * len(vindex["fft_methods"])
                         * len(vindex["references"]) * len(EXPECTED_OCULAR_MODES)
                         * sum(len(set((v or {}).get("interpolation_tokens", {})
                                       .values()))
                               for v in vrec.values())),
        # The same count without the dedup, so the page can state the saving
        # rather than carry a hardcoded ratio that goes stale on the next run.
        n_dense_arrays=(len(vindex["epochs_s"]) * len(vindex["fft_methods"])
                        * len(vindex["references"]) * len(EXPECTED_OCULAR_MODES)
                        * len(EXPECTED_INTERPOLATION_MODES) * len(vrec)),
        interpolation_modes=list(vindex["interpolation_modes"]),
        interpolation_sources=list(vindex.get("interpolation_sources")
                                   or ["off", "on", "manual"]),
        frontal_pair=list(vindex.get("interpolation_frontal_pair") or ["F1", "F2"]),
        manual_interpolation=vindex.get("manual_interpolation") or {},
        # mode -> stored-array token, for every entry in the archive. See
        # interp_token() in pipeline.py for why the archive is keyed by the
        # channel list rather than by the mode name.
        interp_tokens={k: (v or {}).get("interpolation_tokens") or {}
                       for k, v in vrec.items()},
        shard_dir=SHARD_DIR_NAME,
        band_bins=band_bins,
        # Time-half-bandwidth product of the DPSS taper set. The page needs it
        # because the multitaper smoothing half-width in Hz is NW divided by the
        # window length in seconds -- it is NOT the fixed +-1 Hz that held while
        # the window was pinned at 4 s. Read from the sweep index when the
        # pipeline recorded it; the fallback is pipeline.MULTITAPER_NW and must
        # be kept in step with it.
        multitaper_nw=float(vindex.get("multitaper_nw", 4.0)),
        # Sliding-window lengths the per-window sigmas were computed at. Read from
        # the sweep index rather than assumed: the page can only offer lengths the
        # sweep actually stored, and its wording follows whichever is chosen.
        # Head motion (2026-09-03). The page thresholds the shipped per-window
        # motion itself, so `motion_k` is a continuous slider rather than a set of
        # stored positions -- unlike the sliding sigma, the motion vector is small
        # enough to send whole.
        ocular_modes=list(vindex["ocular_modes"]),
        motion_sources=list(vindex.get("motion_sources") or []),
        default_motion_source=vindex.get("default_motion_source"),
        motion_columns=list(vindex.get("motion_columns") or []),
        motion_min_samples=vindex.get("motion_min_samples"),
        motion_k=vindex.get("motion_k"),
        robust_slide_choices_s=[float(x) for x in vindex["robust_slide_choices_s"]],
        robust_slide_default_s=float(vindex.get("robust_slide_default_s",
                                                vindex["robust_slide_choices_s"][-1])),
        # Baseline segments the reader can subtract against, their bounds in the
        # 6-minute block, and which one the page opens on.
        baseline_segments={k: list(v) for k, v in SEGMENTS.items()},
        baseline_segment_order=SEG_ORDER,
        default_baseline_segment=vindex.get("default_baseline_segment",
                                            SEG_ORDER[0]),
        # Carried for the JSON export, so a downloaded file names the pipeline
        # run it came from and the discontinuity rule that shaped its windows.
        # An export that cannot say which run produced it is not reproducible.
        # From the sweep index, not cognitive_load.json: every other field the
        # export cites for provenance -- excluded_windows, times_s,
        # window_max_gap_s -- comes from the index, and naming the run from a
        # second file reintroduces exactly the two-file coupling the motion
        # reason was moved out of.
        run_meta=dict(
            generated_utc=(vindex.get("run_meta") or {}).get("generated_utc")
                          or run_meta.get("generated_utc"),
            bad_channel_detection=_det,
        ),
        window_max_gap_s=vindex.get("window_max_gap_s"),
        inline={},
    )

    # ---- the default window length goes inside the file ---------------------
    missing_total = []
    for fft, shard in build_epoch_shards(default_epoch, vindex,
                                         task_keys, base_keys).items():
        missing_total += shard["missing"]
        payload["inline"][f"e{int(float(default_epoch)):02d}_{fft}"] = shard

    html = TEMPLATE.replace("__DATA__", json.dumps(payload, separators=(",", ":")))

    # ---- every other window length goes beside it ---------------------------
    # Built to a temporary directory and swapped in, so a failure part-way
    # through the 36-file write cannot leave dashboard.html pointing at a folder
    # that is missing shards the previous build had.
    shard_dir = ROOT / SHARD_DIR_NAME
    staging = ROOT / (SHARD_DIR_NAME + ".tmp")
    _remove_with_retry(staging)
    staging.mkdir(parents=True, exist_ok=True)
    shard_bytes, n_shards = 0, 0
    for epoch in vindex["epochs_s"]:
        ek = f"{float(epoch):g}"
        if ek == default_epoch:
            continue
        for fft, data in build_epoch_shards(ek, vindex, task_keys, base_keys).items():
            missing_total += data["missing"]
            js = (f"__eegShard({int(float(ek))},{json.dumps(fft)},"
                  f"{json.dumps(data, separators=(',', ':'))});\n")
            path = staging / f"e{int(float(ek)):02d}_{fft}.js"
            path.write_text(js, encoding="utf-8")
            shard_bytes += path.stat().st_size
            n_shards += 1

    # Only now is the previous build replaced, and only now is the HTML that
    # points at it written.
    swap_warnings = swap_directory_contents(staging, shard_dir)
    (ROOT / "dashboard.html").write_text(html, encoding="utf-8")

    html_bytes = (ROOT / "dashboard.html").stat().st_size

    def _bseg(r, segment):
        """This recording's payload for one baseline segment, or None."""
        b = r["base"]
        return None if b.get("missing") else (b.get("segs") or {}).get(segment)

    default_seg = payload["default_baseline_segment"]
    n_base = sum(1 for r in recordings if _bseg(r, default_seg))
    n_flagged = sum(1 for r in recordings
                    for sp in [_bseg(r, default_seg)] if sp
                    and (sp["implausible"] or r["base"]["interpolated"]))

    def _ok(amp, bad, chans):
        return all(amp.get(c) is not None and PLAUSIBLE_UV[0] <= amp[c] <= PLAUSIBLE_UV[1]
                   and c not in bad for c in chans)

    def _usable(chans, segment):
        """
        Recordings where BOTH sides of the subtraction are plausible and measured,
        for one baseline segment. Evaluated per segment because a segment carries
        its own amplitudes -- the eyes-closed slice can be a different electrode
        picture from the rest phase, so the defensible n is not the same number.
        """
        out = []
        for r in recordings:
            sp = _bseg(r, segment)
            if sp and _ok(r["amp"], r["bad"], chans) \
                   and _ok(sp["amp"], r["base"]["interpolated"], chans):
                out.append(r)
        return out

    total_mb = (html_bytes + shard_bytes) / 1024 / 1024
    print(f"Wrote {ROOT/'dashboard.html'} ({html_bytes/1024/1024:.1f} MB)")
    print(f"Wrote {shard_dir} ({shard_bytes/1024/1024:.0f} MB, {n_shards} files)")
    print(f"  TOTAL DELIVERABLE {total_mb:.0f} MB. dashboard.html alone is "
          f"self-contained at the {payload['default_epoch']} s default; the folder "
          f"is only needed to move the window-length slider.")
    if total_mb > 100:
        print(f"  NOTE: {total_mb:.0f} MB will not go through email and will sync to "
              f"any cloud-backed folder. Reduce EPOCH_CHOICES_S or FFT_METHODS in "
              f"pipeline.py if that matters more than the sweep does.")
    for w in swap_warnings:
        print(f"  NOTE: {w} (the build itself is complete)")
    if ref_failed:
        print(f"  WARNING: {len(ref_failed)} recording(s) where a reference could "
              f"not be applied by the pipeline; those variants are absent:")
        for k, v in list(ref_failed.items())[:5]:
            print(f"    - {k}: {'; '.join(v)}")
    if missing_total:
        kinds = sorted({m["why"] for m in missing_total})
        print(f"  WARNING: {len(missing_total)} array(s) the sweep did not produce; "
              f"the dashboard reports them as missing rather than as empty:")
        for w in kinds:
            print(f"    - {w} ({sum(1 for m in missing_total if m['why'] == w)})")
    print(f"  {len(recordings)} task recordings, "
          f"{payload['n_precomputed_cells']} precomputed processing combinations")
    print(f"  {n_base}/{len(recordings)} have a usable {default_seg} baseline "
          f"({n_flagged} flagged: outside the derived band, or interpolated)")
    for s in SEG_ORDER:
        if s == default_seg:
            continue
        have = sum(1 for r in recordings if _bseg(r, s))
        print(f"  {have}/{len(recordings)} have a usable {s} baseline "
              f"({SEGMENTS[s][0]:.0f}-{SEGMENTS[s][1]:.0f} s of the block)")
    print(f"  {len(excluded_upstream)} excluded upstream")
    print(f"  the sweep's default cell reproduces MNE's own estimator on all "
          f"{len(vrec)} swept recordings, all {len(vindex['value_keys'])} columns")

    print("\n  EFFECTIVE n for baseline-corrected work at the DEFAULT settings --")
    print("  recordings where BOTH the task and its baseline are plausible AND")
    print("  measured on that measure's own channels. This, not 22, governs a")
    print("  group analysis. NOTE the band is DERIVED FROM THIS RUN, so this count")
    print("  is relative, not absolute: adding noisier recordings widens the band")
    print("  and would RAISE it. It is the count you can defend against this")
    print("  dataset's own spread, not against an external standard. One column")
    print("  per baseline segment, because each carries its own amplitudes:")
    print(f"    {'':<26} " + " ".join(f"{s:>12}" for s in SEG_ORDER))
    measures = (("Frontal midline theta", ["F1", "F2"]),
                ("Parietal alpha", ["Pz"]),
                ("Parietal beta asymmetry", ["P3", "P4"]),
                ("Frontal alpha asymmetry", ["F1", "F2"]),
                ("Cognitive load index", ["F1", "F2", "Pz"]),
                ("ALL five channels", CH5))
    for label, chans in measures:
        cells = []
        for s in SEG_ORDER:
            u = _usable(chans, s)
            cells.append(f"{len(u):>2} rec/{len({r['participant'] for r in u}):>2}p")
        print(f"    {label:<26} " + " ".join(f"{c:>12}" for c in cells))
    ps_def = sorted({r["participant"] for r in _usable(CH5, default_seg)})
    print(f"    (participants clearing all five channels on the {default_seg} "
          f"baseline: {', '.join(ps_def) if ps_def else 'none'})")


TEMPLATE = r"""<meta charset="utf-8">
<!-- Without this a phone lays the page out at its default 980 px and then zooms
     out to fit, so every control is rendered small rather than narrow and the
     control banks really do swallow the screen. With it the layout gets the
     real device width and the flex rows, auto-fill grids and text columns
     below can reflow to it. -->
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Galea EEG Measures</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap">
<style>
:root{
  color-scheme:light;
  --page:#f4f6f7; --surface:#fbfcfc; --surface-2:#eef1f3;
  --ink:#0e1417; --ink-2:#4c565c; --ink-muted:#7d888e;
  --rule:#dde3e6; --rule-strong:#c3ccd1;
  --series:#2a78d6; --series-soft:rgba(42,120,214,.13);
  --good:#0ca30c; --warning:#fab219; --critical:#d03b3b;
  --good-bg:rgba(12,163,12,.10); --warning-bg:rgba(250,178,25,.16); --critical-bg:rgba(208,59,59,.11);
  --warning-ink:#8a5c00;
  --shadow:0 1px 2px rgba(14,20,23,.05),0 1px 8px rgba(14,20,23,.04);
  --sans:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,"SFMono-Regular",Menlo,monospace;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    color-scheme:dark;
    --page:#101315; --surface:#191d20; --surface-2:#21262a;
    --ink:#f2f5f6; --ink-2:#b8c1c6; --ink-muted:#7d888e;
    --rule:#262c30; --rule-strong:#394247;
    --series:#3987e5; --series-soft:rgba(57,135,229,.18);
    --good-bg:rgba(12,163,12,.16); --warning-bg:rgba(250,178,25,.14); --critical-bg:rgba(208,59,59,.16);
    --warning-ink:#fab219;
    --shadow:0 1px 2px rgba(0,0,0,.4),0 1px 10px rgba(0,0,0,.25);
  }
}
:root[data-theme="dark"]{
  color-scheme:dark;
  --page:#101315; --surface:#191d20; --surface-2:#21262a;
  --ink:#f2f5f6; --ink-2:#b8c1c6; --ink-muted:#7d888e;
  --rule:#262c30; --rule-strong:#394247;
  --series:#3987e5; --series-soft:rgba(57,135,229,.18);
  --good-bg:rgba(12,163,12,.16); --warning-bg:rgba(250,178,25,.14); --critical-bg:rgba(208,59,59,.16);
  --warning-ink:#fab219;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 1px 10px rgba(0,0,0,.25);
}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font-family:var(--sans);
  font-size:15px;line-height:1.55;-webkit-font-smoothing:antialiased}
.wrap{max-width:1320px;margin:0 auto;padding:0 24px 72px}
h1,h2,h3{text-wrap:balance;margin:0}

header.mast{padding:36px 0 0}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.12em;text-transform:uppercase;
  color:var(--ink-muted);margin-bottom:10px}
h1{font-size:31px;font-weight:600;letter-spacing:-.02em;line-height:1.15}
.formula{font-family:var(--mono);font-size:14px;color:var(--ink-2);margin-top:12px;
  display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.frac{display:inline-flex;flex-direction:column;text-align:center;line-height:1.25}
.frac .num{border-bottom:1.5px solid var(--rule-strong);padding:0 8px 1px}
.frac .den{padding:1px 8px 0}
.lede{color:var(--ink-2);max-width:74ch;margin-top:14px;font-size:14.5px}
.lede strong{color:var(--ink);font-weight:600}
code{font-family:var(--mono);font-size:12.5px;background:var(--surface-2);
  padding:1px 5px;border-radius:3px}

/* ---- tab bar ---- */
.tabrow{position:relative;display:flex;align-items:flex-end;gap:16px;
  margin-top:26px;border-bottom:1px solid var(--rule)}
.tabrow .tabs{flex:1 1 auto;min-width:0;margin-top:0;border-bottom:0}
.export{flex:none;font-family:var(--mono);font-size:12px;color:var(--ink-2);
  background:var(--surface);border:1px solid var(--rule);border-radius:3px;
  padding:5px 12px;margin-bottom:7px;cursor:pointer;white-space:nowrap;
  transition:color .12s,border-color .12s}
.export:hover{color:var(--series);border-color:var(--series)}
.export-status{position:absolute;right:0;top:100%;margin-top:4px;
  font-family:var(--mono);font-size:11.5px;color:var(--ink-muted);
  max-width:48ch;line-height:1.35;text-align:right;pointer-events:none}
.export:focus-visible{outline:2px solid var(--series);outline-offset:2px}
.tabs{display:flex;gap:2px;margin-top:26px;border-bottom:1px solid var(--rule);
  overflow-x:auto;scrollbar-width:thin}
.tabs button{font-family:var(--sans);font-size:13.5px;font-weight:500;color:var(--ink-2);
  background:none;border:0;border-bottom:2px solid transparent;padding:10px 15px;
  cursor:pointer;white-space:nowrap;transition:color .12s,border-color .12s}
.tabs button:hover{color:var(--ink)}
.tabs button[aria-selected="true"]{color:var(--ink);font-weight:600;
  border-bottom-color:var(--series)}
.tabs button:focus-visible{outline:2px solid var(--series);outline-offset:-2px}

.controls{position:sticky;top:0;z-index:30;background:var(--page);
  border-bottom:1px solid var(--rule);padding:14px 0;margin-bottom:26px}
.controls-inner{display:flex;gap:24px;flex-wrap:wrap;align-items:flex-end}
.ctl{display:flex;flex-direction:column;gap:6px}
.ctl-label{font-family:var(--mono);font-size:10.5px;letter-spacing:.1em;
  text-transform:uppercase;color:var(--ink-muted)}
/* `flex-wrap:wrap` is load-bearing, not cosmetic. The buttons are
   `white-space:nowrap`, so without it the widest group -- Artifact rejection,
   seven buttons, 712 px -- cannot shrink below its content and forces the whole
   page wider than a phone screen. The page has a `width=device-width` viewport,
   so that overflow becomes a sideways pan rather than a zoomed-out page, and
   `.controls` is sticky vertically only, so the control bar pans away with it. */
.seg{display:flex;flex-wrap:wrap;background:var(--surface-2);
  border:1px solid var(--rule);border-radius:7px;padding:2px;gap:2px}
.seg button{font-family:var(--sans);font-size:13px;font-weight:500;color:var(--ink-2);
  background:none;border:0;padding:6px 13px;border-radius:5px;cursor:pointer;
  white-space:nowrap;transition:background .12s,color .12s}
.seg button:hover:not(:disabled){color:var(--ink)}
.seg button[aria-pressed="true"],.seg button[aria-checked="true"]{background:var(--surface);color:var(--ink);
  box-shadow:var(--shadow);font-weight:600}
.seg button:disabled{opacity:.38;cursor:not-allowed}
.seg button:focus-visible{outline:2px solid var(--series);outline-offset:1px}
.ctl-note{font-size:11px;color:var(--ink-muted);max-width:23ch;line-height:1.35}
/* The frontal-pair checkbox. Sized to sit level with the segmented controls
   beside it rather than to match a form field, and capped in width so the
   explanation under it does not stretch the control row. */
.chk{display:flex;align-items:flex-start;gap:8px;cursor:pointer;
  font-size:12.5px;line-height:1.35;color:var(--ink-2);max-width:30ch;
  background:var(--surface-2);border:1px solid var(--rule);border-radius:7px;
  padding:8px 10px}
.chk:hover{color:var(--ink)}
.chk input{margin:1px 0 0 0;accent-color:var(--series);cursor:pointer;flex:0 0 auto}
.chk input:focus-visible{outline:2px solid var(--series);outline-offset:2px}
/* A hover/focus tooltip, not a block. Absolutely positioned so revealing it
   never reflows the control row, and above the panels so it is readable over
   them. */
.ctl-tip{position:relative}
.chk-q{display:inline-flex;align-items:center;justify-content:center;
  width:14px;height:14px;flex:0 0 auto;margin-left:2px;border-radius:50%;
  border:1px solid var(--rule);color:var(--ink-muted);
  font-size:10px;font-weight:600;line-height:1}
.chk:hover .chk-q{border-color:var(--series);color:var(--series)}
/* NO opacity transition. The tooltip starts visibility:hidden, and a
   transition queued on an element that was hidden does not run -- measured:
   visibility flipped to visible while computed opacity stayed 0, so the tooltip
   was "shown" and invisible. `display` is the honest toggle here. */
.chk-help{position:absolute;top:calc(100% + 6px);left:0;z-index:60;
  width:max-content;max-width:min(42ch, 70vw);padding:10px 12px;
  background:var(--surface);border:1px solid var(--rule);border-radius:7px;
  box-shadow:0 6px 24px -8px rgba(0,0,0,.45);
  font-size:11.5px;color:var(--ink-2);line-height:1.45;
  display:none}
.ctl-tip:hover .chk-help,
.ctl-tip:focus-within .chk-help{display:block}

.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));gap:1px;
  background:var(--rule);border:1px solid var(--rule);border-radius:10px;
  overflow:hidden;margin-bottom:14px}
.tile{background:var(--surface);padding:15px 18px}
.tile .k{font-family:var(--mono);font-size:10.5px;letter-spacing:.09em;
  text-transform:uppercase;color:var(--ink-muted)}
.tile .v{font-family:var(--mono);font-size:25px;font-weight:500;letter-spacing:-.02em;
  margin-top:5px;font-variant-numeric:tabular-nums}
.tile .s{font-size:12.5px;color:var(--ink-2);margin-top:2px}

/* Plain-English readout of the current selection. Deliberately prose, not a
   settings dump: the toggles above are compact enough to be ambiguous, and the
   method notes at the foot of the page describe the method in general rather
   than what is on screen right now. */
/* ---- collapsible disclosure panels ----------------------------------------
   The plain-English readout and the warnings are both long, and both are
   rebuilt on every control change. The <details> element therefore lives in the
   STATIC skeleton and only its body is replaced, so the open/closed state
   survives a re-render without having to be tracked in state and coerced. Both
   default to closed: they are reference material, not the first thing to read.
   -------------------------------------------------------------------------- */
.disclose{border:1px solid var(--rule);border-radius:8px;background:var(--surface);
  margin-bottom:14px}
.disclose.plain-wrap{border-left:3px solid var(--series)}
.disclose.warn-wrap{border-left:3px solid var(--warning);background:var(--warning-bg)}
.disclose > summary{list-style:none;cursor:pointer;padding:11px 16px;display:flex;
  align-items:center;gap:8px;font-family:var(--mono);font-size:12px;font-weight:600;
  letter-spacing:.04em;text-transform:uppercase;color:var(--ink-muted);
  -webkit-user-select:none;user-select:none;border-radius:6px}
.disclose > summary::-webkit-details-marker{display:none}
.disclose > summary::marker{content:""}
.disclose > summary:hover{color:var(--ink-2)}
.disclose > summary:focus-visible{outline:2px solid var(--series);outline-offset:-2px}
.caret{flex:none;width:7px;height:7px;border-right:2px solid currentColor;
  border-bottom:2px solid currentColor;transform:rotate(-45deg);margin:0 4px 2px 1px;
  transition:transform .13s}
.disclose[open] > summary .caret{transform:rotate(45deg);margin-bottom:0}
.disclose-n{margin-left:auto;font-family:var(--sans);font-size:11px;font-weight:600;
  padding:1px 8px;border-radius:9px;letter-spacing:0;text-transform:none;
  background:var(--surface-2);color:var(--ink-muted);border:1px solid var(--rule)}
.warn-wrap .disclose-n{background:var(--surface);color:var(--warning-ink);
  border-color:var(--warning)}

.plain{padding:2px 20px 15px}
.plain p{margin:0 0 9px;font-size:14px;color:var(--ink-2);max-width:88ch;line-height:1.62}
.plain p:last-child{margin-bottom:0}
.plain b{color:var(--ink);font-weight:600}
.plain .warn{color:var(--critical);font-weight:600}

.banner{border:1px solid var(--rule);border-left:3px solid var(--warning);
  background:var(--warning-bg);border-radius:8px;padding:12px 16px;margin-bottom:12px;
  font-size:13.5px;color:var(--ink)}
.banner b{font-weight:600}
.banners{padding:2px 16px 16px}
.banners .banner:last-child{margin-bottom:0}

.sec{display:flex;align-items:baseline;justify-content:space-between;gap:16px;
  margin:0 0 14px;flex-wrap:wrap}
.sec h2{font-size:17px;font-weight:600;letter-spacing:-.01em}
.sec .hint{font-size:13px;color:var(--ink-muted);max-width:62ch}

.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px}
.panel{background:var(--surface);border:1px solid var(--rule);border-radius:10px;
  overflow:hidden;cursor:pointer;position:relative;text-align:left;padding:0;
  font-family:inherit;color:inherit;transition:border-color .13s,transform .13s;
  display:flex;flex-direction:column}
.panel:hover{border-color:var(--rule-strong);transform:translateY(-1px)}
.panel:focus-visible{outline:2px solid var(--series);outline-offset:2px}
.stripe{height:3px;width:100%}
.stripe.good{background:var(--good)} .stripe.warning{background:var(--warning)}
.stripe.critical{background:var(--critical)}
.p-head{padding:11px 13px 8px;display:flex;justify-content:space-between;
  align-items:flex-start;gap:10px}
.p-id{font-family:var(--mono);font-size:13px;font-weight:600;letter-spacing:-.01em}
.p-cond{font-size:11.5px;color:var(--ink-muted);margin-top:1px}
.p-med{font-family:var(--mono);font-size:15px;font-weight:500;text-align:right;
  font-variant-numeric:tabular-nums;line-height:1.2}
.p-med small{display:block;font-size:10px;letter-spacing:.07em;text-transform:uppercase;
  color:var(--ink-muted);font-weight:400}
.p-chart{padding:0 8px}
.p-foot{padding:7px 13px 10px;display:flex;justify-content:space-between;
  align-items:center;gap:8px;font-size:11.5px;color:var(--ink-muted);
  font-family:var(--mono);border-top:1px solid var(--rule);margin-top:6px}
.badge{display:inline-flex;align-items:center;gap:5px;font-family:var(--sans);
  font-size:11px;font-weight:600;padding:2px 7px;border-radius:4px;white-space:nowrap}
.badge.good{background:var(--good-bg);color:var(--good)}
.badge.warning{background:var(--warning-bg);color:var(--warning-ink)}
.badge.critical{background:var(--critical-bg);color:var(--critical)}
.badge svg{width:11px;height:11px;flex:none}
.chip{font-family:var(--mono);font-size:10px;padding:1px 5px;border-radius:3px;
  border:1px solid var(--rule-strong);color:var(--ink-muted)}
.chip.flag{border-color:var(--critical);color:var(--critical)}
.chip.flag.warning{border-color:var(--warning);color:var(--warning-ink)}

.detail{background:var(--surface);border:1px solid var(--rule);border-radius:10px;
  margin-bottom:30px;overflow:hidden}
.d-head{padding:16px 20px;border-bottom:1px solid var(--rule);display:flex;
  justify-content:space-between;align-items:flex-start;gap:16px;flex-wrap:wrap}
.d-title{font-family:var(--mono);font-size:18px;font-weight:600;letter-spacing:-.01em}
.d-sub{font-size:13px;color:var(--ink-2);margin-top:3px}
.d-body{padding:8px 12px 4px}
.d-meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));
  gap:1px;background:var(--rule);border-top:1px solid var(--rule)}
.d-meta div{background:var(--surface);padding:11px 16px}
.d-meta .k{font-family:var(--mono);font-size:10px;letter-spacing:.09em;
  text-transform:uppercase;color:var(--ink-muted)}
.d-meta .v{font-family:var(--mono);font-size:13.5px;margin-top:3px;
  font-variant-numeric:tabular-nums}
.d-note{padding:11px 20px;background:var(--warning-bg);border-top:1px solid var(--rule);
  font-size:13px;color:var(--ink)}
/* Baseline reference viewer, inside an expanded panel. Visually subordinate to
   the task chart above it -- tinted ground, smaller type -- because it is
   context for that chart, not a second result. */
.d-base{border-top:1px solid var(--rule);background:var(--surface-2)}
.d-base-head{padding:13px 20px 9px;display:flex;justify-content:space-between;
  align-items:flex-start;gap:16px;flex-wrap:wrap}
.d-base-title{font-size:13.5px;font-weight:600;color:var(--ink)}
.d-base-range{font-family:var(--mono);font-size:11px;color:var(--ink-muted);
  font-weight:400;margin-left:7px}
.d-base-sub{font-size:12px;color:var(--ink-2);margin-top:3px}
.d-base-body{padding:0 12px}
/* The stat text is wrapped in a single <span> so this flex row does not inject
   its 8px gap between the words of one sentence. */
.d-base-foot{padding:5px 20px 13px;font-size:12px;color:var(--ink-2);
  display:flex;gap:8px;align-items:baseline;flex-wrap:wrap}
/* .warn is scoped to .plain elsewhere; the viewer needs the same emphasis
   outside it, and this is the warning that fires on a single-window baseline. */
.warn-ink{color:var(--warning-ink);font-weight:600}
.d-base-hint{margin-left:auto;font-family:var(--mono);font-size:11px;
  color:var(--ink-muted);white-space:nowrap}
.d-base-seg button{font-size:12px;padding:5px 10px}
.close{background:none;border:1px solid var(--rule);border-radius:6px;color:var(--ink-2);
  font-family:var(--sans);font-size:12.5px;padding:5px 11px;cursor:pointer}
.close:hover{color:var(--ink);border-color:var(--rule-strong)}
.close:focus-visible{outline:2px solid var(--series);outline-offset:1px}

.grid-line{stroke:var(--rule);stroke-width:1}
.axis-line{stroke:var(--rule-strong);stroke-width:1}
.zeroline{stroke:var(--ink-2);stroke-width:1.1;opacity:.75}
.tick{font-family:var(--mono);font-size:9.5px;fill:var(--ink-muted)}
.tick-lg{font-family:var(--mono);font-size:11px;fill:var(--ink-muted)}
.trace{fill:none;stroke:var(--series);stroke-width:1.4;stroke-linejoin:round;
  stroke-linecap:round}
.trace-lg{stroke-width:1.7}
.trace-imputed{stroke-dasharray:2.5 3;stroke:var(--ink-muted);stroke-width:1.2}
.medline{stroke:var(--ink-muted);stroke-width:1;stroke-dasharray:3 3;opacity:.6}
.area{fill:var(--series-soft);stroke:none}

.tablewrap{overflow-x:auto;border:1px solid var(--rule);border-radius:10px;
  background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:8px 13px;text-align:left;border-bottom:1px solid var(--rule);
  white-space:nowrap}
th{font-family:var(--mono);font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;
  color:var(--ink-muted);font-weight:500;position:sticky;top:0;background:var(--surface)}
td.num{font-family:var(--mono);font-variant-numeric:tabular-nums;text-align:right}
tbody tr:last-child td{border-bottom:0}
tbody tr:hover{background:var(--surface-2)}
.empty{padding:16px 20px;color:var(--ink-muted);font-size:13px}

/* Multi-column rather than grid. As a grid these six blocks sat in rows, and a
   row is as tall as its tallest member -- one 2,600 px column of prose stretched
   its three neighbours to match and pushed the last two blocks a full screen
   below the fold, which is the whitespace this replaces. Columns pack them by
   height instead, so the section ends where its content ends. */
.method{margin-top:34px;border-top:1px solid var(--rule);padding-top:22px;
  columns:4 260px;column-gap:26px}
/* The blocks flow ACROSS column breaks rather than being kept whole. Keeping
   them whole left one block setting the height of the whole section and a
   fourth column standing empty -- and that block is a SINGLE 2,600 px
   paragraph, so holding paragraphs together has the same effect as holding
   blocks together. Both are therefore allowed to break, which is what a
   multi-column text layout is for; `widows`/`orphans` keep a break from
   stranding one line, and `break-after:avoid` keeps a heading with the text it
   introduces. Reading order still runs down each column in turn. */
.method > div{margin:0 0 22px}
.method h3{font-size:13px;font-weight:600;margin-bottom:7px;break-after:avoid}
.method p{font-size:13px;color:var(--ink-2);margin:0 0 9px;max-width:60ch;
  widows:2;orphans:2}
.tooltip{position:fixed;pointer-events:none;z-index:60;background:var(--surface);
  border:1px solid var(--rule-strong);border-radius:6px;box-shadow:var(--shadow);
  padding:7px 10px;font-family:var(--mono);font-size:11.5px;opacity:0;
  transition:opacity .1s;font-variant-numeric:tabular-nums}
.tooltip.on{opacity:1}
.tooltip .tt-v{font-size:14px;font-weight:600;color:var(--ink)}
.tooltip .tt-k{color:var(--ink-muted);font-size:10.5px}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}

/* ---- collapsing the control bank ------------------------------------------
   The two banks run to ~350 px, and the bar is sticky, so on a laptop they hold
   a third of the viewport permanently and on a phone rather more than that.
   Collapsing leaves the one-line summary below, which is what the banks were
   being read for most of the time anyway. */
.ctl-bar-head{display:flex;align-items:center;gap:14px;min-width:0}
.controls:not(.collapsed) .ctl-bar-head{margin-bottom:13px}
.controls.collapsed .controls-inner{display:none}
.ctl-toggle{display:inline-flex;align-items:center;gap:4px;flex:none;cursor:pointer;
  font-family:var(--mono);font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;
  color:var(--ink-muted);background:var(--surface-2);border:1px solid var(--rule);
  border-radius:6px;padding:4px 10px 4px 8px}
.ctl-toggle:hover{color:var(--ink-2);border-color:var(--rule-strong)}
.ctl-toggle:focus-visible{outline:2px solid var(--series);outline-offset:2px}
.ctl-toggle .caret{transform:rotate(45deg);margin:0 2px 2px 1px}
.controls.collapsed .ctl-toggle .caret{transform:rotate(-45deg);margin-bottom:0}
/* The summary is the only readout of what is applied while the banks are shut,
   so it may shrink and ellipsize but must never wrap the bar onto a second line. */
.ctl-summary{font-family:var(--mono);font-size:11.5px;color:var(--ink-muted);
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;min-width:0}
.ctl-summary b{color:var(--ink-2);font-weight:600}
.controls:not(.collapsed) .ctl-summary{display:none}

/* ---- control rows: two banks, values above thresholds ---- */
.controls-inner{flex-direction:column;align-items:stretch;gap:13px}
.ctl-row{display:flex;gap:22px;flex-wrap:wrap;align-items:flex-end}
.ctl-row + .ctl-row{border-top:1px solid var(--rule);padding-top:13px}
.slider{display:flex;align-items:center;gap:9px;height:33px}
.slider input[type=range]{width:126px;accent-color:var(--series);cursor:pointer}
.slider input[type=range]:disabled{cursor:not-allowed}
.slider input[type=range]:focus-visible{outline:2px solid var(--series);outline-offset:3px}
.slider .val{font-family:var(--mono);font-size:12.5px;min-width:56px;
  font-variant-numeric:tabular-nums;color:var(--ink)}
/* .gatebox went with the amplitude gate's checkbox on 2026-09-02. `.chk` is
   live again since 2026-09-08 -- the frontal-pair checkbox -- and is defined
   with the other control styles above. */
.ctl.off{opacity:.5}
#shard-state:not([hidden]){margin-bottom:26px}

/* ---- small screens ---------------------------------------------------------
   The stylesheet had no width media query at all. That was survivable while the
   page had no viewport meta and a phone simply rendered it at 980 px and zoomed
   out; with `width=device-width` the layout gets a real 375 px and needs to
   answer for it. Nothing here changes the desktop rendering.
   -------------------------------------------------------------------------- */
@media (max-width:760px){
  .wrap{padding-left:16px;padding-right:16px}
  /* A wrapped control bank is a lot of rows, so tighten the pieces that repeat. */
  .ctl-row{gap:14px}
  .ctl-row + .ctl-row{padding-top:11px}
  .controls-inner{gap:11px}
  .seg button{padding:6px 10px;font-size:12.5px}
  .slider input[type=range]{width:104px}
  .slider .val{min-width:48px}
  /* One trace per row reads better than two cramped ones. */
  .grid{grid-template-columns:1fr}
  .d-head{padding:13px 15px}
  .d-meta{grid-template-columns:repeat(auto-fit,minmax(150px,1fr))}
  .plain{padding:2px 15px 14px}
  .banners{padding:2px 12px 13px}
  .method{column-gap:0}
}
</style>

<div class="wrap">
<header class="mast">
  <div class="eyebrow" id="mast-eyebrow"></div>
  <h1 id="mast-title"></h1>
  <div class="formula" id="mast-formula"></div>
  <p class="lede" id="mast-lede"></p>
  <div class="tabrow">
    <div class="tabs" role="tablist" id="tabs"></div>
    <button type="button" id="export-btn" class="export"
      aria-describedby="export-status"
      title="Download every task value currently plotted, with the settings that produced it">
      Export JSON</button>
    <span id="export-status" role="status" aria-live="polite" class="export-status"></span>
  </div>
</header>

<div class="controls" id="controls-bar">
  <div class="ctl-bar-head">
    <button type="button" class="ctl-toggle" id="ctl-toggle" aria-expanded="true"
      aria-controls="controls"
      title="Collapse the control banks to free up screen space. The settings stay applied.">
      <span class="caret" aria-hidden="true"></span><span id="ctl-toggle-label">Controls</span>
    </button>
    <div class="ctl-summary" id="ctl-summary"></div>
  </div>
  <div class="controls-inner" id="controls"></div>
</div>

<div id="tabpanel" role="tabpanel" aria-labelledby="tab-selected" tabindex="-1">
<div id="shard-state" role="status" aria-live="polite" hidden></div>
<details class="disclose plain-wrap" id="plain-wrap">
  <summary><span class="caret" aria-hidden="true"></span>What this page is showing right now</summary>
  <div class="plain" id="plain"></div>
</details>
<div class="tiles" id="tiles"></div>
<details class="disclose warn-wrap" id="warn-wrap" hidden>
  <summary><span class="caret" aria-hidden="true"></span>Warnings<span
    class="disclose-n" id="warn-count"></span></summary>
  <div class="banners" id="banners"></div>
</details>
<div id="detail-slot"></div>

<div class="sec">
  <h2>Per-recording traces</h2>
  <span class="hint" id="grid-hint"></span>
</div>
<div class="grid" id="grid"></div>

<div class="sec" style="margin-top:34px">
  <h2>Excluded recordings</h2>
  <span class="hint" id="excl-hint"></span>
</div>
<div class="tablewrap"><table id="excluded"></table><div class="empty" id="excluded-empty" hidden></div></div>

<div class="sec" style="margin-top:34px">
  <h2>Table view</h2>
  <span class="hint">Same data, for reading exact values and for screen readers.</span>
</div>
<div class="tablewrap"><table id="table"></table></div>

<div class="method" id="method"></div>
</div>
</div>

<div class="tooltip" id="tt"></div>

<script>
const DATA = __DATA__;
const PL = DATA.plausible_uv;
const AB = DATA.amplitude_bound || {};
/* One phrase for what the band IS, used everywhere it is quoted. It is derived
   from this dataset, so the wording never claims a physiological standard. */
const BAND_TXT = `${PL[0]}–${PL[1]} µV robust SD`;
const BAND_WHY = `the band this dataset's own channel amplitudes define
  (median ${AB.median_uv} µV, ±${AB.k} robust σ in log space,
  fitted on ${AB.n_measurements} pre-interpolation, hardware-referenced channel
  measurements from this run)`;

/* Column names of the 7-wide value matrix and the 5-wide peak matrix, taken
   from the pipeline's own index.json rather than hardcoded, so a change to
   VARIANT_VALUE_KEYS cannot silently misalign the columns here. */
const VK = DATA.value_keys, PC = DATA.peak_channels, W = VK.length, K = PC.length;
const COL = {}; VK.forEach((k,i)=>COL[k]=i);
/* Reordering VARIANT_VALUE_KEYS is handled by reading the names; a RENAME would
   leave COL.x undefined, and V[i*W + undefined] is NaN -- a silently blank tab.
   Fail loudly instead. */
for(const k of ["fm_theta","parietal_alpha","parietal_beta_asym",
                "frontal_alpha_asym","theta_f1","theta_f2","alpha_holm_pz"])
  if(COL[k]===undefined) throw new Error("value column missing from the sweep: "+k);
const MT_NW = DATA.multitaper_nw;
/* Sliding-window lengths for the windowed robust mode, from the pipeline. The
   page can only offer lengths the sweep actually computed, because each one is a
   stored per-window sigma column rather than something derivable here. Every
   phrase naming the length is generated from the current choice, so the wording
   moves with the toggle. */
/* Head motion, from the helmet's IMU. MC maps a column name to its index in the
   shipped (n_windows x 3) matrix, so the page never addresses a column by
   position. The last column is the aux sample count behind that window. */
const MOT_SRC = DATA.motion_sources || [];
const MOT_COLS = DATA.motion_columns || [];
const MC = Object.fromEntries(MOT_COLS.map((c,i)=>[c,i]));
const MOT_K = DATA.motion_k || {min:1,max:8,step:0.25,default:3};
const MOT_MIN = DATA.motion_min_samples;
const MOT_LABEL = {accel_jerk:"Accelerometer", gyro:"Gyroscope", union:"Either"};
/* A THIRD rule, not a third sensor channel: reject a window if EITHER source
   would reject it, each against its own per-recording threshold. It is offered
   because the two disagree far more than their +0.84 rank correlation suggests
   -- at the k=3 default they shared only 129 of the 418 windows either one
   rejected (Jaccard 0.31) -- so "which sensor" is a real analysis choice rather
   than a formality, and the union is the sensitive end of it. (Those figures
   come from the study this code was written for, measured over its 19 task
   recordings with usable motion. Expect different numbers on another dataset;
   the argument for offering the union does not depend on them.)

   It lives here rather than in MOTION_SOURCES because MOTION_SOURCES names the
   COLUMNS the pipeline ships in the motion matrix, and there is no union column:
   this thresholds quantities the archive already carries, exactly as the sigma
   and cap sliders do, so it needs no pipeline run. */
const MOT_UNION = "union";
const MOT_RULES = MOT_SRC.length > 1 ? MOT_SRC.concat([MOT_UNION]) : MOT_SRC;
const isMotUnion = () => motSrc() === MOT_UNION;
const motSrc = () => st().motSrc || DATA.default_motion_source || MOT_SRC[0];
const SLIDE_CHOICES = DATA.robust_slide_choices_s;
const SLIDE_DEFAULT = DATA.robust_slide_default_s;
const SL_LABEL = () => `${+st().slideS} s`;
const SL_SPAN  = () => `the ${+st().slideS} s centred on it`;
/* Baseline segments: which part of the 6-minute baseline block a task is
   referenced against. Bounds and ordering come from the pipeline. */
const SEGB = DATA.baseline_segments;
const SEG_ORDER = DATA.baseline_segment_order;
const SEG_DEFAULT = DATA.default_baseline_segment;
const SEG_LABEL = {rest:"Resting (eyes open)", math:"Mental math",
                   eyes_closed:"Eyes closed"};
const SEG_SHORT = {rest:"resting", math:"mental math", eyes_closed:"eyes closed"};
/* What each segment IS, in the reader's terms -- because subtracting the math
   segment is not a "baseline correction" in the sense the resting one is, and a
   reader who does not notice that will read every sign backwards. */
const SEG_BLURB = {
  rest: `the last two minutes of the block, sitting quietly with eyes open. This is
    the conventional resting reference, and the one every published number here was
    computed against.`,
  math: `the first two minutes of the block, doing mental arithmetic. This is an
    <b>active-task</b> reference, not a resting one: subtracting it asks &ldquo;how did
    this task compare with deliberate mental effort&rdquo;, not &ldquo;how did it compare
    with rest&rdquo;. A value near zero means the task looked like doing sums, which is
    not the same as looking calm.`,
  eyes_closed: `the last 15 seconds of the middle two minutes, with the eyes closed.
    <b>15 seconds is 3 windows at the 4-second default, and a single window at 8
    seconds or longer</b> &mdash; a median over one window is that window. It is also the
    tail of a phase of deliberate eye movements, so it carries the highest ocular risk
    of the three. Treat it as indicative, not as a stable level.`
};
const segOf = () => st().baseSeg || SEG_DEFAULT;
/* Accessors rather than bare lookups, so a segment the pipeline adds later that
   these dicts do not know about degrades to its own name and its real bounds
   instead of rendering the string "undefined" across a dozen places. */
const segShort = (s) => SEG_SHORT[s || segOf()] || (s || segOf()).replace(/_/g, " ");
const segLabel = (s) => SEG_LABEL[s || segOf()] || (s || segOf()).replace(/_/g, " ");
const segBounds = (s) => SEGB[s || segOf()] || [0, 0];
const segLen = (s) => { const b = segBounds(s); return b[1] - b[0]; };
const segBlurb = (s) => SEG_BLURB[s || segOf()] ||
  `${segBounds(s)[0]}&ndash;${segBounds(s)[1]}&nbsp;s of the baseline block.`;
/* The per-segment payload for a recording under the CURRENT segment choice, or
   null where this baseline has no such segment. */
function baseSeg(rec){
  const b = rec.base;
  if(!b || b.missing) return null;
  return (b.segs && b.segs[segOf()]) || null;
}

const detailSegOf = () => st().detailSeg || SEG_DEFAULT;

/* The baseline segment's OWN time course, in the measure's own units.

   Never baseline-corrected -- a baseline cannot be referenced against itself --
   so this is the raw level whose median is exactly what the correction
   subtracts. Available whether or not correction is on, which is the point: the
   reader can look at what they would be subtracting before deciding to.

   Every array comes from that segment's own archive entry, so the peaks, the
   sigma and the window grid belong to the segment being drawn. Windows outside
   the segment's own index are dropped, so the trace shows the samples the median
   was actually taken over and nothing else. */
function baselineSeries(rec, segment){
  const b = rec.base;
  if(!b || b.missing)
    return {missing:true, reason:(b && b.reason) || "no paired baseline recording"};
  const sp = (b.segs || {})[segment];
  if(!sp) return {missing:true,
    reason:`this baseline block has no ${segShort(segment)} segment`};
  const B = bValsOf(sp.key);
  if(!B) return {missing:true,
    reason:"the sweep holds no baseline values for this processing combination"};
  const n = B.length / W;
  if(!Number.isInteger(n) || n < 1)
    return {missing:true, reason:"no windows in this segment"};
  const vec = measureVector(B, n, rec);
  if(!vec) return {missing:true, reason:"this measure is not available here"};
  const chans = maskChannels(rec);
  const bexcl = exclSet(sp.excl);
  const keep = maskFor(rec, st().art, n, bPeaksOf(sp.key),
                       sp.sigma ? sp.sigma[st().oc] : null,
                       chans, bSlideOf(sp.key), bMotionOf(sp.key), bexcl);
  /* Refused on a missing or empty index, matching baselineStats exactly. The
     two are meant to be the same rule, and treating a null index as "every
     window counts" would let the viewer show a median the correction has just
     refused to compute. */
  const idx = sp.idx ? sp.idx[ekey()] : null;
  if(!idx || !idx.length) return {missing:true,
    reason:`this segment holds no windows at the ${st().epoch}-second window length`};
  const inSeg = new Uint8Array(n);
  for(const i of idx){ if(i < n) inSeg[i] = 1; }
  const vals = new Array(n);
  let kept = 0;
  for(let i=0;i<n;i++){
    /* bexcl for the same reason baselineStats applies it: this function draws
       the reference graph for the median that function computes, and a window in
       one but not the other would put a point on the chart that is not in the
       number beside it. */
    const bad = !inSeg[i] || (bexcl && bexcl.has(i))
                || (keep && !keep[i]) || !isFinite(vec[i]);
    vals[i] = bad ? null : vec[i];
    if(!bad) kept++;
  }
  const live = vals.filter(v => v != null);
  /* imputed is all-false and stays that way: the pipeline never imputes a
     baseline, and a median has no gaps to fill. drawChart wants the array. */
  return {vals, imputed:new Array(n).fill(false), n, kept, nAvail: idx.length,
          spKey: sp.key,
          median: live.length ? median(vals) : null,
          /* Read by the footer. A null mask means the criterion could not be
             EVALUATED, which is not the same as nothing being rejected -- the
             chart is then unfiltered, and saying "30 of 30 kept, rejection rule
             applied" over it is the exact failure this page fixed elsewhere. */
          maskRefused: keep === null && maskRule(st().art) !== "none"};
}

const SL = DATA.sliders;
/* SL.amplitude_gate_uv, and the GATE_FLOOR that was read from it, went with the
   amplitude gate on 2026-09-02 -- the pipeline no longer ships that slider. */
const SHARD_DIR = DATA.shard_dir;

/* =======================================================================
   Measure definitions.

   `kind` drives the maths, not just the labels:
     ratio  - the Holm index. Strictly positive, so a log axis is available.
     power  - absolute band power. Strictly positive; log axis available.
     asym   - ALREADY a difference of natural logs, so routinely negative.
              A log axis is undefined on it, and a log-domain baseline
              correction is undefined too: subtraction IS the log-domain
              operation for these, which is why they offer one Subtract
              button rather than a Raw/Log pair.
   ======================================================================= */
const MEAS = {
  index: {
    tab:"Cognitive load index",
    title:"Cognitive load across task recordings",
    kind:"ratio",
    channels:["F1","F2","Pz"],
    formula:`<span>Brain load index =</span>
      <span class="frac"><span class="num">absolute theta power (4&ndash;8&nbsp;Hz), midline frontal</span><span class="den">absolute alpha power (8&ndash;12&nbsp;Hz) at Pz</span></span>`,
    lede:`Holm et al. (2009) define brain load as frontal theta over parietal alpha at
      <strong>Fz</strong> and <strong>Pz</strong>. This montage has no Fz, so the numerator comes from
      <strong>F1 and F2</strong> &mdash; the two sites flanking it &mdash; averaged in the time domain.
      Read every value as theta(F1,F2)/alpha(Pz), not theta&nbsp;Fz/alpha&nbsp;Pz. Higher means more load.
      <strong>Every task recording is shown</strong>, including those whose electrodes are not
      inside that band &mdash; nothing is withheld, so read each panel's signal-quality badge
      and the electrode amplitudes before trusting a value.`,
    unit:"ratio"
  },
  fm_theta: {
    tab:"Frontal midline theta",
    title:"Frontal midline theta across task recordings",
    kind:"power",
    channels:["F1","F2"],
    formula:`<span>Frontal midline theta = absolute power, 4&ndash;8&nbsp;Hz, of &frac12;(F1&nbsp;+&nbsp;F2)</span>`,
    lede:`Theta power of the <strong>time-domain mean of F1 and F2</strong> &mdash; the same midline
      derivation the cognitive-load index uses as its numerator, plotted on its own. Every task
      recording appears, including those whose electrodes sit outside that band; it always
      uses the full F1+F2 mean. Check each panel's badge before trusting a value.`,
    unit:"power"
  },
  parietal_alpha: {
    tab:"Parietal alpha",
    title:"Parietal alpha across task recordings",
    kind:"power",
    channels:["Pz"],
    formula:`<span>Parietal alpha = absolute power, 8&ndash;13&nbsp;Hz, at Pz</span>`,
    lede:`Absolute alpha power at <strong>Pz</strong>. Note the band: this tab uses the example
      pipeline's <strong>8&ndash;13&nbsp;Hz</strong>, whereas the cognitive-load index denominator uses
      Holm's narrower <strong>8&ndash;12&nbsp;Hz</strong>. The two are close but not the same quantity,
      so this trace is not simply the index's denominator.`,
    unit:"power"
  },
  parietal_beta_asym: {
    tab:"Parietal beta asymmetry",
    title:"Parietal beta asymmetry across task recordings",
    kind:"asym",
    channels:["P3","P4"],
    formula:`<span>Parietal beta asymmetry = ln(beta power P4) &minus; ln(beta power P3) &middot; 13&ndash;30&nbsp;Hz</span>`,
    lede:`Log-power difference between the right and left parietal sites in the beta band.
      <strong>Positive means P4 (right) exceeds P3 (left).</strong> Values are already natural-log
      units, so they are routinely negative and a log axis does not apply. Both channels are
      required &mdash; a difference has no single-channel substitute.`,
    unit:"ln ratio"
  },
  frontal_alpha_asym: {
    tab:"Frontal alpha asymmetry",
    title:"Frontal alpha asymmetry across task recordings",
    kind:"asym",
    channels:["F1","F2"],
    formula:`<span>Frontal alpha asymmetry = ln(alpha power F2) &minus; ln(alpha power F1) &middot; 8&ndash;13&nbsp;Hz</span>`,
    lede:`Log-power difference between the right and left frontal sites in the alpha band.
      <strong>Positive means F2 (right) exceeds F1 (left).</strong> Values are already natural-log
      units and are routinely negative, so a log axis does not apply. Both channels are required.
      Alpha here is the example pipeline's <strong>8&ndash;13&nbsp;Hz</strong>.`,
    unit:"ln ratio"
  }
};
const ORDER = ["index","fm_theta","parietal_alpha","parietal_beta_asym","frontal_alpha_asym"];

const ART = [{id:"none",label:"None"},
             {id:"robust",label:"Robust"},{id:"robust_imputed",label:"Robust + impute"},
             {id:"robust_windowed",label:"Windowed Robust"},
             {id:"holm_strict",label:"Absolute cap"},{id:"holm_imputed",label:"Cap + impute"},
             {id:"motion",label:"Head motion"}];

/* Two derived properties of an artifact mode, so a mode is defined once and every
   branch reads it rather than string-matching. `robust_imputed` was added
   2026-09-02: it applies the robust rejection and then the SAME gap-filling that
   `holm_imputed` does. Imputation is a display step layered on a rejection, not a
   rejection of its own, so a mode has a rejection rule and, separately, a flag for
   whether the holes are filled. This is entirely browser-side -- the pipeline
   already ships the peaks and sigma the robust mask needs -- so no re-run or
   re-sweep is involved.

   `robust_windowed` was added 2026-09-02 at the user's request. Same comparison,
   local yardstick: sigma is recomputed from a centred sliding window and each
   analysis window is judged against its own. Unlike `robust_imputed` this one DID
   need a pipeline run -- sigma is the MAD of the raw samples and the page only
   ever sees per-window peaks -- so it reads the per-window sigma matrices the
   shards carry, shipped by pipeline.robust_sigma_sliding_uv.

   Both robust rules share the k-sigma SLIDER and every branch that keys off
   "robust", which is why maskRule folds them together; where the two differ
   (which sigma, and the wording) the code tests the mode id itself. */
const IMPUTED = new Set(["holm_imputed", "robust_imputed"]);
const WINDOWED = new Set(["robust_windowed"]);
const maskRule = m => m === "none" ? "none"
                    : m === "motion" ? "motion"
                    : (m === "robust" || m === "robust_imputed"
                       || m === "robust_windowed") ? "robust" : "cap";
const isMotion = (m) => (m === undefined ? st().art : m) === "motion";
const isWindowed = m => WINDOWED.has(m === undefined ? st().art : m);
const isImputed = () => IMPUTED.has(st().art);
/* Built from the archive rather than written out, so a mode the pipeline
   swept can never be missing from the control and a mode it did not sweep can
   never be offered. `ica` arrived on 2026-09-05 and would otherwise have been
   present in the data and absent from the page. */
const OC_LABEL = {none:"None", eog_regression:"EOG regression", ica:"ICA"};
/* `.length`, not truthiness: an empty array is truthy, so `|| [...]` would have
   rendered a labelled control with no buttons. The build refuses an archive whose
   ocular modes disagree with this page, so in practice this fallback is
   unreachable -- it is here so a hand-edited payload degrades to something
   usable rather than to an empty control. */
const OC = (DATA.ocular_modes && DATA.ocular_modes.length
              ? DATA.ocular_modes : ["none"])
             .map(id => ({id, label: OC_LABEL[id] || id}));
const REF = [{id:"hardware",label:"Hardware (SRB2)"},{id:"average",label:"Average"},
             {id:"rest",label:"REST"}];
/* Where the list of channels to interpolate comes from. The frontal-pair
   checkbox beside it is the SECOND choice, and the two together name one of the
   five swept modes -- see interpMode(). Split into a selector plus a checkbox
   rather than five buttons because they are independent questions, and because
   "off, but keep the frontal pair" is not a position: there is nothing to keep. */
const INT = [{id:"off",label:"Off"},
             {id:"on",label:"Automatic"},
             {id:"manual",label:"Manual list"}];
/* `.length`, not truthiness: an empty array is truthy, so `|| [...]` would keep
   a `frontal_pair: []` payload and print " and  are being held back". The same
   trap is documented on OC above. */
const FRONTAL = (DATA.frontal_pair && DATA.frontal_pair.length)
              ? DATA.frontal_pair : ["F1","F2"];

/* "one 20-minute task recording" used to be written into two places on the
   page. It described the study this code was written for, and it rendered
   verbatim over whatever tree the page was actually built from -- six 10-minute
   recordings, in the template's own example build. Derived instead, and only
   when the recordings agree on a length: a tree with mixed durations gets the
   plain phrase rather than a median that describes none of them. Trailing space
   is part of the value so the callers read `${TASK_LEN}task`. */
const TASK_LEN = (()=>{
  const d = DATA.recordings.map(r=>r.duration_s)
                           .filter(v=>typeof v === "number" && v > 0)
                           .sort((a,b)=>a-b);
  if(!d.length) return "";
  const lo = d[0], hi = d[d.length-1], med = d[(d.length-1)>>1];
  if(hi > lo * 1.25) return "";
  const mins = med / 60;
  return `${mins >= 2 ? Math.round(mins) : +mins.toFixed(1)}-minute `;
})();
/* Did the frontal-pair checkbox actually bite on this recording? The question is
   about the SOURCE list the mode started from -- the automatic list under
   `automatic`, the manual one under `manual` -- not about what detection found.
   Testing detection instead reported "the pair is being held back" on every
   recording with F1 and F2 flagged, including ones whose manual list is empty
   and where the rule was a strict no-op. One definition, three callers. */
function frontalHeldBack(rec){
  if(st().interpSrc === "off" || st().frontalPair) return false;
  const src = (rec && rec.interpLists && rec.interpLists[st().interpSrc]) || [];
  return FRONTAL.every(c => src.includes(c));
}
const FFT = [{id:"hann",label:"Hann"},{id:"multitaper",label:"Multitaper"},
             {id:"welch",label:"Welch"},{id:"boxcar",label:"Boxcar"}];
const SC  = [{id:"log",label:"Log"},{id:"linear",label:"Linear"}];
/* The "Strict subset" control that stood here is gone with the amplitude gate:
   the strict subset was defined as the recordings whose frontal channels both
   passed that gate, and with no gate there is no subset to switch to. The
   pipeline still labels recordings `strict_subset` for anyone who wants it. */

const REF_LABEL = {hardware:"hardware SRB2/earlobe", average:"average of 10 EEG channels",
                   rest:"REST (reference at infinity)"};
const FFT_LABEL = {hann:"Hann", multitaper:"multitaper", welch:"Welch", boxcar:"boxcar"};

/* Per-tab state, so switching tabs and back restores what you had selected.
   SUPERSEDED 2026-09-05 -- see the block immediately below, which is authoritative.
   The defaults used to reproduce the pre-sweep pipeline (interpolation on, no
   ocular correction, hardware reference, 4 s windows, Hann taper); they no
   longer do, and the sliders alone still hold their pre-sweep values (robust
   distance 5, absolute cap 70 uV). The `gateOn`/`gateMax`/`sub` keys went with
   the amplitude gate and the strict subset on 2026-09-02.

   `base` defaults to baseline subtraction as of 2026-09-02, at the user's
   request -- previously "off". Log ratio where it is defined, which is the
   two power measures and the index; the two ASYMMETRIES are already differences
   of logs and go negative, so a log ratio of them is undefined and they default
   to the plain subtraction that is their equivalent. `baseSeg` picks WHICH
   segment of the baseline block is subtracted.

   Note what this default changes: the page now opens on a DERIVED quantity, and
   a recording with no usable baseline segment drops out of the opening view
   rather than appearing uncorrected. The tiles and the excluded table say so. */
/* The opening cell, set by the researcher on 2026-09-05: interpolation on,
   EOG regression on, average reference, Welch estimator, 4 s windows, head-motion
   rejection using Either source at 3 sigma, and a log-ratio correction against
   the open-eye resting baseline.

   TWO THINGS A READER OF THIS BLOCK SHOULD KNOW.

   1. It is NO LONGER the cell the pipeline verifies. verify_default_variant
      checks the sweep against MNE's own estimator on ONE combination -- 4 s,
      Hann, hardware reference, interpolation on, ocular none -- and
      build_dashboard.py refuses to build if that check fails. That guarantee
      still holds, and it still covers the cell it always did; it simply is not
      the cell the page now opens on. The opening view is reachable from the
      verified one by moving four controls, and every value in it comes from the
      same swept archive, but no independent estimator has been compared against
      this particular combination. Moving the verification to match would need a
      pipeline re-run.

   2. The two ASYMMETRY tabs cannot take a log ratio and are left on subtraction.
      They are already differences of logarithms and go negative; ln of a
      negative number is undefined, so "log ratio baseline correction" has no
      meaning there. `baseOpts` offers those two tabs only Off and Subtract, and
      coerceState would overwrite a log setting anyway. Subtraction IS the
      log-domain operation for a quantity that is already a log difference, so
      this is the same correction expressed the only way it can be. */
const S = {};
for(const m of ORDER){
  S[m] = {art:"motion", oc:"eog_regression", ref:"average",
          /* `interpSrc` + `frontalPair` together name one swept interpolation
             mode; interpMode() composes them. Two fields rather than one so the
             checkbox keeps its position while the reader moves the selector --
             collapsing them into a single mode string loses it the moment the
             reader passes through Off. */
          interpSrc:"on", frontalPair:true,
          fft:"welch", epoch:4,
          sigmaK: SL.robust_sigma.default, holmCap: SL.holm_cap_uv.default,
          /* Sliding-window length for the windowed robust mode. Only the lengths
             the pipeline swept can be offered, so this is a toggle, not a slider. */
          slideS: SLIDE_DEFAULT,
          /* Head-motion rejection: which IMU stream, and how many robust sigma
             of that recording's own motion count as an outlier. `union` is the
             Either rule -- reject where either stream flags the window. Guarded
             on MOT_RULES rather than named directly, so an archive shipping a
             single motion column cannot open on a rule it cannot evaluate. */
          motSrc: MOT_RULES.includes(MOT_UNION) ? MOT_UNION
                  : (DATA.default_motion_source || MOT_SRC[0]),
          motK: MOT_K.default,
          epochIdx: DATA.epochs_s.indexOf(4),
          scale: MEAS[m].kind==="asym" ? "linear" : "log",
          base: MEAS[m].kind==="asym" ? "raw" : "log",
          baseSeg: SEG_DEFAULT,
          /* Which baseline segment the EXPANDED panel graphs. Separate from
             `baseSeg` on purpose: looking at a reference is not the same act as
             subtracting one, so the viewer works with correction off, and
             arrowing through the three does not silently re-correct the data.
             It follows `baseSeg` when that changes, so the two do not drift
             apart without the reader asking for it. */
          detailSeg: SEG_DEFAULT, open:null};
}
let tab = "index";
const st = () => S[tab];
const M  = () => MEAS[tab];

const $ = s => document.querySelector(s);
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const fmt = v => {
  if(v==null || !isFinite(v)) return "—";
  const a = Math.abs(v);
  if(a===0) return "0";
  if(a>=100) return v.toFixed(0);
  if(a>=10)  return v.toFixed(1);
  if(a>=1)   return v.toFixed(2);
  if(a>=0.01) return v.toFixed(3);
  return v.toExponential(1);
};
const median = a => { const s=a.filter(v=>v!=null && isFinite(v)).sort((x,y)=>x-y);
  return s.length ? (s.length%2 ? s[(s.length-1)/2] : (s[s.length/2-1]+s[s.length/2])/2) : null; };

/* =======================================================================
   Variant store.

   The pipeline sweeps 720 processing combinations. Only the 4 s epoch is
   embedded in this file; the other nine epoch lengths sit in `dashboard_data/`
   beside it and are fetched, as plain <script> tags so this works from a
   file:// URL, the first time a slider position asks for them. Values arrive as
   base64 little-endian float32 -- the same 4-byte floats the pipeline wrote --
   and are decoded once and cached.
   ======================================================================= */
const STORE = Object.create(null);
const SHARD_FAILED = Object.create(null);
/* A SET, not one slot. With a single slot, moving the slider 4 -> 1 -> 2 -> 1
   left `pending` holding "e02_hann", so the still-loading 15 MB e01_hann was
   requested a second time; and an error on an abandoned request cleared the slot
   for a live one, causing the same duplication. */
const pending = new Set();
/* Keeping every shard ever loaded retained ~300 MB after exploring the
   10 x 4 grid, because a base64 string is UTF-16 in the engine and the decoded
   Float32Array sits beside it. Keep the inline default plus the last few. */
const KEEP_SHARDS = 3;
const shardOrder = [];

const sk = (e,f) => "e" + String(e).padStart(2,"0") + "_" + f;
const ekey = () => String(st().epoch);

window.__eegShard = function(e, f, payload){
  const k = sk(e,f);
  STORE[k] = payload;
  pending.delete(k);
  shardOrder.push(k);
  evictShards();
  /* A shard the user has already moved past should not redraw the page under
     them, nor steal focus through focusKey(). */
  if(k === needShard()) render();
};
for(const k in DATA.inline) STORE[k] = DATA.inline[k];
const INLINE = new Set(Object.keys(DATA.inline));

function evictShards(){
  while(shardOrder.length > KEEP_SHARDS){
    const old = shardOrder.shift();
    if(INLINE.has(old) || old === needShard() || shardOrder.includes(old)) continue;
    const sh = STORE[old];
    /* Iterate the shard's OWN slots rather than a hardcoded list. The list said
       ["v","p","bv","bp"] and the sliding-sigma slots "s"/"bs" were added without
       it, so their Float32Arrays stayed reachable through `_decoded` for the life
       of the tab -- the exact retention KEEP_SHARDS exists to bound. Deriving the
       slots from the object means the next one added is freed automatically. */
    if(sh) for(const grp in sh){
      const g = sh[grp];
      if(!g || typeof g !== "object" || Array.isArray(g)) continue;   // skip `missing`
      for(const vkey in g){
        const inner = g[vkey];
        if(!inner || typeof inner !== "object") continue;
        for(const rk in inner) _decoded.delete(inner[rk]);
      }
    }
    delete STORE[old];
  }
}

const _decoded = new Map();
function decode(b64){
  if(b64 == null) return null;
  let out = _decoded.get(b64);
  if(out) return out;
  const bin = atob(b64), buf = new ArrayBuffer(bin.length), u8 = new Uint8Array(buf);
  for(let i=0;i<bin.length;i++) u8[i] = bin.charCodeAt(i);
  out = new Float32Array(buf);
  _decoded.set(b64, out);
  return out;
}

function needShard(){ return sk(st().epoch, st().fft); }
function shard(){ return STORE[needShard()] || null; }

/* Returns true when the current selection is ready to draw. */
function ensureShard(){
  const k = needShard();
  if(STORE[k]) return true;
  if(SHARD_FAILED[k] || pending.has(k)) return false;
  pending.add(k);
  const el = document.createElement("script");
  el.src = SHARD_DIR + "/" + k + ".js";
  const give_up = why => {
    if(STORE[k]) return;
    SHARD_FAILED[k] = why;
    pending.delete(k);
    el.remove();
    render();
  };
  el.onerror = () => give_up("the file could not be read");
  /* A file that loads but never calls __eegShard -- a truncated write, an
     interrupted build, a server returning an HTML error page with status 200 --
     fires `load`, not `error`. Without this the page said "Loading..." forever
     over a blank body, which is a false statement about what it is doing. */
  el.onload = () => give_up("the file loaded but contained no data for this setting");
  document.head.appendChild(el);
  return false;
}

/* Let a transient failure be retried without reloading the whole page. */
window.__eegRetry = function(){
  for(const k in SHARD_FAILED) delete SHARD_FAILED[k];
  render();
};

/* The five swept interpolation modes, composed from the two controls. `off`
   takes no frontal variant -- nothing is interpolated, so there is nothing for
   the checkbox to withhold -- which is why the suffix is only ever appended to
   `on` and `manual`. */
function interpMode(){
  const src = st().interpSrc;
  return src === "off" ? "off" : (st().frontalPair ? src : src + "_keepfrontal");
}

/* Which stored array this control position names, for one archive entry.

   The archive is keyed by WHICH CHANNELS were interpolated, not by the mode that
   asked for them: two modes that reach the same list are one signal and one
   array. So a variant key cannot be built from the controls alone -- it needs
   the entry -- which is why vk() takes a key where it used to take nothing.

   Returns null rather than falling back when the map has no entry for this mode.
   Only a stale archive can do that, and build_dashboard.py refuses to build from
   one; a wrong array would be far worse than no array, and silently reading
   another mode's numbers is exactly what a fallback here would do. */
function interpTokenOf(key, mode){
  const by = DATA.interp_tokens[key];
  const m = mode === undefined ? interpMode() : mode;
  return (by && by[m] !== undefined) ? by[m] : null;
}
function vk(key){
  const t = interpTokenOf(key);
  return t === null ? null : t + "|" + st().oc + "|" + st().ref;
}
/* What the current mode actually replaces in this recording. Drives every
   sentence the page writes about what is synthetic; the token above drives which
   numbers it draws. Kept separate because they answer different questions and
   two recordings can share a token while differing in nothing else. */
function interpChansOf(rec, mode){
  const by = (rec && rec.interpLists) || null;
  if(!by) return [];
  return by[mode === undefined ? interpMode() : mode] || [];
}
function valsOf(key){ const s=shard(); const k=vk(key);
  const g=(s&&k!==null)?s.v[k]:null; return g?decode(g[key]):null; }
function peaksOf(key){ const s=shard(); const g=s&&s.p[st().oc]; return g?decode(g[key]):null; }
/* The BASELINE segment's OWN token, not the task's. Identical by construction --
   one list per group, threaded through step10_baseline -- and checked at build
   time; read from its own entry anyway, because plotting a baseline array under
   the task's key is precisely the asymmetry the paired rule exists to prevent
   and it would be invisible on screen. */
function bValsOf(key){ const s=shard(); const k=vk(key);
  const g=(s&&k!==null)?s.bv[k]:null; return g?decode(g[key]):null; }
function bPeaksOf(key){ const s=shard(); const g=s&&s.bp[st().oc]; return g?decode(g[key]):null; }
/* Per-window sigma for the windowed robust mode, keyed "<ocular>|<length>" --
   the same row order as the peaks, so one window number indexes both. */
const slideKey = () => `${st().oc}|${(+st().slideS)}`;
function slideOf(key){ const s=shard(); const g=s&&s.s[slideKey()]; return g?decode(g[key]):null; }
function bSlideOf(key){ const s=shard(); const g=s&&s.bs[slideKey()]; return g?decode(g[key]):null; }
/* Per-window head motion, (n_windows x 3) flattened. No variant dimension. */
function motionOf(key){ const s=shard(); return (s&&s.mo&&s.mo[key])?decode(s.mo[key]):null; }
function bMotionOf(key){ const s=shard(); return (s&&s.bmo&&s.bmo[key])?decode(s.bmo[key]):null; }

/* ---------- availability rules ---------- */

const logAllowed = () => M().kind!=="asym" && st().base==="off";
const effScale   = () => logAllowed() ? st().scale : "linear";

const baseOpts = () => M().kind==="asym"
  ? [{id:"off",label:"Off"},{id:"raw",label:"Subtract"}]
  : [{id:"off",label:"Off"},{id:"raw",label:"Raw Δ"},{id:"log",label:"Log ratio"}];

/* The amplitude gate was removed browser-side on 2026-09-02, and from the
   pipeline the same day, so nothing upstream withholds a recording either.
   Nothing is withheld for its electrode amplitudes; the index numerator is
   always the full F1+F2 midline mean; signal quality is surfaced through the
   per-panel badges instead.

   `frontalUsed` keeps its name because its call sites read naturally as "which
   frontal channels is this index built from", and the answer is still worth
   stating on the panel even though it no longer varies. `indexComputable` and
   `gateFails` were removed with the gate -- both had become constants nothing
   called. */
function frontalUsed(rec){ return ["F1","F2"]; }
/* Which channels' amplitude the quality badge and the masks read, for this tab. */
function maskChannels(rec){
  return tab==="index" ? ["F1","F2","Pz"] : M().channels;
}

/* Recordings a tab can show at all, before any baseline requirement. */
function candidates(){
  if(!shard()) return [];
  return DATA.recordings.filter(r => !!valsOf(r.key));
}

/* ---------- artifact masks ----------

   Both criteria threshold the SAME per-window peak amplitudes, measured on the
   pre-interpolation signal in the hardware reference, so moving the reference,
   interpolation or estimator toggle never changes which windows survive -- the
   toggle stays a clean A/B over one fixed set of retained windows.

     robust        peak <= k x sigma, where sigma is that recording's own
                   1.4826 x MAD for that channel. k = 5 is the pipeline's fixed
                   rule. Because sigma rescales with each recording's own noise,
                   retention here compares WITHIN a recording, never between.
     windowed      the same comparison against a sigma recomputed inside each
                   window CENTRED on the window being judged, so each window is
                   compared with its own neighbourhood. The yardstick is now
                   local, so retention is comparable neither between recordings
                   NOR between moments of one recording.
     absolute cap  peak <= a fixed microvolt ceiling, the same number for every
                   recording. 70 uV is Holm's own criterion; unlike the robust
                   rule this one IS comparable across recordings.

   `sigma` is the whole-recording map {channel: sigma}; `slide` the per-WINDOW
   matrix (n_windows x 5, same row order as `peaks`) for the currently chosen
   sliding length. Only one of the two is read, per mode.                       */
/* The head-motion rule, kept separate because it reads none of the EEG: it
   thresholds the recording's own per-window motion against that recording's own
   distribution of it.

   RELATIVE, deliberately. The IMU's units are not documented anywhere in this
   repository -- the accelerometer implies a scale where 1 g is about 0.5 units,
   and the gyroscope is plausibly deg/s but nothing says so -- and an absolute cut
   in an unnamed unit is precisely the unsourced-threshold problem the amplitude
   band turned out to be. A per-recording criterion needs no unit.

   The threshold is median + k x 1.4826 x MAD over the windows that HAVE motion.
   A window whose aux coverage fell below the pipeline's floor carries NaN and is
   NOT ASSESSED: it is neither rejected nor counted as clean, and the caller is
   told how many such windows there were rather than silently keeping them. */
/* One source's threshold and per-window values. Split out of motionMask so the
   union rule can ask for both without duplicating the criterion -- two copies of
   a threshold rule is how the two drift apart. */
function motionOneSource(motion, n, col){
  const K = MOT_COLS.length;
  const v = new Float64Array(n);
  const live = [];
  for(let i=0;i<n;i++){ v[i] = motion[i*K + col];
    if(isFinite(v[i])) live.push(v[i]); }
  if(live.length < 3) return null;
  const med = median(live);
  const mad = median(live.map(x => Math.abs(x - med)));
  const sd = 1.4826 * mad;
  /* Zero dispersion means every assessed window moved identically, so nothing is
     an outlier. Reject nothing rather than divide by zero. */
  return {v, med, thr: sd > 0 ? med + (+st().motK) * sd : Infinity};
}

function motionMask(motion, n, excl){
  if(!motion || motion.length / MOT_COLS.length < n) return null;
  const K = MOT_COLS.length;

  /* One rule may consult several columns. Every part must be evaluable or the
     whole mask is refused -- a union silently falling back to one source would
     be a different criterion wearing the union's label. */
  const srcs = isMotUnion() ? MOT_SRC : [motSrc()];
  const parts = [];
  for(const s of srcs){
    const col = MC[s];
    if(col === undefined) return null;
    const p = motionOneSource(motion, n, col);
    if(!p) return null;
    parts.push(Object.assign({src: s}, p));
  }

  const keep = new Uint8Array(n);
  /* A NaN here has TWO possible causes and they are opposites, so counting them
     together produced a readout that was wrong about half the windows it
     described. motion_matrix NaNs a window when it has FEWER than
     motion_min_samples aux samples (the sensor did not cover it) OR when its
     wall-clock span runs past MOTION_MAX_SPAN_FACTOR x its nominal duration --
     which happens when the window straddles an interval step 2 excised, and
     which comes with FAR MORE aux samples than normal, not fewer. On
     p02/task_ai_speedscore one window straddles the 449 s excision and carries
     23,020 aux samples against a median of 203. Calling that "the sensor did not
     cover them" was the exact inverse of what happened, so the two are counted
     separately here using the n_aux_samples column, which the pipeline ships
     beside the values so the reason is visible rather than inferred. */
  const nax = MC["n_aux_samples"];
  let unassessed = 0, uncovered = 0, straddle = 0;
  for(let i=0;i<n;i++){
    /* A structurally excluded window is not eligible for this criterion and must
       not be tallied by it. Without this the p02 splice window is reported twice
       under two incompatible stories -- once as "excluded outright" and once as
       "NOT ASSESSED, kept for want of evidence" -- and the motion banner
       describes a window that is no longer on screen. keep[i]=0 is harmless:
       buildFrame and baselineStats both skip excluded indices before reading it. */
    if(excl && excl.has(i)){ keep[i] = 0; continue; }
    /* Unassessed if ANY part lacks a value. motion_matrix NaNs every source of a
       window together, so under one source this is the same set as before; the
       `some` is there so it stays true if that ever stops being the case. */
    if(parts.some(p => !isFinite(p.v[i]))){
      keep[i] = 1; unassessed++;                       // not assessed -> not rejected
      const na = nax === undefined ? NaN : motion[i*K + nax];
      if(isFinite(na) && na >= MOT_MIN) straddle++; else uncovered++;
    }
    /* UNION = reject if ANY source rejects, i.e. keep only where ALL agree. */
    else keep[i] = parts.every(p => p.v[i] <= p.thr) ? 1 : 0;
  }
  keep.unassessed = unassessed;
  keep.uncovered = uncovered;
  keep.straddle = straddle;
  keep.parts = parts.map(p => ({src: p.src, thr: p.thr, med: p.med}));
  /* Single scalars only where there IS a single source. Under the union they are
     null rather than one arbitrary part's value, so a readout that has not been
     taught about `parts` shows an em dash instead of a number that silently
     describes half the rule. */
  keep.threshold = parts.length === 1 ? parts[0].thr : null;
  keep.median = parts.length === 1 ? parts[0].med : null;
  return keep;
}

/* Why a window went unassessed, worded from the counts rather than assumed.
   Both causes can occur in one recording, so both are named when both are
   present -- and neither is described as the other. */
function motWhyShort(s){
  if(s.straddle && s.uncovered) return "excised-interval straddle / no sensor coverage";
  if(s.straddle) return "straddles an excised interval";
  return "no sensor coverage";
}
function motWhyLong(s){
  const parts = [];
  if(s.uncovered)
    parts.push(`${s.uncovered} had fewer than ${MOT_MIN} sensor samples inside them`);
  if(s.straddle)
    parts.push(`${s.straddle} ${s.straddle===1?"straddles":"straddle"} an interval step 2
      excised, so the window spans
      far more wall-clock than its own duration and the sensor samples between its first
      and last EEG sample cover recording that was deliberately cut out`);
  return parts.join("; ");
}

function maskFor(rec, mode, n, peaks, sigma, chans, slide, motion, excl){
  const rule = maskRule(mode);
  if(rule === "none") return null;
  if(rule === "motion") return motionMask(motion, n, excl);
  /* A short peak matrix reads past its end, and `undefined > threshold` is
     false, so every window would be retained and the page would report 100%
     kept. Refuse instead, and let the caller say the criterion could not be
     evaluated rather than that nothing was rejected. */
  if(!peaks || peaks.length / K < n) return null;
  const idx = chans.map(c => PC.indexOf(c));
  const win = isWindowed(mode);
  /* Same refusal for the windowed mode, and the same reason: no per-window sigma
     means no threshold, and silently falling back to the whole-recording sigma
     would label a DIFFERENT rule with this mode's name. The length check is part
     of it -- a short sigma matrix would read undefined and retain everything. */
  if(win && (!slide || slide.length / K < n)) return null;

  /* Windowed: the threshold is a straight lookup at this window's own row. The
     tiled version this replaced needed a block index, a midpoint rule to decide
     which block a straddling window belonged to, and a clamp for the folded
     trailing remainder. A centred sliding window has no block to belong to, so
     all of that is gone -- window i is judged against the sigma computed for
     window i. */
  const capThr = st().holmCap;
  const flat = (rule === "robust" && !win)
    ? chans.map(c => { const v = sigma ? sigma[c] : null;
                       return (v == null || !(v > 0)) ? Infinity : st().sigmaK * v; })
    : (rule === "robust" ? null : chans.map(() => capThr));

  const keep = new Uint8Array(n);
  const k4 = st().sigmaK;
  for(let i=0;i<n;i++){
    let ok = 1;
    for(let k=0;k<idx.length;k++){
      const col = idx[k];
      let thr;
      if(win){
        const sg = slide[i*K + col];
        thr = (sg > 0) ? k4 * sg : Infinity;
      } else {
        thr = flat[k];
      }
      if(peaks[i*K + col] > thr){ ok = 0; break; }
    }
    keep[i] = ok;
  }
  return keep;
}

/* Measure value per window, out of the 7-column matrix.

   The index numerator is the F1+F2 midline mean for every recording. It used to
   be selected per recording -- falling back to COL.theta_f1 or COL.theta_f2
   where the amplitude gate dropped a frontal channel -- and that selection went
   with the gate on 2026-09-02, along with the empty-set guard that went with it.
   Those two columns are still shipped but nothing reads them now. */
function measureVector(V, n, rec){
  const out = new Float64Array(n);
  if(tab === "index"){
    const c = COL.fm_theta;
    for(let i=0;i<n;i++){
      const d = V[i*W + COL.alpha_holm_pz];
      out[i] = d > 0 ? V[i*W + c] / d : NaN;
    }
    return out;
  }
  const c = COL[tab];
  for(let i=0;i<n;i++) out[i] = V[i*W + c];
  return out;
}

/* Median level of this recording's own baseline, in the SELECTED SEGMENT, under
   the CURRENT toggles -- including the sliders. Recomputed rather than looked
   up, so the baseline and the task it is subtracted from are always measured by
   exactly the same rule -- because subtracting a baseline built from a different
   derivation would not be a baseline correction of the same quantity. For the
   index that means the F1+F2 mean on both sides; before the gate's removal the
   baseline had to follow whichever frontal set the task's gate had selected.

   Every array here comes from the segment's own archive entry, so switching
   segment switches the peaks, the sigma and the window grid together. Mixing one
   segment's sigma with another's peaks would silently invent a fourth rule. */
function baselineStats(rec){
  const sp = baseSeg(rec);
  if(!sp) return null;
  const B = bValsOf(sp.key);
  const idx = sp.idx ? sp.idx[ekey()] : null;
  if(!B || !idx || !idx.length) return null;
  const n = B.length / W;
  const chans = maskChannels(rec);
  const vec = measureVector(B, n, rec);
  if(!vec) return null;
  /* The same rule as the task side. No baseline segment in this dataset
     contains a discontinuity -- the excision and the dropout are both inside
     task recordings -- but a baseline median must be computed by the same rule
     as the series it is subtracted from, or the two sides stop being the same
     quantity. That is the defect the 2026-09-04 pairing change existed to fix,
     one level up; leaving it out here would reintroduce it one level down. */
  const bexcl = exclSet(sp.excl);
  const keep = maskFor(rec, st().art, n, bPeaksOf(sp.key),
                       sp.sigma ? sp.sigma[st().oc] : null,
                       chans, bSlideOf(sp.key), bMotionOf(sp.key), bexcl);
  const v = [];
  for(const i of idx){
    if(i >= n) continue;
    if(bexcl && bexcl.has(i)) continue;
    if(keep && !keep[i]) continue;
    if(isFinite(vec[i])) v.push(vec[i]);
  }
  if(!v.length) return {median:null, n_windows:0, se:null, n_avail:idx.length};
  const med = median(v);
  /* SE(median) ~ 1.2533 * sigma / sqrt(n), sigma from the MAD so one wild
     window cannot set it. Identical to the pipeline's _median_dispersion.
     On the 15 s eyes-closed segment n is 3 at the 4 s default and 1 above it,
     so this is null or nearly meaningless there -- which is the point of
     reporting it beside the median rather than only the median. */
  let se = null;
  if(v.length > 1){
    const mad = median(v.map(x => Math.abs(x - med)));
    se = 1.2533 * (1.4826 * mad) / Math.sqrt(v.length);
  }
  return {median:med, n_windows:v.length, se, n_avail:idx.length};
}

/* ---------- per-render frame cache ----------
   render() touches every recording from four different renderers; without this
   the 1 s epoch would decode and re-mask 1,200 windows x 22 recordings x 4. */
let FRAME = new Map();
function frame(rec){
  let f = FRAME.get(rec.key);
  if(f === undefined){ f = buildFrame(rec); FRAME.set(rec.key, f); }
  return f;
}

/* Window indices this recording can never plot at the current epoch, as a Set.
   Their samples are not contiguous in real time, so their spectra are taken
   across a step in the signal. Excluded under EVERY artifact mode, `none`
   included -- "no rejection" means "do not judge the EEG", not "plot a window
   that is two recordings spliced together". See WINDOW_MAX_GAP_SEC. */
function exclSet(src){
  const a = src && src[ekey()];
  return (a && a.length) ? new Set(a) : null;
}

function buildFrame(rec){
  const V = valsOf(rec.key);
  if(!V) return null;
  const n = V.length / W;
  /* new Array(n) throws RangeError on a fractional n, which would escape
     render() and leave the page half-drawn. */
  if(!Number.isInteger(n) || n < 1) return null;
  const chans = maskChannels(rec);
  const vec = measureVector(V, n, rec);
  if(!vec) return null;

  /* Declared before maskFor, which consumes it. */
  const excl = exclSet(rec.excl);

  const keep = maskFor(rec, st().art, n, peaksOf(rec.key),
                       rec.sigma ? rec.sigma[st().oc] : null, chans,
                       slideOf(rec.key), motionOf(rec.key), excl);

  /* maskFor returns null for two DIFFERENT reasons, and conflating them is how a
     broken control comes to look like clean data: "no rejection was asked for"
     (mode none) and "the rejection could not be evaluated" (peaks too short, or
     no per-window sigma). Only the second is a refusal. Until 2026-09-02 the
     comment on maskFor claimed buildFrame drew this distinction; it did not, and
     a refused mask was reported as 100% kept. It is drawn here now, and every
     retention readout checks it. */
  const maskRefused = keep === null && maskRule(st().art) !== "none";

  /* Head motion leaves a window UNASSESSED when the IMU did not cover it. Such
     a window is kept -- there is no evidence against it -- but it has not passed
     anything, and counting it in "windows surviving rejection" would report an
     unmeasured window as a clean one. Carried onto the frame so every readout can
     say so; the same reasoning as maskRefused, one level down. */
  const unassessed = (keep && keep.unassessed) || 0;
  const uncovered = (keep && keep.uncovered) || 0;
  const straddle = (keep && keep.straddle) || 0;
  const motThr = keep && isFinite(keep.threshold) ? keep.threshold : null;
  const motParts = (keep && keep.parts) || null;
  const motMed = keep && keep.median != null ? keep.median : null;

  let vals = new Array(n);
  const imputed = new Array(n).fill(false);
  let kept = 0, eligible = 0;
  for(let i=0;i<n;i++){
    if(excl && excl.has(i)){ vals[i] = null; continue; }
    eligible++;
    const bad = (keep && !keep[i]) || !isFinite(vec[i]);
    vals[i] = bad ? null : vec[i];
    if(!bad) kept++;
  }
  /* Denominator is the ELIGIBLE windows, not every window. An excluded window
     was never offered to the criterion, so counting it as something the
     criterion discarded would attribute a splice to the rejection rule. */
  const retention = eligible ? 100 * kept / eligible : 0;
  /* n - eligible, not excl.size: the two agree today because the pipeline trims
     `discontinuous` to n, but a count derived from the loop that actually ran
     cannot drift from it. */
  const excluded = n - eligible;

  /* Imputation happens in the MEASURED domain, before any baseline transform,
     so the filled values reconstruct the quantity that was measured. Identical
     fill for both imputed modes -- only which windows were rejected first differs
     (absolute cap vs the robust per-recording threshold).

     RUN BY RUN, not once across the whole recording. An excluded window is a
     break in the time course, and interpolation must not reach across it. Doing
     the fill globally and then re-voiding the excluded index -- the first version
     of this -- looked right and was not: with windows [ok, EXCLUDED, rejected,
     ok] the global pass fills indices 1 and 2 from a straight line between 0 and
     3, the re-void clears 1, and index 2 keeps a value interpolated from the far
     side of a 449 s cut. The leading and trailing flat fills had the same reach.
     Segmenting first makes it structural: no fill can see past a break. */
  if(isImputed()){
    let s0 = 0;
    while(s0 < n){
      if(excl && excl.has(s0)){ s0++; continue; }
      let e0 = s0;
      while(e0 < n && !(excl && excl.has(e0))) e0++;      // run is [s0, e0)
      let last = -1;
      for(let i=s0;i<e0;i++){
        if(vals[i] != null){
          if(last >= 0 && i - last > 1){
            const a = vals[last], b = vals[i];
            for(let j=last+1;j<i;j++){ vals[j] = a + (b-a)*(j-last)/(i-last); imputed[j] = true; }
          } else if(last < 0 && i > s0){
            for(let j=s0;j<i;j++){ vals[j] = vals[i]; imputed[j] = true; }
          }
          last = i;
        }
      }
      if(last >= 0) for(let j=last+1;j<e0;j++){ vals[j] = vals[last]; imputed[j] = true; }
      s0 = e0;
    }
  }

  const f = {n, vals, imputed, chans, baseline:null, noBaseline:false, maskRefused,
             unassessed, uncovered, straddle, motThr, motMed, motParts,
             /* What the artifact criterion alone retained, before any baseline
                transform. Reported separately from `counts` because a log-ratio
                baseline drops any corrected value <= 0, so the two are genuinely
                different quantities and printing one as a percentage of the
                other was arithmetic nonsense.

                `retention` is meaningless when maskRefused -- no criterion ran --
                so read maskRefused first. */
             surviving: kept, retention, eligible, excluded};
  if(st().base !== "off"){
    const bm = baselineStats(rec);
    f.baseline = bm;
    if(!bm || bm.median == null){
      f.noBaseline = true;
      f.vals = new Array(n).fill(null);
    } else {
      const b = bm.median;
      if(st().base === "log" && M().kind !== "asym")
        f.vals = vals.map(v => (v == null || !(v > 0) || !(b > 0)) ? null : Math.log(v) - Math.log(b));
      else
        f.vals = vals.map(v => v == null ? null : v - b);
    }
  }

  let measured = 0, fab = 0;
  for(let i=0;i<n;i++){
    if(f.vals[i] == null) continue;
    imputed[i] ? fab++ : measured++;
  }
  f.counts = {measured, fabricated:fab, total:n};
  /* Percentage of what is actually PLOTTED, which is what the "x / y" beside it
     counts. `retention` above remains the rejection-rule figure. */
  f.plottedPct = n ? 100 * (measured + fab) / n : 0;
  return f;
}

/* Amplitudes of the data ACTUALLY PLOTTED, in the current interpolation,
   ocular and reference combination -- so the quality badge describes what is
   on screen rather than what step 5 saw. */
function displayedAmp(rec){
  /* A null token means "no array for this mode", not "fall back to the
     as-recorded amplitudes" -- returning rec.amp there hands back hardware-
     referenced figures while the badge beside them names the average reference.
     Guarded like its sibling valsOf. */
  const k = vk(rec.key);
  return (k !== null && rec.ampDisp && rec.ampDisp[k]) || rec.amp;
}

/* null means "no figure to give" -- either no frame at all, or the criterion
   could not be evaluated. Never report a refused mask as 100% kept. */
function retentionFor(rec){ const f = frame(rec);
  return (f && !f.maskRefused) ? f.retention : null; }
function maskRefused(rec){ const f = frame(rec); return !!(f && f.maskRefused); }

/* When does the sliding window stop being local? Only when EVERY window's span
   already covers the whole signal, and the spans are centred and truncated, not
   slid inwards. The first window is centred at epoch/2 and reaches epoch/2 + L/2;
   for it to reach the end needs T <= (L + epoch)/2, i.e. 2T - epoch <= L.

   The obvious test -- duration <= L -- is wrong by nearly a factor of two, and
   wrong in the unsafe direction: at L = 30 s and a 4 s epoch it would call a 28 s
   recording degenerate, when in fact its first window calibrates on 0-17 s and
   its last on 11-28 s, spans differing by more than half their length. */
function slideDegenerate(rec){
  const T = rec.duration_s || (counts(rec).total * st().epoch);
  return T ? (2*T - st().epoch) <= (+st().slideS) : false;
}

/* The microvolt range the windowed threshold actually took across a recording,
   per channel. One number would be a fiction -- the whole point of the mode is
   that it moves -- and the spread is what tells a reader how much to distrust
   retention under it. Returns null where the sigma matrix is unavailable. */
function slideRange(key, chans, isBase){
  const M2 = isBase ? bSlideOf(key) : slideOf(key);
  if(!M2) return null;
  const n = M2.length / K;
  return chans.map(c => {
    const col = PC.indexOf(c);
    let lo = Infinity, hi = -Infinity;
    for(let i=0;i<n;i++){ const v = M2[i*K + col];
      if(v > 0){ if(v < lo) lo = v; if(v > hi) hi = v; } }
    return isFinite(lo) ? [st().sigmaK*lo, st().sigmaK*hi] : null;
  });
}
function counts(rec){ const f = frame(rec); return f ? f.counts : {measured:0,fabricated:0,total:0}; }

/* ---------- per-recording signal quality, per measure ---------- */
function quality(rec){
  const chans = maskChannels(rec);
  const amp = displayedAmp(rec);
  const bad = chans.filter(c => rec.bad.includes(c));
  const done = interpChansOf(rec);
  const interp = chans.filter(c => done.includes(c));
  /* Flagged by detection and NOT replaced under this mode -- which now happens
     three ways, not one: interpolation off, the frontal pair held back, or a
     manual list that omits it. */
  const leftIn = bad.filter(c => !done.includes(c));
  /* Split by CAUSE, because the two are not the same kind of thing and only one
     of them is a fault.
       heldBack -- on the list this mode WOULD have replaced, and kept anyway
                   because the frontal checkbox is unticked.
       omitted  -- flagged by detection, and this mode never intended to replace
                   it (the manual list does not name it, or interpolation is off).
     frontalHeldBack() reads the SOURCE list, so under `manual` heldBack is
     non-empty only when the manual list itself asked for both frontals. */
  /* From the SOURCE list, not from `leftIn`. `leftIn` is built out of the
     DETECTOR's verdict, and under the manual list the operative judgement about
     which electrodes are bad is the list itself -- p14 ai names F1 and F2 while
     detection flags nothing there, so deriving this from `leftIn` silently
     dropped exactly the case the researcher had made a judgement about. */
  const srcList = (rec.interpLists && rec.interpLists[st().interpSrc]) || [];
  const heldBack = frontalHeldBack(rec)
    ? chans.filter(c => FRONTAL.includes(c) && srcList.includes(c)
                        && !done.includes(c))
    : [];
  const omitted  = leftIn.filter(c => !heldBack.includes(c));
  /* Which of those colours the badge. Under the manual list, disagreeing with
     the detector is the mode working as intended -- but a channel the manual
     list DID name and the checkbox then held back is broken-and-kept in the
     ordinary sense, so it still counts. Under `off` and `automatic`, everything
     left in counts. */
  const leftInFault = st().interpSrc === "manual" ? heldBack : leftIn;
  /* Replaced although detection never flagged it. Only `manual` can do this,
     and saying nothing about it would leave a reader thinking every spline on
     screen was earned by the detector. */
  const extra = interp.filter(c => !rec.bad.includes(c));
  const synth = interp.length > 0 && interp.length === chans.length;
  const impl = chans.filter(c => !interp.includes(c)
                              && !(amp[c] >= PL[0] && amp[c] <= PL[1]));
  const refTxt = st().ref === "hardware" ? "" : ` in the ${REF_LABEL[st().ref]} reference`;

  const parts = [];
  /* "both sites" only when there ARE two. The index tab reads three channels
     and parietal alpha one, and the sentence was printed for all of them. */
  if(synth) parts.push(`every channel this measure reads (${interp.join(", ")}) is a spline reconstruction from neighbouring electrodes${
    chans.length === 2 ? " — with both sites rebuilt from the same donors this value carries no information about this participant"
                       : " — nothing of this participant's own signal reaches this value"}`);
  else if(interp.length) parts.push(`${interp.join(", ")} was interpolated from neighbouring electrodes, so this measure is partly synthetic`);
  if(extra.length) parts.push(`${extra.join(", ")} was interpolated by the manual list although bad-channel detection did not flag it`);
  /* GROUPED by reason. A single parenthetical on the whole join printed one
     cause for a list that can carry two -- on p08 ai the frontal pair is held
     back AND P3 is simply absent from the manual list, and whichever branch won
     was asserted of both. */
  /* `heldBack` too, not just `leftIn`. On p14 ai the manual list names F1 and F2
     while detection flags nothing, so leftIn is empty and this block was skipped
     -- the badge said "Broken, kept" and nothing on the panel said why. */
  if(leftIn.length || heldBack.length){
    const held = heldBack, other = omitted;
    if(held.length) parts.push(st().interpSrc === "manual"
      ? `${held.join(" and ")} ${held.length>1?"are":"is"} on the manual list but ${held.length>1?"are":"is"} being read as recorded, because you asked for the ${FRONTAL.join(" and ")} pair to be held back`
      : `${held.join(" and ")} failed bad-channel detection and ${held.length>1?"are":"is"} being read as recorded, because you asked for the ${FRONTAL.join(" and ")} pair to be held back`);
    if(other.length){
      const many = other.length > 1;
      parts.push(st().interpSrc === "manual"
        ? `${other.join(", ")} ${many?"are":"is"} read as recorded: bad-channel detection flagged ${many?"them":"it"}, and the manual list you have selected does not replace ${many?"them":"it"}`
        : `${other.join(", ")} failed bad-channel detection and ${many?"are":"is"} being read as recorded (interpolation is off)`);
    }
  }
  if(impl.length) parts.push(`${impl.map(c=>`${c} ${fmt(amp[c])} µV`).join(", ")} outside ${BAND_TXT}${refTxt} — unusual for this dataset, which is not the same as impossible`);

  if(!parts.length)
    return {level:"good", text:"Measured", impl:[], interp:[],
      reason:`${chans.join(", ")} within ${BAND_TXT}${refTxt}, none interpolated`};

  /* Nothing is withheld for signal quality any more, so every badge says so. */
  const tail = " Nothing is withheld for it here.";
  /* CRITICALS FIRST, then warnings. This chain picks the label to show, so it
     has to be ordered by severity -- and it stopped being when "High amplitude"
     was demoted to a warning on 2026-09-10: it sat above "Broken, kept" and a
     recording that was BOTH out of band AND keeping a flagged electrode came
     out amber. Measured on p03 ai and p08 agent with the frontal pair held
     back, both of which are exactly that case. */
  if(synth)       return {level:"critical", text:"Fully synthetic", impl, interp, reason:parts.join("; ")+"."+tail};
  if(leftInFault.length)
                  return {level:"critical", text:"Broken, kept",    impl, interp, reason:parts.join("; ")+"."+tail};
  if(impl.length) return {level:"warning",  text:"High amplitude",  impl, interp, reason:parts.join("; ")+"."+tail};
  if(interp.length) return {level:"warning", text:"Partly synthetic", impl, interp, reason:parts.join("; ")+"."+tail};
  /* Nothing was interpolated, nothing is out of band, and the only remark is
     that the manual list disagrees with the detector. That is a description of
     the mode, not a defect, so it gets the neutral badge.
     `speak` carries the reason past the detail panel's `level !== "good"` guard,
     which otherwise drops it: demoting the badge would then have deleted the
     sentence naming the electrodes being read as recorded, which is the one
     thing a reader in this mode most needs to see. */
  return {level:"good", text:"Measured", speak:true, impl, interp,
          reason:parts.join("; ")+"."+tail};
}

/* Severity per condition, on the same scale the task badge uses, so that the
   same fact is the same colour wherever it appears: an amplitude outside the
   derived band is amber on the badge, so it is amber here. `missing` is the one
   critical kind and has no task equivalent -- with no usable baseline segment
   there is no denominator, so nothing on the panel is baseline-corrected at all.
   `few` and `noisy` say a real measurement is shaky, which is a caution rather
   than a verdict. */
const BASELINE_LEVEL = {missing:"critical", amp:"warning",
                        few:"warning", noisy:"warning"};

/* Everything wrong with the BASELINE being subtracted THAT THE TASK BADGE DOES
   NOT ALREADY SAY, worst first. Anything the two share -- interpolation always,
   amplitude when the task is out of band too -- belongs to the badge, not here. */
function baselineFlag(rec){
  if(st().base==="off") return null;
  const b = rec.base;
  if(!b || b.missing) return {kind:"missing", level:BASELINE_LEVEL.missing,
    text:b ? b.reason : "no baseline"};
  const sp = baseSeg(rec);
  if(!sp) return {kind:"missing", level:BASELINE_LEVEL.missing,
    text:`this baseline block has no ${segShort()} segment — its ${b.block_s}s span does not reach ${segBounds()[0]}–${segBounds()[1]}s`};
  const chans = maskChannels(rec);
  const f = frame(rec);
  const bm = f ? f.baseline : null;
  const lvl = segShort();

  /* INTERPOLATION IS NEVER REPORTED HERE, because it is never news. The
     bad-channel list is decided once per task/baseline pair, so the baseline
     replaces exactly the channels the task does -- the builder refuses to emit a
     panel where it does not (see the interpolation_tokens check beside `base`).
     The task badge already says "Partly synthetic" or "Fully synthetic" about
     those channels; a second chip beside it made one fact read as two problems.
     `bdone` is still needed to keep `impl` off replaced channels: an
     interpolated channel's amplitude is the spline's, not the electrode's. */
  const bdone = (b.interpLists && b.interpLists[interpMode()]) || [];
  const binterp = chans.filter(c => bdone.includes(c));
  /* AMPLITUDE ONLY WHEN THE TASK IS NOT ALREADY SAYING IT. quality() hands back
     the channels it flagged so this can subtract them, rather than re-deriving
     the out-of-band test here and drifting away from it. Suppressed wholesale
     rather than per channel: the reader has already been told this recording's
     amplitudes are out of band, and which segment it happened in is a detail the
     panel note still carries. */
  const tq = quality(rec);
  const impl = (tq.impl || []).length ? []
             : chans.filter(c => !binterp.includes(c) && (sp.implausible||[]).includes(c));
  /* Two different "too few windows" facts, and only the first is a property of
     THIS recording:

       thin   the segment had windows and the rejection rule took most of them.
              True of any segment, and the only signal that distinguishes a
              15 s baseline that survived intact from one reduced to a single
              window -- which the earlier draft of this hid completely.
       few    fewer than 10 windows on a segment long enough to have had 10.
              Suppressed on a short-by-design segment (eyes_closed is 15 s and
              can hold at most 3 windows at the 4 s default), because there it
              would fire on every recording and say nothing about any of them. */
  const canBeFew = segLen() >= 60;
  const few = canBeFew && bm && bm.n_windows != null && bm.n_windows < 10;
  const thin = bm && bm.n_windows != null && bm.n_avail != null
               && bm.n_avail > 1 && bm.n_windows < bm.n_avail
               && (bm.n_windows === 1 || bm.n_windows < 0.5 * bm.n_avail);
  const noisy = bm && bm.se != null && bm.median != null && Math.abs(bm.median) > 0
                && bm.se > 0.25*Math.abs(bm.median);

  const parts = [];
  if(impl.length)   parts.push(`baseline ${impl.map(c=>`${c} ${fmt(sp.amp[c])} µV`).join(", ")} outside ${BAND_TXT} in the ${lvl} segment`);
  if(few)           parts.push(`${lvl} median rests on only ${bm.n_windows} window${bm.n_windows===1?"":"s"}`);
  else if(thin)     parts.push(`the rejection rule cut this ${lvl} baseline to ${bm.n_windows} of its ${bm.n_avail} window${bm.n_avail===1?"":"s"}${bm.n_windows===1?" — the median IS that one window":""}`);
  if(noisy)         parts.push(`${lvl} median is unstable (± ${fmt(bm.se)} on ${fmt(bm.median)})`);
  if(!parts.length) return null;
  const kind = impl.length ? "amp" : (few || thin) ? "few" : "noisy";
  return {kind, level:BASELINE_LEVEL[kind], text:parts.join("; ")};
}

/* ---------- chart ---------- */
function niceTicks(lo, hi, log){
  if(log){
    const t=[];
    for(let e=Math.floor(Math.log10(lo)); e<=Math.ceil(Math.log10(hi)); e++)
      for(const m of [1,2,5]) t.push(m*Math.pow(10,e));
    const inr = t.filter(v=>v>=lo*0.99 && v<=hi*1.01).sort((a,b)=>a-b);
    return inr.length>=2 ? inr : [lo,hi];
  }
  const span=hi-lo, step=Math.pow(10,Math.floor(Math.log10(span/4)));
  const mult=[1,2,2.5,5,10].find(m=>span/(step*m)<=5)||10, s=step*mult;
  const t=[]; for(let v=Math.ceil(lo/s)*s; v<=hi; v+=s) t.push(+v.toFixed(6));
  return t;
}

/* `opts.series` draws something other than the task's corrected trace -- the
   baseline viewer passes a baselineSeries. `opts.log`, `opts.corrected`, `opts.xAt`
   and `opts.xFmt` let that caller state its own axis rules, because a raw
   baseline level and a log-ratio-corrected task are different quantities and
   inheriting the task's would mislabel both. */
function drawChart(rec, opts){
  const {w,h,large} = opts;
  const ser = opts.series || frame(rec);
  const pad = large ? {t:14,r:16,b:26,l:60} : {t:8,r:8,b:16,l:42};
  const iw = w-pad.l-pad.r, ih = h-pad.t-pad.b;

  const blank = msg => `<svg viewBox="0 0 ${w} ${h}" width="100%" height="${h}" role="img"
      aria-label="${esc(msg)}"><text x="${w/2}" y="${h/2}" text-anchor="middle"
      class="${large?'tick-lg':'tick'}">${esc(msg)}</text></svg>`;

  if(!ser) return blank("not available in this combination");
  if(ser.missing) return blank(ser.reason);
  if(ser.noBaseline) return blank(`no ${segShort()} baseline for this recording`);

  const log = opts.log !== undefined ? opts.log : effScale()==="log";
  /* Whether zero is a meaningful line on this chart. True for a corrected task
     (zero = matched the baseline) and for the asymmetries (zero = sides equal);
     false for a raw baseline level, where zero is just the bottom of a power
     axis and drawing it would invent a reference point. */
  const corrected = opts.corrected !== undefined ? opts.corrected
                                                 : (st().base!=="off");
  const vals = ser.vals, imputed = ser.imputed;
  const finite = vals.filter(v=>v!=null && isFinite(v) && (!log || v>0));
  if(!finite.length) return blank("no window survives this criterion");

  let lo = Math.min(...finite), hi = Math.max(...finite);
  if(hi===lo){ hi = lo + (Math.abs(lo)*0.1 || 1); lo = lo - (Math.abs(lo)*0.1 || 1); }
  if(log){ const p=Math.max((Math.log10(hi)-Math.log10(lo))*0.08,0.05);
    lo=Math.pow(10,Math.log10(lo)-p); hi=Math.pow(10,Math.log10(hi)+p); }
  else { const p=Math.max((hi-lo)*0.08,1e-9); lo=lo-p; hi=hi+p;
    if(corrected || M().kind==="asym"){ lo=Math.min(lo,0); hi=Math.max(hi,0); } }

  const X = i => pad.l + (vals.length<2?0:i/(vals.length-1))*iw;
  const Y = v => { const t = log ? (Math.log10(v)-Math.log10(lo))/(Math.log10(hi)-Math.log10(lo))
                                 : (v-lo)/(hi-lo); return pad.t + ih - Math.max(0,Math.min(1,t))*ih; };
  const usable = i => { const v=vals[i]; return v!=null && isFinite(v) && (!log || v>0); };

  /* Split into runs of consecutive usable points, tagged measured vs imputed.
     The segment joining a measured window to an interpolated one is drawn EXACTLY
     ONCE, and always by the imputed run, so it takes the dashed style. */
  const runs=[]; let cur=null;
  for(let i=0;i<vals.length;i++){
    if(!usable(i)){ cur=null; continue; }
    const kind = imputed[i] ? "imp" : "meas";
    if(!cur || cur.kind!==kind){
      const bridge = usable(i-1);
      if(cur && bridge && cur.kind==="imp") cur.pts.push([X(i),Y(vals[i])]);
      const started={kind,pts:[]}; runs.push(started);
      if(bridge && kind==="imp") started.pts.push([X(i-1),Y(vals[i-1])]);
      cur=started;
    }
    cur.pts.push([X(i),Y(vals[i])]);
  }

  const ticks = niceTicks(lo,hi,log);
  const g  = ticks.map(v=>`<line class="grid-line" x1="${pad.l}" y1="${Y(v).toFixed(1)}" x2="${w-pad.r}" y2="${Y(v).toFixed(1)}"/>`).join("");
  const tk = ticks.map(v=>`<text class="${large?'tick-lg':'tick'}" x="${pad.l-6}" y="${(Y(v)+3.2).toFixed(1)}" text-anchor="end">${fmt(v)}</text>`).join("");

  const zero = (!log && lo<=0 && hi>=0 && (corrected || M().kind==="asym"))
    ? `<line class="zeroline" x1="${pad.l}" y1="${Y(0).toFixed(1)}" x2="${w-pad.r}" y2="${Y(0).toFixed(1)}"/>` : "";

  const med = median(vals);
  const medl = med!=null && (!log||med>0) ? `<line class="medline" x1="${pad.l}" y1="${Y(med).toFixed(1)}" x2="${w-pad.r}" y2="${Y(med).toFixed(1)}"/>` : "";

  const base = (!log && lo<=0 && hi>=0) ? Y(0) : (pad.t+ih);
  const area = runs.filter(r=>r.kind==="meas" && r.pts.length>1).map(r=>{
    const d = r.pts.map((p,i)=>`${i?'L':'M'}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join("");
    return `<path class="area" d="${d}L${r.pts[r.pts.length-1][0].toFixed(1)},${base.toFixed(1)}L${r.pts[0][0].toFixed(1)},${base.toFixed(1)}Z"/>`;
  }).join("");

  const lines = runs.filter(r=>r.pts.length>1).map(r=>{
    const d = r.pts.map((p,i)=>`${i?'L':'M'}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join("");
    return `<path class="trace ${large?'trace-lg':''} ${r.kind==='imp'?'trace-imputed':''}" d="${d}"/>`;
  }).join("");

  const dots = runs.filter(r=>r.pts.length===1).map(r=>
    `<circle cx="${r.pts[0][0].toFixed(1)}" cy="${r.pts[0][1].toFixed(1)}" r="${large?2.4:1.7}"
       fill="${r.kind==='imp'?'var(--ink-muted)':'var(--series)'}"/>`).join("");

  /* Axis labels come from the RECORDED window times, not from spreading the
     total duration evenly over window indices: p02/task_ai_speedscore has a
     449 s excised splice, which the even-spacing assumption puts 4 minutes
     wrong. Times are per epoch length, because the grid moves with it. */
  let xax = "";
  if(large){
    const times = rec.times ? rec.times[ekey()] : null;
    const total = rec.elapsed_s || (vals.length-1)*st().epoch;
    const at = opts.xAt ||
      (i => (times && times[i]!=null) ? times[i] : total*i/Math.max(1,vals.length-1));
    const xf = opts.xFmt || (t => `${(t/60).toFixed(0)} min`);
    /* Deduped, and placed at X(i) rather than at the tick fraction. A short
       series repeats indices -- the 15 s eyes-closed segment holds ONE window at
       an 8 s epoch, which printed the same label five times across the axis with
       the single point drawn at the far left. Distinct indices only, each label
       over the point it names. */
    const tix = [...new Set([0,0.25,0.5,0.75,1]
      .map(f => Math.round(f*(vals.length-1))))];
    xax = tix.map((i,k)=>{
      const anchor = tix.length===1 ? 'start'
                   : k===0 ? 'start' : k===tix.length-1 ? 'end' : 'middle';
      return `<text class="tick-lg" x="${X(i).toFixed(1)}" y="${(h-8).toFixed(1)}" text-anchor="${anchor}">${xf(at(i))}</text>`;
    }).join("");
  }

  return `<svg viewBox="0 0 ${w} ${h}" width="100%" height="${h}" preserveAspectRatio="none"
      role="img" aria-label="${esc(opts.label || (MEAS[tab].tab + " over the task"))}"
      data-n="${vals.length}" data-padl="${pad.l}" data-padr="${pad.r}">
    ${g}${medl}${zero}${area}${lines}${dots}
    <line class="axis-line" x1="${pad.l}" y1="${pad.t+ih}" x2="${w-pad.r}" y2="${pad.t+ih}"/>
    ${tk}${xax}
  </svg>`;
}

const ICON = {
  good:'<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 8.5l3.2 3.2L13 5"/></svg>',
  warning:'<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M8 3.5v5.2"/><circle cx="8" cy="12.2" r="1.1" fill="currentColor" stroke="none"/></svg>',
  critical:'<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M4 4l8 8M12 4l-8 8"/></svg>'
};

/* ---------- rendering ---------- */

function renderHead(){
  const m = M();
  $("#mast-title").textContent = m.title;
  $("#mast-formula").innerHTML = m.formula +
    ` <span>&middot; ${st().epoch}&nbsp;s windows &middot; ${FFT_LABEL[st().fft]} taper</span>`;
  /* No pointer to an analysis-decisions file here. This page is built from
     whatever tree it is given, and the document that recorded those decisions
     for the original study does not travel with this template -- naming it put
     a dead reference in the masthead of every build. */
  $("#mast-lede").innerHTML = m.lede;
  $("#mast-eyebrow").textContent =
    `OpenBCI Galea · ${DATA.n_participants} participants · ${DATA.recordings.length} task recordings`;
  $("#tabs").innerHTML = ORDER.map(id=>
    `<button type="button" role="tab" ${tab===id?'id="tab-selected"':""} data-tab="${id}" aria-selected="${tab===id}" aria-controls="tabpanel" tabindex="${tab===id?0:-1}">${esc(MEAS[id].tab)}</button>`).join("");
}

function segHTML(label, id, items, key, disabled, current, describedBy){
  const sel = current===undefined ? st()[key] : current;
  const desc = describedBy ? ` aria-describedby="${describedBy}"` : "";
  /* radiogroup/radio rather than group + aria-pressed: these choices are
     mutually exclusive, and aria-pressed announced them as four independent
     toggles. */
  return `<div class="ctl${disabled?" off":""}"><span class="ctl-label" id="lbl-${id}">${label}</span>
    <div class="seg" role="radiogroup" aria-labelledby="lbl-${id}"${desc}>${
      items.map(i=>`<button type="button" role="radio" data-k="${key}" data-v="${i.id}"
        aria-checked="${sel===i.id}" aria-pressed="${sel===i.id}"${desc} ${disabled?'disabled aria-disabled="true"':""}>${i.label}</button>`).join("")
    }</div></div>`;
}

/* A real checkbox, not a two-position segmented control: this is one boolean
   question about the control beside it, and rendering it as On/Off buttons put
   it in the same visual class as the mutually-exclusive choices, which it is
   not. `hidden` rather than `disabled` when interpolation is off -- a disabled
   checkbox invites the reader to work out why it is disabled, when the answer
   is that the question does not arise. */
function checkHTML(label, key, text, hidden, help){
  if(hidden) return "";
  const on = !!st()[key];
  return `<div class="ctl ctl-tip"><span class="ctl-label" id="lbl-${key}">${label}</span>
    <label class="chk" tabindex="-1"><input type="checkbox" data-c="${key}" ${on?"checked":""}
      aria-describedby="help-${key}">
      <span>${text}</span><span class="chk-q" aria-hidden="true">?</span></label>
    <div class="chk-help" id="help-${key}" role="tooltip">${help}</div></div>`;
}

function rangeHTML(label, id, key, spec, fmtv, disabled){
  return `<div class="ctl${disabled?" off":""}"><span class="ctl-label" id="lbl-${id}">${label}</span>
    <div class="slider"><input type="range" id="rng-${id}" data-r="${key}"
      min="${spec.min}" max="${spec.max}" step="${spec.step}" value="${st()[key]}"
      aria-labelledby="lbl-${id}" ${disabled?"disabled":""}>
      <span class="val" id="val-${id}" aria-live="polite">${fmtv(st()[key])}</span></div></div>`;
}

/* =======================================================================
   Export what is on screen.

   The file is the PLOTTED series -- after the artifact criterion, after
   imputation, after the baseline transform -- because that is what "the data
   shown" means, and a reader who wanted the raw sweep can take the .npz. Every
   control that shaped those numbers is written beside them, so a file can be
   reproduced without remembering how the page was set. Nulls are preserved
   rather than dropped: position in the array is the window index, and a
   compacted array would silently renumber the time course.
   ======================================================================= */

function exportPayload(){
  const m = M(), recs = candidates(), s = st();
  const ruleNow = maskRule(s.art);
  /* The archive is float32, so beyond about 7 significant digits every further
     figure is conversion noise from the widening to float64. Publishing 17 of
     them would present that noise as measurement, and doubles the file. */
  const q = v => (v == null || !isFinite(v)) ? null : +v.toPrecision(7);
  const rows = [];
  let unavailable = 0;
  for(const r of recs){
    const f = frame(r);
    /* A recording whose measure cannot be built in this combination gets a panel
       on screen saying so. Dropping it silently here would make two exports of
       the same tab differ in row count for a reason the file does not contain. */
    if(!f){ unavailable++;
      rows.push({key:r.key, participant:r.participant, recording:r.recording,
                 arm:r.arm, framing:r.framing, available:false,
                 reason:"this measure is not available in this combination"});
      continue; }
    const sp = baseSeg(r);
    const times = (r.times && r.times[ekey()]) || null;
    const excl = exclSet(r.excl);
    /* No criterion ran at all under `none`, so there is no retention figure to
       give. Reporting 100% here would be the same false claim the page itself
       was corrected for in 2026-09-02: a rule that never ran, described as one
       that kept everything. */
    const noRule = ruleNow === "none" || f.maskRefused;
    rows.push({
      key: r.key, participant: r.participant, recording: r.recording,
      arm: r.arm, framing: r.framing,
      available: true,
      n_windows: f.n,
      /* Refuse a short times array rather than truncating into a silently
         misaligned one -- index position is the schema's whole contract. */
      times_s: (times && times.length >= f.n) ? times.slice(0, f.n) : null,
      values: Array.from(f.vals, q),
      /* A window imputed before the baseline transform can be nulled by it
         (no usable baseline, or a non-positive log ratio). Reporting it as
         imputed would contradict counts.imputed, which is counted afterwards. */
      imputed: f.imputed ? Array.from(f.imputed, (b,i)=> !!b && f.vals[i] != null)
                         : null,
      excluded_windows: excl ? [...excl].filter(i=>i < f.n).sort((a,b)=>a-b) : [],
      counts: {
        measured: f.counts.measured, imputed: f.counts.fabricated,
        plotted: f.counts.measured + f.counts.fabricated,
        eligible: f.eligible, excluded_outright: f.excluded, total: f.counts.total
      },
      criterion_evaluated: !noRule,
      criterion_retention_pct_of_eligible: noRule ? null : q(f.retention),
      windows_not_assessed: f.unassessed || 0,
      windows_not_assessed_no_sensor_coverage: f.uncovered || 0,
      windows_not_assessed_spans_discontinuity: f.straddle || 0,
      /* What the toggle WOULD replace, and what it actually did. With
         interpolation off nothing is synthetic, and a consumer filtering on
         "has interpolated channels" would otherwise get the wrong answer. */
      bad_channels_detected: r.interpChans || [],
      /* What this mode ACTUALLY replaced, which since 2026-09-08 is not always
         the detected list: the manual mode can interpolate a channel detection
         never flagged, and can leave a flagged one alone. A consumer filtering
         on "has interpolated channels" needs this one, not the one above. */
      channels_interpolated: interpChansOf(r),
      /* Both, because `quality` is judged on the amplitudes AS PLOTTED in the
         current reference, which are not the as-recorded ones once the
         reference is average or REST. */
      amplitude_uv_as_recorded: r.amp,
      amplitude_uv_as_plotted: displayedAmp(r),
      quality: quality(r).level,
      quality_label: quality(r).text,
      baseline: (s.base === "off") ? null : {
        segment: segOf(),
        segment_bounds_s: sp ? sp.bounds : null,
        median: f.baseline ? q(f.baseline.median) : null,
        n_windows: f.baseline ? f.baseline.n_windows : null,
        se: f.baseline ? q(f.baseline.se) : null,
        usable: !f.noBaseline
      }
    });
  }
  return {
    exported_utc: new Date().toISOString(),
    schema: "galea-dashboard-export/1",
    source: {
      pipeline_run_utc: (DATA.run_meta && DATA.run_meta.generated_utc) || null,
      bad_channel_detection: (DATA.run_meta && DATA.run_meta.bad_channel_detection)
                             || null,
      note: "Values are as plotted: criterion applied, imputation applied, "
          + "baseline transform applied. The unprocessed sweep is in "
          + "outputs/variants/*.npz. Values are rounded to 7 significant "
          + "figures, the precision of the float32 archive.",
      retention_note: "criterion_retention_pct_of_eligible is the artifact "
          + "criterion's own figure, measured BEFORE the baseline transform, "
          + "and counts a window the motion rule could not assess as kept. "
          + "counts.plotted is the post-transform figure, so the two do not "
          + "have to agree."
    },
    measure: {
      id: tab, label: m.tab, title: m.title, kind: m.kind,
      channels: m.channels,
      unit: st().base === "off" ? m.unit
          : (st().base === "log" && m.kind !== "asym")
            ? "ln ratio to baseline (dimensionless)"
            : m.unit + " difference from baseline"
    },
    settings: {
      /* Both halves of the control, plus the swept mode they compose into, so
         a consumer can round-trip the selection without re-deriving it. */
      interpolation: interpMode(),
      interpolation_source: s.interpSrc,
      /* "on the list", not "flagged": under the manual source the list is the
         user's own and need not agree with detection at all. */
      interpolate_frontal_pair_when_both_on_the_list: s.interpSrc === "off"
        ? null : !!s.frontalPair,
      frontal_pair: FRONTAL,
      ocular: s.oc, reference: s.ref,
      spectral_estimator: s.fft, window_length_s: s.epoch,
      artifact_rejection: s.art,
      robust_sigma_k: maskRule(s.art) === "robust" ? s.sigmaK : null,
      /* Coerced: the segmented control writes a string, so an export taken
         after touching it would not compare equal to one taken before. */
      robust_slide_s: isWindowed() ? +s.slideS : null,
      absolute_cap_uv: maskRule(s.art) === "cap" ? s.holmCap : null,
      motion_source: isMotion() ? s.motSrc : null,
      motion_sigma_k: isMotion() ? s.motK : null,
      baseline_correction: s.base,
      baseline_segment: s.base === "off" ? null : segOf(),
      scale: effScale()
    },
    window_exclusion: {
      rule: "windows whose samples are not contiguous in real time are dropped "
          + "under every artifact mode, including none",
      max_gap_s: DATA.window_max_gap_s != null ? DATA.window_max_gap_s : null
    },
    n_recordings: rows.length,
    n_recordings_unavailable: unavailable,
    recordings: rows
  };
}

function exportFilename(){
  const s = st();
  const bits = [tab, s.epoch + "s", s.fft, s.ref, "interp-" + interpMode(),
                "oc-" + s.oc, "art-" + s.art];
  /* The CONTINUOUS controls belong in the name too. Without them, two exports
     at robust k=2.5 and k=7.0 differ in which windows are null yet produce
     names identical but for the timestamp -- which defeats the point of naming
     a file after its settings. Only the ones that apply to the current mode are
     appended, matching how exportPayload nulls the inapplicable ones. */
  const num = v => String(v).replace(".", "p");
  if(maskRule(s.art) === "robust") bits.push("k" + num(s.sigmaK));
  if(isWindowed())                 bits.push("slide" + num(s.slideS) + "s");
  if(maskRule(s.art) === "cap")    bits.push("cap" + Math.round(s.holmCap) + "uv");
  if(isMotion())                   bits.push(s.motSrc, "k" + num(s.motK));
  bits.push("base-" + (s.base === "off" ? "off" : s.base + "-" + segOf()));
  /* Keep the Z. Slicing to seconds dropped the only marker saying this is UTC
     rather than the reader's local time, and the filename is what people sort
     by. Milliseconds stay so two clicks in one second cannot collide. */
  const t = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 23) + "Z";
  return "galea_" + bits.join("_").replace(/[^A-Za-z0-9_.-]/g, "") + "_" + t + ".json";
}

/* The status goes in a live region rather than into the button's own label.
   A button's accessible name IS its text, so rewriting it renames the control
   -- for a few seconds the page holds a button called "Downloaded" -- and an
   accessible-name change on an already-focused element is not reliably
   announced. The page already uses this pattern for #shard-state. */
let exportTimer = null;
function setExportStatus(msg){
  const el = document.getElementById("export-status");
  if(el) el.textContent = msg;
}

function doExport(btn){
  let url = null, a = null, ok = false;
  clearTimeout(exportTimer);
  setExportStatus("");
  /* candidates() is empty while a window-length shard is still loading, and a
     well-formed file with nothing in it is worse than no file. */
  if(!candidates().length){
    setExportStatus("Nothing to export yet — the data for this window "
                  + "length is still loading. Try again in a moment.");
    return;
  }
  try{
    const json = JSON.stringify(exportPayload(), null, 2);
    const blob = new Blob([json], {type:"application/json"});
    url = URL.createObjectURL(blob);
    a = document.createElement("a");
    a.href = url; a.download = exportFilename();
    document.body.appendChild(a);
    a.click();
    ok = true;
  }catch(err){
    console.error(err);
    /* Name a fix the reader can act on. This page is opened from disk, and the
       one thing that reliably works when a browser declines a blob download
       from a file:// document is serving the folder over HTTP. */
    setExportStatus("Export failed. Try serving this folder "
                  + "(python -m http.server) and opening it over http://. "
                  + "Details in the browser console.");
  }finally{
    if(a && a.parentNode) a.parentNode.removeChild(a);
    /* Revoking immediately can cancel the download in some browsers. */
    if(url) setTimeout(()=>URL.revokeObjectURL(url), 30000);
    if(ok){
      /* "Download started", not "Downloaded". a.click() does not throw when a
         browser silently declines the download, so success cannot be observed
         from here -- and the catch above can never see the very failure its
         message describes. Claiming a completed download would be asserting
         something this code has no way to know. */
      setExportStatus("Download started \u2014 check your downloads folder.");
      exportTimer = setTimeout(()=>setExportStatus(""), 6000);
    }
    /* A failure message is left up until the next click: one that clears itself
       after a couple of seconds is gone before it can be acted on. */
  }
}

function renderControls(){
  const m = M();

  const row1 =
      segHTML("Bad-channel interpolation","interp",INT,"interpSrc",false)
    /* The second, independent question, so it is its own control rather than
       two more buttons in the row above. Hidden when interpolation is off:
       there is nothing to hold back, and a live checkbox that changes nothing
       is worse than an absent one. */
    + checkHTML("Frontal pair",
                "frontalPair",
                `Interpolate ${FRONTAL.join(" and ")} when <b>both</b> are on the list`,
                st().interpSrc === "off",
                `Unchecked, ${FRONTAL.join(" and ")} are left exactly as recorded and the
                 rest of the list is still interpolated. Only cases where BOTH are on
                 the list are affected &mdash; with one of them flagged this changes
                 nothing, because one frontal channel rebuilt from nine others,
                 its partner included, is an ordinary interpolation.
                 <b>Why you might uncheck it:</b> ${FRONTAL.join(" and ")} are the whole
                 numerator of the cognitive-load index and the two sides of frontal
                 alpha asymmetry. Rebuilding both at once reconstructs the frontal
                 signal entirely from central, parietal and occipital electrodes,
                 and rebuilds both sites from the same donors &mdash; which drives
                 the asymmetry towards zero by construction.`)
    + segHTML("Ocular correction","oc",OC,"oc",false)
    + segHTML("Reference","ref",REF,"ref",false)
    + segHTML("Spectral estimator","fft",FFT,"fft",false)
    /* Indexed into DATA.epochs_s rather than using the value as the position,
       so a non-contiguous EPOCH_CHOICES_S (say [1,2,4,8]) cannot offer the user
       a window length the pipeline never swept. */
    + rangeHTML("Window length","epoch","epochIdx",
                {min:0, max:DATA.epochs_s.length-1, step:1},
                i=>DATA.epochs_s[i]+" s", false);

  let row2 = segHTML("Artifact rejection","art",ART,"art",false);
  if(maskRule(st().art)==="robust")
    row2 += rangeHTML("Robust distance","sig","sigmaK",SL.robust_sigma,
                      v=>(+v).toFixed(2)+" × σ", false);
  /* Shown only for the windowed mode, because it means nothing to the other two
     -- unlike the baseline-segment control, which is disabled rather than hidden
     because a reader needs to know the choice exists even when it is inactive.
     Here the whole mode is the choice, so the control appears with it. */
  if(isWindowed())
    /* NOT "Window length": the epoch slider one row above is already called that,
       and the two mean different things -- that one sets what a plotted point
       covers, this one how much signal its threshold is calibrated on. Two
       controls with the same visible name also give assistive tech two
       radiogroups with the same announced label. */
    row2 += segHTML("Calibration window","slid",
                    SLIDE_CHOICES.map(v=>({id:String(v),label:`${+v} s`})),
                    "slideS", false, String(+st().slideS));
  if(maskRule(st().art)==="cap")
    row2 += rangeHTML("Absolute cap","cap","holmCap",SL.holm_cap_uv,
                      v=>(+v).toFixed(0)+" µV", false);

  if(isMotion()){
    row2 += rangeHTML("Motion threshold","mot","motK",MOT_K,
                      v=>(+v).toFixed(2)+" × σ", false);
    if(MOT_RULES.length > 1)
      row2 += segHTML("Motion source","msrc",
                      MOT_RULES.map(x=>({id:x,label:MOT_LABEL[x]||x})),
                      "motSrc", false);
  }
  row2 += segHTML("Baseline correction","base",baseOpts(),"base",false);
  /* Which segment of the baseline block is subtracted. Only meaningful while a
     correction is on, so it is DISABLED rather than hidden when it is off --
     hiding it would make the row jump and would hide the fact that a choice
     exists at all. */
  {
    const segOpts = SEG_ORDER.map(s => ({id:s, label:segLabel(s)}));
    const off = st().base === "off";
    row2 += segHTML("Baseline segment","bseg",segOpts,"baseSeg",off,
                    st().baseSeg, off?"bseg-note":null);
    if(off)
      row2 += `<div class="ctl"><span class="ctl-label">&nbsp;</span>
        <div class="ctl-note" id="bseg-note">no segment is subtracted while baseline
        correction is off</div></div>`;
  }
  row2 += segHTML("Scale","sc",SC,"scale",!logAllowed(),effScale(),logAllowed()?null:"scale-note");
  if(!logAllowed())
    row2 += `<div class="ctl"><span class="ctl-label">&nbsp;</span><div class="ctl-note" id="scale-note">${
      m.kind==="asym"
        ? "log axis undefined: values are already ln differences and go negative"
        : "log axis undefined: a baseline-corrected value can be negative"
    }</div></div>`;
  $("#controls").innerHTML = `<div class="ctl-row">${row1}</div><div class="ctl-row">${row2}</div>`;
}

/* =======================================================================
   Plain-English description of what is currently on screen.
   ======================================================================= */

const PLAIN_MEASURE = {
  index:
    `how much stronger the slow <b>theta</b> waves over the forehead were than the
     <b>alpha</b> waves at the back of the head. Holm and colleagues proposed this
     ratio as an index of mental workload, so a higher point means that, by this
     measure, the person was working harder during that window`,
  fm_theta:
    `how much <b>theta</b>-band activity there was over the middle of the forehead.
     Theta means slow waves, 4 to 8 cycles per second. The two electrodes either
     side of the midline, F1 and F2, are averaged into a single signal first and
     then measured as one`,
  parietal_alpha:
    `how much <b>alpha</b>-band activity there was at Pz, the electrode on top of the
     head towards the back. Alpha means waves of 8 to 13 cycles per second, which
     tend to be strongest when someone is relaxed and not concentrating hard`,
  parietal_beta_asym:
    `how much more <b>beta</b>-band activity there was on the right side at the back
     of the head (P4) than on the left (P3). Beta means fast waves, 13 to 30 cycles
     per second. A point at zero means the two sides matched; above zero means the
     right side was stronger, below zero means the left side was`,
  frontal_alpha_asym:
    `how much more <b>alpha</b>-band activity there was on the right side of the
     forehead (F2) than on the left (F1). Alpha means waves of 8 to 13 cycles per
     second. A point at zero means the two sides matched; above zero means the
     right side was stronger, below zero means the left side was`
};

function plainArtifact(){
  const a = st().art;
  const sig = (+st().sigmaK).toFixed(2);
  /* Shared tail for both imputed modes. */
  const fill = ` The holes the discarded windows leave are then filled by drawing a straight line
      between the surviving points on either side; runs at the very start or end, which have nothing
      to interpolate between, are filled flat with the nearest surviving value. <b>Dashed segments are
      invented numbers</b>, not measurements &mdash; do not read their shape as a time course.`;

  if(a==="none")
    return `<b>Artifact rejection is off</b>, so every window the recording can offer is drawn
      &mdash; including windows where the electrode was plainly picking up movement, muscle
      tension or electrical noise rather than brain activity. Useful for seeing where the bad
      stretches are; not a safe basis for reading the measure. One thing is still withheld
      here, in two recordings: a window whose own samples are not contiguous in real time. Its
      spectrum would be taken across a step in the signal rather than across anything an
      electrode measured, so it is dropped in every mode including this one.`;

  if(isMotion(a))
    return `<b>Head-motion rejection is on at ${(+st().motK).toFixed(2)}&nbsp;&sigma;.</b> This one
      does not look at the EEG at all. The helmet carries an inertial sensor, and this rule throws
      away any window during which the head was moving unusually much <em>for this recording</em> --
      more than ${(+st().motK).toFixed(2)} robust standard deviations above that recording's own
      typical movement, measured from the ${
        isMotUnion() ? "accelerometer OR the gyroscope &mdash; whichever flags it first, "
                     + "each against its own threshold"
                     : (MOT_LABEL[motSrc()]||motSrc())}.
      <b>Why relative and not an absolute amount of movement:</b> the sensor's units are not
      documented anywhere in this project, so an absolute cut would be a number invented rather than
      measured. A per-recording comparison needs no units.
      <b>What this is worth, measured on these recordings:</b> at the default 3&nbsp;&sigma; this rule
      removes about <b>5% of windows</b>, and the ones it removes carry several times the frontal
      theta and about <b>3&times; the parietal alpha</b> of the ones it keeps (the theta figure is a
      median of per-recording ratios and lands between 6&times; and 8&times; depending on which
      recordings are counted; on individual recordings it ranges enormously). Movement really does
      inflate the quantities these tabs plot.
      <b>But it is not finding anything the amplitude rules miss.</b> Of the windows in the top
      tenth of movement, an average of only <b>1% survive <em>Robust</em></b> at its default
      5&nbsp;&sigma; &mdash; the two rules are largely selecting the same windows. Movement also
      predicts the largest voltage excursion in a window <em>well</em> &mdash; in the study
      this page's code was written for, a median rank correlation of +0.67, positive on every
      recording with a working sensor. Those are that dataset's figures, not this one's.
      It is the same fact seen from the other side.
      <b>The reason to use it is independence.</b> Every other rule thresholds the EEG whose
      spectrum is then plotted, so a window is judged by the same data it contributes; the motion
      sensor is separate evidence about the same moment. Rejection modes are exclusive here, so
      this runs <em>instead of</em> an amplitude rule &mdash; treat it as a second opinion, not as
      wider coverage.
      Windows the sensor did not cover are kept and counted separately &mdash; they are unmeasured,
      not clean, and the panel says how many.`;

  if(isWindowed(a))
    return `<b>Windowed robust rejection is on at ${sig}&nbsp;&sigma;, over ${SL_LABEL()}.</b> The same
      comparison as Robust, but the yardstick is rebuilt <em>for every window</em> from
      ${SL_SPAN()}: a window is thrown away if any electrode feeding this measure swung further
      than ${sig} times how much that electrode was typically swinging in the
      ${+st().slideS}&nbsp;seconds around that moment. The window sits at the CENTRE of its own
      calibration, so it has the same amount of context on each side &mdash; except at the very
      start and end, where the span is cut short rather than slid along &mdash; and the threshold
      moves smoothly instead of jumping at fixed boundaries.
      <b>This is the local version of an already-local rule, and it compounds the same caveat.</b>
      Robust cannot be compared between recordings; this one cannot be compared between
      <em>moments</em> either, because each moment sets the threshold it is then judged against.
      A stretch of solid artifact raises its own bar and can keep nearly every window &mdash;
      so read a kept window as &ldquo;unremarkable for its neighbourhood&rdquo;, never as clean.
      A shorter window follows the signal more closely and is more easily fooled by a sustained
      bad patch; a longer one is steadier and slower to notice one.
      Use it to find <em>local</em> excursions that a recording-wide threshold smooths over;
      use the absolute cap for anything you need to compare.`;

  if(maskRule(a)==="robust"){
    const base = `<b>Robust rejection is on at ${sig}&nbsp;&sigma;${a==="robust_imputed"?", with the gaps filled in":""}.</b> Within each
      recording separately, a window is thrown away if any electrode feeding this measure swung
      further than ${sig} times that recording's own typical swing.
      Because the yardstick is set from each recording's own noise, a recording that was noisy
      throughout gets a lenient threshold and a clean one gets a strict threshold &mdash; so the
      &ldquo;windows kept&rdquo; figures can be compared <em>within</em> a recording but
      <b>not between recordings</b>. The pipeline's own fixed rule is 5&nbsp;&sigma;.`;
    return a==="robust_imputed" ? base + fill : base;
  }

  const cap = (+st().holmCap).toFixed(0);
  const holm = cap === "70" ? " That is Holm's own published criterion." : "";
  const base = `<b>An absolute &plusmn;${cap}&nbsp;&micro;V cap is on${a==="holm_imputed"?", with the gaps filled in":""}.</b> A window is thrown away if any
      electrode feeding this measure went past ${cap} microvolts, in either direction, at any instant
      inside it.${holm} Unlike the robust rule this threshold is the same number for every recording,
      so retention under it <em>is</em> comparable between recordings.${
        cap === "70" ? ` On this headset a 70&nbsp;&micro;V cap discards the large majority of
        windows, so the surviving points are sparse; raise the slider to see how quickly they come
        back.` : ` Holm's own criterion is 70&nbsp;&micro;V, which on this headset discards the large
        majority of windows; you are ${(+cap) > 70 ? "above" : "below"} it.`}`;
  return a==="holm_imputed" ? base + fill : base;
}

/* One branch per mode, not a boolean. The two-branch version described the
   regression whenever the mode was not "none", so choosing ICA -- which forms no
   estimate and subtracts nothing -- rendered a sentence about subtracting an
   estimate. */
/* Where did applying the pair's ICA make a channel LOUDER?

   ica.apply is an oblique projection, not an orthogonal one, so it is not
   guaranteed to reduce anything. The decomposition is fitted on a join that is
   about 10:1 task-weighted, and where one member of a pair has a broken
   electrode and the other does not, the component being removed is the broken
   member's artifact -- projecting that spatial pattern out of the quiet member
   INJECTS a scaled copy of it instead of removing anything.

   Measured on the run of 2026-09-08: p09/baseline_agent_personal gains 25% on P3
   and smaller amounts on eight of its other nine channels -- ONE of the ten
   recordings ICA acts on. It was two until that day; p02/baseline_agent_personal
   gained 179% on F1 and 236% on F2 and was excluded with its task, but it remains
   the clearest measurement of the mechanism: ICA works on variance, so PLAIN SD is
   what matters, and p02's task carried F1/F2 at 19,100 and 18,900 uV against 844
   and 735 in the rest crop it was joined to -- 23x and 26x, where no other
   electrode differed by more than 9x. The join is 91% task by sample count, and
   nothing stops this recurring on any pair whose halves differ by an order of
   magnitude on the channels a removed component describes. F1 and F2 are the frontal-theta numerator, so an
   injected baseline corrupts the denominator of every log ratio on that
   recording -- which is why this is surfaced rather than left in the QC. */
function icaInflation(rec){
  /* Both sides. On this dataset the injection lands on the BASELINE of a pair
     whose task has a broken electrode, and the baseline is the denominator, so
     checking only the plotted recording would have found nothing at all. */
  const look = (q, where) => {
    const by = (q && q.sd_reduction_by_channel_pct) || null;
    if(!by || !q.n_components_removed) return null;
    let worst = null;
    for(const [ch, v] of Object.entries(by))
      if(v != null && v < -1 && (!worst || v < worst[1])) worst = [ch, v];
    return worst ? {ch: worst[0], pct: -worst[1], where} : null;
  };
  return look((rec.ocQc || {}).ica, "this recording")
      || look((rec.baseOcQc || {}).ica, "its baseline");
}

/* Is the correction the reader is looking at untrustworthy on THIS recording?
   Only the regression can be degenerate; ICA reports its own cost differently
   and `none` cannot be wrong about a correction it did not make. */
function ocularFlag(rec){
  const q = (rec.ocQc || {})[st().oc];
  if(!q) return null;
  if(st().oc === "eog_regression" && q.degenerate)
    return `the EOG regression is numerically degenerate here (largest coefficient
            ${q.max_abs_coefficient}, where a physical one is below 1) &mdash; the two eye
            sensors are nearly collinear, so this correction is not trustworthy on this
            recording`;
  if(st().oc === "ica"){
    if(q.failed) return "ICA failed on this recording, so nothing was removed";
    if(!q.n_components_removed)
      return `no component reached the EOG threshold, so ICA removed nothing here and
              this is identical to no correction`;
    /* Worst channel, not just the total. The total is a share of all ten
       channels' variance, and where a frontal pair is broken that total IS the
       broken pair -- 0.014% total while P3 lost 58% of its amplitude. */
    const inf = icaInflation(rec);
    if(inf)
      return `<b>this correction made ${inf.ch} LOUDER by ${inf.pct.toFixed(0)}% in
              ${inf.where}.</b> The decomposition is fitted on the task joined to its
              baseline, and where one half has a broken electrode the other does not,
              removing that half's artifact injects a scaled copy of it into the other
              instead of taking anything away.${
              inf.where === "its baseline"
                ? " Every value on this panel is divided by that baseline, so the"
                + " whole series is affected, not one electrode."
                : ""} Treat this recording's ICA series as corrupted, not corrected`;
    const by = q.sd_reduction_by_channel_pct || {};
    let worst = null;
    for(const [ch, v] of Object.entries(by))
      if(v != null && (!worst || v > worst[1])) worst = [ch, v];
    return `ICA removed ${q.n_components_removed} of the ten components`
      + (q.variance_removed_pct != null && q.variance_removed_pct > 0
          ? `, ${q.variance_removed_pct}% of the total variance` : "")
      + (worst && worst[1] > 0
          ? ` &mdash; the largest single-electrode cost is ${worst[0]}, down
              ${worst[1].toFixed(1)}%` : "")
      + ". Brain signal is removed along with the artifact.";
  }
  return null;
}

function plainOcular(){
  if(st().oc === "none")
    return `<b>Eye-blink correction is off</b>, so blinks and eye movements are still in the
       signal. That matters most for the forehead electrodes, where a blink produces a
       large slow deflection that looks much like theta.`;
  if(st().oc === "ica")
    return `<b>Eye-blink correction is on, by ICA.</b> The recording is separated into as
       many independent components as it has electrodes, any component whose time course
       tracks the eye sensors is discarded, and the rest are recombined.
       <span class="warn">This montage has only ten electrodes</span>, so a blink is
       spread across components that also carry brain activity &mdash; discarding one
       takes real signal with it, most of all the slow frontal activity that frontal
       midline theta is made of. The decomposition is fitted once per task/baseline
       pair so both sides of a ratio lose the same components &mdash; but that means a
       pair whose two halves differ sharply in electrode quality can have one half's
       artifact <em>injected</em> into the other, which happens on ${
       (function(){const n=DATA.recordings.filter(r=>icaInflation(r)).length;
        return n===1 ? "one recording here and is called out on its panel"
                     : `${n} recordings here and is called out on their panels`;})()}.
       Read this mode as an experiment, not a cleaner answer.`;
  return `<b>Eye-blink correction is on.</b> The headset's two eye sensors are used to
       estimate how much of each EEG electrode's signal came from blinks and eye
       movements, and that estimate is subtracted out.`;
}

/* The stacked-reconstruction warning belongs in the interpolation-ON branch.
   It was written into the OFF branch behind a `st().interp==="on"` guard, so it
   could never render: the branch only runs when interpolation is off, and the
   guard only passes when it is on. It shipped in dashboard.html as unreachable
   text -- present in the file, invisible on the page -- while the notes
   claimed it was surfaced. Caught by review 2026-09-08. */
function plainInterp(){
  /* The frontal-pair sentence is written once and appended to both the automatic
     and the manual branch, because the rule is identical in both and two copies
     would be two things to keep in step. */
  const frontal = st().interpSrc==="off" ? "" : (st().frontalPair
    ? ` ${FRONTAL.join(" and ")} are interpolated like any other channel when both
        are on the list; untick the box beside the control to see those recordings
        with the frontal pair left as recorded.`
    : ` <span class="warn">${FRONTAL.join(" and ")} are being held back:</span> where both
        are on the list they are left exactly as recorded and the rest of the list is
        still interpolated. Those two electrodes are the whole numerator of the
        cognitive-load index and the two sides of frontal alpha asymmetry, so this is the
        difference between reading two failed electrodes and reading a frontal signal
        rebuilt entirely from central, parietal and occipital ones. Recordings with only
        one of the two flagged are unaffected.`);
  if(st().interpSrc === "manual")
    return `<b>Bad channels are being interpolated from a manual list.</b> The channels
       replaced here were written down per participant and arm rather than detected: the
       list has no derivation in the pipeline and does not have to agree with what
       bad-channel detection found, and on several recordings it does not &mdash; it asks
       for channels detection never flagged, and passes over channels it did. Each panel
       says what was replaced in that recording. Read this mode as the researcher's own
       call about which electrodes to rebuild, not as a measurement.${frontal}${
       st().oc==="ica" ? ` <span class="warn">You have also stacked two reconstructions:</span>
       ICA discards whole components and interpolation then rebuilds channels from the ones
       left.` : ""}`;
  return st().interpSrc!=="off"
    ? `<b>Bad-channel interpolation is on</b>, which is what the pipeline has always done: an
       electrode whose amplitude was wildly out of step with the other nine is replaced by a
       spherical-spline estimate built from its neighbours. The panel badges say when a value
       you are looking at is partly or wholly such a reconstruction.${frontal}${
       st().oc==="ica" ? ` <span class="warn">You have stacked two reconstructions:</span>
       ICA discards whole components and interpolation then rebuilds channels from the ones
       left, so this view is further from the measured signal than either setting alone
       &mdash; on ten electrodes there is not much left to rebuild from. Nothing here fails
       or warns on its own, because interpolation and re-referencing are fixed functions of
       head geometry and will happily transform a signal that no longer spans the space they
       assume.` : ""}`
    : `<b>Bad-channel interpolation is off.</b> A failed electrode is left exactly as it recorded,
       so nothing on screen is synthetic &mdash; but nothing is repaired either, and a broken
       electrode feeds its own measure directly.
       ${st().ref!=="hardware" ? `<span class="warn">With a re-referenced view that spreads:</span>
       both the average and REST references mix every channel into every other, so one broken
       electrode moves all ten.` : ""}`;
}

function plainReference(){
  if(st().ref==="hardware")
    return `Voltages are measured <b>against the earlobe sensor</b>, which is how the headset
      recorded them, and how the cognitive-load index is defined.`;
  const asymWarn = M().kind==="asym"
    ? ` <span class="warn">This is worth pausing on for a left-versus-right measure:</span>
        because every electrode is folded into the reference, the two sides are no longer two
        independent measurements, and a difference between them is not a clean comparison of two
        spots on the head.`
    : "";
  if(st().ref==="average")
    return `Voltages have been <b>re-measured against the average of all ten EEG electrodes</b>
      rather than the earlobe. Every electrode is folded into that average, so a single broken
      electrode shifts all ten.` + asymWarn;
  return `Voltages have been <b>re-referenced by REST</b>, which estimates what the electrodes
    would have read against a reference infinitely far from the head. It does that through an
    assumed model of head geometry, not a measurement, and with only ten electrodes that
    estimate is loosely determined &mdash; read it as a check on whether a conclusion depends on
    the reference, not as a better measurement.` + asymWarn;
}

function plainSpectral(){
  const e = st().epoch;
  const bins = DATA.band_bins[e + "|" + st().fft] || null;
  /* Resolution comes from the payload, not from 1/epoch: Welch splits the window
     into half-length segments, so at 1 s it resolves 2 Hz, not 1. */
  const res = bins && bins.freq_resolution_hz != null
    ? bins.freq_resolution_hz : 1 / e;
  /* The index denominator is Holm's 8-12 Hz; the other tabs use 8-13. Quoting
     one band's bin count on the other tab was quietly wrong about the very
     distinction the index tab's own lede makes a point of. */
  const alphaBins = bins ? (tab === "index" ? bins.alpha_holm : bins.alpha) : null;
  const alphaLabel = tab === "index" ? "8\u201312" : "8\u201313";

  /* DPSS half-bandwidth in HERTZ is NW divided by the window length in seconds.
     It is not a constant. This text said "about +-1 Hz, a 25% widening", which
     was true only while the window was pinned at 4 s; at 1 s the smoothing
     kernel is +-4 Hz, wider than the theta band it is smoothing. */
  /* Shipped by the pipeline per (window, estimator) so the two cannot drift;
     MT_NW / e is the same quantity and is the fallback. */
  const halfBW = (bins && bins.smoothing_half_bandwidth_hz)
    ? bins.smoothing_half_bandwidth_hz : MT_NW / e;
  const est = {
    hann: `a single <b>Hann-tapered</b> Fourier transform of the whole window, which is the method
           Holm et al. specify and the pipeline's default`,
    multitaper: `an average of seven <b>DPSS multitapers</b>, which steadies the estimate but
           smooths the spectrum over <b>&plusmn;${halfBW.toFixed(2)}&nbsp;Hz</b> at this window
           length &mdash; the smoothing width is 4&nbsp;Hz divided by the window length in seconds,
           so it widens as you shorten the window`,
    welch: `<b>Welch's method</b>: the window is split into overlapping half-length pieces and their
           spectra averaged, which steadies the estimate at half the frequency resolution`,
    boxcar: `<b>no taper at all</b>. This has the sharpest frequency resolution and the worst leakage:
           a strong rhythm just outside a band spills into it`
  }[st().fft];

  let out = `Each point covers <b>${e} second${e===1?"":"s"}</b> of recording and is computed with
    ${est}. At this setting the spectrum is resolved in steps of about
    ${(+res).toFixed(2)}&nbsp;Hz${
      bins ? `, which puts ${bins.theta} bin${bins.theta===1?"":"s"} inside the
      4&ndash;8&nbsp;Hz theta band and ${alphaBins} inside ${alphaLabel}&nbsp;Hz alpha` : ""}.
    Shorter windows give more points and follow the task more closely; longer windows give a
    steadier number per point but fewer of them.`;

  if(st().fft === "multitaper" && halfBW >= 2)
    out += ` <span class="warn">At this window length the multitaper smoothing is
      &plusmn;${halfBW.toFixed(1)}&nbsp;Hz, which is as wide as the 4&nbsp;Hz bands being
      measured.</span> Theta and alpha are then being read from very nearly the same smeared
      spectrum, and a ratio between them is not measuring a difference between bands. Lengthen
      the window or choose a different estimator before reading anything into this view.`;
  return out;
}


function plainBaseline(){
  const b = st().base;
  if(b==="off")
    return `<b>Baseline correction is off</b>, so these are raw values, not compared
      against anything. Two people with different skull thickness or electrode contact
      can differ several-fold for reasons that have nothing to do with the task, so be
      careful reading one recording against another here.`;

  const seg = segOf(), lvl = segShort(), bnd = segBounds();
  const mins = t => (t % 60 === 0) ? `${t/60}:00` : `${Math.floor(t/60)}:${String(t%60).padStart(2,"0")}`;

  const what = `Each point has had <b>that person's own ${lvl} level subtracted from
    it</b>. That level is the middle (median) value of ${segBlurb()} It is taken from
    <b>${mins(bnd[0])}&ndash;${mins(bnd[1])}</b> of their matching baseline recording.`;

  const other = ` The other references are one click away on <b>Baseline segment</b>,
    and they are genuinely different questions &mdash; not different estimates of one
    answer. Switching changes what zero means.`;

  const scale = (b==="log" && M().kind!=="asym")
    ? ` Because you have picked <b>log ratio</b>, the point is the task value divided by
        the ${lvl} value and then log-transformed: <b>0 means the task matched ${lvl}</b>,
        +0.69 means it was double, &minus;0.69 means half.`
    : ` <b>0 means the task matched ${lvl}</b>, above 0 means more, below 0 means
        less, measured as a straight difference in this measure's own units.`;

  const tracks = ` The <b>window length, estimator, reference, interpolation, rejection rule and
    its threshold are all applied to the baseline windows too</b>, so the two sides of the subtraction
    are always measured the same way and moving any of those controls moves both. Since 2026-09-04
    that holds for <b>which channels are interpolated</b> as well: one bad-channel list is decided
    for each task and its own baseline together &mdash; from the task joined to the baseline's
    resting phase &mdash; and the same channels are replaced by a spline in both. Before that the
    two recordings were judged separately against a purely relative criterion, and they could
    disagree, so a task could be corrected against a baseline in which the very channel driving the
    measure was real where the task's was synthetic, or the reverse. Each segment is
    cropped and calibrated on its own samples, so the rejection rule is judged against the segment
    it is applied to. One consequence of deciding on the resting phase: choosing the
    <b>math</b> or <b>eyes-closed</b> baseline segment still uses channels chosen on <b>rest</b>. A baseline whose own channels sit outside the band is flagged on the panel rather
    than withheld, so a clean task can still be corrected against a questionable level
    &mdash; watch the baseline flag. It reports only what the panel&rsquo;s own badge does
    <i>not</i> already say: because the pair shares one channel list, interpolation is never
    flagged a second time here, and an out-of-band baseline is called out only on a
    recording whose task amplitudes were in band. Its colour matches the severity of what
    it found.`;

  const warn = (b==="raw" && M().kind!=="asym")
    ? ` <span class="warn">Worth knowing in this mode:</span> subtracting one absolute
        power from another keeps the original units, so a recording whose electrodes
        were noisy &mdash; on either side of the subtraction &mdash; carries that noise
        into its corrected value at full weight. Corrected values in this view run
        <b>${plainRange()}</b>. <b>Log ratio</b> divides instead of subtracting, which
        cancels the per-person scale and is the safer choice for comparing people.`
    : "";

  return what + other + scale + tracks + warn;
}

function plainRange(){
  let lo = Infinity, hi = -Infinity;
  for(const r of candidates()){
    const s = frame(r);
    if(!s || s.noBaseline) continue;
    for(const v of s.vals){ if(v==null) continue; if(v<lo) lo=v; if(v>hi) hi=v; }
  }
  return isFinite(lo) ? `from about ${fmt(lo)} to ${fmt(hi)}` : "over no surviving windows";
}

function plainScale(){
  if(effScale()==="log")
    return `The vertical axis is <b>logarithmic</b>, so equal distances up the axis mean
      equal multiples rather than equal amounts.`;
  if(M().kind==="asym")
    return `The vertical axis is <b>linear</b>, and a horizontal line marks zero &mdash;
      the point at which the two sides are equal. A logarithmic axis is not available
      here because these values are already a log difference and are routinely negative.`;
  if(st().base!=="off")
    return `The vertical axis is <b>linear</b>, and a horizontal line marks zero &mdash;
      the ${segShort()} level. A logarithmic axis is not available while the baseline
      is subtracted, because a value below that level is negative.${
        st().base==="log"
          ? ` <b>The values themselves are already logarithmic</b> in this mode, though:
              they are log ratios, so equal distances up this linear axis DO mean equal
              multiples &mdash; +0.69 is double the ${segShort()} level wherever it appears.`
          : ""}`;
  return `The vertical axis is <b>linear</b>, so equal distances mean equal amounts.`;
}

/* Why a recording drops out of a baseline-corrected view. FOUR reasons, kept
   apart because they call for different responses from the reader:

     noRec   no paired baseline recording at all
     noSeg   the baseline exists but its block is too short to contain the
             SELECTED segment -- fixed by choosing a different segment
     noData  the sweep holds no baseline values for this processing combination
             (a reference the pipeline could not build). A gap in the DATA: no
             control brings it back
     noWin   the segment exists and has data, but every window was rejected --
             the only one of the four that loosening the rejection rule fixes

   noData used to be folded into noWin, so the page told the reader to loosen a
   threshold that could not possibly help, while the excluded-recordings table
   simultaneously told them it was a gap in the data. */
function baselineDropReasons(recs){
  let noRec=0, noSeg=0, noData=0, noWin=0;
  for(const r of recs){
    const s = frame(r);
    if(!s || !s.noBaseline) continue;
    if(r.base && r.base.missing){ noRec++; continue; }
    const sp = baseSeg(r);
    if(!sp){ noSeg++; continue; }
    if(!bValsOf(sp.key)){ noData++; continue; }
    const ix = sp.idx ? sp.idx[ekey()] : null;
    if(!ix || !ix.length){ noData++; continue; }
    noWin++;
  }
  return {noRec, noSeg, noData, noWin, total:noRec+noSeg+noData+noWin};
}

function renderPlain(){
  const recs = candidates();
  const shown = recs.filter(r=>{ const s=frame(r); return s && !s.noBaseline; });
  const missing = recs.length - shown.length;
  const drop = baselineDropReasons(recs);
  const gatedOut = DATA.recordings.length - recs.length;
  const flagged = shown.filter(r=>quality(r).level!=="good").length;
  const bflag = st().base==="off" ? 0 : shown.map(r=>baselineFlag(r)).filter(Boolean).length;
  /* `e` alongside `t`: `t` counts every window in the recording and `e` only the
     ones a criterion could ever keep. Reporting "measured of total" while the
     percentage beside it came from the eligible pool is how the retention
     readout came to print a fraction that disagreed with its own percentage. */
  const c = shown.reduce((a,r)=>{const k=counts(r); const s=frame(r);
    a.m+=k.measured; a.t+=k.total; a.e+=(s && s.eligible!=null) ? s.eligible : k.total;
    return a;},{m:0,t:0,e:0});

  const p1 = `Each small chart below is <b>one ${TASK_LEN}task recording</b>. The recording
    is chopped into ${st().epoch}-second windows, and every point on the chart is one of those
    windows. How high the point sits is ${PLAIN_MEASURE[tab]}.`;

  const p2 = [plainSpectral(), plainInterp(), plainOcular(), plainReference()].join(" ");
  const p3 = plainArtifact();
  const p4 = plainBaseline() + " " + plainScale();

  /* Count what is actually PLOTTED, not what has data upstream. With baseline
     correction on by default, a recording can pass every processing setting and
     still show nothing because its baseline segment is unusable -- so a flat
     "all N pass" was true of the sweep and false of the screen. */
  const bits = [];
  if(gatedOut)
    bits.push(`<b>${recs.length} of the ${DATA.recordings.length} task recordings</b> pass the
      settings above; ${gatedOut} ${gatedOut===1?"is":"are"} held back and ${gatedOut===1?"is":"are"}
      listed, with the reason, under <em>Excluded recordings</em>.`);
  else if(missing)
    bits.push(`All ${recs.length} task recordings pass the processing settings above, but
      <b>${shown.length} of them ${shown.length===1?"is":"are"} plotted</b> &mdash;
      ${missing} ${missing===1?"has":"have"} no usable ${segShort()} baseline to subtract,
      for the reason${missing===1?"":"s"} below.`);
  else
    bits.push(`<b>All ${recs.length} task recordings</b> pass the current settings.`);

  if(flagged)
    bits.push(`<span class="warn">${flagged} of the ${shown.length} on screen ${flagged===1?"is":"are"}
      flagged</span> &mdash; an electrode this measure depends on sits outside the amplitude band
      this dataset defines, or was reconstructed from its neighbours, or was left in place broken.
      Each panel says which, and the exact amplitudes are in the table below.`);

  const lvl = segShort();
  if(drop.noRec)
    bits.push(`<span class="warn">${drop.noRec} recording${drop.noRec===1?" is":"s are"}
      hidden</span> because ${drop.noRec===1?"it has":"they have"} no usable paired
      baseline recording at all.`);
  if(drop.noSeg)
    bits.push(`<span class="warn">${drop.noSeg} recording${drop.noSeg===1?" is":"s are"}
      hidden</span> because ${drop.noSeg===1?"its":"their"} baseline block is too short to
      contain the <b>${lvl}</b> segment. <b>Choosing a different baseline segment brings
      ${drop.noSeg===1?"it":"them"} back</b> &mdash; no other control will.`);
  if(drop.noData)
    bits.push(`<span class="warn">${drop.noData} recording${drop.noData===1?" is":"s are"}
      hidden</span> because the sweep holds no ${lvl} baseline values for this processing
      combination. <b>That is a gap in the data, not a consequence of the controls</b>, so
      loosening the rejection rule will not bring ${drop.noData===1?"it":"them"} back;
      changing reference, interpolation or estimator may.`);
  if(drop.noWin)
    bits.push(`<span class="warn">${drop.noWin} recording${drop.noWin===1?" is":"s are"}
      hidden</span> because, under the rejection rule you have selected, <b>not one
      ${lvl} window survives</b> in ${drop.noWin===1?"its":"their"} baseline &mdash; so there is
      no level left to subtract. ${drop.noWin===1?"It comes":"They come"} back if
      you loosen the rule or its threshold.`);

  if(missing===0 && st().base!=="off")
    bits.push(`Every recording shown has a usable ${lvl} baseline.`);

  if(c.e) bits.push(`Across everything on screen, <b>${c.m.toLocaleString()} of
    ${c.e.toLocaleString()}</b> windows survive the current rejection setting.${
    c.t > c.e ? ` A further ${(c.t-c.e).toLocaleString()} never reached it &mdash; their
    samples are not contiguous in real time.` : ""}`);

  if(st().oc !== "none"){
    /* Only where the correction MISBEHAVED. ocularFlag returns text for every
       recording under ICA -- the normal "removed N components" report -- so
       counting truthiness here warned that all 22 were suspect, every time. */
    const bad = shown.filter(r=>{
      const q = (r.ocQc || {})[st().oc];
      if(!q) return false;
      return st().oc === "ica" ? !!(q.failed || !q.n_components_removed)
                               : !!q.degenerate;
    }).length;
    const inflated = shown.filter(r=>st().oc==="ica" && icaInflation(r));
    if(inflated.length)
      bits.push(`<span class="warn">On ${inflated.length} recording${
        inflated.length===1?"":"s"} this ICA correction makes an electrode LOUDER
        rather than quieter</span> &mdash; ${inflated.map(r=>{
          const i = icaInflation(r);
          return `${esc(r.key)} (${i.ch} +${i.pct.toFixed(0)}% in ${i.where})`;
        }).join(", ")}.
        The decomposition is fitted per pair, so a partner's broken electrode can be
        injected into a recording that did not have one. Those series are corrupted,
        not corrected.`);
    if(bad) bits.push(`<span class="warn">${bad} of the recordings shown carry a
      caveat on the ocular correction you have selected</span> &mdash; open a panel to
      see which and why.`);
  }
  if(bflag) bits.push(`<span class="warn">${bflag} of the baselines ${bflag===1?"is itself":"are themselves"}
    flagged</span> &mdash; either a baseline electrode sits outside the band, or the median rests
    on very few surviving ${segShort()} windows. A clean task compared against a bad baseline gives
    a wrong answer that the corrected line alone will not reveal.`);

  /* No heading here: the <summary> of the surrounding <details> is the heading,
     and repeating it inside would be read out twice. */
  $("#plain").innerHTML =
    `<p>${p1}</p><p>${p2}</p><p>${p3}</p><p>${p4}</p><p>${bits.join(" ")}</p>`;
  $("#plain-wrap").hidden = false;   /* undoes the hide in render()'s not-ready branch */
}

function renderTiles(){
  const recs = candidates();
  const shown = recs.filter(r=>{ const s=frame(r); return s && !s.noBaseline; });
  const meds = shown.map(r=>median(frame(r).vals)).filter(v=>v!=null);
  /* `e` is the ELIGIBLE pool -- every window minus the ones excluded outright for
     not being contiguous in real time. The tile's percentage divides by it, so it
     must be accumulated here and not only in renderPlain, which keeps its own
     accumulator over the same recordings. */
  const c = shown.reduce((a,r)=>{const k=counts(r); const s=frame(r);
    a.m+=k.measured; a.f+=k.fabricated; a.t+=k.total;
    a.e+=(s && s.eligible!=null) ? s.eligible : k.total; return a;},{m:0,f:0,t:0,e:0});

  const flagged = shown.filter(r=>quality(r).level!=="good").length;
  const withValue = shown.filter(r=>{const s=frame(r);
    return s && s.vals.some(v=>v!=null);}).length;
  const drop = baselineDropReasons(recs);

  const t3 = isImputed()
    ? ["Windows fabricated", c.f ? c.f.toLocaleString() : "0",
       c.f ? `interpolated, not measured (${(100*c.f/(c.m+c.f)).toFixed(0)}% of what is plotted)`
           : "no window needed filling at this threshold"]
    : ["Recordings flagged", String(flagged),
       flagged ? `of ${shown.length} shown, an electrode is outside the band, interpolated or broken`
               : "every shown recording has in-band, measured electrodes"];

  const label = st().base==="off" ? "Median value"
              : st().base==="log" ? `Median ln(task/${segShort()})`
              : `Median Δ from ${segShort()}`;

  const dropTxt = drop.noWin
    ? `${drop.noWin} hidden: no ${segShort()} window survives this rejection rule`
    : drop.noSeg ? `${drop.noSeg} hidden: baseline block has no ${segShort()} segment`
    : drop.noRec ? `${drop.noRec} hidden: no paired baseline recording`
    : `every recording has a usable ${segShort()} baseline`;

  const sub1 = st().base!=="off" ? dropTxt
    : tab==="index" ? "F1+F2 midline mean, every recording"
    : "every task recording shown";

  const tiles=[
    ["Recordings shown", String(shown.length), sub1],
    ["Windows measured", c.m.toLocaleString(),
      c.e ? `of ${c.e.toLocaleString()} ${st().epoch}-second windows (${(100*c.m/c.e).toFixed(0)}%)`
          + (c.t > c.e ? `, ${(c.t-c.e).toLocaleString()} excluded outright` : "") : "—"],
    t3,
    [label, meds.length?fmt(median(meds)):"—",
      meds.length
        ? `over ${withValue}${withValue!==shown.length?` of ${shown.length}`:""} recording${withValue===1?"":"s"} · range ${fmt(Math.min(...meds))}–${fmt(Math.max(...meds))}`
        : "no recording retains a window in this mode"]
  ];
  $("#tiles").innerHTML = tiles.map(([k,v,s])=>
    `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${s}</div></div>`).join("");

  /* ---- banners ---- */
  const B = [];
  if(isImputed() && c.f)
    B.push(`<b>${(100*c.f/(c.m+c.f)).toFixed(0)}% of the plotted values in this mode are interpolated, not measured.</b>
      The rejected windows are filled by linear interpolation, and leading or trailing gaps are filled
      flat with the nearest value rather than extrapolated. Dashed segments mark fabricated windows.`);
  else if(st().art==="none"){
    const nx = candidates().reduce((a,r)=>{const f=frame(r); return a+((f&&f.excluded)||0);},0);
    B.push(`<b>No artifact rejection in this mode.</b> Every window is plotted, including
      those an amplitude criterion would discard. Useful for seeing where the artifacts are;
      not a basis for interpreting the measure.${
      nx ? ` <b class="warn-ink">With one exception:</b> ${nx} window${nx===1?"":"s"} on screen
      ${nx===1?"is":"are"} withheld even here, because ${nx===1?"its":"their"} samples are not
      contiguous in real time &mdash; ${nx===1?"it spans":"they span"} a cut or a dropout, so the
      spectrum would be taken across a step in the signal. That is a property of the recording,
      not a judgement about the EEG, which is why no rejection setting brings
      ${nx===1?"it":"them"} back.` : ""}`);
  }
  if(isMotion()){
    /* Three things a reader of this mode must be able to see without opening a
       panel: how much was actually rejected, how much was never assessed, and
       whether any recording has no IMU at all. None of them were visible before;
       `motionOk` was shipped and unread, and the unassessed count was computed
       and discarded. */
    const shownM = candidates().filter(r=>{const f=frame(r); return f && !f.noBaseline;});
    /* These two OVERLAP, and reporting them as separate tallies said "1 recording
       has no usable motion data. 1 recording has no IMU file at all." of what was
       one recording counted twice: a recording whose sensor is unusable yields an
       all-NaN motion array, which yields fewer than 3 assessed windows, which
       refuses the mask. So the unusable ones are named once, and `refusedOther`
       carries only recordings refused for some OTHER reason. */
    const noMot = candidates().filter(r=>!r.motionOk);
    const refusedOther = shownM.filter(r=>maskRefused(r) && r.motionOk);
    const unass = shownM.reduce((a,r)=>{const f=frame(r); return a+((f&&f.unassessed)||0);},0);
    const unc = shownM.reduce((a,r)=>{const f=frame(r); return a+((f&&f.uncovered)||0);},0);
    const strd = shownM.reduce((a,r)=>{const f=frame(r); return a+((f&&f.straddle)||0);},0);
    const totw = shownM.reduce((a,r)=>a+counts(r).total,0);
    B.push(`<b>Rejection here is driven by the helmet's motion sensor, not by the EEG.</b>
      A window goes if the head moved more than ${(+st().motK).toFixed(2)}&nbsp;&sigma; above that
      recording's own typical movement, measured from the
      ${isMotUnion() ? "accelerometer OR the gyroscope, each against its own threshold "
                      + "&mdash; the sensitive setting: it rejects a window either one flags"
                    : (MOT_LABEL[motSrc()]||motSrc())}. The threshold is relative because the sensor's units are
      undocumented, so retention is <b>not comparable between recordings</b> &mdash; the same
      caveat the robust rules carry.${
        unass ? ` <b class="warn-ink">${unass} of the ${totw.toLocaleString()} windows on screen
        (${(100*unass/Math.max(1,totw)).toFixed(1)}%) could not be assessed</b> &mdash; ${
          strd && unc ? `${unc} the sensor did not cover, and ${strd} straddle an interval
          step&nbsp;2 excised, which gives a window far MORE sensor samples than normal rather
          than fewer`
          : strd ? `${strd===1?"it straddles":"they straddle"} an interval step&nbsp;2 excised,
          so the window spans far more wall-clock than its own duration &mdash; that is far MORE
          sensor samples than normal, not fewer`
          : `the sensor did not cover ${unass===1?"it":"them"}`}. ${
          unass===1 ? "It is kept, because there is no evidence against it, but it has"
                    : "They are kept, because there is no evidence against them, but they have"
        } not passed anything.` : ""}${
        noMot.length ? ` <b class="warn-ink">${noMot.length}
        recording${noMot.length===1?" has":"s have"} no usable motion data</b>, so nothing is
        filtered there: ${noMot.map(r=>`${r.key}${r.motionWhy?` &mdash; ${r.motionWhy}`:""}`)
        .join("; ")}.` : ""}${
        refusedOther.length ? ` A further ${refusedOther.length}
        recording${refusedOther.length===1?" has":"s have"} too few assessable windows at this
        window length for a threshold to be computed, so nothing is filtered there either.` : ""}`);
  }
  if(isWindowed()){
    /* Only claim locality where the signal is actually longer than the sliding
       window; a recording shorter than it gives every window the same span --
       the whole signal -- and is plain Robust wearing this label. */
    const shownW = candidates().filter(r=>{const s=frame(r); return s && !s.noBaseline;});
    const short = shownW.filter(slideDegenerate).length;
    B.push(`<b>The threshold is rebuilt for every window</b>
      (${(+st().sigmaK).toFixed(2)}&nbsp;&times;&nbsp;1.4826&nbsp;&times;&nbsp;MAD of ${SL_SPAN()}), so its
      microvolt value differs for every recording, every channel <b>and every window</b>.
      Retention here is <b>comparable neither between recordings nor between moments of one
      recording</b>: a stretch that was noisy throughout raises the bar it is judged against and
      can retain almost everything. The absolute cap is the comparable one.
      At the two ends the span is truncated to the samples that exist rather than shifted inwards,
      so the first and last windows are calibrated on as little as half of ${SL_LABEL()}.${
        short ? ` <b>${short} of the ${shownW.length} recordings shown ${short===1?"is":"are"}
        short enough that every window's span covers the whole signal</b>, so for
        ${short===1?"it":"them"} this mode is identical to Robust.` : ""}`);
  }
  else if(maskRule(st().art)==="robust")
    B.push(`<b>The robust threshold is self-calibrated per recording</b>
      (${(+st().sigmaK).toFixed(2)}&nbsp;&times;&nbsp;1.4826&nbsp;&times;&nbsp;MAD), so its microvolt value differs
      for every recording and every channel. Retention in this mode measures within-recording
      cleanliness and is <b>not comparable between recordings</b>. The absolute cap is the
      comparable one.`);

  B.push(`<b>No signal-quality gate.</b> Every task recording is plotted, whatever its electrodes
      were doing, so nothing is withheld &mdash; but nothing is vouched for either.
      <span class="badge critical">${ICON.critical}Fully synthetic</span> is the one to stop at:
      every channel that measure reads is a spline reconstruction, so the value carries none of
      this participant's own signal.
      <span class="badge warning">${ICON.warning}High amplitude</span> is a caution rather than a
      verdict &mdash; an electrode outside the band derived from this dataset is unusual here,
      which is not the same as impossible. Read the badge and the amplitudes before trusting a
      value.`);

  /* Every MODE, not the detected list: under `manual` a recording with nothing
     detected can still have channels replaced, so counting detections claimed
     the traces were identical either way when they are not. */
  const noBad = candidates().filter(
    r => (DATA.interpolation_modes||[]).every(m => !((r.interpLists||{})[m]||[]).length)).length;

  /* What the frontal-pair checkbox does to the recordings on screen, right now.
     It is a strict no-op unless BOTH F1 and F2 are on the current source's list,
     which on this dataset is 5 groups of 20 -- so most of the time unticking it
     changes nothing, and the reader deserves to be told that rather than left
     wondering whether the control works. Where it does bite it usually empties
     the list outright, which makes it an off switch for that recording rather
     than a partial interpolation; that is worth saying too. */
  if(st().interpSrc !== "off"){
    const shownI = candidates();
    const bites = shownI.filter(r => {
      const src = (r.interpLists||{})[st().interpSrc] || [];
      return FRONTAL.every(c => src.includes(c));
    });
    const emptied = bites.filter(r => {
      const src = (r.interpLists||{})[st().interpSrc] || [];
      return src.every(c => FRONTAL.includes(c));
    });
    if(shownI.length && !bites.length)
      B.push(`<b>The ${FRONTAL.join("/")} checkbox does nothing on this view.</b> It only
        acts where <em>both</em> ${FRONTAL.join(" and ")} are on the list this mode
        interpolates, and none of the ${shownI.length} recordings shown is in that
        position. Ticking or unticking it leaves every trace identical.`);
    else if(bites.length)
      B.push(`<b>The ${FRONTAL.join("/")} checkbox affects ${bites.length} of the
        ${shownI.length} recordings shown</b> &mdash; ${bites.map(r=>esc(r.participant)+" "+esc(r.arm)
          /* sbx ran No AI under both framings, so participant + arm alone repeats */
          + (bites.filter(o=>o.participant===r.participant && o.arm===r.arm).length > 1
             ? " "+esc(r.framing) : "")).join(", ")}
        &mdash; because only those have both ${FRONTAL.join(" and ")} on the list this
        mode interpolates. The rest are unchanged either way.${
        emptied.length ? ` On ${emptied.length} of them the frontal pair is the
        <em>whole</em> list, so unticking leaves nothing to interpolate at all and the
        result is identical to turning interpolation off.` : ""}`);
  }
  if(noBad === candidates().length && candidates().length)
    B.push(`<b>The interpolation toggle does nothing on this view.</b> None of the
      ${noBad} recordings on screen has a channel that failed bad-channel detection,
      so there is nothing to replace and the On/Off traces are identical. That is the
      data, not a broken control.`);
  else if(noBad)
    B.push(`<b>The interpolation toggle does nothing for ${noBad} of the
      ${candidates().length} recordings on screen</b>, which have no failed channel to
      replace. Their traces are identical either way.`);

  if(st().interpSrc==="off")
    B.push(`<b>Interpolation is off.</b> Channels that failed bad-channel detection are being read as
      recorded rather than rebuilt from their neighbours. Nothing on screen is synthetic; nothing is
      repaired either.${st().ref!=="hardware" ? ` Because the ${REF_LABEL[st().ref]} reference mixes
      all ten channels, a single broken electrode is now moving every trace on this page.` : ""}`);

  if(st().ref==="rest")
    B.push(`<b>REST is a model-based reference.</b> It reconstructs the signal against a reference at
      infinity through a spherical head model fitted to this montage. With ten electrodes the
      reconstruction is loosely constrained, so treat differences from the hardware reference as a
      sensitivity check rather than a correction.`);

  if(st().fft==="multitaper"){
    const hb = MT_NW / st().epoch;
    B.push(`<b>The multitaper estimator widens every band, by
      &plusmn;${hb.toFixed(2)}&nbsp;Hz at a ${st().epoch}&nbsp;second window.</b> The smoothing
      width is ${MT_NW} divided by the window length in seconds, so it grows as the window
      shortens: &plusmn;0.4&nbsp;Hz at 10&nbsp;s, &plusmn;1&nbsp;Hz at 4&nbsp;s,
      &plusmn;4&nbsp;Hz at 1&nbsp;s. Energy just outside theta or alpha is counted inside it.
      This was the pipeline's estimator before 2026-08-27 and was changed for exactly this
      reason; it is here so the size of that effect is visible.
      ${hb >= 2 ? `<b>At ${st().epoch}&nbsp;s the smoothing is at least as wide as the bands
      themselves, so the measures on this page are not separating theta from alpha.</b>` : ""}`);
  }
  else if(st().fft==="boxcar")
    B.push(`<b>The boxcar estimator applies no taper.</b> Frequency resolution is at its sharpest and
      spectral leakage at its worst: a strong rhythm just outside a band spills energy into it. It is
      the null comparison that shows what the Hann taper is doing, not a recommended setting.`);

  if(st().epoch <= 2)
    B.push(`<b>At ${st().epoch}&nbsp;second${st().epoch===1?"":"s"} the frequency grid is
      ${(1/st().epoch).toFixed(1)}&nbsp;Hz coarse${st().fft==="welch" ? ", and Welch halves that again" : ""}.</b>
      A 4&nbsp;Hz band is then only a handful of bins wide, so each point is a much rougher estimate of
      band power than the 4&nbsp;second default &mdash; more points, each worth less.`);

  if(st().base!=="off"){
    const missing = recs.length - shown.length;
    const flaggedB = shown.map(r=>baselineFlag(r)).filter(Boolean).length;
    const sg = segOf(), lvl = segShort(), bd = segBounds();
    B.push(`<b>Each point is the task value minus the median of that recording's
      ${lvl} baseline</b> (${bd[0]}&ndash;${bd[1]}&nbsp;s of the paired 6-minute block). Every
      processing and rejection setting above is applied to those baseline windows too.
      ${st().base==="log" && M().kind!=="asym"
        ? `In <b>log ratio</b> mode the plotted value is ln(task) &minus; ln(${lvl} median), so 0 means the task matched ${lvl} and +0.69 means double.`
        : `In <b>raw</b> mode the plotted value is a difference in the measure's own units, so 0 means the task matched ${lvl}.`}
      ${missing ? `<b>${missing} recording${missing===1?"":"s"} dropped</b> for want of a usable baseline. ` : ""}
      ${flaggedB ? `<b>${flaggedB} baseline${flaggedB===1?" is":"s are"} flagged</b> &mdash; see the panels.` : ""}`);
    if(sg === "math")
      B.push(`<b>Mental math is an active-task reference, not a resting one.</b> Zero here does
        not mean &ldquo;calm&rdquo;; it means the task looked like doing arithmetic. A value below
        zero is a task that engaged this measure LESS than deliberate mental effort did, which for
        a workload measure is a different claim from being below rest &mdash; and the two can point
        opposite ways on the same recording. Switch to <b>Resting</b> if what you want is the
        conventional baseline correction.`);
    /* Short-segment warning, driven by the segment's real length rather than by
       its name, so it follows the pipeline if a bound moves and fires for any
       future segment that is equally thin. */
    if(segLen() <= 30){
      const cap = Math.floor(segLen() / st().epoch);
      const nw = shown.map(r=>{const f=frame(r); return f&&f.baseline?f.baseline.n_windows:null;})
                      .filter(v=>v!=null);
      const worst = nw.length ? Math.min(...nw) : null;
      const unused = segLen() - cap*st().epoch;
      B.push(`<b>The ${lvl} segment is ${segLen()} seconds long.</b> At the current
        ${st().epoch}&nbsp;s window that is at most ${cap} window${cap===1?"":"s"}
        per recording${worst!=null?`, and the thinnest baseline on screen rests on ${worst}`:""}.
        ${unused > 0 ? `<b class="warn-ink">Only ${cap*st().epoch} of its ${segLen()} seconds are
          used</b> &mdash; the remaining ${unused % 1 ? unused.toFixed(1) : unused}&nbsp;s do not fill a
          whole ${st().epoch}&nbsp;s window, so they are dropped from the median as well as from the
          chart. A window length that divides ${segLen()} (1, 3 or 5&nbsp;s here) uses all of it.` : ""}
        ${cap<=1?"<b>A median over one window IS that window</b>, and no dispersion can be computed for it."
                :"A median over so few windows is not a stable level."}
        ${sg==="eyes_closed"?`This slice is also the tail of a phase of deliberate eye movements, so it
        carries the most ocular contamination of the three, and alpha rises with the eyes closed &mdash;
        expect a systematic shift against the other two references, not a small one.`:""}
        <b>Read it as indicative only.</b>`);
    }
  }
  $("#banners").innerHTML = B.map(b=>`<div class="banner">${b}</div>`).join("");
  /* The whole disclosure goes away when there is nothing to say, rather than
     sitting there as an empty "Warnings" bar inviting a pointless click. The
     count is on the summary so the bar is worth reading while shut. */
  $("#warn-wrap").hidden = B.length === 0;
  $("#warn-count").textContent = B.length;
}

function renderGrid(){
  const recs = candidates();
  /* The grid deliberately keeps a panel for a recording with no usable baseline
     -- blank, with the reason in its footer -- rather than dropping it, so a
     reader scanning the grid sees that it exists and why it is empty. But the
     tiles count only what is plotted, so the hint says which number is which;
     otherwise the page shows "20 recordings shown" above 22 panels. */
  const blank = recs.filter(r=>{const s=frame(r); return !s || s.noBaseline;}).length;
  $("#grid-hint").textContent =
    `Each panel is one ${TASK_LEN}task. Select a panel for the full trace, the electrode amplitudes behind it, and its signal-quality readout.`
    + (blank ? `  ${blank} of the ${recs.length} panels ${blank===1?"is":"are"} empty: `
             + `${blank===1?"that recording has":"those recordings have"} no usable `
             + `${segShort()} baseline, and each says so in its footer.` : "");

  $("#grid").innerHTML = recs.map(r=>{
    const s = frame(r);
    const q = quality(r);
    const bf = baselineFlag(r);
    const k = counts(r);
    const med = s && !s.noBaseline ? median(s.vals) : null;
    const foot = s && s.noBaseline ? `no ${segShort()} baseline`
      : isImputed() ? `${k.measured} real · ${k.fabricated} filled`
      : `${k.measured}/${k.total} windows`;
    const sub = tab==="index"
      ? `${r.framing} framing &middot; ${frontalUsed(r).join("+")}`
      : `${r.framing} framing &middot; ${M().channels.join(", ")}`;
    return `<button class="panel" data-key="${esc(r.key)}" type="button"
        aria-label="${esc(r.participant)} ${esc(r.arm)} ${esc(r.framing)}, median ${med==null?'not available':fmt(med)}, ${esc(q.text)}">
      <div class="stripe ${q.level}"></div>
      <div class="p-head">
        <div><div class="p-id">${esc(r.participant)} &middot; ${esc(r.arm)}</div>
             <div class="p-cond">${sub}</div></div>
        <div class="p-med">${fmt(med)}<small>median</small></div>
      </div>
      <div class="p-chart">${drawChart(r,{w:300,h:82,large:false})}</div>
      <div class="p-foot"><span>${foot}</span>
        <span style="display:flex;gap:5px;align-items:center">
        ${bf?`<span class="chip flag ${bf.level||"critical"}" title="${esc(bf.text)}">baseline</span>`:""}
        <span class="badge ${q.level}">${ICON[q.level]}${q.text}</span></span></div>
    </button>`;
  }).join("");
}

function renderDetail(){
  const slot = $("#detail-slot");
  const r = st().open ? DATA.recordings.find(x=>x.key===st().open) : null;
  if(!r || !candidates().includes(r)){ slot.innerHTML=""; return; }
  const s = frame(r);
  const q = quality(r);
  const bf = baselineFlag(r);
  const k = counts(r);
  const kept = s && !s.noBaseline ? s.vals.filter(v=>v!=null) : [];
  const ret = retentionFor(r);
  const bm = st().base!=="off" && s ? s.baseline : null;

  const chans = maskChannels(r);
  const meta = [
    [st().base==="off" ? "Median" : (st().base==="log" ? `Median ln(task/${segShort()})` : `Median Δ from ${segShort()}`),
      kept.length?fmt(median(s.vals)):"—"],
    ["Range", kept.length?`${fmt(Math.min(...kept))} – ${fmt(Math.max(...kept))}`:"—"],
    ["Windows plotted", `${k.measured + k.fabricated} / ${k.total}` +
      (s ? ` (${s.plottedPct.toFixed(1)}%)` : "")],
    ...(ocularFlag(r) ? [["Ocular correction", ocularFlag(r)]] : []),
    ...(s && s.excluded ? [["Windows excluded outright",
      `${s.excluded} of ${k.total} — ${s.excluded===1?"its samples are":"their samples are"}
       not contiguous in real time, so ${s.excluded===1?"its spectrum would be":"their spectra would be"}
       taken across a step in the signal. Dropped under every artifact mode,
       including "None", and never imputed.`]] : []),
    /* "could not be evaluated" is said out loud rather than shown as a dash
       beside a full-looking chart -- the chart in that case is UNFILTERED. */
    ["Windows surviving rejection", maskRefused(r)
      ? "criterion could not be evaluated \u2014 nothing was filtered"
      : ret!=null
        ? `${s.surviving} / ${s.eligible} (${ret.toFixed(1)}%)` +
          (s.unassessed ? ` \u2014 ${s.unassessed} NOT ASSESSED (${motWhyShort(s)}), kept for want of evidence rather than because they passed` : "")
        : "\u2014"],
  ];
  if(isImputed()) meta.push(["Windows fabricated", String(k.fabricated)]);
  meta.push(["Derivation", tab==="index"
      ? `theta mean(${frontalUsed(r).join(",")}) / alpha Pz`
      : tab==="fm_theta" ? "mean(F1,F2), time domain"
      : tab==="parietal_alpha" ? "Pz"
      : tab==="parietal_beta_asym" ? "ln(P4) − ln(P3)" : "ln(F2) − ln(F1)"]);
  meta.push(["Spectral estimate", `${st().epoch} s window · ${FFT_LABEL[st().fft]}`]);
  meta.push(["Reference", REF_LABEL[st().ref]]);
  /* Name the mode AND what it did here. "on" told a reader which button was
     pressed and nothing about this recording; with a manual list in play the
     two can disagree completely -- p09 interpolates O1 under `manual` and
     nothing under `automatic`. */
  {
    const done = interpChansOf(r), src = st().interpSrc;
    const held = (!st().frontalPair && FRONTAL.every(c => (r.interpLists[src]||[]).includes(c)))
               ? ` · ${FRONTAL.join(" and ")} held back at your request` : "";
    meta.push(["Interpolation",
      src === "off" ? "off — every channel read as recorded"
      : `${src === "manual" ? "manual list" : "automatic"} — ${
          done.length ? done.join(", ") + " replaced by a spline" : "nothing replaced here"}${held}`]);
  }

  if(isWindowed()){
    /* One number per channel would be a fiction here -- the threshold is
       recomputed for every window -- so report the RANGE it took across the
       recording, which is what a reader needs in order to distrust retention
       under it. */
    const rng = slideRange(r.key, chans, false);
    meta.push([`Windowed threshold, ${(+st().sigmaK).toFixed(2)} × σ of the ${+st().slideS} s around each window`,
      rng == null ? "—" : rng.map(p2 =>
        p2 == null ? "—" : p2[0].toFixed(0) + (p2[1] > p2[0] ? "–" + p2[1].toFixed(0) : "")
      ).join(" / ")+" µV"]);
    /* How much of the recording each threshold actually saw. A signal shorter
       than the sliding length gives every window the same span -- the whole
       signal -- so the mode degenerates to plain Robust, and the reader should
       not have to infer that from a range that happens not to vary. */
    meta.push(["Calibration span",
      slideDegenerate(r)
        ? `${(+st().slideS)} s window over ${(r.duration_s||0).toFixed(0)} s of signal — every window sees all of it, so this is identical to Robust here`
        : `${(+st().slideS)} s centred on each window, truncated at the two ends`]);
  } else if(isMotion()){
    /* Every other rejection mode shows the number it is actually cutting at;
       this one did not, so a reader could not see what "3 sigma of this
       recording's movement" came out as. Shown as a bare number: the sensor's
       units are not documented, and naming one here would be inventing it. */
    /* Under the union there are TWO thresholds and no single number describes
       the rule, so both are named. Printing one of them under a label that says
       "union" would describe half the criterion. */
    meta.push([`Motion threshold, ${(+st().motK).toFixed(2)} × σ of ${
        isMotUnion() ? "each source (reject if EITHER exceeds)"
                     : (MOT_LABEL[motSrc()]||motSrc())}`,
      (s && s.motParts && s.motParts.length)
        ? s.motParts.map(p => `${MOT_LABEL[p.src]||p.src} ${
            isFinite(p.thr) ? p.thr.toPrecision(3) : "\u2014"} (median ${
            p.med!=null ? p.med.toPrecision(3) : "\u2014"})`).join("; ")
          + ", sensor units"
        : "\u2014"]);
    if(s && s.unassessed)
      meta.push(["Windows not assessed",
        `${s.unassessed} of ${k.total} — ${motWhyLong(s)}`]);
  } else if(maskRule(st().art)==="robust"){
    const sg = (r.sigma||{})[st().oc] || {};
    meta.push([`Robust threshold, ${(+st().sigmaK).toFixed(2)} × σ`,
      chans.map(c=> sg[c]!=null ? (st().sigmaK*sg[c]).toFixed(0) : "—").join(" / ")+" µV"]);
  } else if(maskRule(st().art)==="cap"){
    meta.push(["Absolute cap", `${(+st().holmCap).toFixed(0)} µV, same for every recording`]);
  }

  const damp = displayedAmp(r);
  meta.push([`${chans.join(" / ")} robust SD, as plotted`,
             chans.map(c=>fmt(damp[c])).join(" / ")+" µV"]);
  meta.push([`${chans.join(" / ")} robust SD, as recorded`,
             chans.map(c=>fmt(r.amp[c])).join(" / ")+" µV"]);
  meta.push(["Task duration", `${(r.duration_s/60).toFixed(2)} min`]);
  meta.push(["Bad channels", r.bad.length ? r.bad.join(", ") : "none"]);
  if(st().base!=="off"){
    /* Name the recording AND the slice of it that produced the level being
       subtracted, so a panel is self-describing: two of the three segments come
       from the same file and differ only by their bounds. */
    if(r.base && !r.base.missing)
      meta.push(["Baseline subtracted",
        `${esc(r.base.key)} · ${segLabel()} ${segBounds()[0]}–${segBounds()[1]} s`]);
    meta.push(["Baseline median", bm && bm.median!=null
      ? `${fmt(bm.median)} ± ${bm.se!=null?fmt(bm.se):"—"} (${bm.n_windows} of ${bm.n_avail} ${segShort()} windows)` : "—"]);
    if(r.base && !r.base.missing)
      meta.push([`Baseline ${chans.join("/")} robust SD, ${segShort()}`,
        chans.map(c=>fmt((baseSeg(r)||{amp:{}}).amp[c])).join(" / ")+" µV"]);
  }

  const notes = [];
  if(q.level!=="good" || q.speak) notes.push(q.reason.endsWith(".")?q.reason:q.reason+".");
  if(bf) notes.push("Baseline: "+bf.text+".");
  if(r.repair) notes.push("Segment repair: "+r.repair+".");
  if(r.n_gaps) notes.push(`${r.n_gaps} internal data gap(s), largest ${r.max_gap_s}s — window times come from the recorded timestamps, not an assumed grid.`);
  (r.notes||[]).forEach(n=>notes.push(n.charAt(0).toUpperCase()+n.slice(1)+"."));
  if(r.base && !r.base.missing && r.base.repair) notes.push("Baseline segment repair: "+r.base.repair+".");

  slot.innerHTML = `<div class="detail">
    <div class="d-head">
      <div><div class="d-title">${esc(r.participant)} &middot; ${esc(r.arm)} &middot; ${esc(r.framing)}</div>
        <div class="d-sub">${esc(r.recording)} &mdash; ${esc(M().tab.toLowerCase())} per ${st().epoch}-second window,
          ${esc(ART.find(a=>a.id===st().art).label)} &middot; ocular ${
            st().oc==="none" ? "uncorrected"
            : st().oc==="ica" ? "ICA" : "regressed"}
          &middot; ${esc(REF_LABEL[st().ref])} &middot; ${esc(FFT_LABEL[st().fft])}
          ${st().base!=="off"?`&middot; baseline ${st().base==="log"?"log ratio":"subtracted"}`:""}
          ${tab==="index"?'&middot; <span class="chip">F1+F2</span>':''}</div></div>
      <div style="display:flex;gap:10px;align-items:center">
        <span class="badge ${q.level}">${ICON[q.level]}${q.text}</span>
        <button class="close" id="close-detail" type="button">Close</button></div>
    </div>
    <div class="d-body" id="big-chart">${drawChart(r,{w:1240,h:290,large:true})}</div>
    <div class="d-meta">${meta.map(([a,b])=>`<div><div class="k">${a}</div><div class="v">${b}</div></div>`).join("")}</div>
    ${notes.length?`<div class="d-note">${notes.map(esc).join(" ")}</div>`:""}
    ${baselineViewerHTML(r)}
  </div>`;
  $("#close-detail").onclick = ()=>{ st().open=null; render(); };
  attachHover(r);
  attachBaselineHover(r);
}

/* The three baseline references, graphed. Always present in an expanded panel,
   whether or not correction is on -- the reader should be able to SEE what a
   reference looks like before deciding to subtract it, and to compare the three
   without changing the analysis. Left/right arrows move between them. */
function baselineViewerHTML(rec){
  /* Which segments this RECORDING actually has. `detailSeg` is per-tab, so it
     can point at a segment the newly opened recording lacks; fall back for
     display rather than mutating state mid-render, which would leave a radio
     both checked and disabled. */
  const has = sname => !!(rec.base && !rec.base.missing && (rec.base.segs||{})[sname]);
  const avail = SEG_ORDER.filter(has);
  const seg = avail.includes(detailSegOf()) ? detailSegOf() : (avail[0] || detailSegOf());
  const bs = baselineSeries(rec, seg);
  const bnd = segBounds(seg);
  const corrOn = st().base !== "off";
  const isCorr = corrOn && seg === segOf();
  const blockS = (rec.base && rec.base.block_s) || null;

  const opts = SEG_ORDER.map(sname =>
    `<button type="button" role="radio" data-k="detailSeg" data-v="${sname}"
      aria-checked="${sname===seg}" aria-pressed="${sname===seg}"
      ${has(sname)?"":'disabled aria-disabled="true" title="this baseline block has no such segment"'}
      >${esc(segLabel(sname))}</button>`).join("");

  /* The axis FOLLOWS THE SCALE CONTROL, via the same effScale() the task chart
     uses. A raw baseline level is positive, so a log axis would be defined for
     it even while the correction forces the task chart linear -- but drawing one
     would put a log chart directly beneath a Scale control that is greyed out
     reading "Linear", with no way for the reader to change it. A slightly worse
     axis beats a control that does not describe what is on screen. */
  const log = effScale()==="log";

  /* What the windows actually COVER, which is not the same as the segment's
     bounds. Windows are whole epochs and a partial tail is dropped (the pipeline
     epochs everything this way), so a segment whose length is not a multiple of
     the window length loses its final seconds -- 3 of the eyes-closed segment's
     15 s at the 4 s default, 7 s at an 8 s window. That tail is missing from this
     chart AND from the median computed beside it, so it is stated rather than
     left to be inferred from an axis that stops early.

     Note also that the x labels are window START times, so the last label always
     reads one window length short of the covered end. That alone is not missing
     data; this line is what distinguishes the two. */
  const cov0 = bnd[0], cov1 = bnd[0] + (bs.missing ? 0 : bs.n * st().epoch);
  const dropped = bnd[1] - cov1;
  const stat = bs.missing ? "" :
    `<span>${bs.kept} of ${bs.nAvail} window${bs.nAvail===1?"":"s"} kept
     &middot; median <b>${bs.median!=null?fmt(bs.median):"&mdash;"}</b>${
       bs.kept===1?' &middot; <b class="warn-ink">the median IS this single window</b>':""}</span>
     <span>&middot; windows cover ${cov0}&ndash;${cov1} s${
       dropped > 0
         ? ` &middot; <b class="warn-ink">the last ${dropped % 1 ? dropped.toFixed(1) : dropped} s
             of this segment (${Math.round(100*dropped/(bnd[1]-bnd[0]))}%) do not fill a
             ${st().epoch} s window and are used neither here nor in the median</b>`
         : ""}</span>`;

  /* What actually carries over from the controls above, stated exactly. The
     rejection rule and every processing setting do; IMPUTATION does not -- a
     baseline is never imputed, here or in the pipeline, because a median has no
     gaps to fill. So in an imputed mode the task chart above is continuous and
     dashed while this one shows real gaps, and that difference is the setting,
     not a defect. */
  /* The windowed mode's threshold on THIS segment, which is not the task's: each
     segment is calibrated on its own samples, and on a segment shorter than the
     calibration window every window sees all of it, so the mode collapses to
     plain Robust there. Uses slideRange's baseline branch. */
  let winNote = "";
  if(isWindowed() && !bs.missing){
    const rng = slideRange(bs.spKey, maskChannels(rec), true);
    /* The segment's TRUE length, from its bounds -- not n_windows x epoch, which
       drops the tail. On the 15 s eyes-closed segment at the 4 s default that
       difference is 12 s vs 15 s, and it flips this claim the wrong way at the
       20 s calibration setting: the page would assert the mode had collapsed to
       plain Robust when window 0 is in fact judged against a different span from
       windows 1 and 2. */
    const segT = bnd[1] - bnd[0];
    winNote = ` &middot; windowed threshold ${rng==null?"—":rng.map(q=>
        q==null?"—":q[0].toFixed(0)+(q[1]>q[0]?"–"+q[1].toFixed(0):"")).join(" / ")} µV${
      (2*segT - st().epoch) <= (+st().slideS)
        ? ` &mdash; short enough that every window's span covers the whole segment, so the
           mode is plain Robust here` : ""}`;
  }

  const applies = bs.maskRefused
    ? `&middot; <b class="warn-ink">the rejection criterion could not be evaluated &mdash;
       this trace is unfiltered</b>`
    : `&middot; ${log?"log":"linear"} axis, ${st().epoch}-second windows
       &middot; the rejection rule and every processing setting above apply here too${
         isImputed()?", except imputation: a baseline is never imputed, so gaps here are real":""}${winNote}`;

  return `<div class="d-base">
    <div class="d-base-head">
      <div><div class="d-base-title">Baseline reference &mdash; ${esc(segLabel(seg))}
        <span class="d-base-range">${bnd[0]}&ndash;${bnd[1]} s of the ${
          blockS?`${(blockS/60).toFixed(1)}-minute`:"baseline"} block</span></div>
        <div class="d-base-sub">${esc(rec.base && !rec.base.missing ? rec.base.key : "no paired baseline")}
          &middot; ${esc(M().tab.toLowerCase())} in its own units, uncorrected
          &middot; ${isCorr ? "<b>this is the level being subtracted above</b>"
                    : corrOn ? `not the segment being subtracted (that is ${esc(segLabel(segOf()))})`
                    : "correction is off &mdash; nothing is being subtracted"}</div></div>
      <div class="seg d-base-seg" role="radiogroup" aria-label="Baseline segment to graph"
      aria-describedby="d-base-hint">${opts}</div>
    </div>
    <div class="d-base-body" id="base-chart">${
      drawChart(rec,{w:1240,h:190,large:true,series:bs,log,corrected:false,
        xAt:i=>bnd[0]+i*st().epoch, xFmt:t=>`${Math.round(t)} s`,
        label:`${M().tab} across the ${segShort(seg)} baseline segment`})}</div>
    <div class="d-base-foot">${stat}${bs.missing?"":applies}
      <span class="d-base-hint" id="d-base-hint">&larr; &rarr; to compare references</span></div>
  </div>`;
}

function attachHover(rec){
  const host = $("#big-chart"); if(!host) return;
  const svg = host.querySelector("svg"); if(!svg || !svg.dataset.n) return;
  const tt = $("#tt");
  const s = frame(rec); if(!s) return;
  const times = rec.times ? rec.times[ekey()] : null;
  const xs = exclSet(rec.excl);        // hoisted: this fires on every mousemove
  const n = +svg.dataset.n, padl=+svg.dataset.padl, padr=+svg.dataset.padr;
  svg.style.cursor="crosshair";
  svg.addEventListener("mousemove", e=>{
    const b = svg.getBoundingClientRect();
    const f = ((e.clientX-b.left)/b.width*1240 - padl)/(1240-padl-padr);
    const i = Math.round(Math.max(0,Math.min(1,f))*(n-1));
    const v = s.vals[i];
    const t = (times && times[i]!=null) ? times[i] : i*st().epoch;
    /* An excluded window is null but was never offered to the criterion, so
       calling it "rejected" would blame the artifact rule for a splice -- and in
       mode `none`, where nothing is rejected at all, would contradict the banner
       directly above the chart. */
    const label = (xs && xs.has(i)) ? "excluded \u2014 samples not contiguous in real time"
      : v==null ? "rejected" : (s.imputed[i] ? fmt(v)+" (interpolated)" : fmt(v));
    tt.innerHTML = `<div class="tt-k">${(t/60).toFixed(2)} min &middot; window ${i+1}</div>
      <div class="tt-v">${label}</div>`;
    tt.style.left = Math.min(e.clientX+14, window.innerWidth-170)+"px";
    tt.style.top = (e.clientY-46)+"px";
    tt.classList.add("on");
  });
  svg.addEventListener("mouseleave", ()=>tt.classList.remove("on"));
}

/* Same tooltip for the baseline chart. x is seconds from the START OF THE BLOCK,
   not of the segment, so the number under the cursor is the one the protocol
   timeline is written in.

   NOMINAL, not recorded: baselines ship no per-window times (write_variant_archives
   sends times_s for tasks only), so this assumes a contiguous grid -- the very
   assumption the task axis avoids because a splice puts it minutes wrong. Safe
   here only because the segment is short and the block is delimited by markers;
   a baseline carrying a `repair` note is the case to distrust, and the panel
   already prints that note. */
function attachBaselineHover(rec){
  const host = $("#base-chart"); if(!host) return;
  const svg = host.querySelector("svg"); if(!svg || !svg.dataset.n) return;
  const tt = $("#tt");
  /* Same fallback the viewer applies, so the tooltip cannot describe a different
     segment from the one drawn. */
  const has = sname => !!(rec.base && !rec.base.missing && (rec.base.segs||{})[sname]);
  const av = SEG_ORDER.filter(has);
  const seg = av.includes(detailSegOf()) ? detailSegOf() : (av[0] || detailSegOf());
  const bs = baselineSeries(rec, seg); if(bs.missing) return;
  const lo = segBounds(seg)[0];
  const n = +svg.dataset.n, padl=+svg.dataset.padl, padr=+svg.dataset.padr;
  svg.style.cursor="crosshair";
  svg.addEventListener("mousemove", e=>{
    const b = svg.getBoundingClientRect();
    const f = ((e.clientX-b.left)/b.width*1240 - padl)/(1240-padl-padr);
    const i = Math.round(Math.max(0,Math.min(1,f))*(n-1));
    const v = bs.vals[i];
    /* The SPAN, not just the start: on a short segment the difference between
       "233 s" and "233-237 s" is the difference between a reader thinking the
       chart ends early and understanding that a point IS a window. */
    const t0 = lo + i*st().epoch;
    tt.innerHTML = `<div class="tt-k">${t0.toFixed(0)}&ndash;${(t0+st().epoch).toFixed(0)} s &middot; window ${i+1} of ${n}</div>
      <div class="tt-v">${v==null?"rejected":fmt(v)}</div>`;
    tt.style.left = Math.min(e.clientX+14, window.innerWidth-170)+"px";
    tt.style.top = (e.clientY-46)+"px";
    tt.classList.add("on");
  });
  svg.addEventListener("mouseleave", ()=>tt.classList.remove("on"));
}

function renderTable(){
  const recs = candidates();
  const isIdx = tab==="index";
  const rows = recs.map(r=>{
    const s = frame(r);
    const kept = s && !s.noBaseline ? s.vals.filter(v=>v!=null) : [];
    const k = counts(r); const q = quality(r); const ret = retentionFor(r);
    const bm = st().base!=="off" && s ? s.baseline : null;
    const chans = maskChannels(r);
    return `<tr><td>${esc(r.participant)}</td><td>${esc(r.arm)}</td><td>${esc(r.framing)}</td>
      <td>${esc(chans.join(", "))}</td>
      <td class="num">${kept.length?fmt(median(s.vals)):"—"}</td>
      <td class="num">${kept.length?fmt(Math.min(...kept)):"—"}</td>
      <td class="num">${kept.length?fmt(Math.max(...kept)):"—"}</td>
      <td class="num">${k.measured}/${k.total}</td>
      ${isImputed()?`<td class="num">${k.fabricated||"—"}</td>`:""}
      <td class="num">${ret!=null?ret.toFixed(1):"—"}</td>
      <td class="num">${chans.map(c=>fmt(displayedAmp(r)[c])).join(" / ")}</td>
      ${st().base!=="off"?`<td class="num">${bm&&bm.median!=null?fmt(bm.median):"—"}</td>
        <td class="num">${bm&&bm.n_windows!=null?bm.n_windows:"—"}</td>`:""}
      <td><span class="badge ${q.level}">${ICON[q.level]}${q.text}</span></td></tr>`;
  }).join("");
  $("#table").innerHTML = `<thead><tr>
    <th>Participant</th><th>Arm</th><th>Framing</th><th>Channels</th>
    <th>${st().base==="off" ? "Median"
           : st().base==="log" ? `Median ln(task/${segShort()})`
           : `Median Δ from ${segShort()}`}</th><th>Min</th><th>Max</th>
    <th>Measured</th>${isImputed()?"<th>Fabricated</th>":""}<th>Kept %</th>
    <th>Robust SD µV</th>
    ${st().base!=="off"?`<th>${segLabel()} med</th><th>${segShort()} win.</th>`:""}
    <th>Quality</th>
  </tr></thead><tbody>${rows}</tbody>`;
}

function renderExcluded(){
  const rows = [];
  DATA.excluded_upstream.forEach(e=>rows.push(
    `<tr><td>${esc(e.participant)}</td><td>${esc(e.recording)}</td><td>Excluded upstream</td>
     <td>—</td><td class="num">—</td><td>${esc(e.reason)}</td></tr>`));

  const shownKeys = new Set(candidates().map(r=>r.key));
  DATA.recordings.forEach(r=>{
    if(shownKeys.has(r.key)) return;
    // Nothing is gated for signal quality any more, so a recording is absent only
    // when the sweep genuinely holds no data for this processing combination.
    let stage, reason;
    const sh = shard();
    /* When vk() is null the archive carries no token for this mode, and the
       shard's `missing` row is keyed by the MODE name instead -- so matching on
       `m.variant === null` would never fire and the page told the reader "no
       windows at this length", which is the false statement the whole `missing`
       mechanism exists to prevent. */
    const _vk = vk(r.key);
    const miss = sh && (sh.missing||[]).find(m => m.key === r.key
      && (_vk === null ? m.mode === interpMode() : m.variant === _vk));
    if(miss){
      stage = "Not produced by the pipeline";
      reason = `${miss.why} (${st().epoch}\u2009s / ${FFT_LABEL[st().fft]} / ${
                  _vk === null ? interpMode() : _vk}). `
             + `This is a gap in the sweep, not a consequence of the controls; `
             + `re-run the pipeline, then build_dashboard.py.`;
    } else {
      stage = "No data in this combination";
      reason = "the pipeline produced no window for this recording at this window length";
    }
    rows.push(`<tr><td>${esc(r.participant)}</td><td>${esc(r.recording)}</td><td>${stage}</td>
      <td>—</td>
      <td class="num">${(tab==="index"?["F1","F2","Pz"]:M().channels).map(c=>`${c} ${fmt(r.amp[c])}`).join(" / ")} µV</td>
      <td>${esc(reason)}</td></tr>`);
  });

  if(st().base!=="off")
    candidates().forEach(r=>{ const s=frame(r); if(!s || !s.noBaseline) return;
      const missing = r.base && r.base.missing;
      const sp0 = baseSeg(r);
      const n = (sp0 && sp0.idx && sp0.idx[ekey()]) ? sp0.idx[ekey()].length : null;
      rows.push(`<tr><td>${esc(r.participant)}</td><td>${esc(r.recording)}</td>
        <td>${missing?"No baseline recording":!sp0?`No ${segShort()} segment`:"No surviving baseline window"}</td>
        <td>${esc(maskChannels(r).join(", "))}</td>
        <td class="num">—</td>
        <td>${missing
          ? esc(r.base.reason||"paired baseline block could not be delimited")
          : !sp0
            ? `the paired baseline block does not extend to
               ${segBounds()[0]}&ndash;${segBounds()[1]}&nbsp;s, so it has no
               ${segShort()} segment. Choose another baseline segment to include
               this recording.`
          : (!bValsOf(sp0.key)
             ? `the sweep holds no baseline values for this processing combination, so no
                ${segShort()} level could be computed. This is a gap in the data, not a
                consequence of the controls.`
             : `the ${segShort()} segment exists${n!=null?` and holds ${n} window${n===1?"":"s"}`:""}, but none
                survives the selected artifact criterion, so there is no level to
                subtract`)}</td></tr>`); });

  $("#excl-hint").textContent =
    "Which recordings dropped out, and why, is part of the result — so nothing is omitted silently. This list moves as you move the controls.";

  const tbl = $("#excluded"), empty = $("#excluded-empty");
  if(rows.length){
    tbl.hidden = false; empty.hidden = true;
    tbl.innerHTML = `<thead><tr><th>Participant</th><th>Recording</th><th>Stage</th>
      <th>Failed channels</th><th>Robust SD, as recorded</th><th>Reason</th></tr></thead><tbody>${rows.join("")}</tbody>`;
  } else {
    tbl.hidden = true; empty.hidden = false; tbl.innerHTML = "";
    empty.textContent = "Nothing excluded in this view.";
  }
}

function renderMethod(){
  $("#method").innerHTML = `
  <div><h3>What the controls change</h3>
    <p>The five controls on the top row change the <em>values</em>: they alter the signal or the
    spectrum, so the pipeline computed every one of the
    ${DATA.n_precomputed_cells.toLocaleString()} combinations in advance. The threshold on the
    second row changes only <em>which windows count</em> &mdash; never which recordings, since
    nothing is withheld &mdash; so this page evaluates it itself and it moves continuously. Every
    default reproduced the pipeline as it ran before these controls existed until
    2026-09-05, when the opening cell was set deliberately instead: interpolation on,
    EOG regression, average reference, Welch, 4&nbsp;s windows, head-motion rejection
    on either sensor at 3&nbsp;&sigma;, and a log-ratio correction against the resting
    baseline. The cell the pipeline verifies against an independent estimator is still
    the old one (hardware reference, Hann, no ocular correction), which is four controls
    away from what you see first.</p>
    <p><strong>Fewer arrays are stored than there are combinations to select.</strong>
    The three interpolation sources and the frontal-pair checkbox make five swept modes,
    but the archive is keyed by <em>which channels were replaced</em> rather than by which
    mode asked for them &mdash; so two modes that reach the same list share one stored
    array. ${(DATA.n_stored_arrays||0).toLocaleString()} value arrays are stored where a
    dense grid over all five modes would have needed
    ${(DATA.n_dense_arrays||0).toLocaleString()}. Nothing is approximated by this: two modes
    share an array only when they are, channel for channel, the same instruction.</p>
    <p><strong>The manual list is a judgement, not a measurement.</strong> It is written
    down per participant and arm, by hand, and nothing in the pipeline derives it or can
    check it against the data. Where it is empty for a recording &mdash; which it is for
    most &mdash; the manual mode interpolates nothing at all, and that is an instruction
    rather than a gap.</p></div>
  <div><h3>Baseline correction</h3>
    <p>Each task is paired with the baseline recording made under the same condition in the same
    session. That block runs six minutes between two markers: two minutes of mental arithmetic,
    two of eye movements, then two of rest. <strong>Three segments of it can be subtracted</strong>,
    chosen with <em>Baseline segment</em>:</p>
    <ul>
      <li><strong>Resting (eyes open)</strong>, 240&ndash;360&nbsp;s &mdash; the conventional
      reference, the default, and the one every published number here was computed against.</li>
      <li><strong>Mental math</strong>, 0&ndash;120&nbsp;s &mdash; an <em>active-task</em>
      reference. Subtracting it asks how the task compared with deliberate mental effort, not with
      rest, so zero means something different and the sign can point the other way.</li>
      <li><strong>Eyes closed</strong>, 225&ndash;240&nbsp;s &mdash; the last 15 seconds of the
      eye-movement phase. <strong>15 seconds is 3 windows at the 4&nbsp;s default and one window
      at 8&nbsp;s or longer</strong>, and it sits at the end of a phase of deliberate eye
      movements, so it is the least stable and most ocular-contaminated of the three.</li>
    </ul>
    <p>The median of the selected segment's windows is subtracted from every task window, and every
    setting above &mdash; window length, estimator, reference, interpolation, rejection rule and
    threshold &mdash; is applied to those windows too. Each segment is cropped, epoched and
    calibrated on <em>its own samples</em>, so a segment's robust threshold and window grid come
    from the segment being subtracted rather than from the block around it. The three are different
    questions, not three estimates of one answer: switching changes what zero means.</p></div>
  <div><h3>Artifact rejection</h3>
    <p><strong>None</strong> plots every window the recording can offer. The one thing no
    rejection setting restores is a window whose own samples are not contiguous in real time:
    one analysed recording contains a discontinuity &mdash; a 449&nbsp;s interval excised from
    p02/task_ai_speedscore and a 3.91&nbsp;s dropout in p04/task_agent_personal &mdash; and a
    window spanning one has its spectrum taken across a step in the signal rather than across
    anything an electrode measured. Those windows are dropped at every window length and under
    every mode, including this one, because that is a property of the recording rather than a
    judgement about the EEG. <strong>Robust</strong> rejects a window whose
    channels exceed that recording's own <em>k</em>&nbsp;&times;&nbsp;1.4826&nbsp;&times;&nbsp;MAD;
    because the threshold rescales with each recording's noise, retention under it is <em>not</em>
    comparable across recordings. <strong>Windowed Robust</strong> applies that same rule with the
    MAD recomputed <em>for every window</em> from the ${SL_LABEL()} centred on it, so a window is
    judged against its own neighbourhood rather than against the whole recording &mdash; which
    catches a local excursion the recording-wide threshold averages away, at the cost of a
    threshold that now varies <em>within</em> a recording as well as between them. A stretch that
    was bad throughout raises its own threshold and keeps most of its windows, so retention in this
    mode is not comparable between moments either. Its per-window &sigma; is the one browser-side
    threshold the page cannot derive from what it already holds, so the pipeline computes it &mdash;
    which is why the length is a fixed toggle (${SLIDE_CHOICES.map(v=>`${+v}`).join(", ")}&nbsp;s)
    rather than a slider: each position is a stored column. The window is CENTRED, so every analysis window has the same
    amount of context on both sides and the threshold moves smoothly; at the two ends the span is
    truncated to the samples that exist rather than shifted inwards, so an edge window is
    calibrated on as little as half the length. A shorter window tracks the signal more closely and
    is more easily fooled by a sustained bad patch; a longer one is steadier and slower to react.
    (Until 2026-09-02 this mode tiled the recording into fixed 60&nbsp;s blocks instead. Two windows
    a second apart on either side of a boundary were then judged against entirely disjoint minutes,
    and a window at a block edge had 59&nbsp;s of context on one side and 1&nbsp;s on the other.)
    Note the span is measured in <em>recorded data</em>, not session time &mdash; where a recording
    has an excised dropout, the samples either side of it are neighbours here.
    <strong>Head motion</strong> is the one rule that does not read the EEG. The helmet carries an
    inertial sensor sampled at 50&nbsp;Hz alongside the 250&nbsp;Hz electrodes, and this rule rejects
    a window whose mean movement exceeds <em>k</em> robust standard deviations above that
    recording's own typical movement. The threshold is relative because the sensor's units are not
    documented in this project &mdash; an absolute cut would be invented rather than measured.
    Three rules are offered over two streams: the <strong>accelerometer</strong>, as the rate of
    change of the acceleration vector (the accelerometer is dominated by gravity, so its
    <em>movement</em> is the informative part, not its magnitude); the <strong>gyroscope</strong>,
    as rotation rate; and <strong>Either</strong>, which rejects a window that either stream flags,
    each against its own threshold.
    <em>Every figure in the rest of this panel was measured on the study this page&rsquo;s code was
    written for, not on the recordings currently loaded. They are kept because they are the
    evidence for offering these three choices at all; read them as that study&rsquo;s results, and
    expect your own to differ.</em>
    The two streams agree less than they look like they should
    &mdash; their values correlate +0.84 within a recording, but at the 3&nbsp;&sigma; default they
    share only 129 of the 418 windows one or other rejects. The accelerometer is the better single
    detector on <em>that</em> dataset: against each recording&rsquo;s own per-window peak amplitude it
    scores &rho; +0.68 to the gyroscope&rsquo;s +0.61, wins in 17 of the 19 recordings that have
    usable motion, and at a matched rejection budget catches about twice as much of each
    recording&rsquo;s loudest tail. <strong>Either</strong> is nonetheless what this page opens on,
    being the sensitive setting: it costs 7.3% of windows against the accelerometer&rsquo;s 5.5%
    and the gyroscope&rsquo;s 4.1%, and picks up what the gyroscope alone would have found. The
    magnetometer is not offered &mdash; it is heavily quantised and measures heading against an
    external field, and it was not evaluated here. Measured on <em>that</em> dataset, windows in the top
    decile of movement carry about 5.2&times; the frontal theta and 2.7&times; the parietal alpha of
    the remainder, and at the default 3&nbsp;&sigma; the rule removes about 5% of windows carrying
    several times the frontal theta of those kept (a median of per-recording ratios, 6&ndash;8&times;
    depending on the recordings counted). <strong>It is not, however, finding windows the
    amplitude rules miss:</strong> a mean of 1% of top-decile-motion windows survive <em>Robust</em>
    at 5&nbsp;&sigma;, and movement correlates with per-window peak amplitude at +0.67 median across
    every recording with a working sensor. The two criteria largely agree. What this rule offers is
    <em>independence</em> &mdash; it reaches that verdict without reading the EEG whose spectrum is
    then plotted, whereas every other rule thresholds the same signal it reports. Windows the sensor
    did not cover, and windows spanning an excised stretch, are kept, counted, and named as
    unassessed on the panel rather than folded into the surviving count. Recordings whose IMU
    reported no variation at all are treated as having no motion data rather than as perfectly
    still. Because the rejection modes are mutually exclusive, this runs instead of an amplitude
    rule rather than on top of one.
    <strong>Absolute cap</strong> applies one fixed microvolt ceiling to
    every recording (Holm's own criterion is 70&nbsp;&micro;V; it is currently at
    ${(+st().holmCap).toFixed(0)}&nbsp;&micro;V) and <em>is</em> comparable.
    All of them are evaluated on the <em>measured</em> signal, before interpolation and before
    re-referencing,
    so changing those toggles never changes which windows survive. <strong>Robust + impute</strong> and
    <strong>Cap + impute</strong> apply those same two rejections and then fill the discarded windows by
    linear interpolation, reporting the fabricated fraction; the filled values are drawn dashed and are
    never counted as measured.</p></div>
  <div><h3>No signal-quality gate</h3>
    <p>Every task recording is shown on every tab; none is withheld for its electrode amplitudes, and
    the index numerator is always the full F1+F2 midline mean. An earlier build gated the index on a
    fixed 1&ndash;50&nbsp;&micro;V robust-SD range, but that bound had no published basis and rejected
    genuinely-measured channels (some clean electrodes here sit above 50&nbsp;&micro;V), so it was
    removed &mdash; from this page and, as of 2026-09-02, from the pipeline itself, which no longer
    refuses an index or falls back to a single frontal channel. That fallback is the reason the
    removal matters beyond how many recordings you see: it silently changed the index's
    <em>derivation</em> on the recordings it touched, so one column held two different quantities.
    Signal quality is surfaced instead of acted on: each panel is badged, and the electrode
    amplitudes are in the table, so a recording built on unusual or interpolated electrodes says
    so plainly.</p>
    <p><strong>What the badge measures against, and where that number comes from.</strong>
    Until 2026-09-03 the badge used a fixed 1&ndash;50&nbsp;&micro;V robust-SD range. That range had
    <strong>no published source</strong>: it was invented for this repository, hardcoded, and
    described in a comment as &ldquo;a plausible scalp-EEG range&rdquo; with nothing behind it. It
    was never derived from electrode data of any kind, and in particular not from <strong>dry</strong>
    electrodes &mdash; this helmet is 100% dry, where contact impedance, drift and movement
    sensitivity all run higher than a gelled cap. The dataset disagreed with it directly: 18.6% of
    all channel-measurements exceeded 50&nbsp;&micro;V, including about 40% of F1 and F2, the index
    channels.</p>
    <p>The band is now <strong>derived from this dataset</strong>: ${BAND_TXT}, being
    ${BAND_WHY}. It is pooled across channels rather than fitted per channel, so a chronically noisy
    electrode cannot make its own noise the standard it is judged by; it is fitted in log space,
    because amplitudes are ratio-scaled; and its spread comes from the lower half of the
    distribution only, because the upper tail is the thing being detected and must not be allowed to
    set its own threshold. ${AB.pct_outside}% of channel-measurements fall outside it.</p>
    <p><strong>Read it as relative, not absolute.</strong> It says &ldquo;unusual for this helmet on
    these recordings&rdquo;, not &ldquo;physiologically implausible&rdquo;. No absolute claim is
    available here without a hardware noise specification or a published dry-electrode reference,
    and asserting one is how the number it replaced came to exist. If every electrode in a dataset
    were bad, this band would call them all ordinary &mdash; which is precisely why it labels rather
    than withholds.</p>
    <p><strong>One known mismatch, worth watching.</strong> The band is fitted on
    <em>hardware-referenced</em> amplitudes, but the badge judges the amplitude of whatever is
    currently plotted. Re-referencing moves that number a long way &mdash; average-referencing takes
    one recording's frontal and parietal channels from about 20 to about 142&nbsp;&micro;V &mdash; so
    under <strong>Average</strong> or <strong>REST</strong> the badge is comparing against a band
    that never saw data on that scale, and will over-flag. This predates the derived band (the old
    fixed range had the same mismatch) and is not corrected here. On the default hardware reference
    the comparison is like-for-like.</p></div>
  <div><h3>References</h3>
    <p><strong>Hardware</strong> keeps the recording's own SRB2/earlobe reference, which is how the
    cognitive-load index is defined. <strong>Average</strong> applies the example pipeline's average of
    the ten EEG channels. <strong>REST</strong> estimates the signal against a reference at infinity
    through a spherical head model. The choice matters most for the two asymmetry measures: any
    reference that mixes channels means a left&ndash;right difference is no longer a difference
    between two independent sites.</p></div>
  <div><h3>Window length and estimator</h3>
    <p>At the current setting the spectrum is resolved in steps of about
    ${(() => { const b = DATA.band_bins[st().epoch+"|"+st().fft];
               return (b && b.freq_resolution_hz != null
                       ? b.freq_resolution_hz : 1/st().epoch).toFixed(2); })()}&nbsp;Hz. Shorter
    windows track the task more closely but estimate band power from very few frequency bins; longer
    windows are steadier but blur transient changes. <strong>Hann</strong> is Holm's method and the
    default; <strong>multitaper</strong> and <strong>Welch</strong> trade frequency resolution for
    stability &mdash; multitaper by
    &plusmn;${(MT_NW/st().epoch).toFixed(2)}&nbsp;Hz at this window length, a width that grows as
    the window shortens; <strong>boxcar</strong> removes the taper and shows what tapering was
    suppressing.</p></div>`;
}

/* Guard against a toggle value the current tab does not offer. */
function coerceState(){
  const s = st();
  if(!ART.some(a=>a.id===s.art)) s.art = "none";
  /* An asym tab has no log-ratio option, so a tab switch carrying "log" lands on
     the equivalent plain subtraction rather than falling back to "off" -- which
     would silently turn the correction off on half the tabs now that it is the
     default. */
  if(!baseOpts().some(b=>b.id===s.base))
    s.base = baseOpts().some(b=>b.id==="raw") ? "raw" : "off";
  if(!SEG_ORDER.includes(s.baseSeg)) s.baseSeg = SEG_DEFAULT;
  if(!SEG_ORDER.includes(s.detailSeg)) s.detailSeg = SEG_DEFAULT;
  /* Only a length the sweep actually computed can be selected -- anything else
     would look up an archive column that does not exist and refuse every mask. */
  if(!SLIDE_CHOICES.some(v => +v === +s.slideS)) s.slideS = SLIDE_DEFAULT;
  /* `oc` is a hardcoded opening value like `motSrc`, and unlike `motSrc` it had
     no guard: an archive that never swept it would open on a mode with no button,
     miss every shard lookup keyed interp|oc|ref, and report every recording as
     missing with no control to move off it. */
  if(!OC.some(o => o.id === s.oc))
    s.oc = OC.some(o => o.id === "none") ? "none" : (OC[0] || {}).id;
  if(!MOT_RULES.includes(s.motSrc)) s.motSrc = DATA.default_motion_source || MOT_SRC[0];
  s.motK = Math.min(MOT_K.max, Math.max(MOT_K.min, +s.motK || MOT_K.default));
  /* The `s.sub` coercion that stood here went with the strict-subset control: it
     was re-creating a state key nothing read, and its comment described turning
     a gate back on that no longer exists. */
  s.epochIdx = Math.min(Math.max(Math.round(s.epochIdx || 0), 0),
                        DATA.epochs_s.length - 1);
  s.epoch = DATA.epochs_s[s.epochIdx];
}

function focusKey(){
  const a = document.activeElement;
  if(!a || a===document.body) return null;
  if(a.dataset && a.dataset.tab) return `[data-tab="${a.dataset.tab}"]`;
  if(a.dataset && a.dataset.k) return `[data-k="${a.dataset.k}"][data-v="${a.dataset.v}"]`;
  if(a.dataset && a.dataset.c) return `input[data-c="${a.dataset.c}"]`;
  if(a.dataset && a.dataset.key) return `.panel[data-key="${CSS.escape(a.dataset.key)}"]`;
  if(a.id) return "#"+a.id;
  return null;
}

/* Everything the current selection needs, or an explanation of why it is not
   here yet. The 4 s window is embedded in this file; the rest is fetched. */
function renderShardState(){
  const k = needShard();
  const wrap = $("#shard-state");
  if(STORE[k]){ wrap.hidden = true; wrap.innerHTML = ""; return; }
  wrap.hidden = false;
  wrap.innerHTML = SHARD_FAILED[k]
    ? `<div class="banner"><b>The ${st().epoch}-second, ${FFT_LABEL[st().fft]} data could not be
       loaded &mdash; ${esc(SHARD_FAILED[k])}.</b>
       <button type="button" class="close" onclick="__eegRetry()">Try again</button>
       Only the ${DATA.default_epoch}-second window length is embedded in this file; the others live in
       <code>${SHARD_DIR}/${k}.js</code> beside it. Three things to check, in order:
       that the <code>${SHARD_DIR}</code> folder was copied along with this file; that
       <code>python build_dashboard.py</code> has been run since the last pipeline run; and,
       if your browser refuses to read neighbouring files from a <code>file://</code> address,
       serving the folder over HTTP instead &mdash;
       <code>python -m http.server</code> in this directory, then open
       <code>http://localhost:8000/dashboard.html</code>. Until then the slider still works at
       ${DATA.default_epoch}&nbsp;s, which is embedded in this file.</div>`
    : `<div class="banner"><b>Loading the ${st().epoch}-second, ${FFT_LABEL[st().fft]} data…</b>
       This window length is not embedded in the page; it is being read from
       <code>${SHARD_DIR}/${k}.js</code>. It loads once per setting.</div>`;
}

function render(){
  const want = focusKey();
  coerceState();
  FRAME = new Map();
  renderHead(); renderControls(); renderControlSummary();
  const ready = ensureShard();
  renderShardState();
  if(!ready){
    for(const id of ["plain","tiles","banners","detail-slot","grid","table"])
      $("#"+id).innerHTML = "";
    /* Emptying #banners and #plain is not enough: both disclosures are in the
       static skeleton, so without this they sit there as open, empty, headed
       boxes for as long as the shard takes to arrive -- or permanently, on a
       file:// origin whose dashboard_data/ folder was not copied along. */
    $("#warn-wrap").hidden = true;
    $("#plain-wrap").hidden = true;
    $("#excluded").innerHTML = ""; $("#excluded").hidden = true;
    $("#excluded-empty").hidden = false;
    $("#excluded-empty").textContent = "waiting for data";
    renderMethod();
  } else {
    renderPlain(); renderTiles(); renderDetail(); renderGrid();
    renderTable(); renderExcluded(); renderMethod();
  }
  if(want){
    const el = document.querySelector(want);
    if(el && !el.disabled) el.focus({preventScroll:true});
  }
}

document.addEventListener("click", e=>{
  const x = e.target.closest("#export-btn");
  if(x){ doExport(x); return; }
  const t = e.target.closest(".tabs button");
  if(t){ tab = t.dataset.tab; render(); window.scrollTo({top:0,behavior:"smooth"}); return; }
  const c = e.target.closest("input[type=checkbox][data-c]");
  if(c){
    st()[c.dataset.c] = c.checked;
    if(st().open && !candidates().some(r=>r.key===st().open)) st().open = null;
    render(); return;
  }
  const b = e.target.closest(".seg button");
  if(b && !b.disabled){
    st()[b.dataset.k] = b.dataset.v;
    /* Changing WHICH segment is subtracted moves the viewer to match, so the two
       cannot drift apart unnoticed. Arrowing the viewer does not move the
       correction -- that direction stays one-way on purpose. */
    if(b.dataset.k === "baseSeg") st().detailSeg = b.dataset.v;
    if(st().open && !candidates().some(r=>r.key===st().open)) st().open = null;
    render(); return;
  }
  const p = e.target.closest(".panel");
  if(p){ st().open = st().open===p.dataset.key ? null : p.dataset.key; render();
    revealDetail(); }
});

/* Bring an expanded panel's TOP edge into view -- its title and chart -- rather
   than its middle. `block:"start"` alone would put that edge underneath the
   sticky control bar, so the margin is set from the bar's measured height at
   scroll time: it changes with the viewport width, with which controls are
   showing, and with whether the banks are collapsed. */
function revealDetail(){
  const d = document.querySelector(".detail");
  if(!d) return;
  const bar = document.querySelector(".controls");
  const barH = bar ? bar.getBoundingClientRect().height : 0;
  /* Clamped, because the bar can be taller than the window -- an expanded bank
     on a phone, or any short desktop window. Unclamped, the margin pushes the
     panel past the bottom of the viewport and a reader who taps a panel is left
     looking at the control bar with their chart somewhere below the fold. At
     ordinary desktop sizes the clamp never binds. */
  d.style.scrollMarginTop = Math.min(barH + 10, innerHeight * 0.45) + "px";
  /* The rest of the page drops its animations under prefers-reduced-motion, in
     the media query at the end of the stylesheet, so this honours it too. */
  const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
  d.scrollIntoView({behavior: still ? "auto" : "smooth", block:"start"});
}

/* Sliders update their readout on every pixel of drag but only re-render on
   release: a full re-render at the 1 s window length redraws 22 charts of
   1,200 points each, which is far too slow to run per input event. */
const RANGE_FMT = {epochIdx: i=>DATA.epochs_s[i]+" s",
                   sigmaK: v=>(+v).toFixed(2)+" × σ",
                   motK: v=>(+v).toFixed(2)+" × σ",
                   holmCap: v=>(+v).toFixed(0)+" µV"};
/* Arrow keys on a range input fire `change` on EVERY keypress, so arrowing the
   window slider from 4 s to 1 s issued three separate multi-megabyte shard
   requests and three full re-renders. Settle first. */
let commitTimer = null;
document.addEventListener("input", e=>{
  const r = e.target.closest("input[type=range]");
  if(!r) return;
  const key = r.dataset.r;
  const out = r.parentElement.querySelector(".val");
  if(out) out.textContent = RANGE_FMT[key](r.value);
});
document.addEventListener("change", e=>{
  /* The #chk-gate handler that stood here went with the amplitude gate. */
  const r = e.target.closest("input[type=range]");
  if(!r) return;
  const key = r.dataset.r, val = r.value;
  clearTimeout(commitTimer);
  commitTimer = setTimeout(()=>{
    st()[key] = key==="epochIdx" ? parseInt(val,10) : parseFloat(val);
    if(st().open && !candidates().some(x=>x.key===st().open)) st().open = null;
    render();
  }, 150);
});

document.addEventListener("keydown", e=>{
  if(e.key==="Escape" && st().open){ st().open=null; render(); return; }
  /* Left/right cycle the expanded panel's baseline reference. Deliberately NOT
     claimed when the tabs have focus (they use the same keys to move between
     measures), nor from a range input (arrows nudge the sliders), nor from
     anywhere in the control row. Everything else, while a panel is open, is
     fair game -- that is where a reader's hands are. */
  if((e.key==="ArrowRight"||e.key==="ArrowLeft") && st().open && $(".d-base")
     && !e.target.closest?.(".tabs") && !e.target.closest?.(".controls")
     && !e.target.closest?.(".tablewrap") && e.target.id !== "close-detail"
     /* The export button keeps focus after a click; arrowing from it used to
        cycle the baseline segment and yank focus to the segment control. */
     && e.target.id !== "export-btn"
     /* Same for the two disclosure headings. They are focusable in their own
        right, and a keyboard reader who has tabbed to "Warnings" and presses
        an arrow means to move within the page, not to re-cut the baseline of
        a panel further down it. */
     && e.target.tagName !== "SUMMARY"
     && !/^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName||"")
     && !e.target.isContentEditable){
    /* `$(".d-base")` rather than st().open alone: renderDetail bails out when the
       open recording has dropped out of `candidates()`, leaving st().open set
       with NO panel drawn. Without this the arrows would preventDefault and
       trigger a full 22-chart re-render against nothing on screen. */
    const r = DATA.recordings.find(x=>x.key===st().open);
    const avail = SEG_ORDER.filter(sname =>
      r && r.base && !r.base.missing && (r.base.segs||{})[sname]);
    if(avail.length < 2) return;
    e.preventDefault();
    const i = Math.max(0, avail.indexOf(detailSegOf()));
    st().detailSeg = avail[(i + (e.key==="ArrowRight"?1:avail.length-1)) % avail.length];
    render();
    document.querySelector(`.d-base-seg button[data-v="${st().detailSeg}"]`)
      ?.focus({preventScroll:true});
    return;
  }
  if((e.key==="ArrowRight"||e.key==="ArrowLeft") && e.target.closest?.(".tabs button")){
    e.preventDefault();
    const i = ORDER.indexOf(tab);
    tab = ORDER[(i + (e.key==="ArrowRight"?1:ORDER.length-1)) % ORDER.length];
    render();
    document.querySelector(`.tabs button[data-tab="${tab}"]`)?.focus({preventScroll:true});
  }
});

/* =======================================================================
   Collapsing the control banks.

   Deliberately NOT part of `st()`: the banks are a property of the window, not
   of the measure being looked at, so collapsing them on one tab and finding
   them collapsed on the next is the behaviour that matches what was asked for.
   It also means no re-render is needed to toggle -- the bar head is outside
   everything render() rewrites, so a class on the wrapper is the whole job.
   ======================================================================= */
const CTL_BAR = $("#controls-bar");

function setControlsCollapsed(on){
  CTL_BAR.classList.toggle("collapsed", on);
  $("#ctl-toggle").setAttribute("aria-expanded", String(!on));
  $("#ctl-toggle-label").textContent = on ? "Show controls" : "Controls";
  /* The tooltip has to move with the label, or a collapsed bar reads
     "Show controls" and then offers to collapse itself on hover. */
  $("#ctl-toggle").title = on
    ? "Show the control banks. The settings below are applied either way."
    : "Collapse the control banks to free up screen space. The settings stay applied.";
  try{ localStorage.setItem("eeg-controls-collapsed", on ? "1" : "0"); }catch(_){}
}

$("#ctl-toggle").addEventListener("click", ()=>{
  setControlsCollapsed(!CTL_BAR.classList.contains("collapsed"));
});

/* The one-line readout shown while the banks are shut. Only the settings that
   change the NUMBERS are listed; the thresholds that merely re-colour windows
   are left out to keep it to one line. Rebuilt on every render so it cannot
   drift from the controls it stands in for. */
function renderControlSummary(){
  const lbl = (opts, id) => (opts.find(o=>o.id===id)||{}).label || id;
  const bits = [
    /* interpSrc alone does NOT name the mode: it and the frontal-pair checkbox
       together select one of five swept interpolation modes, so two different
       stored arrays would otherwise print the same summary line. Only mentioned
       when it can bite -- with interpolation off there is nothing to hold back,
       and the checkbox is a no-op unless BOTH frontal channels are on the list. */
    lbl(INT, st().interpSrc) +
      (st().interpSrc !== "off" && !st().frontalPair
        ? ` (${FRONTAL.join("+")} held back)` : ""),
    lbl(OC, st().oc),
    lbl(REF, st().ref),
    lbl(FFT, st().fft),
    st().epoch + " s",
    lbl(ART, st().art),
    st().base === "off" ? "no baseline"
      : `${lbl(baseOpts(), st().base)} vs ${segLabel(st().baseSeg)}`,
  ];
  $("#ctl-summary").innerHTML = bits.map(b=>`<b>${b}</b>`).join(" &middot; ");
}

/* Restore the collapsed state before the first paint so the bar does not flash
   open. Default COLLAPSED on a narrow screen, where an expanded bank is taller
   than the whole viewport and a first-time visitor would otherwise have to
   scroll past it to reach any content -- which is the case the collapse exists
   for. A stored choice always wins over the default, and a browser with storage
   blocked just gets the width-based default. */
let ctlStart = matchMedia("(max-width:760px)").matches;
try{
  const saved = localStorage.getItem("eeg-controls-collapsed");
  if(saved !== null) ctlStart = saved === "1";
}catch(_){}
setControlsCollapsed(ctlStart);

render();
</script>
"""


if __name__ == "__main__":
    main()

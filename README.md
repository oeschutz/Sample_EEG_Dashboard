# Dashboard template

A self-contained folder that lets another group run this dashboard on their own
recordings, with none of ours.

There are two routes to a dashboard, and they are independent:

* **From synthetic data** — `generate_template_data.py` fabricates a valid
  `outputs/` tree and `build_dashboard.py` turns it into a page. No EEG, no MNE,
  a few seconds. This is how you see the interface and check your own exporter
  against a known-good tree.
* **From real recordings** — `pipeline.py` processes raw Galea recordings into
  the same `outputs/` tree, and `build_dashboard.py` turns that into a page. This
  needs the recordings, MNE, and about forty minutes, depending on the number of recordings.

Both routes end at the same four files, so *The short version* below applies to
either. The two build sections are marked **Route 1** and **Route 2**; the two
sections after them apply to both.

| File | What it is |
|---|---|
| `build_dashboard.py` | The builder: an `outputs/` tree in, a dashboard out. It has to travel with this folder, and nothing here runs without it. |
| `dashboard_template.json` | The input contract for `build_dashboard.py`, field by field, annotated with whether the dashboard actually reads each one. Valid JSON; underscore keys are annotations. |
| `generate_template_data.py` | Writes a complete, valid `outputs/` tree from nothing. Seeded, so two runs are identical apart from the `generated_utc` stamp. |
| `dashboard_template.html` | A working dashboard built from that synthetic tree. 5.6 MB, opens from disk by double-clicking it. |
| `pipeline.py` | The real pipeline: raw recordings in, an `outputs/` tree out. Only Route 2 uses it. |

The synthetic route requires Python 3 and numpy, and nothing else.
`build_dashboard.py` does not import MNE; `generate_template_data.py` imports
nothing outside the standard library but numpy. The real route needs more — see
*Requirements* under it.

The shipped `dashboard_template.html` travels without its `dashboard_data/`
folder, so only the 4 s window length is live in it. The other nine slider
positions show a banner saying where they should have been. Rebuild, below, to
get them.

## The short version

`build_dashboard.py` never touches a raw recording. It reads four things:

```
outputs/cognitive_load.json          one entry per task recording
outputs/baselines.json               one entry per baseline, carrying its segments
outputs/variants/index.json          the sweep manifest
outputs/variants/*_eNN.N.npz         the numbers, two files per window length
```

If your processing code can emit those four, you get the dashboard — whatever
produced the numbers. `dashboard_template.json` documents every field.
`generate_template_data.py` produces a known-good example to diff against.

## Route 1 — rebuilding from synthetic data

From this folder:

```bash
python generate_template_data.py --out template_run
cp build_dashboard.py template_run/
cd template_run && python build_dashboard.py
```

Both steps take a few seconds. That writes `template_run/dashboard.html`
(5.6 MB, self-contained at the 4 s default) and `template_run/dashboard_data/`
(57 MB in 36 files, the other nine window lengths, fetched on demand as
`<script>` tags so the slider still works from a `file://` address). 62 MB
together. Copy the HTML over `dashboard_template.html` to refresh the shipped
copy — it will be byte-identical to the current one apart from the
`generated_utc` stamp.

`build_dashboard.py` reads `outputs/` from **its own directory** and writes
beside itself, which is why it is copied next to the generated tree rather than
run in place. Run it here in `template/`, where there is no `outputs/`, and it
exits with a `FileNotFoundError` on `outputs/cognitive_load.json`.

### What the synthetic dataset contains

Three analysed participants, two cells each: six task recordings and six
baselines, plus one participant excluded upstream so the excluded table has
something to print. Tasks are 10 minutes, baselines 6 minutes with the usual
three segments.

Each cell exercises a different part of the interface:

| Cell | What it shows |
|---|---|
| `t01` agent | Both frontal channels flagged — the frontal-pair checkbox bites |
| `t01` ai | A clean pair: every interpolation mode resolves to one array |
| `t02` agent | The manual list asks for a channel detection never flagged |
| `t02` ai | The manual list omits a channel detection did flag |
| `t03` none | The no-AI control arm, with an empty manual list |
| `t03` none (speedscore) | No usable IMU — the head-motion rule cannot run |
| `t04` | Excluded upstream, with a reason |

The build prints `WARNING: 160 array(s) the sweep did not produce`. That is the
`t03` no-IMU pair, deliberately: four swept units × ten window lengths × four
estimators. It is the missing-data path working, not a fault. It also prints
`2 excluded upstream` — that is `t04`, one participant, counted as its task and
its baseline.

## Route 2 — running the real pipeline on your own recordings

Line numbers below point into the copies of `pipeline.py` and
`build_dashboard.py` in this folder.

### Requirements

`pipeline.py` needs far more than the synthetic route. The versions the last full
run was made with:

```
Python 3.13.7, MNE 1.11.0, NumPy 2.3.3, SciPy 1.16.2, pandas, matplotlib
```

There is no lock file, and **specify the MNE version**. Its spectral defaults
have changed between versions before, which can cause real defects in the
analysis.

### Three tables to fill in, already emptied for you

`pipeline.py` keeps its per-participant decisions as module-level constants and
validates them against the recordings it discovers. The copy in this folder ships
with all of them **empty**, so it runs on your recordings without edits: nothing
is excluded, and the manual interpolation modes interpolate nothing.

| Constant | Line | What it is for |
|---|---|---|
| `MANUAL_INTERPOLATION` | `pipeline.py:653` | Hand-picked channels to interpolate, per `(participant, arm)`. Feeds the `manual` and `manual_keepfrontal` interpolation modes; while it is empty those two resolve to the same arrays as `off`. |
| `EXCLUDED_RECORDINGS` | `pipeline.py:972` | Named recordings to drop, keeping the rest of that participant. Name both halves of a pair. |
| `EXCLUDED_PARTICIPANTS` | `pipeline.py:963` | Participants to drop entirely, with `EXCLUDED_PARTICIPANT_REASONS` at `:964` for the reason the dashboard prints. |

Fill them in and the validators become strict, which is the point of them:

- `assert_manual_interpolation()` (`pipeline.py:778`) exits on an entry that
  matches no discovered recording, names an unknown channel, repeats one, would
  interpolate every channel, or is keyed by an arm that matches two pairs. A
  typo would otherwise interpolate nothing and ship as a result rather than as
  an error — an absent group legitimately means "interpolate nothing here", so a
  miss is indistinguishable from an instruction.
- `assert_excluded_recordings()` (`pipeline.py:983`) exits on a key naming no
  real recording, and on a pair excluded from one side only. The surviving half
  would otherwise be swept and baseline-corrected against a partner that no
  longer exists.

Both run over the discovered recordings before any processing starts, so a
mistake costs you a second rather than forty minutes.

### The data it expects

A folder called `EEG Recordings/` beside `pipeline.py`. Discovery is a single
glob in `discover_recordings()` (`pipeline.py:1138`):

```
EEG Recordings/*/*/openbci-raw-exg_*.txt
```

Everything else is derived from where that file sits, so the layout is fixed at
exactly three levels:

```
EEG Recordings/
  galea_session_p02-20260522-165312/     session folder
    task_agent_personal/                 recording folder
      openbci-raw-exg_<stamp>.txt          the EEG
      openbci-raw-aux_<stamp>.txt          the IMU  (optional)
      openbci-packet-loss_<stamp>.txt      the loss log (optional)
    baseline_agent_personal/
      openbci-raw-exg_<stamp>.txt
      ...
  galea_session_p03-.../
    ...
```

- **The session folder must match `galea_session_<pid>-...`.** The participant id
  is everything between `galea_session_` and the first hyphen — `p02`, `sbx`,
  anything without a hyphen in it. A folder that does not match this pattern
  raises an `AttributeError` on a failed regex match, not a readable error.
- **The recording folder name is the recording name**, and a name starting with
  `task` makes it a task; anything else is treated as a baseline.
- **`<stamp>` is whatever follows `openbci-raw-exg_`**, and the aux and
  packet-loss files beside it must carry the same stamp. Only the exg file is
  globbed for; the other two are constructed from its name.
- The recording **key**, used everywhere downstream and in `--only`, is
  `<pid>/<recording folder name>` — for example `p02/task_agent_personal`.

### File formats

All three are OpenBCI GUI text files, read with pandas.

**`openbci-raw-exg_<stamp>.txt`** — `pd.read_csv(path, skiprows=4)`. Four header
lines, then a CSV. The columns that are read by name:

| Column | Used for |
|---|---|
| `Timestamp` | float seconds. Every window time comes from these, never from an assumed grid |
| `Timestamp (Formatted)` | only by the p02 interval excision |
| `Marker` | segmentation; the value looked for is `8.0` (`TASK_MARKER`, `pipeline.py:944`) |

Channels are taken **positionally**, from columns 1–18, in this order
(`pipeline.py:105`):

```
1-4   EMG        5-6  EOG        7-8  EMG        9-18  EEG
```

Of those, the ten EEG columns are picked by having `EEG` in the column name, and
are renamed to the montage (`pipeline.py:578`):

```
F1 F2 C3 C4 P3 P4 O1 O2 Cz Pz
```

`assert_montage_names()` (`pipeline.py:760`) compares that list against what MNE
actually holds and raises `SystemExit` on a mismatch, because the manual
interpolation lists and the frontal-pair rule are written in terms of those
names. Sampling rate is 250 Hz, hardware reference SRB2 / earlobe.

> **There is no Fz**, and the whole analysis is shaped around that. Holm's index
> is defined at Fz and Pz; the frontal numerator here is the time-domain mean of
> F1 and F2, averaged sample-by-sample before the FFT. A rig with a real Fz, or a
> different montage, is a change to the analysis and not just to a constant.

**`openbci-raw-aux_<stamp>.txt`** — `pd.read_csv(path, skiprows=4)`, same shape.
Read for the `Head motion` artifact mode, at 50 Hz, by name:

```
Timestamp, Accelerometer X, Accelerometer Y, Accelerometer Z,
           Gyroscope X,     Gyroscope Y,     Gyroscope Z
```

It is genuinely optional. A missing or unreadable aux file, or one whose IMU
never varies, leaves the recording processed and the head-motion rule unable to
run on it — that is the case the synthetic `t03 none (speedscore)` cell exercises,
and it is why a good build prints a `did not produce` warning rather than failing.
The aux file carries no markers; timestamps are the only alignment.

**`openbci-packet-loss_<stamp>.txt`** — plain text. Seven header lines; every
non-blank line after that counts as one loss event. Read inside a `try`, so a
missing file costs only that count.

### Markers and block structure

Both recording types are delimited by two `Marker 8.0` entries in the `Marker`
column.

- A **task** is what lies between them, nominally 1200 s (`TASK_BLOCK_SEC`).
  Everything outside is discarded.
- A **baseline** is a 360 s block (`BASELINE_BLOCK_SEC`) structured as 0–2 min
  mental arithmetic, 2–4 min deliberate eye movements, 4–6 min rest. Three
  segments of it are swept, and the dashboard chooses between them
  (`BASELINE_SEGMENTS`, `pipeline.py:915`):

  | Segment | Bounds | What it is |
  |---|---|---|
  | `rest` | 240–360 s | open-eye resting. The default. |
  | `math` | 0–120 s | mental arithmetic — an active-task reference, not a resting one |
  | `eyes_closed` | 225–240 s | the last 15 s of the eye-movement phase |

  Each is cropped, epoched and calibrated on its own samples. If your baseline
  block is not built this way, those three windows point at the wrong minutes and
  every baseline-corrected number on the page is wrong. Change the bounds rather
  than living with them.

Two marker cases to know about, both in `step2_segment()` (`pipeline.py:1218`):

- **One marker** is repaired rather than dropped, if the recording runs a full
  block past it: the block becomes `marker → marker + 1200 s` for a task, or
  `+ 360 s` for a baseline. Otherwise the recording is dropped with
  `cannot delimit the segment`.
- **Three or more markers** fall through to `else: a, b = mi[0], mi[1]`
  (`pipeline.py:1253`) and **the extras are ignored**. This template assumes your
  recordings carry no false starts or double presses; the original study's
  per-recording repairs for those have been removed. It is not silent — a
  segment that comes out well short of 20 minutes is recorded `clean: false`
  with a `task segment is N min, expected ~20 min` note — but the segment is
  still built and swept. If your recordings do have extra markers, this is the
  branch to change.

### Naming conventions, which are load-bearing

The arm and the framing are parsed out of the recording folder name. Nothing else
records them. Three functions read it:

| Function | Where | What it returns |
|---|---|---|
| `recording_arm()` | `pipeline.py:656` | `agent` / `ai` / `none`, from `^(?:task\|baseline)_(agent\|ai\|none)_`. **Raises `ValueError`** on anything else rather than guessing |
| `recording_condition()` | `pipeline.py:675` | `<arm>_<framing>`, e.g. `none_personal`. Strips a trailing `_1` / `_2` file-split suffix |
| `baseline_key_for_task()` | `pipeline.py:5149` | the paired baseline: `task_agent_personal` → `<pid>/baseline_agent_personal` |

and one more decides what the page says:

| Function | Where | What it returns |
|---|---|---|
| `condition_label()` | `build_dashboard.py:163` | the display labels — Agent / AI / No AI, Personal / Speedscore |

Note the asymmetry between the last two: `recording_arm()` raises on a name
it cannot parse, but `condition_label()` falls back — anything without
`_agent_` or `_none_` is labelled **AI**, and anything without `speedscore` is
labelled **Personal**. A misnamed recording therefore stops the pipeline but
would have been mislabelled rather than caught on the page.

So the grammar is:

| Part | Values | Meaning |
|---|---|---|
| prefix | `task_` / `baseline_` | which of the pair this is |
| arm | `agent` / `ai` / `none` | Agent / AI / No AI |
| framing | `personal` / `speedscore` | Personal / Speedscore |
| suffix | `_1` / `_2` | optional, for a recording split across two files |

Pairing matters as much as the tokens: `task_agent_personal` and
`baseline_agent_personal` under the same participant are one **pair**, and
bad-channel detection runs once per pair, on the task concatenated with the
`rest` segment of its own baseline, interpolating the same channels in both. A
task whose baseline is missing has no denominator.

If your study has different arms or framings, rename your recordings to these
tokens or edit all four functions. `recording_arm()` raising rather than
defaulting means a rename fails loudly, which is the intended behaviour.

### How much data

**Ten recordings is a hard floor.** The amplitude band that decides every quality
label is fitted from the run's own channel measurements rather than from a fixed
range. Below 100 channel measurements — ten recordings — no band is fitted, every
plausibility label reports *"not assessed"*, and `build_dashboard.py` refuses to
build.

Read the band as relative in any case: it means "unusual for this helmet on these
recordings", not "physiologically implausible". Adding noisier recordings widens
it and *raises* the count of recordings that pass.

For scale, the real run behind the original dashboard is 12 participants, 22
tasks and 22 baselines, from 49 recordings discovered across 12 session folders.

### Running it

From the folder holding `EEG Recordings/`, with `pipeline.py` and
`build_dashboard.py` in it:

```bash
python pipeline.py
python build_dashboard.py
```

`pipeline.py` takes about 40 minutes for 49 recordings and writes `outputs/`.
`build_dashboard.py` takes about 3 seconds and writes `dashboard.html` and
`dashboard_data/` **into its own directory**, which is the same constraint as on
the synthetic route.

`pipeline.py` writes more than the dashboard reads. Alongside the four files
listed under *The short version*, it produces two the dashboard ignores:

```
outputs/qc_steps.csv     per-recording clean/unclean verdict per step
outputs/qc_steps.json    per-channel amplitudes, bad-channel z-scores, step notes
```

Expect a much larger deliverable than the synthetic one. At 22 tasks and 22
baselines it is `dashboard.html` at 32.1 MB plus `dashboard_data/` at 333 MB —
**365 MB**, against 62 MB for the synthetic tree. `build_dashboard.py` prints the
total and warns above 100 MB; on a dataset that size the warning always fires.

### Smoke tests, and the trap in them

```bash
python pipeline.py --only p09/task
python pipeline.py --limit 2
```

A filtered run marks `run_meta.partial_run` and diverts everything it writes, so
it cannot clobber a complete run: the JSON and QC files go to `outputs/partial/`
and the sweep to `outputs/variants_partial/` (`pipeline.py:4532`, `:4904`).
`build_dashboard.py` **refuses to build from it**. That refusal is deliberate — a
filtered run also fits the amplitude band on whatever it processed, and stamps
`partial_run: true` inside `amplitude_bound` so the flag travels with the number
rather than sitting beside it.

### What stops a run

- **No recording succeeded** — `SystemExit`. Per-recording errors are otherwise
  caught and recorded, so one bad recording does not cost a forty-minute run; but
  a run in which everything died used to exit 0, print "0 recordings", and
  overwrite `outputs/` on the way out.
- **More than half failed** on a complete run — `SystemExit`, same reason.
- **A montage mismatch** — `SystemExit` from `assert_montage_names()`.
- **An unparseable recording name** — reported by `assert_manual_interpolation()`
  before processing starts, rather than surfacing mid-sweep as a per-recording
  error row.
- **The default cell does not reproduce MNE's own estimator.** Every run asserts
  that its default cell — 4 s, Hann, hardware reference, interpolation on, ocular
  none — matches MNE, and `build_dashboard.py` refuses to build if it does not.

### Per-recording judgement calls you should expect

The original dataset needed several, all approved by the researcher, and yours
probably will too. These are the kind of thing no naming convention fixes:

- a participant dropped entirely, because one task was split across two files
  with ~5.4 minutes missing between them;
- a task whose markers are 27.65 minutes apart rather than 20, with the excess
  interval excised and the remainder treated as continuous;
- a task with three markers, where the pair used is the first and the third;
- a baseline with only one marker;
- a task with only a start marker.

**This template handles the last two and none of the first three.** The
single-marker repair is generic, so it survives; the rest were per-recording
decisions keyed by literal recording name, and they have been removed rather
than shipped as someone else's arbitrary constants. If you hit one of the first
three you are writing the repair yourself, in `step2_segment()`.

The one piece of that machinery that is generic and does survive: window times
come from recorded timestamps and never from an assumed grid, because spreading
a duration evenly over window indices puts a window minutes out of place once a
recording has any discontinuity in it.

Windows whose samples are not contiguous in real time are dropped from every
measure, at every window length, under every artifact mode including `None` — the
spectrum of a window spanning a splice is taken across a step in the signal
rather than across anything an electrode measured.

## Two things to fix before you publish a build of your own

**The page states our findings as fact.** About 15 passages in
`build_dashboard.py`'s template are claims about the Sarma Lab's dataset —
*"positive on all 21 recordings with a working sensor"*, *"one 20-minute task
recording"*, and similar. They render verbatim on a dashboard built from your
data, where they are not true. Grep for `this dataset` (12 hits), `recordings`
next to a number, and `20-minute`.

**The arm and framing come from the recording name**, on both routes.
`condition_label()` at `build_dashboard.py:163` parses `_agent_` / `_ai_` /
`_none_` and `personal` / `speedscore` out of the folder name. Rename your
recordings to match, or edit that function — see *Naming conventions* under
Route 2.

## Putting a real recording in

A third option, between the two routes: keep the synthetic tree and swap one unit
of it for a real one.

See `REAL_RECORDING_SLOT` at the bottom of `generate_template_data.py`. One
consented recording processed by `pipeline.py --only` can replace one synthetic
unit: the tree is assembled per recording, so a real entry and a synthetic one
sit side by side as long as both satisfy the contract. The amplitude band is
fitted over whatever the tree holds, so it must be recomputed after the merge.

The refusal to build from a partial run, described under Route 2, is not in your
way here: it blocks building *directly* from a filtered run’s archive. Merging
that run’s arrays into a complete tree and rebuilding the band is the supported
path.

Whether a real recording can be published at all is a consent question, not a
technical one, and it is deliberately left open.

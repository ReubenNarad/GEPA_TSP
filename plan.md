# Concorde + QSopt + GEPA Integration Plan (Linux)

This guide captures the current end-to-end workflow for building Concorde with local QSopt support, maintaining deterministic evaluation datasets, and running GEPA-driven heuristic experimentation in isolated sandboxes.

---

## 1. Prerequisites

Tested on Ubuntu 22.04. Install build and Python tooling:

```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential clang make autoconf automake libtool pkg-config \
    gfortran libgmp-dev libbz2-dev zlib1g-dev \
    python3 python3-venv python3-pip curl git rsync
```

Optional Python venv:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

---

## 2. Repository Layout (authoritative)

```
GEPA_TSP/
├── concorde/                # Concorde source tree + build artifacts
│   ├── concorde/            # extracted co031219 sources
│   └── install/             # staging area for rebuilt binaries
├── qsopt/                   # QSopt sources and local installs
│   ├── src/                 # qsopt_ex source tree (for local builds)
│   ├── install/             # qsopt_ex installation prefix
│   └── original/            # copy of pre-existing qsopt.a/qsopt.h
├── data/
│   └── eval/                # persistent evaluation datasets + metadata
├── runs/                    # per-evaluation artifacts (gitignored)
├── scripts/                 # helper scripts (build, evaluation, sandbox workflow)
└── plan.md                  # this document (keep updated)
```

`runs/`, build directories, and sandboxes stay out of version control.

---

## 3. QSopt Options

### 3.1 Local qsopt_ex Build (preferred for reproducibility)

1. Download unpacked Debian tarball (working mirrors as of Oct 2025):
   ```bash
   curl -L -o qsopt-ex.tar.gz https://deb.debian.org/debian/pool/main/q/qsopt-ex/qsopt-ex_2.5.10.3.orig.tar.gz
   tar -xzf qsopt-ex.tar.gz
   mv qsopt-ex-2.5.10.3 qsopt/src
   ```
2. Bootstrap and configure with prefix:
   ```bash
   cd qsopt/src
   ./bootstrap
   ./configure --prefix="$(pwd)/../install"
   ```
3. **Patch requirement:** `qsopt_ex/trace.h` is missing an `ILL_IFTRACE2` definition in release 2.5.10.3. Ensure the following block appears:
   ```c
   #ifndef NDEBUG
   #define ILL_IFTRACE        if (TRACE) QSlog
   #define ILL_IFTRACE2       if (TRACE > 1) QSlog
   #define ILL_IFDOTRACE      if (TRACE)
   #else
   #define ILL_IFTRACE        if (0) QSlog
   #define ILL_IFTRACE2       if (0) QSlog
   #define ILL_IFDOTRACE      if (0)
   #endif
   ```
4. Build and install:
   ```bash
   make -j"$(nproc)"
   make install
   ```
5. Sanity-check:
   ```bash
   ../install/bin/esolver --help | head
   ```

### 3.2 Legacy QSopt Library (Concorde compatibility)

Copy the machine’s existing QSopt static library/header into the project to avoid mutating system installs:

```bash
mkdir -p qsopt/original
cp -p /home/rnarad/qsopt/qsopt.a qsopt/original/
cp -p /home/rnarad/qsopt/qsopt.h qsopt/original/
```

Concorde links against this archival copy by default (`--with-qsopt=qsopt/original`).

---

## 4. Concorde Build & Staging

1. Fetch sources:
   ```bash
   curl -L -o concorde.tar.gz https://www.math.uwaterloo.ca/tsp/concorde/downloads/codes/src/co031219.tgz
   tar -xzf concorde.tar.gz -C concorde
   ```
2. Configure to link against the local QSopt archive:
   ```bash
   cd concorde/concorde
   ./configure --with-qsopt=/home/rnarad/GEPA_TSP/qsopt/original --prefix="$(pwd)/../install"
   ```
3. Build:
   ```bash
   make -j"$(nproc)"
   ```
4. Manually stage binaries (Concorde lacks `make install`):
   ```bash
   mkdir -p ../install/bin ../install/include ../install/share/concorde/examples
   cp -p TSP/concorde LINKERN/linkern TOOLS/* ../install/bin/
   cp -p concorde.h ../install/include/
   ```
5. Provide minimal smoke test instance (already added):
   - `concorde/install/share/concorde/examples/5city.tsp`
6. Verify:
   ```bash
   concorde/install/bin/concorde concorde/install/share/concorde/examples/5city.tsp
   ```

Rebuild workflow after editing sources:

```bash
cd concorde/concorde
make -j"$(nproc)"
cp -p TSP/concorde LINKERN/linkern TOOLS/* ../install/bin/
```

---

## 5. Deterministic Evaluation Datasets

All benchmark instances live under `data/eval/` and are discovered via `data/eval/metadata.json`. Each entry records the instance id, relative file path, node count, generator metadata, and the `split` label used by evaluation scripts.

Current maintained splits:

- **toy20** – ten 20-city Euclidean problems (unique integer coordinates). These are the original “smoke test” instances; Concorde solves them instantly at the root.
- **toy200** – ten 200-city Euclidean problems. GEPA’s simple threshold tweak improved linkern’s runtime (~9 ms → ~9.1 ms) even though branch-and-bound still stops at the root.
- **tsplib_random** – a growing bank of explicit-weight “TSPLIB-style” instances (currently 200/300/400 nodes plus ten additional 300-node cases). These FULL_MATRIX problems routinely force Concorde to explore deeper trees (average bbnodes > 1). Generated locally via `random.randint` because direct TSPLIB downloads are blocked; once networking returns, drop real TSPLIB files into `data/eval/tsplib/` and use `scripts/add_tsplib_eval.py` to register them under this split (or a new, canonical split).

To add more problems, append entries to `metadata.json` (or use the helper scripts) and give them a unique `split` label. Never overwrite existing IDs; the evaluation harness assumes entries are immutable.

---

## 6. Evaluation Runner

`scripts/run_concorde_eval.py` orchestrates benchmarking:

```bash
python3 scripts/run_concorde_eval.py \
    --binary concorde/install/bin/concorde \
    --metadata data/eval/metadata.json \
    --split toy20 \
    --repeats 3 \
    --label baseline
```

Features:
- Filters by `--split` or explicit `--ids`; optional `--max-instances`.
- Captures wall-clock runtime, Concorde’s reported total time, branch-and-bound nodes, bounds, and LP progression.
- Saves artifacts under `runs/eval/<timestamp>_<label>/`:
  - `config.json` – binary path, selected instances, repeats, env metadata.
  - `results.jsonl` – per-instance structured output (stdout/stderr included).
  - `instances/<id>_rN.stdout/.stderr` – raw logs.
  - `summary.json` – aggregate metrics.
- Gracefully handles timeouts (`--timeout`), parse failures, and missing files.

Use this runner both manually and from the GEPA bridge to keep metrics consistent.

---

## 7. GEPA Candidate Evaluation Pipeline

### 7.1 Sentinel-Based Code Replacement

Instrument `concorde/concorde/LINKERN/linkern.c` with clear markers (planned):

```c
/* BEGIN LLM HEURISTIC BLOCK */
... // default implementation
/* END LLM HEURISTIC BLOCK */
```

Scripts can surgically swap only the enclosed region, minimizing merge noise and simplifying restoration.

### 7.2 Sandbox Architecture (per candidate)

Avoid running candidates directly in the canonical source tree to prevent concurrent interference:

1. Baseline copy:
   ```bash
   SANDBOX_DIR=/tmp/gepa/<candidate-id>
   rsync -a --exclude 'runs' --exclude '.git' --exclude 'sandboxes' \
       /home/rnarad/GEPA_TSP/ "$SANDBOX_DIR/"
   ```
2. Inject candidate snippet into the sandbox’s `linkern.c` (between sentinel comments).
3. Rebuild within the sandbox:
   ```bash
   make -C "$SANDBOX_DIR/concorde/concorde" -j"$(nproc)"
   ```
4. Run evaluations using the sandbox binary (choose the target split explicitly, e.g. `toy200` or `tsplib_random`):
   ```bash
   python3 scripts/run_concorde_eval.py \
       --binary "$SANDBOX_DIR/concorde/install/bin/concorde" \
       --metadata /home/rnarad/GEPA_TSP/data/eval/metadata.json \
       --split toy200 \
       --label candidate_<id>
   ```
5. Copy artifacts (candidate code, build log, run directory) back into the main repo under an archival folder (see GEPA automation below for current layout).
6. Remove sandbox when finished to reclaim disk: `rm -rf "$SANDBOX_DIR"`.

Advantages:
- Supports true parallelism (no races on object files or binaries).
- Maintains a pristine reference tree.
- Keeps each candidate’s artifacts self-contained.

### 7.3 Automation Hooks

`src/heuristic_bridge.py` exposes:

```python
evaluate_candidate(
    code: Optional[str],
    label: str,
    split: str = "toy20",
    repeats: int = 1,
    timeout: Optional[float] = None,
    sandbox_root: Optional[Path] = None,
    keep_sandbox: bool = False,
    environment: Optional[Dict[str, Any]] = None,
)
```

Return structure includes build logs, evaluation stdout/stderr, the `run_dir` path, and summary metrics. When GEPA invokes it we now pass a dedicated `run_root`, so artifacts end up under `runs/gepa/<timestamp>_<label>/eval/.../` with the top-level run directory consolidating baseline, candidate, and log outputs.

`scripts/evaluate_heuristic_candidate.py` wraps this for CLI usage, accepting candidate code from a file/STDIN or evaluating the default block (`--use-default`). Use it for manual smoke tests or to wire GEPA quickly before a custom metric.

`src/gepa_metric.py` builds on this, providing `evaluate_and_score(...)` that returns `(score, feedback, result)` where the score is `-average_wall_time_sec` and the feedback string summarizes status, metrics, tail of build/evaluation logs, and artifact path—perfect input for DSPy’s `ScoreWithFeedback`.

Additional helpers:
- `scripts/run_gepa_tsplib.sh` – one-shot 10 step GEPA session against `tsplib_random`; produces a consolidated run directory plus an automatically generated rollout plot.
- `scripts/plot_gepa_metrics.py` – post-hoc visualization tool (scatter plot of average wall time vs GEPA iteration) for any run directory.

---

## 8. Testing & Validation Checklist

- `concorde/install/bin/concorde` solves the bundled `5city.tsp`, the toy20 set, and the explicit-weight splits without errors.
- `scripts/run_concorde_eval.py --split <split>` matches expectations across `toy20`, `toy200`, and `tsplib_random` (check wall times and `bbnodes`).
- Sandboxed builds replicate the same results (no path dependencies).
- `data/eval/metadata.json` stays alphabetically sorted; verify changes via review script/CI.
- GEPA end-to-end: `./scripts/run_gepa_tsplib.sh` completes, produces staged artifacts under `runs/gepa/<timestamp>_tsplib_run/`, and emits `gepa_plot.png` with reasonable metrics.

---

## 9. Next Enhancements

1. Adjust both the student and reflector prompts to encourage substantive LK rewrites (not just threshold tweaks) while preserving the contract—explicitly allow reorganizing the loop, alternative queue heuristics, etc.
2. Incorporate branch-and-bound node counts directly into the GEPA metric (fewer nodes preferred), either as part of the scalar score or as a multi-objective to help balance speed vs. search effort.
3. Source real TSPLIB instances (once networking hurdles are resolved) and register them via `scripts/add_tsplib_eval.py` so we benchmark against canonical problems.
4. Implement caching/deduplication keyed by candidate hash + dataset split to avoid re-running identical heuristics.
5. Provide a reusable instance generator script for future splits (currently ad-hoc).

Keep this document updated whenever the workflow changes so the Linux setup remains reproducible.

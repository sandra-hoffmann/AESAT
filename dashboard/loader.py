"""Reads AESAT run folders into DataFrames.

A run folder is any subfolder of the results directory that holds a config.json.
Every file except config.json is optional, so runs from older versions of the
loop (no queries.jsonl, baseline.json or progress.jsonl) load as well.

Nothing here depends on Streamlit; the app caches load_run by the folder's
signature, so an unchanged run is parsed once.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

# default: work/results next to the loop's main script, as in the loop itself
RESULTS_DIR = Path(os.environ.get(
    "AESAT_RESULTS_DIR",
    Path(__file__).resolve().parent.parent / "work" / "results"))

PROMPT_NAMES = ("language", "idea_prompt", "formula_min", "formula_repeat",
                "crossover", "innovation", "analysis_prompt", "fidelity")
MODEL_ROLES = ("idea", "formula", "fidelity", "crossover", "innovation", "analysis")
LOCAL_TZ = "Europe/Berlin"

# a running loop writes an event at least this often, except while a candidate
# is measured (see run_status)
STALE_AFTER = timedelta(minutes=15)


@dataclass
class Run:
    name: str
    path: Path
    config: dict
    candidates: pd.DataFrame          # one row per candidate
    attempts: pd.DataFrame            # one row per translation attempt
    generations: dict                 # generation -> {idea, analysis, crossover_rounds}
    queries: pd.DataFrame             # queries.jsonl, empty for old runs
    baseline: dict | None             # baseline.json of the run, None for old runs
    progress: pd.DataFrame            # progress.jsonl, empty for old runs
    prompts: dict = field(default_factory=dict)
    status: str = "unknown"           # running | finished | failed | aborted
    last_activity: datetime | None = None


# --------------------------------------------------------------------------- files

def list_runs(results_dir: Path = RESULTS_DIR) -> list[Path]:
    """Every run folder, newest first. A folder counts as a run once its
    config.json exists, which the loop writes before anything else."""
    if not results_dir.is_dir():
        return []
    runs = [p for p in results_dir.iterdir() if p.is_dir() and (p / "config.json").is_file()]
    return sorted(runs, key=lambda p: p.name, reverse=True)


def run_signature(path: Path) -> tuple:
    """(name, size, mtime) of every file in the run folder. Changes whenever the
    loop writes something, so it serves as the cache key of load_run."""
    sig = []
    for p in sorted(path.iterdir()):
        if p.is_file():
            st = p.stat()
            sig.append((p.name, st.st_size, st.st_mtime_ns))
    return tuple(sig)


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _read_jsonl(path: Path) -> list[dict]:
    """All complete lines of a jsonl file. A line the loop is still writing is
    skipped instead of failing the load."""
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


# --------------------------------------------------------------------------- parsing

def _clean_text(text):
    return text.strip() if isinstance(text, str) else text


ERROR_CATEGORIES = [
    # (pattern in the error text, category shown in the dashboard)
    ("CANNOT TRANSLATE", "refused"),
    ("no EXPRESSION", "no expression in response"),
    ("unknown heuristic term", "unknown term"),
    ("missing a valid operand", "too many operators"),
    ("ended early", "expression ended early"),
    ("already complete before", "too few operators"),
    ("different params", "wrong parameters"),
]


def error_category(error: str | None) -> str | None:
    """A short category for a translation error, from the solver's message or
    the loop's own error text."""
    if not error:
        return None
    for pattern, category in ERROR_CATEGORIES:
        if pattern in error:
            return category
    if error.startswith("Fail:"):
        return "other parser error"
    return "other"


def translation_status(origin: str, formula, attempts: list[dict]) -> str:
    """How the candidate's formula came about: accepted on the first try,
    accepted after retries, or why it was given up."""
    if origin == "default":
        return "default"
    if formula is not None:
        if origin == "crossover" or len(attempts) <= 1:
            return "accepted"
        return "accepted after retry"
    if not attempts:
        return "failed (no details)"
    last = attempts[-1]
    if last.get("fidelity") == "MISMATCH":
        return "fidelity mismatch"
    cat = error_category(last.get("error"))
    if cat == "refused":
        return "refused"
    if cat == "no expression in response":
        return "no expression"
    if origin == "crossover":
        return "invalid hybrid"
    return "parser error"


def _candidate_rows(gen_file: dict) -> tuple[list[dict], list[dict]]:
    cands, atts = [], []
    for c in gen_file.get("candidates", []):
        attempts = c.get("attempts") or []
        cands.append({
            "id": c["id"],
            "generation": c.get("generation", gen_file.get("generation")),
            "origin": c.get("origin"),
            "parents": list(c.get("parents") or []),
            "idea": _clean_text(c.get("idea")),
            "formula": c.get("formula"),
            "score": c.get("score"),
            "wrong_answers": c.get("wrong_answers"),
            "errors": c.get("errors"),
            "duplicate_of": c.get("duplicate_of"),
            "n_attempts": len(attempts),
            "translation": translation_status(c.get("origin"), c.get("formula"), attempts),
            "pending": False,
        })
        for i, a in enumerate(attempts):
            atts.append({
                "candidate_id": c["id"],
                "generation": cands[-1]["generation"],
                "origin": c.get("origin"),
                "attempt": i,
                "formula": a.get("formula"),
                "error": a.get("error"),
                "error_category": error_category(a.get("error")),
                "fidelity": a.get("fidelity"),
                "fidelity_reason": a.get("fidelity_reason"),
                "response": a.get("response") or a.get("fidelity_response"),
            })
    return cands, atts


CANDIDATE_COLUMNS = ["id", "generation", "origin", "parents", "idea", "formula", "score",
                     "wrong_answers", "errors", "duplicate_of", "n_attempts", "translation", "pending"]
ATTEMPT_COLUMNS = ["candidate_id", "generation", "origin", "attempt", "formula", "error",
                   "error_category", "fidelity", "fidelity_reason", "response"]


def _int_keys(d: dict | None) -> dict:
    """JSON turns the integer ids in innovation and fitness into strings."""
    out = {}
    for k, v in (d or {}).items():
        try:
            out[int(k)] = v
        except (TypeError, ValueError):
            out[k] = v
    return out


def _parse_ts(ts: str):
    """A progress timestamp as an aware datetime. Timestamps without an offset
    come from a container running on UTC."""
    try:
        dt = datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _pending_candidates(progress: pd.DataFrame, known: set) -> list[dict]:
    """Candidates measured in the current generation, from progress.jsonl.

    The generation file is only written when the generation ends; until then
    these rows let the live views show the measurements. Formula and idea are
    not in the log, so they stay empty until the generation file replaces the
    row."""
    if progress.empty or "event" not in progress:
        return []
    origin = {}
    if "candidate_id" in progress:
        for _, e in progress[progress["event"] == "measure_started"].iterrows():
            origin[e["candidate_id"]] = e.get("origin")
    out = []
    for _, e in progress[progress["event"].isin(["candidate_measured", "candidate_duplicate"])].iterrows():
        cid = e.get("candidate_id")
        if pd.isna(cid) or int(cid) in known:
            continue
        cid = int(cid)
        dup = e.get("duplicate_of") if e["event"] == "candidate_duplicate" else None
        out.append({
            "id": cid, "generation": int(e["generation"]), "origin": origin.get(cid),
            "parents": [], "idea": None, "formula": None,
            "score": e.get("score"), "wrong_answers": e.get("wrong_answers"), "errors": e.get("errors"),
            "duplicate_of": None if dup is None or pd.isna(dup) else int(dup),
            "n_attempts": None, "translation": "in progress", "pending": True,
        })
        known.add(cid)
    return out


# --------------------------------------------------------------------------- run

def load_run(path: Path, loop_running: bool | None = None) -> Run:
    """Everything in one run folder. loop_running says whether a loop process
    is alive in this container (None if that cannot be checked); see
    run_status."""
    config = _read_json(path / "config.json") or {}

    cands, atts, generations = [], [], {}
    for gp in sorted(path.glob("generation_*.json")):
        g = _read_json(gp)
        if not g:
            continue
        c, a = _candidate_rows(g)
        cands += c
        atts += a
        gen = g.get("generation")
        generations[gen] = {
            "idea": _clean_text(g.get("idea")),
            "analysis": _clean_text(g.get("analysis")),
            "crossover_rounds": [
                {**r, "innovation": _int_keys(r.get("innovation")),
                 "fitness": _int_keys(r.get("fitness"))}
                for r in g.get("crossover_rounds") or []],
            "has_rounds": "crossover_rounds" in g,
        }

    # summary.jsonl fills in candidates whose generation file is missing
    known = {c["id"] for c in cands}
    for s in _read_jsonl(path / "summary.jsonl"):
        if s.get("id") in known:
            continue
        cands.append({
            "id": s["id"], "generation": s.get("generation"), "origin": s.get("origin"),
            "parents": list(s.get("parents") or []), "idea": None, "formula": s.get("formula"),
            "score": s.get("score"), "wrong_answers": s.get("wrong_answers"),
            "errors": s.get("errors"), "duplicate_of": s.get("duplicate_of"),
            "n_attempts": None,
            "translation": translation_status(s.get("origin"), s.get("formula"), []),
            "pending": False,
        })
        known.add(s["id"])

    progress = pd.DataFrame(_read_jsonl(path / "progress.jsonl"))
    if not progress.empty:
        progress["ts"] = progress["ts"].map(_parse_ts)
        progress = progress.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    cands += _pending_candidates(progress, known)

    candidates = pd.DataFrame(cands, columns=CANDIDATE_COLUMNS)
    if not candidates.empty:
        candidates = candidates.sort_values("id").reset_index(drop=True)
        for col in ("score",):
            candidates[col] = pd.to_numeric(candidates[col], errors="coerce")
        for col in ("wrong_answers", "errors", "duplicate_of", "n_attempts"):
            candidates[col] = pd.to_numeric(candidates[col], errors="coerce").astype("Int64")
    attempts = pd.DataFrame(atts, columns=ATTEMPT_COLUMNS)

    queries = pd.DataFrame(_read_jsonl(path / "queries.jsonl"))
    if not queries.empty:
        for col in ("baseline_backtracks", "backtracks", "ratio", "runtime_s"):
            if col in queries:
                queries[col] = pd.to_numeric(queries[col], errors="coerce")

    prompts = {}
    for name in PROMPT_NAMES:
        p = path / f"{name}.txt"
        if p.is_file():
            prompts[name] = p.read_text()

    run = Run(name=path.name, path=path, config=config, candidates=candidates,
              attempts=attempts, generations=generations, queries=queries,
              baseline=_read_json(path / "baseline.json"), progress=progress,
              prompts=prompts)
    run.status, run.last_activity = run_status(run, loop_running)
    return run


def last_modified(path: Path) -> datetime | None:
    times = [p.stat().st_mtime for p in path.iterdir() if p.is_file()]
    if not times:
        return None
    return datetime.fromtimestamp(max(times), tz=timezone.utc)


def generations_done(run: Run) -> int:
    """Highest generation with a written generation file or summary entry."""
    if run.candidates.empty:
        return 0
    return int(run.candidates["generation"].max())


def run_status(run: Run, loop_running: bool | None) -> tuple[str, datetime | None]:
    """finished, failed, running or aborted.

    The end events of progress.jsonl decide where they exist. Otherwise a run
    counts as running if a loop process is alive and this run was active
    recently. During a measurement the loop is silent for up to n_queries x
    timeout seconds, so the allowed silence grows while a measure_started has
    no matching end event.
    """
    last = last_modified(run.path)
    events = run.progress["event"].tolist() if not run.progress.empty else []
    if "run_finished" in events:
        return "finished", last
    if "run_failed" in events:
        return "failed", last

    planned = run.config.get("generations")
    if not events and planned is not None and generations_done(run) >= planned:
        return "finished", last
    if loop_running is False:
        return "aborted", last

    allowed = STALE_AFTER
    if events and events[-1] == "measure_started":
        n = (run.baseline or {}).get("n_queries") or 50
        allowed = max(allowed, timedelta(seconds=n * run.config.get("timeout_s", 30)) + STALE_AFTER)
    if last is not None and datetime.now(timezone.utc) - last <= allowed:
        return "running", last
    return "aborted", last


def loop_process_count() -> int | None:
    """Number of AESAT loop processes in this container, from /proc. None where
    /proc is unavailable (e.g. when the dashboard runs on macOS)."""
    proc = Path("/proc")
    if not proc.is_dir():
        return None
    n = 0
    for p in proc.iterdir():
        if not p.name.isdigit():
            continue
        try:
            cmd = (p / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore")
        except OSError:
            continue
        if "heureka_optimization_main" in cmd and "streamlit" not in cmd:
            n += 1
    return n


def loop_running_for(paths: list[Path]) -> dict[Path, bool | None]:
    """For each run folder whether its loop process may still be alive.

    A process found in this container belongs to the newest run. No process
    found proves nothing: the loop may run in another container (e.g. one
    PyCharm starts per run), so then the status falls back to the time of the
    last write (None)."""
    n = loop_process_count()
    if not n:
        return {p: None for p in paths}
    newest = max(paths, key=lambda p: p.name) if paths else None
    return {p: (p == newest) for p in paths}


# --------------------------------------------------------------------------- config

def models_label(config: dict) -> str:
    """The models of a run in one string: one name if every role used the same
    model, otherwise role=model pairs."""
    pairs = [(r, config.get(f"{r}_model")) for r in MODEL_ROLES if config.get(f"{r}_model")]
    names = {m for _, m in pairs}
    if len(names) == 1:
        return names.pop()
    return ", ".join(f"{r}={m}" for r, m in pairs)


def to_local(dt: datetime | None):
    if dt is None:
        return None
    return pd.Timestamp(dt).tz_convert(LOCAL_TZ)


RUN_NAME = re.compile(r"run_(\d{8}_\d{6})")


def run_label(run: Run) -> str:
    """Run name plus status, for selectors."""
    return f"{run.name} ({run.status})"

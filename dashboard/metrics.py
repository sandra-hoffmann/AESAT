"""Derived numbers for the dashboard. Every function takes a loader.Run or its
DataFrames and returns new DataFrames; nothing here reads files."""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd

from loader import Run, generations_done, models_label

# terms of the expression language, in the order of language.txt
TERMS = ["scc", "indeg", "outdeg", "inpath", "outpath", "pm", "extdegrat", "cb", "ceig",
         "deg", "path"]
PARAM_COUNT = {"deg": 2, "path": 4}
OPERATORS = {"+", "-", "*", "/", "^"}
NEAR_TIMEOUT = 0.8          # share of the timeout above which a solved query counts as close


# --------------------------------------------------------------------------- formulas

def _is_number(tok: str) -> bool:
    try:
        float(tok)
        return True
    except ValueError:
        return False


def formula_key(formula) -> str | None:
    """Same normalisation as population.formula_key in the loop: every number in
    one notation. Two formulas with the same key are the same heuristic."""
    if not isinstance(formula, str):
        return None
    return " ".join(repr(float(t)) if _is_number(t) else t for t in formula.split())


def structure_signature(formula) -> str | None:
    """The formula with every number replaced by #, so variants of one idea and
    crossovers that only exchange weights share a signature."""
    if not isinstance(formula, str):
        return None
    return " ".join("#" if _is_number(t) else t for t in formula.split())


def terms_used(formula) -> list[str]:
    """The language terms a formula uses, in TERMS order."""
    if not isinstance(formula, str):
        return []
    toks = set(formula.split())
    return [t for t in TERMS if t in toks]


# --------------------------------------------------------------------------- queries

def query_outcomes(run: Run) -> pd.DataFrame:
    """queries.jsonl with an outcome per line: win, loss, tie, timeout, solver
    error or wrong answer. Empty for runs without per-query results."""
    q = run.queries
    if q.empty:
        return q
    q = q.copy()
    base = (run.baseline or {}).get("baseline_values", {})
    timeout = run.config.get("timeout_s", 30)

    def outcome(r):
        if r.get("timed_out"):
            return "timeout"
        if isinstance(r.get("error"), str):
            return "solver error"
        expected = base.get(r["query"], {}).get("accepted")
        wrong = (expected is not None and r.get("accepted") is not None
                 and r["accepted"] != expected) or (not base and r["ratio"] == 0.001)
        if wrong:
            return "wrong answer"
        if r["ratio"] > 1:
            return "win"
        if r["ratio"] < 1:
            return "loss"
        return "tie"

    q["outcome"] = q.apply(outcome, axis=1)
    q["solved"] = q["outcome"].isin(["win", "loss", "tie"])
    q["near_timeout"] = q["solved"] & (q["runtime_s"] >= NEAR_TIMEOUT * timeout)
    q["log10_ratio"] = np.log10(q["ratio"].where(q["ratio"] > 0))
    return q


def per_candidate_query_stats(run: Run) -> pd.DataFrame:
    """Per measured candidate: queries compared, solved, wins, losses, ties,
    timeouts, near-timeout queries and the score over solved queries only.
    The solved-only score separates the heuristic's quality from the penalties
    for timeouts and wrong answers."""
    q = query_outcomes(run)
    if q.empty:
        return pd.DataFrame(columns=["candidate_id"])
    rows = []
    for cid, g in q.groupby("candidate_id"):
        solved = g[g["solved"]]
        rows.append({
            "candidate_id": cid,
            "n_queries": len(g),
            "solved": len(solved),
            "wins": int((g["outcome"] == "win").sum()),
            "losses": int((g["outcome"] == "loss").sum()),
            "ties": int((g["outcome"] == "tie").sum()),
            "timeouts": int((g["outcome"] == "timeout").sum()),
            "near_timeout": int(g["near_timeout"].sum()),
            "solved_score": (math.exp(np.log(solved["ratio"]).mean()) if len(solved) else np.nan),
            "runtime_s": g["runtime_s"].sum(),
            "_results": tuple(zip(g["query"], g["backtracks"].fillna(-1), g["timed_out"])),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- candidates

def enrich(run: Run) -> pd.DataFrame:
    """The candidate table with everything the views need:
    key, signature and terms of the formula; crossover gain; per-query
    statistics; repeat_of for runs before duplicate detection; same_results_as
    for different formulas that produced identical results."""
    c = run.candidates.copy()
    if c.empty:
        return c
    c["key"] = c["formula"].map(formula_key)
    c["signature"] = c["formula"].map(structure_signature)
    c["terms"] = c["formula"].map(terms_used)
    c["measured"] = c["score"].notna()

    # first candidate per formula; later ones are repeats, whether or not the
    # loop already detected them (duplicate_of)
    first = {}
    repeat_of = []
    for _, r in c.iterrows():
        k = r["key"]
        if not isinstance(k, str):
            repeat_of.append(pd.NA)
            continue
        if k in first:
            repeat_of.append(first[k])
        else:
            first[k] = r["id"]
            repeat_of.append(pd.NA)
    c["repeat_of"] = pd.array(repeat_of, dtype="Int64")
    # a duplicate detected by the loop counts as a repeat even without a formula
    # (rows of the running generation, taken from progress.jsonl)
    c["repeat_of"] = c["repeat_of"].fillna(c["duplicate_of"])
    c["is_repeat"] = c["repeat_of"].notna()

    # a duplicate logged in progress.jsonl carries no score; it is the original's
    orig_score = dict(zip(c["id"], c["score"]))
    c["score"] = c.apply(lambda r: orig_score.get(int(r["duplicate_of"]), r["score"])
                         if pd.isna(r["score"]) and pd.notna(r["duplicate_of"]) else r["score"], axis=1)
    c["measured"] = c["score"].notna()

    score = dict(zip(c["id"], c["score"]))
    def gain(r):
        ps = [score.get(p) for p in r["parents"] if score.get(p) is not None and not pd.isna(score.get(p))]
        if r["origin"] != "crossover" or not ps or pd.isna(r["score"]):
            return np.nan
        return r["score"] - max(ps)
    c["crossover_gain"] = c.apply(gain, axis=1)
    c["best_parent_score"] = c.apply(
        lambda r: max([score[p] for p in r["parents"] if p in score and not pd.isna(score[p])], default=np.nan),
        axis=1)

    stats = per_candidate_query_stats(run)
    if not stats.empty:
        c = c.merge(stats, how="left", left_on="id", right_on="candidate_id").drop(columns="candidate_id")
        # a detected duplicate was not measured again; it inherits the original's numbers
        by_id = stats.set_index("candidate_id")
        for col in ("n_queries", "solved", "wins", "losses", "ties", "timeouts", "near_timeout",
                    "solved_score", "_results"):
            src = c["duplicate_of"].map(lambda d: by_id[col].get(d) if pd.notna(d) and d in by_id.index else None)
            c[col] = c[col].where(c[col].notna(), src)
        for col in ("n_queries", "solved", "wins", "losses", "ties", "timeouts", "near_timeout",
                    "solved_score", "runtime_s"):
            c[col] = pd.to_numeric(c[col], errors="coerce")
    else:
        for col in ("n_queries", "solved", "wins", "losses", "ties", "timeouts", "near_timeout",
                    "solved_score", "runtime_s", "_results"):
            c[col] = np.nan

    c["same_results_as"] = _same_results(c)
    return c


def _same_results(c: pd.DataFrame) -> pd.Series:
    """For each measured candidate, the first other candidate with a different
    formula but identical results: the same backtracks on every query where
    per-query results exist, otherwise the same score, errors and wrong answers.
    Such formulas are very likely the same ordering written differently, e.g.
    all weights scaled by one factor."""
    groups = defaultdict(list)
    for _, r in c[c["measured"] & ~c["is_repeat"]].iterrows():
        res = r.get("_results")
        if isinstance(res, tuple):
            fp = ("q", res)
        else:
            fp = ("s", round(r["score"], 12), r["errors"], r["wrong_answers"])
        groups[fp].append(r["id"])
    out = {}
    for ids in groups.values():
        if len(ids) > 1:
            for i in ids[1:]:
                out[i] = ids[0]
    return c["id"].map(out).astype("Int64")


def best_curve(cands: pd.DataFrame) -> pd.DataFrame:
    """Best score so far per generation, and the number of measurements needed
    to get there (repeats excluded, since they cost nothing with duplicate
    detection and were noise before it)."""
    m = cands[cands["measured"]]
    if m.empty:
        return pd.DataFrame(columns=["generation", "best_score", "evaluations"])
    per_gen = m.groupby("generation").agg(gen_best=("score", "max"))
    evals = cands[cands["measured"] & ~cands["is_repeat"]].groupby("generation").size()
    gens = range(int(cands["generation"].min()), int(cands["generation"].max()) + 1)
    rows, best, n = [], -np.inf, 0
    for g in gens:
        if g in per_gen.index:
            best = max(best, per_gen.loc[g, "gen_best"])
        n += int(evals.get(g, 0))
        rows.append({"generation": g, "best_score": best if best > -np.inf else np.nan,
                     "evaluations": n})
    return pd.DataFrame(rows)


def translation_by_generation(cands: pd.DataFrame) -> pd.DataFrame:
    """Count of variants per generation and translation outcome. Crossovers are
    left out: their formulas are not translated from an idea."""
    v = cands[(cands["origin"] == "variant") & ~cands["pending"].fillna(False).astype(bool)]
    if v.empty:
        return pd.DataFrame(columns=["generation", "translation", "count"])
    return v.groupby(["generation", "translation"]).size().reset_index(name="count")


def term_usage(cands: pd.DataFrame) -> pd.DataFrame:
    """How many measured candidates of each generation use each term."""
    m = cands[cands["measured"] & (cands["origin"] != "default")]
    rows = [{"generation": r["generation"], "term": t} for _, r in m.iterrows() for t in r["terms"]]
    if not rows:
        return pd.DataFrame(columns=["generation", "term", "count"])
    return pd.DataFrame(rows).groupby(["generation", "term"]).size().reset_index(name="count")


def term_effect(cands: pd.DataFrame) -> pd.DataFrame:
    """Median score of measured candidates with and without each term. A
    descriptive comparison: terms appear together, so a difference is not the
    effect of the term alone."""
    m = cands[cands["measured"] & ~cands["is_repeat"] & (cands["origin"] != "default")]
    rows = []
    for t in TERMS:
        has = m["terms"].map(lambda ts: t in ts)
        if has.sum() == 0:
            continue
        rows.append({"term": t, "with": m.loc[has, "score"].median(),
                     "without": m.loc[~has, "score"].median() if (~has).any() else np.nan,
                     "n_with": int(has.sum()), "n_without": int((~has).sum())})
    return pd.DataFrame(rows)


def ancestors(cands: pd.DataFrame, root: int, max_depth: int = 30) -> set[int]:
    parents = dict(zip(cands["id"], cands["parents"]))
    seen, todo = set(), [(root, 0)]
    while todo:
        cid, d = todo.pop()
        if cid in seen or d > max_depth:
            continue
        seen.add(cid)
        for p in parents.get(cid, []) or []:
            todo.append((p, d + 1))
    return seen


def lineage_dot(cands: pd.DataFrame, root: int, dark: bool = False) -> str:
    """Graphviz DOT of the crossover ancestry of one candidate. Each node shows
    id, generation and score; edges point from parent to child."""
    ids = ancestors(cands, root)
    sub = cands[cands["id"].isin(ids)]
    ink, muted, fill_v, fill_c, fill_d = (
        ("#f0efec", "#898781", "#1c5cab", "#b54a1f", "#383835") if dark else
        ("#0b0b0b", "#898781", "#cde2fb", "#fbd9c9", "#f0efec"))
    lines = ["digraph G {", "rankdir=LR;", 'bgcolor="transparent";',
             f'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, color="{muted}", fontcolor="{ink}"];',
             f'edge [color="{muted}"];']
    for _, r in sub.iterrows():
        score = "-" if pd.isna(r["score"]) else f"{r['score']:.3f}"
        fill = {"variant": fill_v, "crossover": fill_c}.get(r["origin"], fill_d)
        pen = ', penwidth=2.5' if r["id"] == root else ""
        lines.append(f'{r["id"]} [label="#{r["id"]}  gen {r["generation"]}\\n{r["origin"]}  {score}", fillcolor="{fill}"{pen}];')
        for p in r["parents"]:
            if p in ids:
                lines.append(f"{p} -> {r['id']};")
    lines.append("}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- timing

def _events(run: Run, name: str, cols: list[str]) -> pd.DataFrame:
    """The progress events of one kind, with the given columns present even
    when no such event or field has been written yet (e.g. right after start)."""
    p = run.progress
    if p.empty or "event" not in p:
        return pd.DataFrame(columns=["ts", "generation", "event", *cols])
    e = p[p["event"] == name].copy()
    for col in cols:
        if col not in e:
            e[col] = np.nan
    return e


def llm_calls(run: Run) -> pd.DataFrame:
    return _events(run, "llm_call", ["role", "seconds", "ok"])


def time_by_generation(run: Run) -> pd.DataFrame:
    """Seconds per generation spent in model calls (per role) and in solver
    measurements, from progress.jsonl."""
    rows = []
    calls = llm_calls(run).dropna(subset=["role", "seconds"])
    for (g, role), s in calls.groupby(["generation", "role"])["seconds"].sum().items():
        rows.append({"generation": g, "part": f"model: {role}", "seconds": s})
    meas = _events(run, "candidate_measured", ["seconds"]).dropna(subset=["seconds"])
    for g, s in meas.groupby("generation")["seconds"].sum().items():
        rows.append({"generation": g, "part": "solver", "seconds": s})
    return pd.DataFrame(rows, columns=["generation", "part", "seconds"])


def run_duration(run: Run) -> float | None:
    """Seconds from the first to the last event, or None without a log."""
    p = run.progress
    if p.empty:
        return None
    return (p["ts"].iloc[-1] - p["ts"].iloc[0]).total_seconds()


def current_activity(run: Run) -> dict:
    """What a running loop is doing now, from the last events: the current
    generation, the candidate being measured and since when."""
    p = run.progress
    if p.empty:
        return {}
    last = p.iloc[-1]
    out = {"generation": int(last["generation"]), "last_event": last["event"], "since": last["ts"]}
    if last["event"] == "measure_started" and pd.notna(last.get("candidate_id")):
        out["measuring"] = int(last["candidate_id"])
    return out


# --------------------------------------------------------------------------- runs

def run_summary(run: Run, cands: pd.DataFrame | None = None) -> dict:
    """One row of the overview table."""
    c = enrich(run) if cands is None else cands
    cfg = run.config
    m = c[c["measured"] & (c["origin"] != "default")] if not c.empty else c
    best = c.loc[c["score"].idxmax()] if not c.empty and c["score"].notna().any() else None
    variants = (c[(c["origin"] == "variant") & ~c["pending"].fillna(False).astype(bool)]
                if not c.empty else c)
    failed = variants[variants["formula"].isna()] if not variants.empty else variants
    duration = run_duration(run)
    return {
        "run": run.name,
        "status": run.status,
        "generations": f"{generations_done(run)}/{cfg.get('generations', '?')}",
        "best score": None if best is None else round(float(best["score"]), 4),
        "best id": None if best is None else int(best["id"]),
        "measured": int((m["measured"] & ~m["is_repeat"]).sum()) if not m.empty else 0,
        "repeats": int(m["is_repeat"].sum()) if not m.empty else 0,
        "translation failed": f"{len(failed)}/{len(variants)}" if len(variants) else "-",
        "duration h": None if duration is None else round(duration / 3600, 2),
        "started": cfg.get("started"),
        "models": models_label(cfg) if cfg else None,
        "task": cfg.get("task"),
        "benchmark": str(cfg.get("benchmark", "")).rstrip("/").split("/")[-1],
        "best formula": None if best is None else best["formula"],
    }


def config_diff(runs: list[Run]) -> pd.DataFrame:
    """Config entries that differ between the runs, one column per run."""
    keys = sorted({k for r in runs for k in r.config})
    rows = []
    for k in keys:
        vals = [r.config.get(k) for r in runs]
        if len({json_str(v) for v in vals}) > 1:
            rows.append({"setting": k, **{r.name: json_str(v) for r, v in zip(runs, vals)}})
    return pd.DataFrame(rows)


def json_str(v) -> str:
    return "-" if v is None else str(v)

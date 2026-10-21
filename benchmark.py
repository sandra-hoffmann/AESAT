#!/usr/bin/env python3
"""benchmark.py.py

Short description of this script.

Usage:
    python benchmark.py.py [options]

Created: 12.10.26
Author: Sandra Hoffmann
"""
import json
from pathlib import Path
from dataclasses import dataclass
from statistics import mean

import math
from math import inf
from tqdm import tqdm

from measure_solver import run_once


@dataclass
class Query:
    af_path: Path
    arg: int
    name: str


def load_queries(path:Path)->list[Query]:
    #load af files from path
    af_paths = sorted(path.glob("*.af"))
    if len(af_paths) == 0:
        raise FileNotFoundError(f"No af files found in {path}")
    queries = []
    for af_path in af_paths:
        arg_path = af_path.with_suffix(".arg")
        if arg_path.exists():
            try:
                arg = int(arg_path.read_text().strip())
            except ValueError:
                print(f"Could not parse {arg_path}. Please check file contents. File {af_path} is skipped")
                continue
        else:
            raise FileNotFoundError(f"No arg file found for {af_path}")
        queries.append(Query(af_path, arg, af_path.stem))
    return queries


def measure_query(query: Query, task, heuristic, template: str, timeout):
    if "-d 1" not in template:
        raise ValueError(f"Template {template} needs to have debug mode active")
    run_result = run_once(template, query.af_path, task, query.arg, timeout, heuristic)
    result = {}
    if run_result.get('timed_out'):
        result['backtracks'] = inf
        result['timed_out'] = True
        result['accepted'] = None
        result['error'] = None
        return result

    if run_result.get('returncode') != 0:
        result['backtracks'] = inf
        result['timed_out'] = False
        result['accepted'] = None
        # the log also carries the header line and the messages that only pass the
        # failure upwards; the first Fail: line names the cause
        result['error'] = [l for l in run_result.get('log', []) if l.startswith("Fail:")]
        return result

    out = run_result.get('out')
    if not out:
        raise ValueError(
            f"Solver returned 0 but produced no output for {query.af_path}. Log: "
            f"{run_result.get('log')}"
        )

    answer = out[0].replace("[", "").replace("]", "").strip()
    if answer == 'NO':
        result['accepted'] = False
    elif answer == 'YES':
        result['accepted'] = True
    else:
        raise ValueError(
            f"Unexpected answer for {query.af_path}. Check solver adheres to ICCMA "
            f"Format and outputs acceptance status first"
        )

    result['backtracks'] = run_result['metrics']['backtracks']
    result['timed_out'] = False
    result['error'] = None
    return result

def measure_baseline(benchmark:list[Query], task, template:str, timeout):
    results = {}
    skipped = 0
    for query in tqdm(benchmark):
        query_measure = {}
        full_result = measure_query(query, task, None, template, timeout)
        if full_result.get('timed_out') or full_result.get('error') is not None:
            skipped += 1
            continue
        query_measure['accepted'] = full_result.get('accepted')
        query_measure['backtracks'] = full_result.get('backtracks')
        results[query.name] = query_measure
    if results == {}:
        raise ValueError("All queries timed out or produced an error.")
    if skipped > 0:
        print(f"{skipped} out of {len(benchmark)} queries skipped due to timeout or solver error")
    return results

def load_baseline(path: Path, task: str, template: str, timeout) -> dict:
    """The baseline measurements for `task`, from disk if they are already there
    and were taken with the same template, otherwise measured and written out.

    One file per benchmark folder holds one entry per task, so measuring DC-PR
    does not discard an earlier DS-PR run.
    """
    baseline_file = path / "baseline.json"
    try:
        with baseline_file.open() as fh:
            stored = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        stored = {"benchmark_path": str(path)}

    entry = stored.get(task)
    if entry is not None and entry["template"] == template:
        return entry["baseline_values"]

    queries = load_queries(path)
    baseline = measure_baseline(queries, task, template, timeout)
    stored[task] = {"template": template, "baseline_values": baseline}
    with baseline_file.open("w") as fh:
        json.dump(stored, fh, indent=2)
    return baseline

def calculate_candidate_score(queries: list[Query], baseline: dict, heuristic: str,
                              task, template, timeout,
                              wrong_answer_penalty=0.001, timeout_penalty=0.01):
    """How a heuristic compares to the default one, as a single number.

    Per query the ratio of baseline backtracks to candidate backtracks; above 1
    means fewer backtracks than the default. A run that aborts or answers
    differently than the baseline gets a fixed bad ratio instead.
    """
    wrong_answers = 0
    errors = 0
    ratios = {}

    for query in tqdm(queries):
        # queries without a baseline value have nothing to compare against
        if query.name not in baseline:
            continue

        full_result = measure_query(query, task, heuristic, template, timeout)

        # an aborted run has neither an answer nor a counter, so it is checked first
        if full_result.get('timed_out') or full_result.get('error') is not None:
            ratios[query.name] = timeout_penalty
            errors += 1
            continue

        # a heuristic only changes the search order, so a different answer is a defect
        if full_result.get('accepted') != baseline[query.name]['accepted']:
            ratios[query.name] = wrong_answer_penalty
            wrong_answers += 1
            continue

        # +1 on both sides to avoid division by 0 errors if no backtracks occur
        backtracks_baseline = baseline[query.name]['backtracks']
        backtracks_query = full_result.get('backtracks')
        ratios[query.name] = (backtracks_baseline + 1) / (backtracks_query + 1)

    if not ratios:
        raise ValueError("No query could be compared against the baseline.")

    # geometric mean
    score = math.exp(mean(math.log(r) for r in ratios.values()))
    return score, wrong_answers, errors

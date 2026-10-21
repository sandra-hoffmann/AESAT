"""Check a generated benchmark against its manifest.

Run once per benchmark folder, right after create_benchmark.

create_benchmark writes the manifest and the exported .af/.arg copies separately,
so the two can diverge without anything failing. The loop scores candidates against
the baseline and never against an expected answer, so a mispaired graph and node
would go unnoticed. This check is the only place it becomes visible.
"""


import json
from pathlib import Path

from benchmark import load_queries, measure_query


def load_expected_answers(path: Path, semantics: str, task: str) -> dict[str, bool]:
    manifest_path = path / "manifest.json"
    try:
        with manifest_path.open() as fh:
            manifest = json.load(fh)
    except json.JSONDecodeError as err:
        raise ValueError(f"{manifest_path} is not valid JSON: {err}") from err

    if manifest["semantics"] != semantics or manifest["task"] != task:
        raise ValueError(
            f"{manifest_path} holds {manifest['semantics']}/{manifest['task']}, "
            f"but {semantics}/{task} was requested. The stored answers do not apply."
        )

    expected_answers = {}
    for query in manifest["queries"]:
        name = query["name"]
        if name in expected_answers:
            raise ValueError(f"Duplicate query name in {manifest_path}: {name}")
        expected_answers[name] = query["expected"]
    return expected_answers


def check_against_manifest(path: Path, semantics: str, task: str,
                           template: str, timeout: float) -> list[dict]:
    queries = load_queries(path)
    expected = load_expected_answers(path, semantics, task)
    problem = f"{task}-{semantics}"   # heureka will DC-PR, das Manifest haelt DC und PR getrennt
    findings = []

    # Namensmengen zuerst. Kostet nichts und deckt den groebsten Fall vor
    # sieben Minuten Rechnerei auf.
    in_folder = {q.name for q in queries}
    in_manifest = set(expected)
    for name in sorted(in_folder - in_manifest):
        findings.append({"name": name, "expected": None, "measured": None,
                         "reason": "query file has no manifest entry"})
    for name in sorted(in_manifest - in_folder):
        findings.append({"name": name, "expected": expected[name], "measured": None,
                         "reason": "manifest entry has no query file"})
    if findings:
        print(f"{len(findings)} names do not line up between folder and manifest")
        for f in findings:
            print(f"  {f['name']}: {f['reason']}")

    matched = differed = unmeasured = 0
    checkable = [q for q in queries if q.name in expected]

    for i, query in enumerate(checkable, 1):
        print(f"[{i}/{len(checkable)}] {query.name}", flush=True)
        result = measure_query(query, problem, None, template, timeout)

        if result["accepted"] is None:
            unmeasured += 1
            reason = "timed out" if result["timed_out"] else f"solver error: {result['error']}"
        elif result["accepted"] != expected[query.name]:
            differed += 1
            reason = "answer differs from manifest"
        else:
            matched += 1
            continue

        finding = {"name": query.name,
                   "expected": expected[query.name],
                   "measured": result["accepted"],
                   "reason": reason}
        findings.append(finding)
        print(f"  FINDING {finding}", flush=True)

    print(f"\n{len(checkable)} of {len(queries)} queries checked: "
          f"{matched} matched, {differed} differed, {unmeasured} not measured")
    print(f"{len(findings)} findings in total")
    return findings


if __name__ == "__main__":
    BENCHMARK = Path("/work/data/G2IO_heureka_dcpr/benchmark_PR_DC_20261016_120429_1_arg_per_graph")
    SEMANTICS = "PR"
    TASK = "DC"
    TEMPLATE = "heureka -p {task} -f {file} -H {heuristic} -a {arg} -d 1"
    TIMEOUT = 120

    findings = check_against_manifest(BENCHMARK, SEMANTICS, TASK, TEMPLATE, TIMEOUT)
    if findings:
        out = BENCHMARK / "manifest_check.json"
        with out.open("w") as fh:
            json.dump(findings, fh, indent=2)
        print(f"\n{len(findings)} findings written to {out}")

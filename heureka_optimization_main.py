from pathlib import Path

import json
import time

from benchmark import load_queries, load_baseline, calculate_candidate_score, Query, load_expected_answers
from config import Config
from file_utils import read_file
from llm_api import GPTCallAPI
from measure_solver import run_once

EXECUTION_START_TIME = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
MINI_AF = Path(__file__).resolve().parent / "test_instance.af"
BENCHMARK = Path("/work/data/G2IO_heureka_dcpr/benchmark_PR_DC_20261016_120429_1_arg_per_graph")
RESULTS = Path(__file__).resolve().parent / "work" / "results"
SOLVER_CMD = "heureka -p {task} -f {file} -H {heuristic} -a {arg} -d 1"
MAX_TRIES = 5
ITERATIONS = 30

# the solver's own heuristic for every semantics except ST; it is iteration 0 and
# the reference every score is relative to, so its score is 1.0 by definition
DEFAULT_HEURISTIC = "+ * scc 1000 * -1 * outpath inpath"
DEFAULT_IDEA = ("Weight the SCC ordering very strongly, then subtract the product "
                "of incoming and outgoing path counts, so that arguments in early "
                "components come first and, within a component, those that are "
                "neither deeply reached nor far-reaching are preferred.")
FIRST_ANALYSIS = ("No candidate has been tried yet. The starting point is the "
                  "solver's default heuristic.")


def extract_expression(response: str) -> str | None:
    """The formula behind the last EXPRESSION: marker, or None if the model did
    not answer in the expected format. A missing marker also covers the
    CANNOT TRANSLATE case, where the idea has no expressible form."""
    if "EXPRESSION:" not in response:
        return None
    return response.rsplit("EXPRESSION:", maxsplit=1)[1].strip().strip("`").strip()


def validate_heuristic(heuristic: str, task: str, template: str, timeout: int = 1) -> str | None:
    """The solver's complaint about `heuristic`, or None if it parses.

    Runs against a six-argument framework, so the search costs nothing and only
    the parser is exercised. Uses the same template as the measurement, so a
    broken template shows up here and not three hundred solver calls later.
    """
    probe = Query(MINI_AF, 1, "mini")
    result = run_once(template, probe.af_path, task, probe.arg, timeout, heuristic)

    # six arguments cannot take a second; a timeout here means something else is wrong
    if result.get("timed_out"):
        raise RuntimeError(f"Validation timed out for: {heuristic}")

    if result.get("returncode") != 0:
        return "\n".join(l for l in result.get("log", []) if l.startswith("Fail:"))
    return None


def generate_heuristic(idea: str, api, first_prompt: str, repeat_prompt: str,
                       task: str, template: str, max_tries: int = MAX_TRIES):
    """A validated formula for `idea`, plus every attempt it took to get there.

    Returns (formula, attempts). `formula` is None when no attempt validated.
    `attempts` holds one entry per try with the formula and the solver's
    complaint; it goes into the iteration's record.
    """
    attempts = []
    prompt = first_prompt.format(idea=idea)

    for _ in range(max_tries):
        response = api.generate(prompt)
        formula = extract_expression(response)

        if formula is None:
            # no marker: the model refused or ignored the format. Asking again
            # with the same idea would only push it to invent something.
            attempts.append({"formula": None, "error": "no EXPRESSION: in response",
                             "response": response})
            return None, attempts

        error = validate_heuristic(formula, task, template)
        attempts.append({"formula": formula, "error": error})
        if error is None:
            return formula, attempts

        print(f"invalid formula: {formula}\n{error}")
        prompt = repeat_prompt.format(idea=idea, formula=formula,
                                      errors_in_heuristic=error)

    return None, attempts


def analyse(api, prompt_template: str, record: dict, best: dict) -> str:
    """What this candidate did differently from the best one so far.

    The text is the only thing the next idea sees of this iteration, so it is
    written for a model, not for a reader.
    """
    prompt = prompt_template.format(
        best_idea=best["idea"], best_formula=best["formula"], best_score=best["score"],
        idea=record["idea"], formula=record["formula"], score=record["score"],
        wrong_answers=record["wrong_answers"], errors=record["errors"],
        attempts=len(record["attempts"]),
    )
    return api.generate(prompt)


def format_tried(records: list[dict]) -> str:
    """One line per candidate, so the idea model can see what is already taken
    without the prompt growing by a page per iteration."""
    return "\n".join(f"{r['formula']} -> {r['score']}" for r in records)


def write_record(run_dir: Path, record: dict) -> None:
    """One file per iteration, plus one line in a summary that can be read
    without opening twenty files."""
    path = run_dir / f"iteration_{record['iteration']:03d}.json"
    path.write_text(json.dumps(record, indent=2))
    with (run_dir / "summary.jsonl").open("a") as fh:
        fh.write(json.dumps({k: record[k] for k in
                             ("iteration", "formula", "score", "wrong_answers", "errors")}) + "\n")


def main():
    task = "DC-PR"
    timeout = 30

    idea_prompt = read_file("prompt_af/idea_prompt.txt")
    formula_prompt = read_file("prompt_af/formula_min.txt")
    repeat_prompt = read_file("prompt_af/formula_repeat.txt")
    analysis_prompt = read_file("prompt_af/analysis_prompt.txt")
    if not all([idea_prompt, formula_prompt, repeat_prompt, analysis_prompt]):
        raise RuntimeError("Could not read one of the prompt templates")

    idea_api = GPTCallAPI(Config.OPENAI_API_BASE, Config.OPENAI_API_KEY, Config.OPENAI_MODEL)
    formula_api = GPTCallAPI(Config.CLAUDE_API_BASE, Config.CLAUDE_API_KEY, Config.CLAUDE_MODEL)
    analysis_api = GPTCallAPI(Config.DEEPSEEK_API_BASE, Config.DEEPSEEK_API_KEY, Config.DEEPSEEK_MODEL)

    run_dir = RESULTS / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(json.dumps({
        "started": EXECUTION_START_TIME,
        "benchmark": str(BENCHMARK),
        "task": task,
        "solver_cmd": SOLVER_CMD,
        "timeout_s": timeout,
        "iterations": ITERATIONS,
        "idea_model": Config.OPENAI_MODEL,
        "formula_model": Config.CLAUDE_MODEL,
        "analysis_model": Config.DEEPSEEK_MODEL,
    }, indent=2))

    queries = load_queries(BENCHMARK)
    expected_answers = load_expected_answers(BENCHMARK)
    baseline = load_baseline(BENCHMARK, task, SOLVER_CMD, timeout)

    best = {"iteration": 0, "idea": DEFAULT_IDEA, "formula": DEFAULT_HEURISTIC,
            "score": 1.0, "wrong_answers": 0, "errors": 0, "attempts": [],
            "analysis": FIRST_ANALYSIS}
    write_record(run_dir, best)

    tried = [best]
    analysis = FIRST_ANALYSIS

    for i in range(1, ITERATIONS + 1):
        print(f"\n=== iteration {i}/{ITERATIONS} ===")

        idea = idea_api.generate(idea_prompt.format(analysis=analysis,
                                                    tried=format_tried(tried)))
        print(f"IDEA: {idea}")

        formula, attempts = generate_heuristic(idea, formula_api, formula_prompt,
                                               repeat_prompt, task, SOLVER_CMD)
        if formula is None:
            # capture failed formula translation in the record
            write_record(run_dir, {"iteration": i, "idea": idea, "formula": None,
                                   "score": None, "wrong_answers": None,
                                   "errors": None, "attempts": attempts,
                                   "analysis": None})
            print("no valid formula, skipping measurement")
            continue

        print(f"FORMULA ({len(attempts)} attempt(s)): {formula}")
        score, wrong_answers, errors = calculate_candidate_score(
            queries, baseline, formula, task, SOLVER_CMD, timeout)
        print(f"SCORE: {score} ({wrong_answers} wrong, {errors} failed)")

        record = {"iteration": i, "idea": idea, "formula": formula, "score": score,
                  "wrong_answers": wrong_answers, "errors": errors,
                  "attempts": attempts, "analysis": None}

        analysis = analyse(analysis_api, analysis_prompt, record, best)
        record["analysis"] = analysis
        print(f"ANALYSIS: {analysis}")

        write_record(run_dir, record)
        tried.append(record)
        if score > best["score"]:
            best = record

    print(f"\nbest: iteration {best['iteration']}, score {best['score']}")
    print(best["formula"])
    print(f"records in {run_dir}")


if __name__ == "__main__":
    main()
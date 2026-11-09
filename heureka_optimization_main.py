from pathlib import Path
from dataclasses import asdict

import json
import re
import random
import time
from datetime import datetime

from benchmark import load_queries, load_baseline, calculate_candidate_score, Query
from config import Config
from file_utils import read_file
from llm_api import GPTCallAPI
from measure_solver import run_once
from population import (Candidate, summary_line, format_tried, fitness, prune,
                        formula_key, add_to_pool, DEFAULT_INNOVATION)

EXECUTION_START_TIME = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
MINI_AF = Path(__file__).resolve().parent / "test_instance.af"
BENCHMARK = Path("/work/data/WS_k4_heureka_dcpr/benchmark_PR_DC_20261109_094228")
RESULTS = Path(__file__).resolve().parent / "work" / "results"
# next to this script, so the loop works from any working directory (PyCharm,
# docker exec, another folder)
PROMPTS = Path(__file__).resolve().parent / "prompt_af"
SOLVER_CMD = "heureka -p {task} -f {file} -H {heuristic} -a {arg} -d 1"

GENERATIONS = 30
N_VARIANTS = 5
MAX_CROSSOVER_ROUNDS = 10   # each round costs one measurement per hybrid
MAX_TRIES = 5              # per variant; a formula that fails twice is suspect anyway
SEED = 42                  # the fitness has a random term; without this no run repeats

# the solver's own heuristic for every semantics except ST; it is generation 0 and
# the reference every score is relative to, so its score is 1.0 by definition
DEFAULT_HEURISTIC = "+ * scc 1000 * -1 * outpath inpath"
DEFAULT_IDEA = ("Weight the SCC ordering very strongly, then subtract the product "
                "of incoming and outgoing path counts, so that arguments in early "
                "components come first and, within a component, those that are "
                "neither deeply reached nor far-reaching are preferred.")
FIRST_ANALYSIS = ("No candidate has been tried yet. The starting point is the "
                  "solver's default heuristic.")

# the progress log of the running loop; set in main so that a crash can still be
# logged from the __main__ block
_active_log = None


class ProgressLog:
    """progress.jsonl: one line per event, appended at once, so the dashboard can
    follow a generation before its file is written. Every line carries a
    timestamp and the current generation."""

    def __init__(self, run_dir: Path):
        self.path = run_dir / "progress.jsonl"
        self.generation = 0

    def event(self, name: str, **fields) -> None:
        # with the UTC offset, so the dashboard can show local time whatever
        # time zone the container runs in
        line = {"ts": datetime.now().astimezone().isoformat(timespec="milliseconds"),
                "generation": self.generation, "event": name, **fields}
        with self.path.open("a") as fh:
            fh.write(json.dumps(line) + "\n")


class TimedAPI:
    """A model client that logs every generate call with its role, model and
    duration. Used for the time and call count per role in the dashboard."""

    def __init__(self, api, role: str, log: ProgressLog):
        self.api = api
        self.role = role
        self.log = log
        self.config = api.config

    def generate(self, prompt: str) -> str:
        start = time.perf_counter()
        try:
            response = self.api.generate(prompt)
        except Exception as e:
            self.log.event("llm_call", role=self.role, model=self.config.model_name,
                           seconds=round(time.perf_counter() - start, 3),
                           ok=False, error=repr(e))
            raise
        self.log.event("llm_call", role=self.role, model=self.config.model_name,
                       seconds=round(time.perf_counter() - start, 3), ok=True,
                       prompt_chars=len(prompt), response_chars=len(response or ""))
        return response


LABELS = ("EXPRESSION:", "PARENTS:", "CANNOT TRANSLATE:")
LIST_MARKER = re.compile(r"^(?:\d+[.)]|[-*•])\s+")


def _clean(line: str) -> str:
    """A response line without surrounding whitespace and bold markers, which
    models like to put around the labels. A list marker ("1.", "2)", "-") is
    removed only in front of a label, since a formula itself can start with
    "* "."""
    line = line.strip().replace("**", "")
    marker = LIST_MARKER.match(line)
    if marker and line[marker.end():].startswith(LABELS):
        line = line[marker.end():]
    return line


def _value_after_label(lines: list[str], index: int, label: str) -> tuple[str | None, int]:
    """The value belonging to the label on lines[index], and the index of the
    line it was taken from.

    The value is the text after the label on the same line or, if that is
    empty, the next line with content. Code fences are skipped, and inline
    backticks are removed. None if the next content is another label or the
    response ends first.
    """
    same = _clean(lines[index])[len(label):].strip().strip("`").strip()
    if same:
        return same, index
    for j in range(index + 1, len(lines)):
        line = _clean(lines[j])
        if line.startswith("```"):
            continue
        line = line.strip("`").strip()
        if not line:
            continue
        if line.startswith(LABELS):
            return None, index
        return line, j
    return None, index


def extract_expressions(response: str) -> list[str]:
    """Every formula after an EXPRESSION: label, on the same or the next line.
    Empty when the model did not answer in the expected format, which also
    covers CANNOT TRANSLATE."""
    lines = response.splitlines()
    out = []
    for i, raw in enumerate(lines):
        if not _clean(raw).startswith("EXPRESSION:"):
            continue
        formula, _ = _value_after_label(lines, i, "EXPRESSION:")
        if formula:
            out.append(formula)
    return out


def extract_cannot_translate(response: str) -> str | None:
    """The reason after a CANNOT TRANSLATE: label, or None if the response has
    none. Separates a deliberate refusal from a response in the wrong format."""
    lines = response.splitlines()
    for i, raw in enumerate(lines):
        if _clean(raw).startswith("CANNOT TRANSLATE:"):
            reason, _ = _value_after_label(lines, i, "CANNOT TRANSLATE:")
            return reason or ""
    return None


def missing_expression_error(response: str, default: str) -> str:
    """The error recorded when a response holds no formula: the model's refusal
    if it gave one, otherwise `default`."""
    reason = extract_cannot_translate(response)
    if reason is None:
        return default
    return f"CANNOT TRANSLATE: {reason}".rstrip()


def extract_hybrids(response: str) -> list[tuple[list[int], str, str]]:
    """Every (parents, rationale, formula) block from a crossover response.

    The rationale is the last non-empty line before PARENTS; it becomes the
    hybrid's idea, which is what the analysis model later reads.
    """
    lines = response.splitlines()
    out = []
    parents: list[int] = []
    rationale = ""
    last_line = ""
    skip_until = -1         # index of a formula line already taken by a label

    for i, raw in enumerate(lines):
        if i <= skip_until:
            continue
        line = _clean(raw)
        if line.startswith("PARENTS:"):
            parents = []
            for part in line[len("PARENTS:"):].split(","):
                part = part.strip()
                if part.lstrip("-").isdigit():
                    parents.append(int(part))
            rationale = last_line
        elif line.startswith("EXPRESSION:"):
            formula, used = _value_after_label(lines, i, "EXPRESSION:")
            skip_until = used
            if formula:
                out.append((parents, rationale, formula))
            parents, rationale = [], ""
        elif line:
            last_line = line
    return out


def parse_innovation(response: str) -> dict[int, float]:
    """The <id>,<rating> lines of an innovation response. Unparseable lines are
    skipped; the caller substitutes the middle of the scale for what is missing."""
    out = {}
    for raw in response.splitlines():
        line = raw.strip()
        if "," not in line:
            continue
        left, _, right = line.partition(",")
        left, right = left.strip(), right.strip()
        if not left.isdigit():
            continue
        try:
            out[int(left)] = float(right)
        except ValueError:
            continue
    return out


def parse_fidelity(response: str) -> tuple[str | None, str]:
    """The verdict ("OK" or "MISMATCH") and reason of a fidelity response.

    The verdict is None when the model did not answer in the expected format.
    The last VERDICT line counts, since the analysis steps come before it.
    """
    verdict, reason_lines, in_reason = None, [], False
    for raw in response.splitlines():
        line = raw.strip().replace("**", "")      # models like to bold the labels
        if line.startswith("VERDICT:"):
            value = line[len("VERDICT:"):].strip().upper()
            if value.startswith("OK"):
                verdict = "OK"
            elif value.startswith("MISMATCH"):
                verdict = "MISMATCH"
            else:
                verdict = None
            reason_lines, in_reason = [], False
        elif line.startswith("REASON:"):
            reason_lines = [line[len("REASON:"):].strip()]
            in_reason = True
        elif in_reason and line:
            reason_lines.append(line)
    return verdict, " ".join(reason_lines).strip()


def check_fidelity(api, prompt: str, idea: str, formula: str) -> tuple[str | None, str, str]:
    """Whether `formula` implements `idea`, judged by a separate model call.

    Returns (verdict, reason, raw response). An API failure gives verdict None,
    so an outage of the checker cannot end the run.
    """
    try:
        response = api.generate(prompt.format(idea=idea, formula=formula))
    except Exception as e:
        print(f"fidelity check failed: {e}")
        return None, "", str(e)
    verdict, reason = parse_fidelity(response)
    if verdict == "MISMATCH" and not reason:
        reason = "The formula does not implement the idea; no reason was given."
    return verdict, reason, response


def format_population(pool: list[Candidate], with_idea: bool = True) -> str:
    """The population as the crossover and innovation prompts see it."""
    blocks = []
    for c in sorted(pool, key=lambda c: -c.score):
        block = f"id {c.id} | score {c.score:.4f} | {c.formula}"
        if with_idea:
            block += f"\n  idea: {c.idea}"
        blocks.append(block)
    return "\n\n".join(blocks)


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


def generate_variants(idea: str, api, first_prompt: str, repeat_prompt: str,
                      fidelity_api, fidelity_prompt: str,
                      task: str, template: str, generation: int, first_id: int,
                      n_variants: int, max_tries: int = MAX_TRIES) -> list[Candidate]:
    """One generation's starting population: n_variants formulas for the same
    idea, each validated by the parser and then checked against the idea. A
    failed translation comes back as a single candidate with formula None, so
    the failure still reaches the log.

    Parser errors and fidelity mismatches share max_tries. A formula that still
    does not match the idea after the last try is dropped, since its idea text
    would mislead the crossover and analysis models.
    """
    response = api.generate(first_prompt.format(idea=idea, n_variants=n_variants))
    formulas = extract_expressions(response)

    if not formulas:
        return [Candidate(id=first_id, generation=generation, origin="variant",
                          parents=[], idea=idea, formula=None,
                          attempts=[{"formula": None,
                                     "error": missing_expression_error(
                                         response, "no EXPRESSION: in response"),
                                     "response": response}],
                          score=None, wrong_answers=None, errors=None)]

    # the same formula twice is one variant and one measurement less, not two
    seen, unique = set(), []
    for f in formulas:
        key = formula_key(f)
        if key not in seen:
            seen.add(key)
            unique.append(f)

    candidates = []
    for offset, formula in enumerate(unique):
        attempts = []
        current = formula

        for attempt_no in range(max_tries):
            error = validate_heuristic(current, task, template)
            attempt = {"formula": current, "error": error}
            attempts.append(attempt)

            if error is None:
                verdict, reason, check_response = check_fidelity(
                    fidelity_api, fidelity_prompt, idea, current)
                attempt["fidelity"] = verdict
                if verdict == "OK":
                    break
                if verdict is None:
                    # checker gave no usable answer; keep the formula unchecked
                    # rather than spend a try on the checker's failure
                    attempt["fidelity_response"] = check_response
                    break
                attempt["fidelity_reason"] = reason
                print(f"unfaithful formula: {current}\n{reason}")
                feedback = ("The formula parses, but it does not implement the "
                            "idea. " + reason)
            else:
                print(f"invalid formula: {current}\n{error}")
                feedback = error

            if attempt_no == max_tries - 1:
                current = None              # out of tries, last one still rejected
                break
            retry = api.generate(repeat_prompt.format(
                idea=idea, formula=current, errors_in_heuristic=feedback))
            fixed = extract_expressions(retry)
            if not fixed:
                attempts.append({"formula": None,
                                 "error": missing_expression_error(
                                     retry, "no EXPRESSION: in retry response"),
                                 "response": retry})
                current = None
                break
            current = fixed[0]

        candidates.append(Candidate(id=first_id + offset, generation=generation,
                                    origin="variant", parents=[], idea=idea,
                                    formula=current, attempts=attempts,
                                    score=None, wrong_answers=None, errors=None))
    return candidates


def generate_hybrids(pool: list[Candidate], api, prompt: str, task: str,
                     template: str, generation: int, first_id: int,
                     n_hybrids: int) -> list[Candidate]:
    """Recombinations of the pool, one model call for all of them.

    An invalid hybrid is kept with formula None rather than retried: a crossover
    that does not parse is cheap to replace next round and not worth a second
    call.
    """
    response = api.generate(prompt.format(n_hybrids=n_hybrids,
                                          population=format_population(pool)))
    hybrids = extract_hybrids(response)
    if not hybrids:
        print("  crossover produced no parseable hybrid")
        return []

    known = {c.id for c in pool}
    candidates = []
    for offset, (parents, rationale, formula) in enumerate(hybrids):
        parents = [p for p in parents if p in known]
        error = validate_heuristic(formula, task, template)
        if error is not None:
            print(f"  invalid hybrid: {formula}\n{error}")
        candidates.append(Candidate(
            id=first_id + offset, generation=generation, origin="crossover",
            parents=parents,
            idea=rationale or f"Crossover of candidates {parents}.",
            formula=None if error else formula,
            attempts=[{"formula": formula, "error": error}],
            score=None, wrong_answers=None, errors=None))
    return candidates


def measure(candidate: Candidate, queries, baseline, task, template, timeout,
            run_dir: Path, measured: dict[str, Candidate],
            log: ProgressLog) -> None:
    """Fill in score, wrong_answers and errors. Candidates without a formula stay
    unmeasured, and an unmeasured candidate can never be selected.

    `measured` maps formula_key to the first candidate measured with that
    formula in this run. A formula that is already in it is not measured again:
    its results are copied and duplicate_of names the original.

    The result of every query is appended to queries.jsonl right away, so the
    dashboard sees a measurement before the generation is written.
    """
    if candidate.formula is None or candidate.score is not None:
        return

    key = formula_key(candidate.formula)
    original = measured.get(key)
    if original is not None:
        candidate.score = original.score
        candidate.wrong_answers = original.wrong_answers
        candidate.errors = original.errors
        candidate.duplicate_of = original.id
        print(f"  [{candidate.id}] duplicate of {original.id}, not measured")
        log.event("candidate_duplicate", candidate_id=candidate.id,
                  duplicate_of=original.id)
        return

    log.event("measure_started", candidate_id=candidate.id, origin=candidate.origin)
    start = time.perf_counter()
    score, wrong_answers, errors, per_query = calculate_candidate_score(
        queries, baseline, candidate.formula, task, template, timeout)
    with (run_dir / "queries.jsonl").open("a") as fh:
        for record in per_query:
            fh.write(json.dumps({"candidate_id": candidate.id, **record}) + "\n")
    candidate.score = score
    candidate.wrong_answers = wrong_answers
    candidate.errors = errors
    measured[key] = candidate
    log.event("candidate_measured", candidate_id=candidate.id,
              seconds=round(time.perf_counter() - start, 3), score=score,
              wrong_answers=wrong_answers, errors=errors)
    print(f"  [{candidate.id}] {score:.4f} "
          f"({wrong_answers} wrong, {errors} failed)  {candidate.formula}")


def analyse(api, prompt_template: str, candidate: Candidate, best: Candidate) -> str:
    """What this candidate did differently from the best one so far.

    The text is the only thing the next idea sees of this generation, so it is
    written for a model, not for a reader.
    """
    prompt = prompt_template.format(
        best_idea=best.idea, best_formula=best.formula, best_score=best.score,
        idea=candidate.idea, formula=candidate.formula, score=candidate.score,
        wrong_answers=candidate.wrong_answers, errors=candidate.errors,
        attempts=len(candidate.attempts),
    )
    return api.generate(prompt)


def write_generation(run_dir: Path, generation: int, idea: str,
                     population: list[Candidate], analysis: str | None,
                     crossover_rounds: list[dict] | None = None) -> None:
    """One file per generation with everything that was tried, plus one summary
    line per candidate so a run can be scanned without opening every file.

    crossover_rounds records pool, innovation ratings, fitness and pruning per
    round, so the selection can be traced afterwards.
    """
    path = run_dir / f"generation_{generation:03d}.json"
    path.write_text(json.dumps({
        "generation": generation,
        "idea": idea,
        "analysis": analysis,
        "candidates": [asdict(c) for c in population],
        "crossover_rounds": crossover_rounds or [],
    }, indent=2))
    with (run_dir / "summary.jsonl").open("a") as fh:
        for c in population:
            fh.write(json.dumps(summary_line(c)) + "\n")


def main():
    global _active_log
    task = "DC-PR"
    timeout = 30
    rng = random.Random(SEED)

    language = read_file(str(PROMPTS / "language.txt"))
    idea_prompt = read_file(str(PROMPTS / "idea_prompt.txt"))
    analysis_prompt = read_file(str(PROMPTS / "analysis_prompt.txt"))
    formula_prompt = language + "\n\n" + read_file(str(PROMPTS / "formula_min.txt"))
    repeat_prompt = language + "\n\n" + read_file(str(PROMPTS / "formula_repeat.txt"))
    crossover_prompt = language + "\n\n" + read_file(str(PROMPTS / "crossover.txt"))
    innovation_prompt = language + "\n\n" + read_file(str(PROMPTS / "innovation.txt"))
    # the language sits in the middle of this prompt, so it is spliced in rather
    # than prepended; .format later sees it exactly as in the other prompts
    fidelity_prompt = read_file(str(PROMPTS / "fidelity.txt")).replace("{language}", language)
    if not all([language, idea_prompt, analysis_prompt, fidelity_prompt]):
        raise RuntimeError("Could not read one of the prompt templates")

    # one client per available model
    oss_api = GPTCallAPI(Config.OPENAI_API_BASE, Config.OPENAI_API_KEY, Config.OPENAI_MODEL)
    mistral_api = GPTCallAPI(Config.CLAUDE_API_BASE, Config.CLAUDE_API_KEY, Config.CLAUDE_MODEL)
    qwen_api = GPTCallAPI(Config.DEEPSEEK_API_BASE, Config.DEEPSEEK_API_KEY, Config.DEEPSEEK_MODEL)

    # which model does which task; change only these lines to swap models
    idea_api = qwen_api
    formula_api = qwen_api
    fidelity_api = qwen_api
    crossover_api = qwen_api
    innovation_api = qwen_api
    analysis_api = qwen_api

    run_dir = RESULTS / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(json.dumps({
        "started": EXECUTION_START_TIME,
        "benchmark": str(BENCHMARK),
        "task": task,
        "solver_cmd": SOLVER_CMD,
        "timeout_s": timeout,
        "generations": GENERATIONS,
        "n_variants": N_VARIANTS,
        "max_crossover_rounds": MAX_CROSSOVER_ROUNDS,
        "max_tries": MAX_TRIES,
        "seed": SEED,
        # read from the clients, so the log always shows what was called
        "idea_model": idea_api.config.model_name,
        "formula_model": formula_api.config.model_name,
        "fidelity_model": fidelity_api.config.model_name,
        "crossover_model": crossover_api.config.model_name,
        "innovation_model": innovation_api.config.model_name,
        "analysis_model": analysis_api.config.model_name,
    }, indent=2))

    # the prompts steer every candidate, so a run is only reproducible with the
    # versions that produced it
    for name in ("language", "idea_prompt", "formula_min", "formula_repeat",
                 "crossover", "innovation", "analysis_prompt", "fidelity"):
        (run_dir / f"{name}.txt").write_text(read_file(str(PROMPTS / f"{name}.txt")))

    log = ProgressLog(run_dir)
    _active_log = log
    log.event("run_started")

    # wrapped after config.json is written, which reads the plain clients
    idea_api = TimedAPI(idea_api, "idea", log)
    formula_api = TimedAPI(formula_api, "formula", log)
    fidelity_api = TimedAPI(fidelity_api, "fidelity", log)
    crossover_api = TimedAPI(crossover_api, "crossover", log)
    innovation_api = TimedAPI(innovation_api, "innovation", log)
    analysis_api = TimedAPI(analysis_api, "analysis", log)

    queries = load_queries(BENCHMARK)
    start = time.perf_counter()
    baseline = load_baseline(BENCHMARK, task, SOLVER_CMD, timeout)
    log.event("baseline_loaded", seconds=round(time.perf_counter() - start, 3),
              n_queries=len(queries), n_baseline=len(baseline))

    # the shared baseline.json is re-measured when the template changes, so each
    # run keeps the values it was scored against; n_queries shows how many
    # queries were skipped for lack of a baseline value
    (run_dir / "baseline.json").write_text(json.dumps({
        "task": task,
        "template": SOLVER_CMD,
        "n_queries": len(queries),
        "baseline_values": baseline,
    }, indent=2))

    best = Candidate(id=0, generation=0, origin="default", parents=[],
                     idea=DEFAULT_IDEA, formula=DEFAULT_HEURISTIC, attempts=[],
                     score=1.0, wrong_answers=0, errors=0)
    write_generation(run_dir, 0, DEFAULT_IDEA, [best], FIRST_ANALYSIS)

    tried = [best]
    # every formula measured in this run, so a duplicate is not measured twice;
    # the default heuristic is in it from the start
    measured_by_formula = {formula_key(best.formula): best}
    analysis = FIRST_ANALYSIS
    next_id = 1

    for generation in range(1, GENERATIONS + 1):
        print(f"\n=== generation {generation}/{GENERATIONS} ===")
        log.generation = generation
        log.event("generation_started")
        generation_start = time.perf_counter()

        idea = idea_api.generate(idea_prompt.format(analysis=analysis,
                                                    tried=format_tried(tried)))
        print(f"IDEA: {idea}")
        log.event("idea", idea=idea.strip())

        produced = generate_variants(idea, formula_api, formula_prompt,
                                     repeat_prompt, fidelity_api, fidelity_prompt,
                                     task, SOLVER_CMD,
                                     generation, next_id, N_VARIANTS)
        next_id += len(produced)

        valid = [c for c in produced if c.formula is not None]
        print(f"{len(valid)}/{len(produced)} variants valid, measuring")
        log.event("variants_translated", n_variants=len(produced), n_valid=len(valid))
        for candidate in valid:
            measure(candidate, queries, baseline, task, SOLVER_CMD, timeout, run_dir,
                    measured_by_formula, log)

        # everything this generation touched, for the record
        all_candidates = list(produced)

        # the elite joins the pool as a parent, but is not logged again - it
        # already has its own entry in the generation that produced it; it comes
        # first, so a variant that repeats its formula is the one left out
        pool = add_to_pool([best] if best.score is not None else [], produced)

        rounds = 0
        final_round_done = False
        crossover_rounds = []       # selection record for the generation file
        while len(pool) > 1 and rounds < MAX_CROSSOVER_ROUNDS:
            if len(pool) == 2 and final_round_done:
                break
            if len(pool) == 2:
                final_round_done = True

            n_hybrids = max(1, len(pool) // 2)
            print(f"crossover round {rounds + 1}: pool {len(pool)}, "
                  f"{n_hybrids} hybrid(s)")

            pool_before = [c.id for c in pool]
            hybrids = generate_hybrids(pool, crossover_api, crossover_prompt, task,
                                       SOLVER_CMD, generation, next_id, n_hybrids)
            next_id += len(hybrids)
            all_candidates.extend(hybrids)

            for candidate in hybrids:
                measure(candidate, queries, baseline, task, SOLVER_CMD, timeout, run_dir,
                        measured_by_formula, log)
            # a hybrid that repeats a pool member's formula adds nothing to select from
            pool = add_to_pool(pool, hybrids)

            innovation = parse_innovation(innovation_api.generate(
                innovation_prompt.format(
                    population=format_population(pool, with_idea=False))))
            fit = {c.id: fitness(c.score,
                                 innovation.get(c.id, DEFAULT_INNOVATION), rng)
                   for c in pool}

            before = len(pool)
            pool = prune(pool, fit)
            print(f"  pruned {before} -> {len(pool)}")
            rounds += 1

            # ids missing from innovation got DEFAULT_INNOVATION
            crossover_rounds.append({
                "round": rounds,
                "pool_before": pool_before,
                "hybrids": [c.id for c in hybrids],
                "innovation": innovation,
                "fitness": fit,
                "pool_after": [c.id for c in pool],
            })
            log.event("crossover_round", round=rounds, pool_before=pool_before,
                      hybrids=[c.id for c in hybrids],
                      pool_after=[c.id for c in pool])

        tried.extend(all_candidates)

        measured = [c for c in all_candidates if c.score is not None]
        if not measured:
            # nothing to compare and nothing to learn from; keep the previous
            # analysis so the next idea is not left without context
            print("no measured candidate this generation")
            write_generation(run_dir, generation, idea, all_candidates, None,
                             crossover_rounds)
            log.event("generation_finished",
                      seconds=round(time.perf_counter() - generation_start, 3),
                      n_candidates=len(all_candidates), best_score=None)
            continue

        generation_best = max(measured, key=lambda c: c.score)
        analysis = analyse(analysis_api, analysis_prompt, generation_best, best)
        print(f"ANALYSIS: {analysis}")

        write_generation(run_dir, generation, idea, all_candidates, analysis,
                         crossover_rounds)

        if generation_best.score > best.score:
            best = generation_best
            print(f"new best: {best.score:.4f}")

        log.event("generation_finished",
                  seconds=round(time.perf_counter() - generation_start, 3),
                  n_candidates=len(all_candidates),
                  best_score=generation_best.score, run_best_id=best.id)

    print(f"\nbest: generation {best.generation}, id {best.id}, score {best.score}")
    print(best.formula)
    print(f"records in {run_dir}")
    log.generation = GENERATIONS
    log.event("run_finished", best_id=best.id, best_score=best.score)


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        # a crash or Ctrl+C still reaches the log, so the dashboard can tell an
        # aborted run from one that is still going
        if _active_log is not None:
            _active_log.event("run_failed", error=repr(e))
        raise
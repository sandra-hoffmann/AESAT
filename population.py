#!/usr/bin/env python3
"""population.py.py

Short description of this script.

Usage:
    python population.py.py [options]

Created: 22.10.26
Author: Sandra Hoffmann
"""
import random
from dataclasses import dataclass, asdict

DEFAULT_INNOVATION = 10.0   # scale is 0-20; an unscored candidate gets the middle


@dataclass
class Candidate:
    id: int
    generation: int
    origin: str                 # "default" | "variant" | "crossover" | "elite"
    parents: list[int]          # empty unless crossover
    idea: str
    formula: str | None         # None if translation gave up
    attempts: list[dict]
    score: float | None         # None means not measured, not measured badly
    wrong_answers: int | None
    errors: int | None
    duplicate_of: int | None = None   # id of the earlier candidate with the same
                                      # formula; its results were copied, not measured


def formula_key(formula: str) -> str:
    """The formula with every number in one notation, so "1000", "1000.0" and
    "1e3" count as the same formula. Used to find duplicates before measuring."""
    tokens = []
    for tok in formula.split():
        try:
            tokens.append(repr(float(tok)))
        except ValueError:
            tokens.append(tok)
    return " ".join(tokens)


def add_to_pool(pool: list, new: list) -> list:
    """`pool` plus the measured candidates of `new` whose formula is not in it
    yet. A duplicate in the pool would be rated and pruned twice."""
    keys = {formula_key(c.formula) for c in pool}
    out = list(pool)
    for c in new:
        if c.score is None:
            continue
        key = formula_key(c.formula)
        if key not in keys:
            keys.add(key)
            out.append(c)
    return out


def summary_line(candidate: Candidate) -> dict:
    """One line for summary.jsonl"""
    return {
        "generation": candidate.generation,
        "id": candidate.id,
        "origin": candidate.origin,
        "parents": candidate.parents,
        "formula": candidate.formula,
        "score": candidate.score,
        "wrong_answers": candidate.wrong_answers,
        "errors": candidate.errors,
        "duplicate_of": candidate.duplicate_of,
    }


def format_tried(candidates: list[Candidate],
                 best_n: int = 10, recent_n: int = 10) -> str:
    """The {tried} block for the idea prompt. Capped so the prompt stops growing."""
    # a formula can recur across generations; keep its best score
    by_formula: dict[str, Candidate] = {}
    for c in candidates:
        if c.formula is None or c.score is None:
            continue
        kept = by_formula.get(c.formula)
        if kept is None or c.score > kept.score:
            by_formula[c.formula] = c

    if not by_formula:
        return "Nothing has been measured yet."

    unique = list(by_formula.values())
    best = sorted(unique, key=lambda c: -c.score)[:best_n]
    recent = sorted(unique, key=lambda c: (c.generation, c.id),
                    reverse=True)[:recent_n]

    chosen = {c.formula: c for c in best}
    chosen.update({c.formula: c for c in recent})

    lines = sorted(chosen.values(), key=lambda c: -c.score)
    return "\n".join(f"{c.formula}  ->  {c.score:.4f}" for c in lines)


def fitness(score: float, innovation: float, rng: random.Random) -> float:
    """Equation 1 of the AESAT paper. Deliberately not greedy: only 40% of the
    decision is performance, so the population does not collapse onto one peak."""
    return 0.4 * score + 0.4 * (innovation / 20.0) + 0.2 * rng.random()


def prune(population: list[Candidate],
          fitness_by_id: dict[int, float]) -> list[Candidate]:
    """Halve the population, keeping the best by pure score whatever q says.

    Unmeasured candidates drop out: not measured is not comparable. They are
    already in the log.
    """
    measured = [c for c in population if c.score is not None]
    if len(measured) <= 1:
        return measured

    best = max(measured, key=lambda c: c.score)
    rest = [c for c in measured if c.id != best.id]
    rest.sort(key=lambda c: fitness_by_id.get(c.id, 0.0), reverse=True)

    keep_count = len(rest) // 2
    return [best] + rest[:keep_count]
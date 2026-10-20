# from config import Config
# from llm_api import GPTCallAPI
# from file_utils import read_file
from pathlib import Path
from benchmark import load_queries, measure_query, measure_baseline, load_baseline, calculate_candidate_score

qs = load_queries(Path("/work/data/G2IO_heureka_dcpr/benchmark_PR_DC_20261012_115628"))

t = "heureka -p {task} -f {file} -H {heuristic} -a {arg} -d 1"
baseline = load_baseline(Path("/work/data/G2IO_heureka_dcpr/benchmark_PR_DC_20261012_115628"), "DC-PR", t, 600)
# result = calculate_candidate_score(qs, baseline, None, "DC-PR", t, 600)
# print(result)
# print(calculate_candidate_score(qs, baseline, "+ scc indeg", "DC-PR", t, 600))
print(calculate_candidate_score(qs, baseline, "+ blubb", "DC-PR", t, 600))

# baseline = measure_baseline(qs, "DC-PR", t, timeout = 600)
# print(baseline)

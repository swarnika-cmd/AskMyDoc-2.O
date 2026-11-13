# Evaluation harness targets. Run inside an environment with requirements-eval.txt installed.
PYTHON ?= python3
PIPELINE ?= P2
IDS ?= pilot
MAX_CALLS ?=
MAX_CALLS_ARG := $(if $(MAX_CALLS),--max-calls $(MAX_CALLS),)

.PHONY: validate test eval-retrieval eval-checks eval-ci eval-ci-baseline eval-llm eval

validate:
	$(PYTHON) eval/validate_golden.py
	$(PYTHON) -m unittest eval.test_validate_golden > /dev/null

test:
	$(PYTHON) -m unittest discover -s eval/tests -t . -v
	$(PYTHON) -m unittest eval.test_validate_golden > /dev/null
	$(PYTHON) -m unittest tests.test_judge_chunk_ids tests.test_llm_reasoning_effort

# All four pipelines on the 130 answerable golden items.
eval-retrieval:
	$(PYTHON) eval/run_retrieval.py --subset full

# HNSW-vs-exact and refusal-gate measurements quoted in the README (eval/results/checks.md).
eval-checks:
	$(PYTHON) eval/run_checks.py

# Fixed 30-item subset (26 answerable) for all four pipelines, then compare recall@5 with the committed baseline.
eval-ci:
	$(PYTHON) eval/run_retrieval.py --subset ci
	$(PYTHON) eval/check_regression.py

# Re-record the baseline from the latest CI-subset run. Only after a reviewed, intended change.
eval-ci-baseline:
	$(PYTHON) eval/run_retrieval.py --subset ci
	cp eval/results/retrieval_ci.json eval/results/baseline_ci.json

# Resumable LLM-judged run. Needs GROQ_API_KEY for uncached calls; fully cached ids run without it.
# See eval/results/llm_progress.json for completed ids and the resume command.
#   make eval-llm PIPELINE=P2 IDS=pilot            (20-item pilot)
#   make eval-llm PIPELINE=P2 IDS=all MAX_CALLS=40  (stop after 40 uncached calls per model)
eval-llm:
	@test -n "$$GROQ_API_KEY" || echo "GROQ_API_KEY is not set: only cached responses can be used"
	$(PYTHON) eval/run_llm.py --pipeline $(PIPELINE) --ids $(IDS) $(MAX_CALLS_ARG)
	$(PYTHON) eval/aggregate_llm.py --pipeline $(PIPELINE) --ids $(IDS)

# validate + retrieval + LLM run when the key exists, else skip the LLM part with a message.
eval: validate eval-retrieval
	@if [ -n "$$GROQ_API_KEY" ]; then $(MAKE) eval-llm; else echo "GROQ_API_KEY not set: skipping the LLM-judged run"; fi

PYTHON ?= python3
VENV   ?= .venv
BIN    := $(VENV)/bin
TORCH_VERSION ?= 2.15.0
CORE := src/docai/domain/*,src/docai/guardrails/*,src/docai/synth/*,src/docai/extract/*,src/docai/eval/*,src/docai/pipeline/*,src/docai/review/*,src/docai/storage/*
NO_EMIT := $(foreach p,torch triton cuda-bindings cuda-pathfinder cuda-toolkit nvidia-cublas nvidia-cuda-cccl nvidia-cuda-cupti nvidia-cuda-nvrtc nvidia-cuda-runtime nvidia-cudnn-cu13 nvidia-cufft nvidia-cufile nvidia-cusolver nvidia-cusparse nvidia-cusparselt-cu13 nvidia-nccl-cu13 nvidia-nvjitlink nvidia-nvshmem-cu13 nvidia-nvtx,--no-emit-package $(p))

.PHONY: install lock test coverage lint format typecheck security check demo serve mcp docker-build docker-up clean

install:  ## venv + hash-checked dev lock + CPU torch + the package (editable)
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install --require-hashes -r requirements-dev.lock
	$(BIN)/pip install --no-deps --index-url https://download.pytorch.org/whl/cpu "torch==$(TORCH_VERSION)" \
	  || $(BIN)/pip install "torch==$(TORCH_VERSION)"
	$(BIN)/pip install --no-deps --no-build-isolation -e .

lock:  ## regenerate requirements.lock and requirements-dev.lock (needs uv)
	uv pip compile pyproject.toml requirements/torch.in --extra api --extra mcp --universal \
	  --python-version 3.12 --generate-hashes $(NO_EMIT) --custom-compile-command "make lock" -o requirements.lock
	uv pip compile pyproject.toml requirements/torch.in --extra api --extra mcp --extra dev --universal \
	  --python-version 3.12 --generate-hashes $(NO_EMIT) --custom-compile-command "make lock" -o requirements-dev.lock

test:  ## run the test suite
	$(BIN)/pytest

coverage:  ## tests with coverage; fails under 85% on the core packages
	$(BIN)/pytest --cov=docai --cov-report=term
	$(BIN)/coverage report --include="$(CORE)" --fail-under=85

lint:  ## ruff lint + format check
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

format:
	$(BIN)/ruff format .
	$(BIN)/ruff check --fix .

typecheck:  ## mypy --strict on the package
	$(BIN)/mypy

security:  ## bandit (medium+) and pip-audit on the locked dependencies
	$(BIN)/bandit -r src -ll
	$(BIN)/pip-audit -r requirements.lock --require-hashes --disable-pip
	$(BIN)/pip-audit -r requirements-dev.lock --require-hashes --disable-pip

check: lint typecheck coverage security  ## everything CI runs (except the Docker build)

demo:  ## offline end-to-end run: synth -> train -> eval -> reports/ + README table (+ run recorded for /dashboard)
	$(BIN)/python -m docai demo --update-readme

serve:  ## API + web UI on http://127.0.0.1:8000
	$(BIN)/python -m docai serve --host 127.0.0.1 --port 8000

mcp:  ## MCP server over stdio
	$(BIN)/python -m docai mcp

docker-build:
	docker build -t ai-guardrails .

docker-up:
	docker compose up --build

clean:
	rm -rf data models var .pytest_cache .ruff_cache .mypy_cache .hypothesis .coverage coverage.xml htmlcov

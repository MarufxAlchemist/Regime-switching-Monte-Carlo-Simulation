# ─────────────────────────────────────────────────────────────────────────────
# Regime-Switching Monte Carlo — Makefile
# Usage: make <target>
# ─────────────────────────────────────────────────────────────────────────────

IMAGE   := regime-mc
COMPOSE := docker compose

.PHONY: help build run test shell clean \
        network contagion regime sentiment cholesky \
        venv venv-install

# ── Default: show help ────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  Regime-switching Monte Carlo — available commands"
	@echo "  ─────────────────────────────────────────────────"
	@echo "  Docker:"
	@echo "    make build       — build the Docker image"
	@echo "    make run         — run the full pipeline (docker)"
	@echo "    make test        — run pytest inside Docker"
	@echo "    make shell       — open interactive bash in container"
	@echo "    make network     — run network_engine.py"
	@echo "    make contagion   — run contagion.py"
	@echo "    make regime      — run hmm_engine.py"
	@echo "    make sentiment   — run sentiment_engine.py"
	@echo "    make cholesky    — run verify_cholesky.py"
	@echo "    make clean       — remove containers and image"
	@echo ""
	@echo "  Local (venv):"
	@echo "    make venv        — create ./venv virtual environment"
	@echo "    make venv-install— install deps into ./venv"
	@echo ""

# ── Docker targets ────────────────────────────────────────────────────────────
build:
	docker build --target runtime -t $(IMAGE):latest .

run:
	$(COMPOSE) --profile pipeline up --build pipeline

test:
	$(COMPOSE) --profile test up --build --abort-on-container-exit test

shell:
	$(COMPOSE) --profile shell run --rm shell

network:
	$(COMPOSE) --profile network up --build network

contagion:
	$(COMPOSE) --profile contagion up --build contagion

regime:
	$(COMPOSE) --profile regime up --build regime

sentiment:
	$(COMPOSE) --profile sentiment up --build sentiment

cholesky:
	$(COMPOSE) --profile verify_cholesky up --build verify_cholesky

clean:
	$(COMPOSE) down --rmi local --volumes --remove-orphans
	docker rmi -f $(IMAGE):latest 2>/dev/null || true

# ── Local venv targets ────────────────────────────────────────────────────────
venv:
	python -m venv venv
	@echo "Virtual environment created at ./venv"
	@echo "Activate with:  .\\venv\\Scripts\\activate  (Windows)"

venv-install: venv
	venv/Scripts/pip install --upgrade pip
	venv/Scripts/pip install -r requirements.txt
	@echo "All dependencies installed into ./venv"

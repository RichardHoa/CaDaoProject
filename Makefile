.PHONY: up down prod dev

# Automatically use the virtual environment's python so you don't even need to activate it
PYTHON := venv/bin/python
DEV_PYTHON := .venv/bin/python
WAITRESS := venv/bin/waitress-serve

prod:
	@echo "Starting server in production mode with waitress via nohup..."
	PYTHONUNBUFFERED=1 nohup $(WAITRESS) --port=4000 step3_server:app > log.txt 2>&1 &
	@echo "Server running in background with Waitress. Logs are being written to log.txt"

dev:
	@echo "Starting server in development mode..."
	$(PYTHON) step3_server.py

dev-local:
	@echo "Starting server in development mode..."
	$(DEV_PYTHON) step3_server.py
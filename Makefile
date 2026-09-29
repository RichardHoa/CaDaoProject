.PHONY: up down prod dev dev-local stop restart status logs

# Absolute directory of this Makefile and project root
PROJECT_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# Executables with absolute paths
PYTHON := $(PROJECT_DIR)/venv/bin/python
DEV_PYTHON := $(PROJECT_DIR)/.venv/bin/python
WAITRESS := $(PROJECT_DIR)/venv/bin/waitress-serve

PORT := 4000
LOG_FILE := $(PROJECT_DIR)/log.txt

prod:
	@echo "Starting server in production mode on port $(PORT)..."
	@cd $(PROJECT_DIR) && PYTHONUNBUFFERED=1 nohup $(WAITRESS) --port=$(PORT) step3_server:app > $(LOG_FILE) 2>&1 &
	@echo "Server running in background from $(PROJECT_DIR)"
	@echo "Logs: $(LOG_FILE)"

dev:
	@echo "Starting server in development mode from $(PROJECT_DIR)..."
	@cd $(PROJECT_DIR) && $(PYTHON) $(PROJECT_DIR)/step3_server.py

dev-local:
	@echo "Starting server in local development mode from $(PROJECT_DIR)..."
	@cd $(PROJECT_DIR) && $(DEV_PYTHON) $(PROJECT_DIR)/step3_server.py

stop: down

down:
	@echo "Stopping running server instances..."
	@-pkill -f "waitress-serve.*step3_server:app" || true
	@-pkill -f "step3_server.py" || true
	@echo "Server stopped."

restart: down prod

status:
	@echo "Checking server status..."
	@pgrep -a -f "step3_server" || echo "No server running."

logs:
	@tail -f $(LOG_FILE)
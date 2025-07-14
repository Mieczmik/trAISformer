#!/bin/bash

PYTHON_SCRIPT="/home/machineblue/repositories/TrAISformer/trAISformer.py"
LOG_FILE="training_logs.log"

# source .venv/bin/activate
nohup /usr/bin/python3 -u "$PYTHON_SCRIPT" > "$LOG_FILE" 2>&1 &
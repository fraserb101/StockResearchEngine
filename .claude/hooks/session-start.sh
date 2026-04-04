#!/bin/bash
set -euo pipefail

# Only run on Claude Code web (remote sessions)
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

# Install Python dependencies
pip install -r "$CLAUDE_PROJECT_DIR/requirements.txt" --quiet

# Load API key from .env file into session environment if it exists
ENV_FILE="$CLAUDE_PROJECT_DIR/.env"
if [ -f "$ENV_FILE" ]; then
  while IFS= read -r line || [ -n "$line" ]; do
    # Skip comments and empty lines
    [[ -z "$line" || "$line" == \#* ]] && continue
    # Strip surrounding quotes from value
    key="${line%%=*}"
    value="${line#*=}"
    value="${value%\"}"
    value="${value#\"}"
    value="${value%\'}"
    value="${value#\'}"
    echo "export ${key}=\"${value}\"" >> "$CLAUDE_ENV_FILE"
  done < "$ENV_FILE"
fi

# Ensure data directories exist
mkdir -p "$CLAUDE_PROJECT_DIR/data/cache" "$CLAUDE_PROJECT_DIR/output"

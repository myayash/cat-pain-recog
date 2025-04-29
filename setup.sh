#!/bin/bash

# exit if any command exit w non zero status 
set -e

echo "SETTING UP PROJECT ENVIRONMENT..."

VENV_DIR=".venv"

# check if venv name already exists
if [ -d "$VENV_DIR" ]; then 
  echo "Virtual environment '$VENV_DIR' already exists."
else
  echo "Creating virtual environment '$VENV_DIR'..."
  python3 -m venv "$VENV_DIR"
  echo "VIRTUAL ENVIRONMENT CREATED"
fi

# activate
echo "ACTIVATING VIRTUAL ENVIRONMENT..."
source "$VENV_DIR/bin/activate"
echo "VIRTUAL ENVIRONMENT ACTIVATED"

# install dependencies from requirements.txt
if [ -f "requirements.txt" ]; then
  echo "Installing dependencies from requirements.txt..."
  python3 -m pip install -r requirements.txt
  echo "DEPENDENCIES INSTALLED"
else
  echo "requirements.txt NOT FOUND"
  exit 1
fi

echo "SETUP COMPLETE"

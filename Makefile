# Detect OS
OS := $(shell uname 2>/dev/null || echo Windows)

# Default target: Run setup and script
all: setup run

# Install dependencies and create virtual environment
setup:
ifeq ($(OS), Windows)
	python -m venv .venv
	.venv\Scripts\pip.exe install -r requirements.txt
else
	python -m venv .venv
	.venv/bin/pip install -r requirements.txt
endif

# Run the script using virtual environment
run:
ifeq ($(OS), Windows)
	.venv\Scripts\python.exe main.py
else
	.venv/bin/python main.py
endif

# Clean up cache files and remove venv
clean:
ifeq ($(OS), Windows)
	rmdir .venv /s /q
	rmdir __pycache__ *.pyc *.pyo /s /q
else
	rm -rf .venv
	rm -rf __pycache__ *.pyc *.pyo
endif

# Reinstall everything and run
rerun: clean all

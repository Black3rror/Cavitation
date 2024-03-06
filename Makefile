# This defines all targets as phony targets, i.e. targets that are always out of date
# This is done to ensure that the commands are always executed, even if a file with the same name exists
# See https://www.gnu.org/software/make/manual/html_node/Phony-Targets.html
# Remove this if you want to use this Makefile for real targets
.PHONY: *

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = cavitation
PROJECT_ROOT = $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python

TF_LOG_LEVEL = 2

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Fully set up the project
setup_project:
	@echo "Creating and setting up the environment..."
	@conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) --no-default-packages -y && \
		conda activate $(PROJECT_NAME) && \
		conda env config vars set PROJECT_NAME="$(PROJECT_NAME)" -n $(PROJECT_NAME) && \
		conda env config vars set PROJECT_ROOT="$(PROJECT_ROOT)" -n $(PROJECT_NAME) && \
		conda env config vars set TF_CPP_MIN_LOG_LEVEL="$(TF_LOG_LEVEL)" -n $(PROJECT_NAME) && \
		python -m pip install -U pip setuptools wheel && \
		python -m pip install -r requirements.txt && \
		python -m pip install -e . && \
		python -m pip install .["dev"] && \
		python -m pip install .["test"]
	@echo "Recommended: Run 'make setup_git' to set up git."
	@echo "Setup completed. Please run 'conda activate $(PROJECT_NAME)' to activate the environment."

setup_git:
	@echo "Setting up git..."
	@git init && \
		git add . && \
		git commit -m "Init cookiecutter project" && \
		pre-commit install
	@echo "Git setup completed."

## Set up python interpreter environment
create_environment:
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) --no-default-packages -y

## Install Python Dependencies
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	$(PYTHON_INTERPRETER) -m pip install -e .

## Install Developer Python Dependencies
dev_requirements: requirements
	$(PYTHON_INTERPRETER) -m pip install .["dev"]

## Install Test Python Dependencies
test_requirements: requirements
	$(PYTHON_INTERPRETER) -m pip install .["test"]

## Delete all compiled Python files
clean:
ifeq ($(OS),Windows_NT)
	@for /r %%i in (*.pyc) do if exist "%%i" del /q "%%i"
	@for /r %%i in (*.pyo) do if exist "%%i" del /q "%%i"
	@for /d /r %%d in (__pycache__) do if exist "%%d" rmdir /s /q "%%d"
else
	@find . -type f -name "*.py[co]" -delete
	@find . -type d -name "__pycache__" -delete
endif

## Remove python interpreter environment
remove_environment:
	conda remove --name $(PROJECT_NAME) --all -y

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

# put project specific rules here

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
help:
ifeq ($(OS),Windows_NT)
	@echo "The help command is not supported on Windows. Please use a Unix-like environment."
else
	@echo "$$(tput bold)Available commands:$$(tput sgr0)"
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
endif

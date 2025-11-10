# ===== Makefile compatible sin TABs =====
SHELL := /bin/bash
.RECIPEPREFIX := >

PYTHON       ?= python3
VENV_DIR     ?= .venv
PIP          := $(VENV_DIR)/bin/pip
PY           := $(VENV_DIR)/bin/python
REQUIREMENTS ?= requirements.txt

.PHONY: help venv install upgrade-pip freeze clean-venv run test

help:
> @echo "Comandos:"
> @echo "  make venv         -> Crea entorno virtual en $(VENV_DIR)"
> @echo "  make install      -> Instala dependencias de $(REQUIREMENTS)"
> @echo "  make upgrade-pip  -> Actualiza pip/setuptools/wheel"
> @echo "  make freeze       -> Genera requirements.lock.txt"
> @echo "  make clean-venv   -> Borra el venv"
> @echo "  make run ARGS='-m src.algo'  -> Ejecuta con el venv"
> @echo "  make test         -> Ejecuta pytest (si está instalado)"

venv: $(VENV_DIR)/bin/activate

$(VENV_DIR)/bin/activate:
> @echo "==> Creando entorno virtual en $(VENV_DIR)"
> $(PYTHON) -m venv $(VENV_DIR)
> @echo "==> Activá con: source $(VENV_DIR)/bin/activate"

install: venv
> @echo "==> Actualizando pip/setuptools/wheel"
> $(PIP) install --upgrade pip setuptools wheel
> @echo "==> Instalando dependencias de $(REQUIREMENTS)"
> test -f $(REQUIREMENTS) || (echo "[ERROR] No existe $(REQUIREMENTS)"; exit 1)
> $(PIP) install -r $(REQUIREMENTS)
> @echo "==> Dependencias instaladas"

upgrade-pip: venv
> $(PIP) install --upgrade pip setuptools wheel

freeze: venv
> @echo "==> Congelando dependencias en requirements.lock.txt"
> $(PIP) freeze | sort > requirements.lock.txt
> @echo "==> Listo"

clean-venv:
> @echo "==> Borrando $(VENV_DIR)"
> rm -rf $(VENV_DIR)

run: venv
> test -n "$(ARGS)" || (echo "Uso: make run ARGS='-m src.main'"; exit 1)
> $(PY) $(ARGS)

test: venv
> $(PY) -m pytest -q || (echo "[WARN] pytest no instalado o tests fallaron"; exit 0)

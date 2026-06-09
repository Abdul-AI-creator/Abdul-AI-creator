# AGENTS.md

## Project overview

This repository is an ML/data-science **notebook portfolio** (38 Jupyter notebooks). There is no web app, API server, or Docker stack. Development means running notebooks locally with Python + Jupyter.

## Standard commands

| Task | Command |
|------|---------|
| Install deps | `pip3 install --break-system-packages -r requirements.txt` |
| Start Jupyter | `jupyter notebook --no-browser --ip=0.0.0.0 --port=8888` |
| Execute a notebook | `jupyter nbconvert --to notebook --execute <notebook>.ipynb --output /tmp/out.ipynb` |
| Validate notebook JSON | `python3 -c "import nbformat,glob; [nbformat.read(open(p),4) for p in glob.glob('*.ipynb')]"` |

Ensure `~/.local/bin` is on `PATH` so `jupyter` is found after pip installs.

## Cursor Cloud specific instructions

- **No long-running services are required by default.** Only start Jupyter when you need the web UI or an interactive kernel.
- **Jupyter installs to `~/.local/bin`.** Add `export PATH="$HOME/.local/bin:$PATH"` in your shell before running `jupyter` commands.
- **Ubuntu 24.04 uses PEP 668.** Use `pip3 install --break-system-packages -r requirements.txt` (or the equivalent single-package installs).
- **There is no lint config or automated test suite.** "Health checks" are: validate `.ipynb` files with `nbformat`, then execute a self-contained notebook (e.g. `SKLEARN_Project_(_L1_based_models_for_Sparse_Signals).ipynb`).
- **Many notebooks need external data.** ~12 notebooks use Kaggle via `opendatasets` (`KAGGLE_USERNAME` / `KAGGLE_KEY` or `~/.kaggle/kaggle.json`). Others expect CSVs that were uploaded in Colab under `/content/` and are not in the repo.
- **Self-contained notebooks** (no Kaggle/manual data): `SKLEARN_Project_(_L1_based_models_for_Sparse_Signals).ipynb`, `Density_Estimation_for_a_Gaussian_mixture.ipynb`, most PyTorch basics notebooks (MNIST/CIFAR auto-download).
- **GPU is optional.** Three DL notebooks benefit from CUDA; CPU fallback works for smoke tests.
- **Large installs:** `torch` and `tensorflow` dominate install time and disk; allow several minutes on first `pip install`.

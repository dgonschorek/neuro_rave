# Removing Inter-Experimental Variability from Functional Data in Systems Neuroscience

Official implementation of:

> D. Gonschorek, L. Höfling, K. P. Szatko, K. Franke, T. Schubert, B. Dunn, P. Berens, D. Klindt, and T. Euler.  
> **Removing Inter-Experimental Variability from Functional Data in Systems Neuroscience**.  
> *Advances in Neural Information Processing Systems (NeurIPS)*, 2021.

Paper: https://proceedings.neurips.cc/paper_files/paper/2021/hash/1e5eeb40a3fce716b244599862fd2200-Abstract.html

---

## Overview

Integrating data across experiments is a common challenge in systems neuroscience. This repository contains the official implementation of the methods introduced in the paper, including data preprocessing, model training, evaluation, and example analyses.

---

## Installation

Clone the repository:

```bash
git clone git@github.com:<username>/<repository>.git
cd <repository>
```

Install dependencies using `uv`:

```bash
uv sync
```

---

## Usage

Run a Python script:

```bash
uv run python scripts/example.py
```

Launch Jupyter Lab:

```bash
uv run jupyter lab
```

---

## Repository Structure

```text
.
├── src/            # Package source code
├── notebooks/      # Example notebooks
├── tests/          # Unit tests
├── docs/           # Documentation
├── pyproject.toml
└── README.md
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{gonschorek2021removing,
  title={Removing Inter-Experimental Variability from Functional Data in Systems Neuroscience},
  author={Gonschorek, Dominic and Höfling, Larissa and Szatko, Klaudia P. and Franke, Katrin and Schubert, Timm and Dunn, Benjamin and Berens, Philipp and Klindt, David and Euler, Thomas},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}
```

---

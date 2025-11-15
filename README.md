## InJecteD — Interpretability for DDPMs on 2D Point Clouds
<img width="202" height="77" alt="image" src="https://github.com/user-attachments/assets/e3d6eb28-e949-4988-9527-c3f072e330f8" />

This repository contains code and datasets used in the paper:

InJecteD: Analyzing Trajectories and Drift Dynamics in Denoising Diffusion Probabilistic Models for 2D Point Cloud Generation

Authors: Sanyam Jain, Khuram Naveed, Illia Oleksiienko, Alexandros Iosifidis, Ruben Pauwels

arXiv: https://arxiv.org/abs/2509.12239
PDF: https://arxiv.org/pdf/2509.12239

Brief summary
-------------
InJecteD is a framework for interpreting Denoising Diffusion Probabilistic Models (DDPMs) by analyzing sample trajectories and drift fields produced during the denoising process for 2D point cloud generation. The codebase implements lightweight DDPM-style models (small MLPs), visualization tools for trajectories and drift fields, statistical metrics (Wasserstein distance, velocity/displacement analysis, clustering), and utilities for morphing datasets (adapted from the "Same Stats" work included in the repo).

Repository layout (important files)
----------------------------------
- `train1.py` — lightweight DDPM-style experiments and visualizations (trex_viz output). Uses a small DenoisingMLP and provides examples of sampling, noising schedules, and drift grid visualizations.
- `train2.py` — extended experiments: noise/trajectory analyses, drift field visualizations, clustering of trajectories, and distribution comparisons (saves outputs under `bullseye/` by default).
- `same_stats.py` — adapted implementation for the "Same Stats, Different Graphs" synthetic dataset morphing (utility code originally from Matejka & Fitzmaurice).
- `DatasaurusDozen.tsv` — collection of 2D example datasets used in experiments (present in repo root).
- `seed_datasets/` — seed dataset CSVs used by `same_stats.py`.
- `generated_datasets/` — example outputs and wide/wide TSVs used for plotting in the paper.

Quick install
-------------
It is recommended to create a Python virtual environment first (venv/conda). Then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Notes on dependencies
---------------------
The code is written for Python 3.8+ and depends on common scientific and ML packages: PyTorch, NumPy, pandas, matplotlib, seaborn, scikit-learn, SciPy, tqdm, docopt, and pytweening. See `requirements.txt` for a minimal list.

Basic usage
-----------
These scripts are intended to be run locally (or in Colab). The datasets (`DatasaurusDozen.tsv` and files in `seed_datasets/`) should be in the working directory when running.

1) Quick visualization / toy training (generate `trex_viz/`):

```bash
# run the example training/visualization (may take significant time if not reduced)
python train1.py
```

2) Extended analyses, trajectory and drift visualizations (generates `bullseye/` outputs by default):

```bash
python train2.py
```

3) `same_stats.py` usage (example morphing script):

```bash
python same_stats.py dino circle
```

Practical tips
--------------
- The training loops in `train1.py` and `train2.py` use many epochs by default (the code shows `n_epochs` like 7000 in examples). For quick experimentation, reduce `n_epochs` or T (timesteps) in the function calls.
- Both `train1.py` and `train2.py` expect `DatasaurusDozen.tsv` in the working directory; `same_stats.py` expects `seed_datasets/*`.
- Visual outputs are written as SVG/PDF/PNG files in `trex_viz/`, `bullseye/`, or the current directory depending on the script.

What this README doesn't do
---------------------------
This README is a concise project entry point. For full experimental reproducibility (exact random seeds, environment, and longer runs used in the paper), please consult the paper (arXiv link above) and the scripts — they are written as runnable notebooks/scripts but assume local compute/visualization.

Citation
--------
If you use this code or the ideas in your research, please cite:

Sanyam Jain, Khuram Naveed, Illia Oleksiienko, Alexandros Iosifidis, Ruben Pauwels. "InJecteD: Analyzing Trajectories and Drift Dynamics in Denoising Diffusion Probabilistic Models for 2D Point Cloud Generation." arXiv:2509.12239 (2025). https://arxiv.org/abs/2509.12239

BibTeX
------
```bibtex
@article{jain2025injecteD,
  title={InJecteD: Analyzing Trajectories and Drift Dynamics in Denoising Diffusion Probabilistic Models for 2D Point Cloud Generation},
  author={Jain, Sanyam and Naveed, K and Oleksiienko, I and Iosifidis, A and Pauwels, R},
  journal={arXiv preprint arXiv:2509.12239},
  year={2025}
}
```

License
-------
No explicit license is included in this repository. If you intend to redistribute or reuse this code publicly, consider adding a license (MIT/Apache-2.0/CC BY) and updating the files accordingly.

Contact
-------

sanyam.jain@hiof.no

Acknowledgements / Origins
--------------------------
Parts of this repository reuse/adapt code and ideas from small DDPM tutorials (see header comments in `train1.py` referencing a tiny-diffusion Colab) and from the "Same Stats" dataset morphing project (see `same_stats.py` docstring). The experiments and writeup are part of the InJecteD paper linked above.

Next steps / Reproducibility tips
--------------------------------
- Add a `LICENSE` file for clarity.  
- Optional: add a `run_quick.sh` that runs a short experiment with lowered epoch/timesteps for CI/quick demos.  
- Consider adding a small notebook demonstrating end-to-end (train → sample → visualize) with reduced compute for reproducibility.

----
Thank you for exploring InJecteD. If you'd like, I can also: (1) add a small demo notebook that runs a trimmed experiment end-to-end, (2) add a `LICENSE`, or (3) create a short `run_quick.sh` with reduced epochs for CI/demo.

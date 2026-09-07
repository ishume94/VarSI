# VarSI: Reducing quantum measurements in qubit-based overlapping grouping methods for quantum energy estimation through better initializations.

[![arXiv](https://img.shields.io/badge/arXiv-2607.02794-b31b1b.svg)](https://arxiv.org/abs/2607.02794)

Scripts for reproducing the paper's main table, frozen core and all-electron molecular results, comparing SI, VarSI-G/O/R/OR, and their ICS initializations.

## Citation

```bibtex
@misc{huidobromeezs2026reducingquantummeasurementsqubitbased,
  title={Reducing quantum measurements in qubit-based overlapping grouping methods for quantum energy estimation through better initializations},
  author={Isaac L. Huidobro-Meezs and Rodrigo A. Vargas-Hernández},
  year={2026},
  eprint={2607.02794},
  archivePrefix={arXiv},
  primaryClass={quant-ph},
  url={https://arxiv.org/abs/2607.02794},
}
```

## Installation

First install [GFlow-VQE / GFlowNets-MOpt](https://github.com/ChemAI-Lab/GFlowNets-MOpt) and its dependencies in the same Python environment you will use for VarSI, following that library's installation instructions.

```bash
git clone https://github.com/ChemAI-Lab/GFlowNets-MOpt.git
cd GFlowNets-MOpt
python -m pip install -e .
```

Run the commands below from the repository root, where `VarSI.py` and `hamiltonians_varsi.py` are located; keep both batch runners under `scripts/`. If copying the runners elsewhere, place both runners, `VarSI.py`, and `hamiltonians_varsi.py` in the same folder. `VarSI_loaded.py` also needs `VarSI.py` alongside it.

## Examples

[VarSI.py](VarSI.py) and [fast_VarSI.py](fast_VarSI.py) run Hamiltonians provided by GFlow-VQE:

```bash
python VarSI.py H4 --wfn CISD --cov-workers 16
python fast_VarSI.py H4 --wfn CISD --cov-workers 16
```

The fast version supports `H2`, `LiH`, `MgO`, `SiO`, `N2`, `H4`, `H6`, `BeH2`, `H2O`, `H2Os`, and `NH3` with Jordan–Wigner mapping. Pass any of these names in place of `H4`.

These GFlow-VQE Hamiltonians use 1 Å bond lengths (nearest-neighbor spacing for hydrogen chains), except `MgO` (1.75 Å), `SiO` (1.5 Å), and `H2Os` (O–H bonds of 1.5 Å). The paper runners define their own geometries; in particular, the paper's stretched H2O uses 2.2 Å.

## Reproducing the main table

Use [scripts/run_varsi_main.py](scripts/run_varsi_main.py) for the paper geometries, with CISD covariances, FCI reporting, STO-3G, and fully commuting groups:

```bash
python -u scripts/run_varsi_main.py --wfn CISD --report-wfn FCI --condition fc \
    --max-sweeps 100 --cov-workers 48 \
    --output-csv main_varsi_results/varsi_main_cisd_fc_sweeps100.csv \
    > varsi_main_cisd_fc_sweeps100.out 2>&1
```

Both JW and BK mappings run by default. For the 500-sweep results, change `--max-sweeps` to `500` and replace `sweeps100` with `sweeps500` in both the CSV and `.out` filenames.

**The main-table NH3 row uses the loaded Hamiltonian** in `ham_lib/nh3_fer.bin`. Replace the generated NH3 results from the batch runner with those from [VarSI_loaded.py](VarSI_loaded.py) or [fast_VarSI_loaded.py](fast_VarSI_loaded.py):

```bash
python VarSI_loaded.py nh3 --tf both --wfn CISD --report-wfn FCI --max-sweeps 100
```

Substitute `fast_VarSI_loaded.py` for the fast version. Repeat with `--max-sweeps 500` for the corresponding NH3 results. 
## Reproducing the all-electron and frozen-core results

Use [scripts/run_varsi_all_electron.py](scripts/run_varsi_all_electron.py) to reproduce the molecular benchmark tables in the supplementary material. This runner calculates results for both all-electron and frozen-core Hamiltonians:

```bash
python -u scripts/run_varsi_all_electron.py --type standard --wfn FCI \
    --condition fc --max-sweeps 100 --cov-workers 48 \
    --output-csv all_electron_varsi_results/varsi_all_electron_standard_FCI_fc_sweeps100.csv \
    > varsi_all_electron_standard_FCI_fc_sweeps100.out 2>&1
```

Use `--wfn CISD` for CISD covariances. `--condition fc` selects fully commuting groups; change it to `qwc` for qubit-wise commuting groups. Update the CSV and `.out` names to match each choice. FCI reporting and both JW/BK mappings are defaults.

Match the published rows by basis, molecule, geometry, mapping, charge, and `frozen_core` (`False` for all-electron, `True` for frozen-core). The built-in linear hydrogen chains use all-electron Hamiltonians only.

## Main options

| Option | Purpose |
| --- | --- |
| `--wfn` | Covariance state: `FCI`, `HF`, or `CISD`; batch runners also accept `CCSD`. |
| `--report-wfn FCI` | Evaluate final costs with FCI; `SAME` uses the covariance state. Available in batch and loaded runners. |
| `--condition fc` | Fully commuting groups (default); `qwc` selects qubit-wise commutativity. |
| `--max-sweeps N` | Refinement sweeps; default `100`. |
| `--mappings JW BK` | Batch mappings; loaded versions use `--tf both`, `jw`, or `bk`. |
| `--cov-workers N` | Number of covariance worker processes. |
| `--output-csv FILE` | Batch results file. |

The all-electron runner additionally requires `--type`; use `standard` for the molecular results above. Each script's `--help` lists further options.

Batch runners save CSVs and print results; loaded versions print results only. Use distinct output files when changing settings, since completed CSV entries are skipped. Check the recorded reference states and failed methods before comparing results. All inputs needed for these two workflows are included; no additional Hamiltonian dataset is required.


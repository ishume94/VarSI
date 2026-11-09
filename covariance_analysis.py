"""Compare FCI Pauli variance and covariance distributions for five molecules.

Run this script from the directory where the plots should be written::

    python covariance_analysis.py

The fixed systems use the all-electron STO-3G Hamiltonians from
``hamiltonians_varsi``.  Variances are the diagonal entries ``C_ii`` for each
nonidentity unit Pauli.  Covariances contain each distinct fully commuting
nonidentity pair once, with ``i < j``.  Every KDE is independently divided by
its maximum so the curves have comparable peak heights.
"""

from __future__ import annotations

import argparse
import math
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path


os.environ.setdefault("MPLCONFIGDIR", "/tmp/varsi_mplconfig")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from openfermion.linalg import get_ground_state, get_sparse_operator
from scipy.stats import gaussian_kde

import hamiltonians_varsi as hv


@dataclass(frozen=True)
class SystemConfig:
    molecule: str
    mapping: str
    label: str
    color: str
    distance: float | None = None


@dataclass(frozen=True)
class PauliTerm:
    index: int
    pauli_tuple: tuple[tuple[int, str], ...]
    ops: tuple[str, ...]


@dataclass(frozen=True)
class DistributionData:
    label: str
    color: str
    variances: np.ndarray
    covariances: np.ndarray


# The order here is also the legend order.  The displayed H6 distance is its
# nearest-neighbor spacing.
SYSTEMS = (
    SystemConfig("BH", "BK", "BH, BK", "#0072B2"),
    SystemConfig("HF", "JW", "HF, JW", "#D55E00"),
    SystemConfig("H2O", "BK", r"H$_2$O, BK", "#009E73"),
    SystemConfig("LiH", "BK", "LiH, BK", "#CC79A7"),
    SystemConfig("H6", "JW", r"H$_6$ (2.0 Å), JW", "#BC3E3E", distance=2.0),
)

OUTPUT_STEMS = {
    "variance": "variance_comparison",
    "covariance": "covariance_comparison",
}

Y_BREAK = (0.2, 0.9)


def available_cpu_count() -> int:
    """Return the number of CPUs available to this process."""

    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bw-adjust",
        type=float,
        default=0.8,
        help="Scott bandwidth multiplier for covariance KDEs (default: 0.8).",
    )
    parser.add_argument(
        "--variance-bw-adjust",
        type=float,
        default=0.2,
        help="Scott bandwidth multiplier for variance KDEs (default: 0.2).",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution (default: 300).")
    parser.add_argument("--no-grid", action="store_true", help="Do not draw the light background grid.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and describe the five Hamiltonians without solving FCI or writing plots.",
    )
    args = parser.parse_args(argv)
    if not math.isfinite(args.bw_adjust) or args.bw_adjust <= 0:
        parser.error("--bw-adjust must be a positive finite number.")
    if not math.isfinite(args.variance_bw_adjust) or args.variance_bw_adjust <= 0:
        parser.error("--variance-bw-adjust must be a positive finite number.")
    if args.dpi < 1:
        parser.error("--dpi must be positive.")
    return args


def make_spec(config: SystemConfig) -> hv.VarsiHamiltonianSpec:
    """Create one of the fixed all-electron STO-3G system specifications."""

    if config.distance is not None:
        distance = float(config.distance)
        geometry = hv.linear_hydrogen_geometry(config.molecule, distance)
        geometry_label = "linear_R{}".format(hv.distance_label(distance))
    else:
        distance, angle = hv.STANDARD_DISTANCES[config.molecule]
        geometry = hv.standard_geometry(config.molecule, distance, angle)
        geometry_label = ""

    return hv.VarsiHamiltonianSpec(
        ham_type="covariance_analysis",
        molecule=config.molecule,
        geometry=geometry,
        bond_distance=hv.distance_label(distance),
        basis_label="sto-3g",
        basis_set="sto-3g",
        mapping=hv.normalize_mapping(config.mapping),
        multiplicity=1,
        spin_state="singlet",
        charge=hv.singlet_charge(config.molecule),
        frozen_core=False,
        geometry_label=geometry_label,
    )


def make_terms(qubit_operator, n_qubits: int) -> list[PauliTerm]:
    """Return stable, indexed nonidentity Pauli terms."""

    pauli_tuples = sorted(term for term in qubit_operator.terms if term)
    terms = []
    for index, pauli_tuple in enumerate(pauli_tuples):
        pauli_tuple = tuple((int(qubit), str(pauli)) for qubit, pauli in pauli_tuple)
        pauli_by_qubit = dict(pauli_tuple)
        terms.append(
            PauliTerm(
                index=index,
                pauli_tuple=pauli_tuple,
                ops=tuple(pauli_by_qubit.get(qubit, "I") for qubit in range(n_qubits)),
            )
        )
    if not terms:
        raise ValueError("The Hamiltonian has no nonidentity Pauli terms.")
    return terms


def terms_fully_commute(left: PauliTerm, right: PauliTerm) -> bool:
    anticommutes = sum(
        left_op != "I" and right_op != "I" and left_op != right_op
        for left_op, right_op in zip(left.ops, right.ops)
    )
    return anticommutes % 2 == 0


def clean_real(value, *, tiny: float = 1.0e-10) -> float:
    value = complex(value)
    if not np.isfinite(value.real) or not np.isfinite(value.imag):
        raise ValueError("Expected a finite value, got {}.".format(value))
    if abs(value.imag) > tiny:
        raise ValueError("Expected a real value, got {}.".format(value))
    return 0.0 if abs(value.real) <= tiny else float(value.real)


def pauli_action(
    term: PauliTerm,
    state: np.ndarray,
    indices: np.ndarray,
    n_qubits: int,
) -> np.ndarray:
    """Apply a unit Pauli as an exact signed permutation in OpenFermion order."""

    flip_mask = 0
    phase = np.ones(state.size, dtype=complex)
    for qubit, pauli in term.pauli_tuple:
        bit = 1 << (n_qubits - 1 - qubit)
        if pauli in "XY":
            flip_mask ^= bit
        if pauli in "YZ":
            phase *= 1 - 2 * ((indices & bit) != 0)
        if pauli == "Y":
            phase *= 1j

    action = np.empty_like(state)
    action[indices ^ flip_mask] = phase * state
    return action


def build_covariance_dictionary(
    terms: list[PauliTerm],
    state,
    n_qubits: int,
    max_workers: int,
):
    """Build exact unit-Pauli covariances using all requested CPU workers."""

    state = np.asarray(state, dtype=complex).reshape(-1)
    dimension = 2**n_qubits
    if state.size != dimension:
        raise ValueError("Expected an FCI vector of size {}, got {}.".format(dimension, state.size))
    norm = float(np.vdot(state, state).real)
    if not np.isclose(norm, 1.0, rtol=0.0, atol=1.0e-10):
        raise ValueError("FCI state is not normalized: norm={:.16g}.".format(norm))
    if max_workers < 1:
        raise ValueError("max_workers must be positive.")

    indices = np.arange(dimension, dtype=np.int64)
    actions = np.empty((len(terms), dimension), dtype=complex)

    def apply(position: int):
        actions[position] = pauli_action(terms[position], state, indices, n_qubits)

    if max_workers == 1:
        for position in range(len(terms)):
            apply(position)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for _ in executor.map(apply, range(len(terms))):
                pass

    single_values = actions @ state.conjugate()
    gram = actions.conjugate() @ actions.T
    expectations = {
        term.index: clean_real(single_values[position])
        for position, term in enumerate(terms)
    }

    covariances = {}
    for left_position, left in enumerate(terms):
        for right_position in range(left_position, len(terms)):
            right = terms[right_position]
            if not terms_fully_commute(left, right):
                continue
            covariances[(left.index, right.index)] = clean_real(
                gram[left_position, right_position]
                - expectations[left.index] * expectations[right.index]
            )
    return covariances, expectations


def covariance_workspace_gib(n_terms: int, n_qubits: int) -> float:
    """Estimate the action, conjugate-action, and Gram matrix workspace."""

    complex_bytes = np.dtype(complex).itemsize
    action_bytes = n_terms * (2**n_qubits) * complex_bytes
    gram_bytes = n_terms * n_terms * complex_bytes
    return (2 * action_bytes + gram_bytes) / (1024**3)


def build_hamiltonian(spec: hv.VarsiHamiltonianSpec) -> hv.BuiltHamiltonian:
    """Build through Tequila without leaving its HDF5 scratch files in the launch directory."""

    launch_directory = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="varsi-covariance-") as scratch_directory:
        try:
            os.chdir(scratch_directory)
            return hv.build_all_electron_hamiltonian(spec)
        finally:
            os.chdir(launch_directory)


def distribution_samples(terms: list[PauliTerm], covariances: dict):
    """Extract diagonal variance and unique off-diagonal covariance samples."""

    variances = []
    covariance_values = []
    for left_position, left in enumerate(terms):
        variance = float(covariances[(left.index, left.index)])
        if variance < -1.0e-8 or variance > 1.0 + 1.0e-8:
            raise ValueError("Pauli variance is outside [0, 1]: {}.".format(variance))
        variance = float(np.clip(variance, 0.0, 1.0))
        variances.append(variance)

        for right in terms[left_position + 1 :]:
            key = (left.index, right.index)
            if key not in covariances:
                continue
            covariance = float(covariances[key])
            if covariance < -1.0 - 1.0e-8 or covariance > 1.0 + 1.0e-8:
                raise ValueError("Pauli covariance is outside [-1, 1]: {}.".format(covariance))
            covariance = float(np.clip(covariance, -1.0, 1.0))
            covariance_values.append(covariance)

    arrays = tuple(
        np.asarray(values, dtype=float)
        for values in (variances, covariance_values)
    )
    if any(not np.all(np.isfinite(values)) for values in arrays):
        raise ValueError("A variance or covariance distribution contains non-finite values.")
    return arrays


def collect_distribution(config: SystemConfig, workers: int, *, dry_run: bool = False):
    spec = make_spec(config)
    print("", flush=True)
    print("Preparing {}".format(config.label), flush=True)
    print(
        "  geometry={} mapping={} frozen_core={} basis={}".format(
            spec.geometry_label or "standard_R{}".format(spec.bond_distance),
            spec.mapping,
            spec.frozen_core,
            spec.basis_label,
        ),
        flush=True,
    )

    built = build_hamiltonian(spec)
    terms = make_terms(built.qubit_operator, built.n_qubits)
    workspace_gib = covariance_workspace_gib(len(terms), built.n_qubits)
    print(
        "  qubits={} nonidentity_terms={} covariance_workspace={:.3f} GiB".format(
            built.n_qubits,
            len(terms),
            workspace_gib,
        ),
        flush=True,
    )
    if dry_run:
        return None

    sparse_hamiltonian = get_sparse_operator(built.qubit_operator, n_qubits=built.n_qubits)
    energy, state = get_ground_state(sparse_hamiltonian)
    state = np.asarray(state, dtype=complex).reshape(-1)
    residual = np.linalg.norm(sparse_hamiltonian.dot(state) - energy * state)
    if residual > 1.0e-7:
        raise ValueError("FCI eigenpair residual is too large: {:.3e}.".format(residual))
    print("  FCI energy={:.12g} residual={:.3e}".format(float(np.real(energy)), residual), flush=True)

    start = time.perf_counter()
    covariances, _ = build_covariance_dictionary(
        terms,
        state,
        built.n_qubits,
        max_workers=workers,
    )
    runtime = time.perf_counter() - start
    variances, covariance_values = distribution_samples(terms, covariances)
    print(
        "  covariance_dictionary={} entries; variances={}; off_diagonal_covariances={}; runtime={:.3f} s".format(
            len(covariances),
            variances.size,
            covariance_values.size,
            runtime,
        ),
        flush=True,
    )
    return DistributionData(
        label=config.label,
        color=config.color,
        variances=variances,
        covariances=covariance_values,
    )


def normalized_kde(samples: np.ndarray, x: np.ndarray, bandwidth_adjust: float) -> np.ndarray:
    if samples.size < 2 or np.ptp(samples) <= 1.0e-12:
        raise ValueError("A KDE requires at least two nonconstant samples.")
    kde = gaussian_kde(
        samples,
        bw_method=lambda estimator: estimator.scotts_factor() * bandwidth_adjust,
    )
    density = np.asarray(kde(x), dtype=float)
    peak = float(np.max(density))
    if not np.all(np.isfinite(density)) or not math.isfinite(peak) or peak <= 0.0:
        raise ValueError("KDE produced an invalid density.")
    return density / peak


def plot_comparison(datasets: list[DistributionData], kind: str, args) -> list[Path]:
    if kind == "variance":
        x_limits = (0.0, 1.0)
        xlabel = r"Variance $C_{ii}$"
        bandwidth_adjust = args.variance_bw_adjust
        sample_attribute = "variances"
        y_break = Y_BREAK
    elif kind == "covariance":
        x_limits = (-1.0, 1.0)
        xlabel = r"Covariance $C_{ij}$"
        bandwidth_adjust = args.bw_adjust
        sample_attribute = "covariances"
        y_break = Y_BREAK
    else:
        raise ValueError("Unknown distribution kind {!r}.".format(kind))

    break_low, break_high = y_break
    figure, axes = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(7.0, 4.5),
        gridspec_kw={"height_ratios": (1.0 - break_high, break_low)},
    )

    x = np.linspace(x_limits[0], x_limits[1], 2049)
    for dataset in datasets:
        samples = getattr(dataset, sample_attribute)
        density = normalized_kde(samples, x, bandwidth_adjust)
        for axis in axes:
            axis.plot(x, density, color=dataset.color, label=dataset.label, linewidth=1.8, zorder=3)
            axis.fill_between(x, density, color=dataset.color, alpha=0.18)

    axes[-1].set_xlabel(xlabel)
    for axis in axes:
        axis.set_xlim(x_limits)
        axis.tick_params(direction="out")
        if not args.no_grid:
            axis.grid(alpha=0.2)

    upper, lower = axes
    ticks = np.linspace(0.0, 1.0, 11)
    upper.set_ylim(break_high, 1.0)
    upper.set_yticks(np.unique(np.append(ticks[ticks > break_high], break_high)))
    lower.set_ylim(0.0, break_low)
    lower.set_yticks(np.unique(np.append(ticks[ticks < break_low], break_low)))
    upper.spines.bottom.set_visible(False)
    lower.spines.top.set_visible(False)
    upper.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    lower.xaxis.tick_bottom()
    break_marker = [(-1, -0.5), (1, 0.5)]
    for axis, y_position in ((upper, 0), (lower, 1)):
        axis.plot(
            [0, 1],
            [y_position, y_position],
            transform=axis.transAxes,
            marker=break_marker,
            markersize=12,
            linestyle="none",
            color="black",
            markeredgecolor="black",
            markeredgewidth=1,
            clip_on=False,
        )
    upper.legend(loc="upper left")
    figure.supylabel("Relative density")

    figure.tight_layout()
    figure.subplots_adjust(hspace=0.15)

    output_directory = Path.cwd()
    paths = []
    for extension in ("png", "svg"):
        path = output_directory / "{}.{}".format(OUTPUT_STEMS[kind], extension)
        figure.savefig(path, dpi=args.dpi, bbox_inches="tight")
        paths.append(path)
        print("Saved {}".format(path), flush=True)
    plt.close(figure)
    return paths


def main(argv=None):
    args = parse_args(argv)
    workers = available_cpu_count()
    print("FCI molecular covariance analysis", flush=True)
    print("Covariance workers={} (all affinity-visible CPUs)".format(workers), flush=True)
    print("Output directory={}".format(Path.cwd()), flush=True)

    datasets = []
    for config in SYSTEMS:
        dataset = collect_distribution(config, workers, dry_run=args.dry_run)
        if dataset is not None:
            datasets.append(dataset)

    if args.dry_run:
        print("Dry run complete; no FCI states, covariances, or plots were generated.", flush=True)
        return []

    paths = []
    for kind in ("variance", "covariance"):
        paths.extend(plot_comparison(datasets, kind, args))
    return paths


if __name__ == "__main__":
    main()

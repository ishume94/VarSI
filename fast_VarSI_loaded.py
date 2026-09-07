"""Standalone fast VarSI runner for fermionic Hamiltonians loaded from ham_lib.

Runs SI, VarSI-G, VarSI-O, VarSI-R, VarSI-OR, and their ICS follow-ups for
Jordan-Wigner and Bravyi-Kitaev mappings.  The fast variants share pairwise
compatibility, covariance, and group-aggregate caches within each run.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import pickle
import re
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path


def configure_runtime_environment():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/varsi_mplconfig")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/varsi_cache")
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    for thread_var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ.setdefault(thread_var, "1")


configure_runtime_environment()

import numpy as np
import tequila as tq
from openfermion import QubitOperator, variance as operator_variance
from openfermion.linalg import get_ground_state, get_sparse_operator
from openfermion.transforms import bravyi_kitaev, bravyi_kitaev_code, jordan_wigner
from openfermion.utils import count_qubits
from tequila.grouping.binary_rep import BinaryHamiltonian
from tequila.hamiltonian import QubitHamiltonian

from gflow_vqe.overlapping_helpers import iterative_coefficient_splitting_from_groups


LOADED_MOLECULE_ORDER = ("nh3",)
DEFAULT_MOLECULE_ORDER = ("nh3",)
DEFAULT_MAX_REFINEMENT_SWEEPS = 100
DEFAULT_COVARIANCE_CHUNKSIZE = 128
ATOMIC_NUMBERS = {
    "h": 1,
    "he": 2,
    "li": 3,
    "be": 4,
    "b": 5,
    "c": 6,
    "n": 7,
    "o": 8,
    "f": 9,
    "ne": 10,
}
ELECTRON_COUNTS = {
    "h2": 2,
    "lih": 4,
    "beh2": 6,
    "h2o": 10,
    "nh3": 10,
    "n2": 14,
}
METHOD_LABELS = {
    "SI": "Sorted insertion baseline",
    "SI-ICS": "ICS initialized from sorted insertion groups",
    "VarSI-G": "VarSI greedy from empty groups",
    "VarSI-G-ICS": "ICS initialized from VarSI greedy groups",
    "VarSI-O": "VarSI ordered from empty groups",
    "VarSI-O-ICS": "ICS initialized from VarSI ordered groups",
    "VarSI-R": "VarSI refinement from sorted insertion",
    "VarSI-R-ICS": "ICS initialized from VarSI-refined sorted insertion groups",
    "VarSI-OR": "VarSI ordered+refined",
    "VarSI-OR-ICS": "ICS initialized from VarSI ordered+refined groups",
}


@dataclass(frozen=True)
class PauliTerm:
    index: int
    pauli_tuple: tuple[tuple[int, str], ...]
    ops: tuple[str, ...]
    coefficient: complex
    word: str
    source_order: int = 0


@dataclass
class MethodResult:
    method: str
    groups: list[list[PauliTerm]]
    variances: list[float]
    eps_sq_m: float
    sample_ratios: list[float]
    runtime_s: float
    accepted_moves: int | None = None
    report_eps_sq_m: float | None = None


@dataclass
class FastVarSIContext:
    """Pairwise caches shared by all grouping methods for one system/state."""

    terms: list[PauliTerm]
    single_variances: np.ndarray
    scaled_covariances: np.ndarray
    compatible: np.ndarray


_ACTION_STATE = None
_ACTION_N_QUBITS = None
_ACTION_TERMS = None


def default_cov_workers():
    return max(1, min(8, os.cpu_count() or 1))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run the standalone fast VarSI grouping/reporting pipeline on "
            "fermionic Hamiltonians loaded from ham_lib."
        )
    )
    parser.add_argument(
        "molecules",
        nargs="*",
        type=lambda value: str(value).lower(),
        help="ham_lib molecule labels to run. Defaults to nh3 when available.",
    )
    parser.add_argument(
        "--all-molecules",
        action="store_true",
        help="Run every molecule discovered from ham_lib/*_fer.bin files.",
    )
    parser.add_argument(
        "--prefix",
        default=".",
        help="Repository/data prefix containing ham_lib (default: current directory).",
    )
    parser.add_argument(
        "--tf",
        type=lambda value: str(value).lower(),
        default="both",
        choices=("both", "bk", "jw"),
        help=(
            "Qubit transform for loaded fermionic Hamiltonians: both, bk, or jw "
            "(default: both)."
        ),
    )
    parser.add_argument(
        "--wfn",
        type=lambda value: str(value).upper(),
        default="FCI",
        choices=("FCI", "HF", "CISD"),
        help="Wavefunction used for the covariance matrix (default: FCI).",
    )
    parser.add_argument(
        "--report-wfn",
        type=lambda value: str(value).upper(),
        default="FCI",
        choices=("FCI", "SAME"),
        help=(
            "Wavefunction used for the second eps^2 M report column. FCI evaluates "
            "the finalized groups directly from the FCI wavefunction; SAME reuses "
            "--wfn and avoids FCI diagonalization (default: FCI)."
        ),
    )
    parser.add_argument(
        "--fci-max-qubits",
        type=int,
        default=16,
        help=(
            "Maximum qubits allowed for FCI diagonalization before the molecule is "
            "skipped with a regular error (default: 16)."
        ),
    )
    parser.add_argument(
        "--condition",
        choices=("fc", "qwc"),
        default="fc",
        help="Fully commuting or qubit-wise commuting groups (default: fc).",
    )
    parser.add_argument(
        "--max-sweeps",
        type=int,
        default=DEFAULT_MAX_REFINEMENT_SWEEPS,
        help="Maximum accepted refinement sweeps (default: 100).",
    )
    parser.add_argument(
        "--allow-new-groups",
        action="store_true",
        help="Also consider moving a term into a new singleton during refinement.",
    )
    parser.add_argument(
        "--ordered-consider-new-groups",
        action="store_true",
        help="Let VarSI-O open a singleton even when a compatible group exists.",
    )
    parser.add_argument(
        "--cov-workers",
        type=int,
        default=None,
        help="Worker processes used to construct Pauli action rows (default: up to 8).",
    )
    parser.add_argument(
        "--cov-chunksize",
        type=int,
        default=DEFAULT_COVARIANCE_CHUNKSIZE,
        help="Maximum Pauli terms in one covariance worker task (default: 128).",
    )
    parser.add_argument(
        "--serial-cov-dict",
        action="store_true",
        help="Compatibility alias for --cov-workers 1.",
    )
    parser.add_argument(
        "--print-groups",
        action="store_true",
        help="Print the Pauli words in every final group.",
    )
    parser.add_argument(
        "--no-ics",
        action="store_true",
        help="Skip iterative coefficient splitting (ICS) for all methods.",
    )
    parser.add_argument(
        "--spin-ordering",
        choices=("interleaved", "blocked"),
        default="interleaved",
        help=(
            "Spin-orbital ordering used by the loaded fermionic Hamiltonian "
            "(default: interleaved)."
        ),
    )
    parser.add_argument(
        "--cisd-validation-tol",
        type=float,
        default=1.0e-8,
        help=(
            "Allowed difference between CISD subspace energy and qubit expectation "
            "(default: 1e-8)."
        ),
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop the batch when one loaded molecule fails instead of continuing.",
    )
    args = parser.parse_args(argv)
    if args.max_sweeps < 1:
        parser.error("--max-sweeps must be at least 1.")
    if args.cov_workers is not None and args.cov_workers < 1:
        parser.error("--cov-workers must be at least 1.")
    if args.cov_chunksize < 1:
        parser.error("--cov-chunksize must be at least 1.")
    if args.cisd_validation_tol <= 0.0:
        parser.error("--cisd-validation-tol must be positive.")
    if args.fci_max_qubits < 1:
        parser.error("--fci-max-qubits must be at least 1.")
    args.tf = ("bk", "jw") if args.tf == "both" else (args.tf,)

    available = discover_loaded_molecules(args.prefix)
    default_molecules = [mol for mol in DEFAULT_MOLECULE_ORDER if mol in available]
    requested = args.molecules or (available if args.all_molecules else default_molecules)
    missing = sorted(set(requested) - set(available))
    if missing:
        parser.error(
            "Unknown loaded molecule(s): {}. Available: {}".format(
                ", ".join(missing),
                ", ".join(available),
            )
        )
    args.molecules = requested
    return args


def discover_loaded_molecules(prefix="."):
    ham_dir = os.path.join(prefix, "ham_lib")
    labels = [
        os.path.basename(path)[: -len("_fer.bin")]
        for path in glob.glob(os.path.join(ham_dir, "*_fer.bin"))
    ]
    known = [mol for mol in LOADED_MOLECULE_ORDER if mol in labels]
    extra = sorted(mol for mol in labels if mol not in LOADED_MOLECULE_ORDER)
    return known + extra


def load_loaded_hamiltonian(mol, tf="bk", prefix="."):
    tf = str(tf).lower()
    path = os.path.join(prefix, "ham_lib", "{}_fer.bin".format(mol))
    with open(path, "rb") as source:
        fermion_hamiltonian = pickle.load(source)

    if tf == "bk":
        qubit_hamiltonian = bravyi_kitaev(fermion_hamiltonian)
    elif tf == "jw":
        qubit_hamiltonian = jordan_wigner(fermion_hamiltonian)
    else:
        raise ValueError("Transformation {} not supported".format(tf))

    tequila_hamiltonian = QubitHamiltonian(qubit_hamiltonian)
    return fermion_hamiltonian, qubit_hamiltonian, tequila_hamiltonian


def infer_electron_count(mol):
    mol = mol.lower()
    if mol in ELECTRON_COUNTS:
        return ELECTRON_COUNTS[mol]

    total = 0
    position = 0
    element_symbols = sorted(ATOMIC_NUMBERS, key=len, reverse=True)
    while position < len(mol):
        symbol = None
        for candidate in element_symbols:
            if mol.startswith(candidate, position):
                symbol = candidate
                break
        if symbol is None:
            raise ValueError(
                "Could not infer electron count from molecule label '{}'.".format(mol)
            )

        position += len(symbol)
        match = re.match(r"\d+", mol[position:])
        if match is None:
            multiplicity = 1
        else:
            multiplicity = int(match.group(0))
            position += len(match.group(0))
        total += ATOMIC_NUMBERS[symbol] * multiplicity
    return total


def spin_label(mode, n_modes, spin_ordering):
    if spin_ordering == "interleaved":
        return mode % 2
    if spin_ordering == "blocked":
        return 0 if mode < n_modes // 2 else 1
    raise ValueError("Unsupported spin ordering '{}'.".format(spin_ordering))


def reference_occupations(n_modes, n_electrons, spin_ordering):
    if n_electrons < 0 or n_electrons > n_modes:
        raise ValueError(
            "Invalid electron count {} for {} spin orbitals.".format(
                n_electrons,
                n_modes,
            )
        )
    if n_electrons % 2:
        raise NotImplementedError("Loaded CISD/HF support expects a closed-shell reference.")

    n_alpha = n_electrons // 2
    n_beta = n_electrons // 2
    occupations = [0] * n_modes
    if spin_ordering == "interleaved":
        for spatial_orbital in range(n_alpha):
            occupations[2 * spatial_orbital] = 1
        for spatial_orbital in range(n_beta):
            occupations[2 * spatial_orbital + 1] = 1
    elif spin_ordering == "blocked":
        n_spatial_orbitals = n_modes // 2
        for spatial_orbital in range(n_alpha):
            occupations[spatial_orbital] = 1
        for spatial_orbital in range(n_beta):
            occupations[n_spatial_orbitals + spatial_orbital] = 1
    else:
        raise ValueError("Unsupported spin ordering '{}'.".format(spin_ordering))
    return tuple(occupations)


def same_spin_counts(removed_modes, added_modes, n_modes, spin_ordering):
    removed = {0: 0, 1: 0}
    added = {0: 0, 1: 0}
    for mode in removed_modes:
        removed[spin_label(mode, n_modes, spin_ordering)] += 1
    for mode in added_modes:
        added[spin_label(mode, n_modes, spin_ordering)] += 1
    return removed == added


def generate_cisd_determinants(n_modes, n_electrons, spin_ordering):
    reference = reference_occupations(n_modes, n_electrons, spin_ordering)
    occupied = [idx for idx, occupation in enumerate(reference) if occupation]
    virtual = [idx for idx, occupation in enumerate(reference) if not occupation]

    determinants = {reference}
    for excitation_rank in (1, 2):
        for removed_modes in combinations(occupied, excitation_rank):
            for added_modes in combinations(virtual, excitation_rank):
                if not same_spin_counts(
                    removed_modes,
                    added_modes,
                    n_modes,
                    spin_ordering,
                ):
                    continue
                determinant = list(reference)
                for mode in removed_modes:
                    determinant[mode] = 0
                for mode in added_modes:
                    determinant[mode] = 1
                determinants.add(tuple(determinant))
    return sorted(determinants)


def occupations_to_sparse_basis_index(occupations):
    index = 0
    n_qubits = len(occupations)
    for qubit, occupied in enumerate(occupations):
        if occupied:
            index |= 1 << (n_qubits - 1 - qubit)
    return index


def qubit_occupations(fermion_occupations, tf):
    tf = str(tf).lower()
    if tf == "jw":
        return tuple(int(value) for value in fermion_occupations)
    if tf != "bk":
        raise ValueError("Unsupported transformation '{}'.".format(tf))
    encoder = bravyi_kitaev_code(len(fermion_occupations)).encoder
    return tuple(
        int(value)
        for value in np.asarray(
            encoder.dot(np.asarray(fermion_occupations)) % 2
        ).ravel()
    )


def sparse_basis_index(fermion_occupations, tf):
    return occupations_to_sparse_basis_index(qubit_occupations(fermion_occupations, tf))


def statevector_from_basis_index(index, n_qubits):
    state = np.zeros(2**n_qubits, dtype=complex)
    state[index] = 1.0
    return state


def apply_qubit_term_to_basis_index(term, basis_index, n_qubits):
    output_index = basis_index
    phase = 1.0 + 0.0j
    for qubit, pauli in term:
        mask = 1 << (n_qubits - 1 - qubit)
        bit = 1 if output_index & mask else 0
        if pauli == "X":
            output_index ^= mask
        elif pauli == "Y":
            phase *= 1.0j if bit == 0 else -1.0j
            output_index ^= mask
        elif pauli == "Z":
            phase *= 1.0 if bit == 0 else -1.0
        else:
            raise ValueError("Unsupported Pauli operator '{}'.".format(pauli))
    return output_index, phase


def basis_state_expectation_value(qubit_hamiltonian, basis_index, n_qubits):
    value = 0.0 + 0.0j
    for term, coefficient in qubit_hamiltonian.terms.items():
        output_index, phase = apply_qubit_term_to_basis_index(term, basis_index, n_qubits)
        if output_index == basis_index:
            value += coefficient * phase
    return value


def build_qubit_subspace_matrix(qubit_hamiltonian, basis_indices, n_qubits):
    index_to_row = {basis_index: row for row, basis_index in enumerate(basis_indices)}
    matrix = np.zeros((len(basis_indices), len(basis_indices)), dtype=complex)
    for column, basis_index in enumerate(basis_indices):
        for term, coefficient in qubit_hamiltonian.terms.items():
            output_index, phase = apply_qubit_term_to_basis_index(term, basis_index, n_qubits)
            row = index_to_row.get(output_index)
            if row is not None:
                matrix[row, column] += coefficient * phase
    return 0.5 * (matrix + matrix.conj().T)


def qubit_operator_support_expectation(qubit_hamiltonian, support, n_qubits):
    value = 0.0 + 0.0j
    for basis_index, amplitude in support.items():
        for term, coefficient in qubit_hamiltonian.terms.items():
            output_index, phase = apply_qubit_term_to_basis_index(term, basis_index, n_qubits)
            output_amplitude = support.get(output_index)
            if output_amplitude is not None:
                value += np.conjugate(output_amplitude) * coefficient * phase * amplitude
    return value


def loaded_hf_energy_and_statevector(
    mol,
    qubit_hamiltonian,
    n_qubits,
    spin_ordering,
    tf,
):
    n_electrons = infer_electron_count(mol)
    reference = reference_occupations(n_qubits, n_electrons, spin_ordering)
    basis_index = sparse_basis_index(reference, tf)
    state = statevector_from_basis_index(basis_index, n_qubits)
    energy = basis_state_expectation_value(qubit_hamiltonian, basis_index, n_qubits)
    return energy, state


def loaded_cisd_statevector(
    mol,
    qubit_hamiltonian,
    n_qubits,
    spin_ordering,
    validation_tol,
    tf,
):
    n_electrons = infer_electron_count(mol)
    determinants = generate_cisd_determinants(n_qubits, n_electrons, spin_ordering)
    basis_indices = [sparse_basis_index(determinant, tf) for determinant in determinants]
    if len(set(basis_indices)) != len(basis_indices):
        raise RuntimeError("{} mapping produced duplicate CISD basis indices.".format(tf.upper()))

    subspace_hamiltonian = build_qubit_subspace_matrix(
        qubit_hamiltonian,
        basis_indices,
        n_qubits,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(subspace_hamiltonian)
    ground_index = int(np.argmin(eigenvalues))
    cisd_energy = float(np.real_if_close(eigenvalues[ground_index]))
    coefficients = eigenvectors[:, ground_index]

    state = np.zeros(2**n_qubits, dtype=complex)
    support = {}
    for basis_index, coefficient in zip(basis_indices, coefficients):
        state[basis_index] = coefficient
        support[basis_index] = coefficient
    state /= np.linalg.norm(state)

    qubit_expectation = qubit_operator_support_expectation(
        qubit_hamiltonian,
        support,
        n_qubits,
    )
    difference = abs(qubit_expectation - cisd_energy)
    if difference > validation_tol:
        raise RuntimeError(
            "CISD qubit-space validation failed for {}: subspace energy {}, "
            "qubit expectation {}, difference {}.".format(
                mol,
                cisd_energy,
                qubit_expectation,
                difference,
            )
        )

    metadata = {
        "basis_size": len(determinants),
        "subspace_energy": cisd_energy,
        "qubit_expectation": qubit_expectation,
        "difference": difference,
    }
    return cisd_energy, state, metadata


def get_loaded_variance_wavefunction(
    mol,
    qubit_hamiltonian,
    method,
    sparse_hamiltonian,
    spin_ordering,
    validation_tol,
    tf,
):
    method = str(method).upper()
    n_qubits = count_qubits(qubit_hamiltonian)
    if method == "FCI":
        if sparse_hamiltonian is None:
            sparse_hamiltonian = get_sparse_operator(
                qubit_hamiltonian,
                n_qubits=n_qubits,
            )
        energy, wavefunction = get_ground_state(sparse_hamiltonian)
        return energy, wavefunction, None
    if method == "HF":
        energy, wavefunction = loaded_hf_energy_and_statevector(
            mol,
            qubit_hamiltonian,
            n_qubits,
            spin_ordering,
            tf,
        )
        return energy, wavefunction, None
    if method == "CISD":
        return loaded_cisd_statevector(
            mol,
            qubit_hamiltonian,
            n_qubits,
            spin_ordering,
            validation_tol,
            tf,
        )
    raise ValueError("Unsupported wavefunction method '{}'.".format(method))


def print_cisd_validation(metadata):
    if metadata is None:
        return
    print("CISD basis size={}".format(metadata["basis_size"]))
    print("CISD subspace Energy={}".format(metadata["subspace_energy"]))
    print("CISD qubit expectation={}".format(metadata["qubit_expectation"]))
    print("CISD qubit expectation difference={}".format(metadata["difference"]))


def clean_complex(value, tiny=1.0e-12):
    value = complex(value)
    real = 0.0 if abs(value.real) < tiny else value.real
    imag = 0.0 if abs(value.imag) < tiny else value.imag
    return complex(real, imag)


def clean_real(value, tiny=1.0e-9):
    value = clean_complex(value, tiny=tiny)
    if abs(value.imag) > tiny:
        raise ValueError("Expected a real value, got {}.".format(value))
    real_value = float(value.real)
    return 0.0 if abs(real_value) < tiny else real_value


def clean_variance(value, tiny=1.0e-7):
    real_value = clean_real(value, tiny=tiny)
    if real_value < 0.0 and abs(real_value) < tiny:
        return 0.0
    if real_value < 0.0:
        raise ValueError("Computed a negative variance: {}.".format(real_value))
    return real_value


def clean_report_variance(value, tiny=1.0e-9):
    """Clean numerical noise without discarding small positive variances."""

    value = complex(value)
    if abs(value.imag) > tiny:
        raise ValueError("Expected a real variance, got {}.".format(value))
    real_value = float(value.real)
    if real_value < 0.0 and abs(real_value) < tiny:
        return 0.0
    if real_value < 0.0:
        raise ValueError("Computed a negative variance: {}.".format(real_value))
    return real_value


def pauli_word(pauli_tuple):
    if not pauli_tuple:
        return "I"
    return " ".join("{}{}".format(pauli, qubit) for qubit, pauli in pauli_tuple)


def make_terms(qubit_operator, n_qubits):
    source_items = list(qubit_operator.terms.items())
    source_order = {pauli_tuple: position for position, (pauli_tuple, _) in enumerate(source_items)}
    items = sorted(source_items, key=lambda item: (bool(item[0]), item[0]))
    terms = []
    for index, (pauli_tuple_value, coefficient) in enumerate(items):
        pauli_tuple_value = tuple((int(qubit), str(pauli)) for qubit, pauli in pauli_tuple_value)
        pauli_by_qubit = dict(pauli_tuple_value)
        terms.append(
            PauliTerm(
                index=index,
                pauli_tuple=pauli_tuple_value,
                ops=tuple(pauli_by_qubit.get(qubit, "I") for qubit in range(n_qubits)),
                coefficient=clean_complex(coefficient),
                word=pauli_word(pauli_tuple_value),
                source_order=source_order[pauli_tuple_value],
            )
        )
    return terms


def terms_fully_commute(term1, term2):
    anticommutes = 0
    for op1, op2 in zip(term1.ops, term2.ops):
        if op1 != "I" and op2 != "I" and op1 != op2:
            anticommutes += 1
    return anticommutes % 2 == 0


def terms_qubit_wise_commute(term1, term2):
    return all(op1 == "I" or op2 == "I" or op1 == op2 for op1, op2 in zip(term1.ops, term2.ops))


def terms_compatible(term1, term2, condition):
    if condition == "fc":
        return terms_fully_commute(term1, term2)
    if condition == "qwc":
        return terms_qubit_wise_commute(term1, term2)
    raise ValueError("Unsupported compatibility condition '{}'.".format(condition))


def tequila_wavefunction_from_array(state_vector):
    return tq.QubitWaveFunction.from_array(np.asarray(state_vector, dtype=complex))


def pauli_hamiltonian_for_term(term):
    return QubitHamiltonian.from_openfermion(QubitOperator(term.pauli_tuple, 1.0))


def wavefunction_array(wfn, dim):
    array = np.asarray(wfn.to_array(), dtype=complex).reshape(-1)
    if array.size != dim:
        raise ValueError("Expected wavefunction array of size {}, got {}.".format(dim, array.size))
    return array


def action_row_for_term(term, reference_wfn, dim):
    return wavefunction_array(pauli_hamiltonian_for_term(term)(reference_wfn), dim)


def _init_action_worker(state_vector, n_qubits, terms):
    global _ACTION_STATE
    global _ACTION_N_QUBITS
    global _ACTION_TERMS
    _ACTION_STATE = tequila_wavefunction_from_array(state_vector)
    _ACTION_N_QUBITS = int(n_qubits)
    _ACTION_TERMS = list(terms)


def _action_rows_chunk(term_positions):
    dim = 2**_ACTION_N_QUBITS
    rows = []
    for position in term_positions:
        rows.append((position, action_row_for_term(_ACTION_TERMS[position], _ACTION_STATE, dim)))
    return rows


def iter_index_chunks(n_items, chunksize):
    for start in range(0, n_items, chunksize):
        yield list(range(start, min(start + chunksize, n_items)))


def build_action_matrix(terms, state_vector, n_qubits, max_workers, chunksize):
    dim = 2**n_qubits
    state_vector = np.asarray(state_vector, dtype=complex).reshape(-1)
    if state_vector.size != dim:
        raise ValueError("Expected statevector size {}, got {}.".format(dim, state_vector.size))
    actions = np.empty((len(terms), dim), dtype=complex)
    if max_workers == 1:
        reference_wfn = tequila_wavefunction_from_array(state_vector)
        for position, term in enumerate(terms):
            actions[position] = action_row_for_term(term, reference_wfn, dim)
        return actions

    automatic_chunksize = max(1, math.ceil(len(terms) / (4 * max_workers)))
    task_chunksize = min(chunksize, automatic_chunksize)
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_action_worker,
        initargs=(state_vector, n_qubits, terms),
    ) as executor:
        for chunk_rows in executor.map(_action_rows_chunk, iter_index_chunks(len(terms), task_chunksize)):
            for position, row in chunk_rows:
                actions[position] = row
    return actions


def build_covariance_dictionary(terms, state_vector, n_qubits, max_workers, chunksize):
    state_vector = np.asarray(state_vector, dtype=complex).reshape(-1)
    actions = build_action_matrix(terms, state_vector, n_qubits, max_workers, chunksize)
    single_values = actions.dot(state_vector.conjugate())
    gram = actions.conjugate().dot(actions.T)
    single_expectations = {
        term.index: clean_complex(single_values[position]) for position, term in enumerate(terms)
    }
    covariances = {}
    for left_position, left in enumerate(terms):
        for right_position in range(left_position, len(terms)):
            right = terms[right_position]
            if not terms_fully_commute(left, right):
                continue
            covariance = clean_complex(
                gram[left_position, right_position]
                - single_expectations[left.index] * single_expectations[right.index]
            )
            covariances[(left.index, right.index)] = covariance
    return covariances, single_expectations


def get_covariance(term1, term2, covariances):
    key = (term1.index, term2.index) if term1.index <= term2.index else (term2.index, term1.index)
    return covariances[key]


def eps_sq_m_from_variances(variances):
    sqrt_sum = sum(math.sqrt(max(float(variance), 0.0)) for variance in variances)
    return sqrt_sum * sqrt_sum


def sample_ratios_from_variances(variances):
    weights = [math.sqrt(max(float(variance), 0.0)) for variance in variances]
    total = sum(weights)
    if total == 0.0:
        return [1.0 / len(weights) for _ in weights]
    return [weight / total for weight in weights]


def build_fast_context(measurable_terms, covariances, condition):
    n_terms = len(measurable_terms)
    scaled_covariances = np.zeros((n_terms, n_terms), dtype=float)
    compatible = np.zeros((n_terms, n_terms), dtype=bool)
    for left_position, left in enumerate(measurable_terms):
        for right_position in range(left_position, n_terms):
            right = measurable_terms[right_position]
            pair_is_compatible = terms_compatible(left, right, condition)
            compatible[left_position, right_position] = pair_is_compatible
            compatible[right_position, left_position] = pair_is_compatible
            if not pair_is_compatible:
                continue
            scaled = clean_real(
                left.coefficient * right.coefficient * get_covariance(left, right, covariances)
            )
            scaled_covariances[left_position, right_position] = scaled
            scaled_covariances[right_position, left_position] = scaled
    return FastVarSIContext(
        terms=list(measurable_terms),
        single_variances=scaled_covariances.diagonal().copy(),
        scaled_covariances=scaled_covariances,
        compatible=compatible,
    )


def fast_group_state(ctx, position_groups):
    """Cache each group's variance, covariance-to-term sums, and compatibility mask."""

    variances = []
    covariance_sums = []
    compatibility_masks = []
    for group in position_groups:
        group_array = np.asarray(group, dtype=int)
        variances.append(
            clean_variance(ctx.scaled_covariances[np.ix_(group_array, group_array)].sum())
        )
        covariance_sums.append(ctx.scaled_covariances[group_array, :].sum(axis=0))
        compatibility_masks.append(ctx.compatible[group_array, :].all(axis=0))
    return variances, covariance_sums, compatibility_masks


def validate_position_groups(ctx, position_groups):
    seen = []
    for group_index, group in enumerate(position_groups):
        for local_index, position in enumerate(group):
            for other_position in group[local_index + 1 :]:
                if not bool(ctx.compatible[position, other_position]):
                    raise ValueError("Group {} contains incompatible terms.".format(group_index))
            seen.append(position)
    expected = set(range(len(ctx.terms)))
    if len(seen) != len(set(seen)) or set(seen) != expected:
        raise ValueError("Grouping is not a non-overlapping cover of all measurable terms.")


def fast_sorted_insertion(ctx):
    ordered_positions = sorted(
        range(len(ctx.terms)),
        key=lambda position: abs(ctx.terms[position].coefficient),
        reverse=True,
    )
    groups = []
    masks = []
    for position in ordered_positions:
        for group_index, mask in enumerate(masks):
            if bool(mask[position]):
                groups[group_index].append(position)
                masks[group_index] = np.logical_and(mask, ctx.compatible[position])
                break
        else:
            groups.append([position])
            masks.append(ctx.compatible[position].copy())
    variances, _, _ = fast_group_state(ctx, groups)
    return groups, variances


def add_position_to_group(
    ctx,
    groups,
    variances,
    covariance_sums,
    masks,
    group_index,
    position,
    new_variance,
):
    groups[group_index].append(position)
    variances[group_index] = clean_variance(new_variance)
    covariance_sums[group_index] = covariance_sums[group_index] + ctx.scaled_covariances[position]
    masks[group_index] = np.logical_and(masks[group_index], ctx.compatible[position])


def open_position_group(ctx, groups, variances, covariance_sums, masks, position):
    groups.append([position])
    variances.append(float(ctx.single_variances[position]))
    covariance_sums.append(ctx.scaled_covariances[position].copy())
    masks.append(ctx.compatible[position].copy())


def fast_varsi_greedy(ctx):
    """Choose the globally best insertion from all remaining terms."""

    remaining_positions = sorted(
        range(len(ctx.terms)),
        key=lambda position: (
            -ctx.single_variances[position],
            ctx.terms[position].source_order,
        ),
    )
    groups = []
    variances = []
    covariance_sums = []
    masks = []

    while remaining_positions:
        if not groups:
            position = remaining_positions.pop(0)
            open_position_group(ctx, groups, variances, covariance_sums, masks, position)
            continue

        current_sqrt_sum = sum(math.sqrt(variance) for variance in variances)
        best_candidate = None

        for remaining_index, position in enumerate(remaining_positions):
            term_variance = float(ctx.single_variances[position])
            compatible_candidates = []

            for group_index, mask in enumerate(masks):
                if not bool(mask[position]):
                    continue
                new_variance = clean_variance(
                    variances[group_index]
                    + term_variance
                    + 2.0 * covariance_sums[group_index][position]
                )
                new_sqrt_sum = (
                    current_sqrt_sum
                    - math.sqrt(variances[group_index])
                    + math.sqrt(new_variance)
                )
                compatible_candidates.append(
                    (new_sqrt_sum * new_sqrt_sum, new_variance, group_index)
                )

            if compatible_candidates:
                metric, new_variance, group_index = min(compatible_candidates)
                candidate = (metric, new_variance, group_index, remaining_index)
            else:
                new_variance = term_variance
                metric = (current_sqrt_sum + math.sqrt(new_variance)) ** 2
                candidate = (metric, new_variance, len(groups), remaining_index)

            if best_candidate is None or candidate < best_candidate:
                best_candidate = candidate

        _, new_variance, group_index, remaining_index = best_candidate
        position = remaining_positions.pop(remaining_index)
        if group_index == len(groups):
            open_position_group(ctx, groups, variances, covariance_sums, masks, position)
        else:
            add_position_to_group(
                ctx,
                groups,
                variances,
                covariance_sums,
                masks,
                group_index,
                position,
                new_variance,
            )

    return groups, variances


def fast_varsi_ordered(ctx, consider_new_groups=False):
    ordered_positions = sorted(
        range(len(ctx.terms)),
        key=lambda position: ctx.single_variances[position],
        reverse=True,
    )
    groups = []
    variances = []
    covariance_sums = []
    masks = []
    for position in ordered_positions:
        term_variance = float(ctx.single_variances[position])
        current_sqrt_sum = sum(math.sqrt(variance) for variance in variances)
        best_metric = None
        best_group_index = None
        best_new_variance = term_variance
        for group_index, mask in enumerate(masks):
            if not bool(mask[position]):
                continue
            new_variance = clean_variance(
                variances[group_index]
                + term_variance
                + 2.0 * covariance_sums[group_index][position]
            )
            new_sqrt_sum = (
                current_sqrt_sum - math.sqrt(variances[group_index]) + math.sqrt(new_variance)
            )
            metric = new_sqrt_sum * new_sqrt_sum
            if best_metric is None or metric < best_metric:
                best_metric = metric
                best_group_index = group_index
                best_new_variance = new_variance
        if best_metric is None or consider_new_groups:
            singleton_metric = (current_sqrt_sum + math.sqrt(term_variance)) ** 2
            if best_metric is None or singleton_metric < best_metric:
                best_metric = singleton_metric
                best_group_index = None
                best_new_variance = term_variance
        if best_group_index is None:
            open_position_group(ctx, groups, variances, covariance_sums, masks, position)
        else:
            add_position_to_group(
                ctx,
                groups,
                variances,
                covariance_sums,
                masks,
                best_group_index,
                position,
                best_new_variance,
            )
    return groups, variances


def fast_refine_groups(
    ctx,
    initial_groups,
    max_sweeps=DEFAULT_MAX_REFINEMENT_SWEEPS,
    allow_new_groups=False,
    tiny=1.0e-9,
):
    groups = [list(group) for group in initial_groups]
    variances, covariance_sums, masks = fast_group_state(ctx, groups)
    accepted_moves = 0
    for _ in range(max_sweeps):
        current_sqrt_sum = sum(math.sqrt(variance) for variance in variances)
        best_metric = current_sqrt_sum * current_sqrt_sum
        best_move = None
        for source_index, group in enumerate(groups):
            source_variance = variances[source_index]
            for term_index, position in enumerate(group):
                term_variance = float(ctx.single_variances[position])
                if len(group) == 1:
                    source_new_variance = None
                else:
                    source_new_variance = clean_variance(
                        source_variance
                        + term_variance
                        - 2.0 * covariance_sums[source_index][position]
                    )
                destination_indices = [
                    index for index in range(len(groups)) if index != source_index
                ]
                if allow_new_groups:
                    destination_indices.append(len(groups))
                for destination_index in destination_indices:
                    destination_is_new = destination_index == len(groups)
                    if not destination_is_new and not bool(masks[destination_index][position]):
                        continue
                    if destination_is_new:
                        destination_old_variance = None
                        destination_new_variance = term_variance
                    else:
                        destination_old_variance = variances[destination_index]
                        destination_new_variance = clean_variance(
                            destination_old_variance
                            + term_variance
                            + 2.0 * covariance_sums[destination_index][position]
                        )
                    sqrt_sum = current_sqrt_sum - math.sqrt(source_variance)
                    if source_new_variance is not None:
                        sqrt_sum += math.sqrt(source_new_variance)
                    if destination_old_variance is not None:
                        sqrt_sum -= math.sqrt(destination_old_variance)
                    sqrt_sum += math.sqrt(destination_new_variance)
                    candidate_metric = sqrt_sum * sqrt_sum
                    if candidate_metric < best_metric - tiny:
                        best_metric = candidate_metric
                        best_move = (
                            source_index,
                            term_index,
                            destination_index,
                            destination_is_new,
                            source_new_variance,
                        )
        if best_move is None:
            break
        source_index, term_index, destination_index, destination_is_new, source_new_variance = (
            best_move
        )
        position = groups[source_index].pop(term_index)
        if source_new_variance is None:
            del groups[source_index]
            if not destination_is_new and destination_index > source_index:
                destination_index -= 1
        if destination_is_new:
            groups.append([position])
        else:
            groups[destination_index].append(position)
        # These are grouping-local caches.  Rebuild them after the accepted move
        # so group deletion/reindexing cannot leave stale aggregate rows.
        variances, covariance_sums, masks = fast_group_state(ctx, groups)
        accepted_moves += 1
    return groups, variances, accepted_moves


def make_fast_result(method, ctx, position_groups, variances, runtime_s, accepted_moves=None):
    validate_position_groups(ctx, position_groups)
    term_groups = [[ctx.terms[position] for position in group] for group in position_groups]
    variances = [float(variance) for variance in variances]
    return MethodResult(
        method=method,
        groups=term_groups,
        variances=variances,
        eps_sq_m=eps_sq_m_from_variances(variances),
        sample_ratios=sample_ratios_from_variances(variances),
        runtime_s=runtime_s,
        accepted_moves=accepted_moves,
    )


def timed_fast_result(method, ctx, grouping_function, accepted_moves=False):
    start = time.perf_counter()
    output = grouping_function()
    runtime_s = time.perf_counter() - start
    if accepted_moves:
        groups, variances, moves = output
    else:
        groups, variances = output
        moves = None
    return make_fast_result(method, ctx, groups, variances, runtime_s, moves)


def run_fast_varsi_methods(ctx, args):
    si_positions = None
    ordered_positions = None

    def record_si():
        nonlocal si_positions
        si_positions, variances = fast_sorted_insertion(ctx)
        return si_positions, variances

    def record_ordered():
        nonlocal ordered_positions
        ordered_positions, variances = fast_varsi_ordered(
            ctx,
            consider_new_groups=args.ordered_consider_new_groups,
        )
        return ordered_positions, variances

    si_result = timed_fast_result("SI", ctx, record_si)
    greedy_result = timed_fast_result("VarSI-G", ctx, lambda: fast_varsi_greedy(ctx))
    ordered_result = timed_fast_result("VarSI-O", ctx, record_ordered)
    refined_result = timed_fast_result(
        "VarSI-R",
        ctx,
        lambda: fast_refine_groups(
            ctx,
            si_positions,
            max_sweeps=args.max_sweeps,
            allow_new_groups=args.allow_new_groups,
        ),
        accepted_moves=True,
    )
    ordered_refined_result = timed_fast_result(
        "VarSI-OR",
        ctx,
        lambda: fast_refine_groups(
            ctx,
            ordered_positions,
            max_sweeps=args.max_sweeps,
            allow_new_groups=args.allow_new_groups,
        ),
        accepted_moves=True,
    )
    # Match VarSI_loaded.py's reporting order even when --no-ics is used.
    return [si_result, greedy_result, ordered_result, ordered_refined_result, refined_result]


def binary_tuple_for_term(term):
    n_qubits = len(term.ops)
    x_bits = [0.0] * n_qubits
    z_bits = [0.0] * n_qubits
    for qubit, pauli in term.pauli_tuple:
        if pauli in ("X", "Y"):
            x_bits[qubit] = 1.0
        if pauli in ("Z", "Y"):
            z_bits[qubit] = 1.0
    return tuple(x_bits + z_bits)


def build_ics_bridge(tequila_hamiltonian, measurable_terms, covariances):
    binary_hamiltonian = BinaryHamiltonian.init_from_qubit_hamiltonian(tequila_hamiltonian)
    binary_terms_by_key = {
        term.binary_tuple(): term
        for term in binary_hamiltonian.binary_terms
        if np.any(term.get_binary())
    }
    pauli_terms_by_key = {binary_tuple_for_term(term): term for term in measurable_terms}
    if set(binary_terms_by_key) != set(pauli_terms_by_key):
        missing = sorted(set(pauli_terms_by_key) - set(binary_terms_by_key))
        extra = sorted(set(binary_terms_by_key) - set(pauli_terms_by_key))
        raise ValueError(
            "Fast/ICS Pauli-term representations do not match. Missing={}, Extra={}.".format(
                missing,
                extra,
            )
        )

    for key, pauli_term in pauli_terms_by_key.items():
        binary_coefficient = binary_terms_by_key[key].get_coeff()
        if not np.isclose(binary_coefficient, pauli_term.coefficient):
            raise ValueError(
                "Fast/ICS coefficients differ for {}: {} vs {}.".format(
                    pauli_term.word,
                    pauli_term.coefficient,
                    binary_coefficient,
                )
            )

    keys_by_index = {term.index: binary_tuple_for_term(term) for term in measurable_terms}
    ics_covariances = {
        (keys_by_index[left_index], keys_by_index[right_index]): covariance
        for (left_index, right_index), covariance in covariances.items()
    }
    return binary_terms_by_key, pauli_terms_by_key, ics_covariances


def pauli_groups_to_binary_groups(groups, binary_terms_by_key):
    return [
        [binary_terms_by_key[binary_tuple_for_term(term)] for term in group]
        for group in groups
    ]


def binary_groups_to_pauli_groups(binary_groups, pauli_terms_by_key):
    pauli_groups = []
    for group in binary_groups:
        binary_terms = group.binary_terms if isinstance(group, BinaryHamiltonian) else list(group)
        pauli_group = []
        for binary_term in binary_terms:
            base_term = pauli_terms_by_key[binary_term.binary_tuple()]
            pauli_group.append(
                PauliTerm(
                    index=base_term.index,
                    pauli_tuple=base_term.pauli_tuple,
                    ops=base_term.ops,
                    coefficient=clean_complex(binary_term.get_coeff()),
                    word=base_term.word,
                    source_order=base_term.source_order,
                )
            )
        pauli_groups.append(pauli_group)
    return pauli_groups


def group_variances_from_covariances(groups, covariances):
    variances = []
    for group in groups:
        variance = 0.0 + 0.0j
        for left in group:
            for right in group:
                variance += (
                    left.coefficient
                    * right.coefficient
                    * get_covariance(left, right, covariances)
                )
        variances.append(clean_report_variance(variance))
    return variances


def validate_ics_groups(groups, pauli_terms_by_key, condition):
    coefficient_totals = {key: 0.0 + 0.0j for key in pauli_terms_by_key}
    for group_index, group in enumerate(groups):
        group_keys = set()
        for term_index, term in enumerate(group):
            key = binary_tuple_for_term(term)
            if key not in coefficient_totals:
                raise ValueError("ICS returned an unknown Pauli term {}.".format(term.word))
            if key in group_keys:
                raise ValueError(
                    "ICS group {} contains duplicate term {}.".format(group_index, term.word)
                )
            group_keys.add(key)
            coefficient_totals[key] += term.coefficient
            for other in group[term_index + 1 :]:
                if not terms_compatible(term, other, condition):
                    raise ValueError(
                        "ICS group {} contains incompatible terms {} and {}.".format(
                            group_index,
                            term.word,
                            other.word,
                        )
                    )

    for key, base_term in pauli_terms_by_key.items():
        if not np.isclose(coefficient_totals[key], base_term.coefficient):
            raise ValueError(
                "ICS split coefficients for {} sum to {}, expected {}.".format(
                    base_term.word,
                    coefficient_totals[key],
                    base_term.coefficient,
                )
            )


def make_ics_result(method, groups, covariances, sample_ratios, runtime_s):
    variances = group_variances_from_covariances(groups, covariances)
    sample_ratios = [float(value) for value in np.asarray(sample_ratios).reshape(-1)]
    if len(sample_ratios) != len(groups):
        raise ValueError(
            "ICS returned {} sample ratios for {} groups.".format(
                len(sample_ratios),
                len(groups),
            )
        )
    if (
        not np.all(np.isfinite(sample_ratios))
        or any(value < -1.0e-12 for value in sample_ratios)
        or not np.isclose(sum(sample_ratios), 1.0)
    ):
        raise ValueError("ICS returned invalid sample ratios: {}.".format(sample_ratios))
    return MethodResult(
        method=method,
        groups=groups,
        variances=variances,
        eps_sq_m=eps_sq_m_from_variances(variances),
        sample_ratios=sample_ratios,
        runtime_s=runtime_s,
    )


def run_ics_methods(
    base_results,
    binary_terms_by_key,
    pauli_terms_by_key,
    covariances,
    ics_covariances,
    condition,
):
    base_results_by_method = {result.method: result for result in base_results}
    expected_base_methods = {"SI", "VarSI-G", "VarSI-O", "VarSI-R", "VarSI-OR"}
    if set(base_results_by_method) != expected_base_methods:
        raise ValueError(
            "Expected base methods {}, got {}.".format(
                sorted(expected_base_methods),
                sorted(base_results_by_method),
            )
        )

    ics_results_by_method = {}
    failed_methods = []
    for base_method in ("SI", "VarSI-G", "VarSI-O", "VarSI-OR", "VarSI-R"):
        base_result = base_results_by_method[base_method]
        method = "{}-ICS".format(base_result.method)
        attempt_start = time.perf_counter()
        helper_runtime_s = None
        try:
            initial_groups = pauli_groups_to_binary_groups(
                base_result.groups,
                binary_terms_by_key,
            )
            helper_start = time.perf_counter()
            binary_groups, sample_ratios = iterative_coefficient_splitting_from_groups(
                initial_groups,
                ics_covariances,
                condition=condition,
            )
            helper_runtime_s = time.perf_counter() - helper_start
            groups = binary_groups_to_pauli_groups(binary_groups, pauli_terms_by_key)
            if len(groups) != len(base_result.groups):
                raise ValueError(
                    "ICS changed the number of groups from {} to {}.".format(
                        len(base_result.groups),
                        len(groups),
                    )
            )
            validate_ics_groups(groups, pauli_terms_by_key, condition)
            ics_results_by_method[method] = make_ics_result(
                method,
                groups,
                covariances,
                sample_ratios,
                helper_runtime_s,
            )
        except Exception as exc:
            failed_methods.append(
                {
                    "method": method,
                    "label": METHOD_LABELS[method],
                    "runtime_s": (
                        helper_runtime_s
                        if helper_runtime_s is not None
                        else time.perf_counter() - attempt_start
                    ),
                    "error": "{}: {}".format(type(exc).__name__, exc),
                }
            )

    result_order = (
        "SI",
        "SI-ICS",
        "VarSI-G",
        "VarSI-G-ICS",
        "VarSI-O",
        "VarSI-OR",
        "VarSI-O-ICS",
        "VarSI-OR-ICS",
        "VarSI-R",
        "VarSI-R-ICS",
    )
    all_results_by_method = dict(base_results_by_method)
    all_results_by_method.update(ics_results_by_method)
    results = [all_results_by_method[method] for method in result_order if method in all_results_by_method]
    return results, failed_methods


def direct_group_variances(groups, state_vector, n_qubits):
    state_vector = np.asarray(state_vector, dtype=complex).reshape(-1)
    variances = []
    for group in groups:
        operator = QubitOperator()
        for term in group:
            operator += QubitOperator(term.pauli_tuple, term.coefficient)
        value = operator_variance(
            get_sparse_operator(operator, n_qubits=n_qubits),
            state_vector,
        )
        variances.append(clean_report_variance(value))
    return variances


def hamiltonian_expectation(terms, single_expectations):
    value = 0.0 + 0.0j
    for term in terms:
        if term.pauli_tuple:
            value += term.coefficient * single_expectations[term.index]
        else:
            value += term.coefficient
    return clean_real(value, tiny=1.0e-7)


def print_result(result, wfn_label, report_label):
    print("")
    print("{}:".format(METHOD_LABELS[result.method]))
    print("  eps^2 M(wfn={})={:.12g}".format(wfn_label, result.eps_sq_m))
    if report_label == "FCI":
        print("  eps^2 M(FCI)={:.12g}".format(result.report_eps_sq_m))
    else:
        print(
            "  eps^2 M(report={})={:.12g}".format(
                report_label,
                result.report_eps_sq_m,
            )
        )
    print("  Number of groups={}".format(len(result.groups)))
    print("  Compatible groups=True")
    print("  Runtime (s)={:.6f}".format(result.runtime_s))
    if result.accepted_moves is not None:
        print("  Accepted moves={}".format(result.accepted_moves))


def print_group_contents(result):
    print("")
    print("{} groups:".format(METHOD_LABELS[result.method]))
    for group_index, group in enumerate(result.groups):
        if result.method.endswith("-ICS"):
            terms = ["{}*{}".format(term.coefficient, term.word) for term in group]
        else:
            terms = [term.word for term in group]
        print("  Group {}: {}".format(group_index, ", ".join(terms)))


def run_loaded_molecule(mol_name, tf, args):
    _, qubit_operator, tequila_hamiltonian = load_loaded_hamiltonian(
        mol_name,
        tf=tf,
        prefix=args.prefix,
    )
    n_qubits = int(count_qubits(qubit_operator))
    terms = make_terms(qubit_operator, n_qubits)
    measurable_terms = [term for term in terms if term.pauli_tuple]

    print("")
    print("=" * 80)
    print("Molecule={}".format(mol_name), flush=True)
    print("Hamiltonian source=ham_lib/{}_fer.bin".format(mol_name), flush=True)
    print("Transformation={}".format(tf), flush=True)
    print(
        "Number of Pauli products to measure: {}".format(len(measurable_terms)),
        flush=True,
    )

    report_method = args.wfn if args.report_wfn == "SAME" else "FCI"
    needs_fci = args.wfn == "FCI" or report_method == "FCI"
    if needs_fci and n_qubits > args.fci_max_qubits:
        raise RuntimeError(
            "FCI requested for {} qubits, which is above --fci-max-qubits {}. "
            "Use --wfn CISD --report-wfn SAME for a CISD-only loaded-Hamiltonian "
            "run, or raise --fci-max-qubits if you really want sparse FCI.".format(
                n_qubits,
                args.fci_max_qubits,
            )
        )
    sparse_hamiltonian = (
        get_sparse_operator(qubit_operator, n_qubits=n_qubits)
        if needs_fci
        else None
    )
    energy, variance_state, metadata = get_loaded_variance_wavefunction(
        mol_name,
        qubit_operator,
        args.wfn,
        sparse_hamiltonian,
        args.spin_ordering,
        args.cisd_validation_tol,
        tf,
    )
    variance_state = np.asarray(variance_state, dtype=complex).reshape(-1)
    print("{} Energy={:.16g}".format(args.wfn, clean_real(energy, tiny=1.0e-7)), flush=True)
    print_cisd_validation(metadata)

    if report_method == args.wfn:
        report_state = variance_state
    else:
        report_energy, report_state, _ = get_loaded_variance_wavefunction(
            mol_name,
            qubit_operator,
            report_method,
            sparse_hamiltonian,
            args.spin_ordering,
            args.cisd_validation_tol,
            tf,
        )
        report_state = np.asarray(report_state, dtype=complex).reshape(-1)
        print(
            "{} Energy={:.16g}".format(
                report_method,
                clean_real(report_energy, tiny=1.0e-7),
            ),
            flush=True,
        )

    print("Compatibility condition={}".format(args.condition), flush=True)
    print("Number of qubits={}".format(n_qubits), flush=True)

    cov_workers = 1 if args.serial_cov_dict else (args.cov_workers or default_cov_workers())
    action_matrix_gib = len(measurable_terms) * (2**n_qubits) * 16 / (1024**3)
    print(
        "Building covariance matrix with {} worker(s); Pauli-action matrix={:.3f} GiB...".format(
            cov_workers,
            action_matrix_gib,
        ),
        flush=True,
    )
    covariance_start = time.perf_counter()
    covariances, single_expectations = build_covariance_dictionary(
        measurable_terms,
        variance_state,
        n_qubits,
        cov_workers,
        args.cov_chunksize,
    )
    covariance_runtime = time.perf_counter() - covariance_start
    action_energy = hamiltonian_expectation(terms, single_expectations)
    if abs(action_energy - clean_real(energy, tiny=1.0e-7)) > 1.0e-7:
        raise ValueError(
            "Pauli-action energy {} does not match reference energy {}.".format(
                action_energy,
                energy,
            )
        )
    covariance_label = (
        "fast serial Pauli-action builder"
        if cov_workers == 1
        else "fast parallel Pauli-action builder ({} workers)".format(cov_workers)
    )
    print(
        "{} covariance dictionary: {} entries built with {} in {:.6f} s".format(
            args.wfn,
            len(covariances),
            covariance_label,
            covariance_runtime,
        ),
        flush=True,
    )
    print("Pauli-action energy check={:.16g}".format(action_energy), flush=True)
    if report_method == args.wfn:
        print(
            "Reporting eps^2 M: reused {} covariance dictionary".format(report_method),
            flush=True,
        )
    else:
        print(
            "Reporting eps^2 M: direct group variances from {} wavefunction; "
            "no reporting covariance dictionary built.".format(report_method),
            flush=True,
        )

    context_start = time.perf_counter()
    context = build_fast_context(measurable_terms, covariances, args.condition)
    context_runtime = time.perf_counter() - context_start
    context_mib = (
        context.scaled_covariances.nbytes
        + context.compatible.nbytes
        + context.single_variances.nbytes
    ) / (1024**2)
    print(
        "Fast context built once and shared by all methods: {:.3f} MiB in {:.6f} s".format(
            context_mib,
            context_runtime,
        ),
        flush=True,
    )

    base_results = run_fast_varsi_methods(context, args)
    if args.no_ics:
        results = base_results
        failed_ics_methods = []
    else:
        binary_terms_by_key, pauli_terms_by_key, ics_covariances = build_ics_bridge(
            tequila_hamiltonian,
            measurable_terms,
            covariances,
        )
        results, failed_ics_methods = run_ics_methods(
            base_results,
            binary_terms_by_key,
            pauli_terms_by_key,
            covariances,
            ics_covariances,
            args.condition,
        )
    for result in results:
        if report_method == args.wfn:
            result.report_eps_sq_m = result.eps_sq_m
        else:
            result.report_eps_sq_m = eps_sq_m_from_variances(
                direct_group_variances(result.groups, report_state, n_qubits)
            )
        print_result(result, args.wfn, report_method)

    if failed_ics_methods:
        print("")
        print("Skipped methods:")
        for failed in failed_ics_methods:
            print(
                "  {} failed after {:.6f} s: {}".format(
                    failed["label"],
                    failed["runtime_s"],
                    failed["error"],
                )
            )

    print("")
    if report_method == "FCI":
        print("Ranking by eps^2 M(FCI) (lowest to highest):")
    else:
        print(
            "Ranking by eps^2 M(report={}) (lowest to highest):".format(
                report_method
            )
        )
    for rank, result in enumerate(
        sorted(results, key=lambda item: item.report_eps_sq_m),
        start=1,
    ):
        print(
            "  {}. {}: {:.12g} Groups: {}".format(
                rank,
                METHOD_LABELS[result.method],
                result.report_eps_sq_m,
                len(result.groups),
            )
        )

    if args.print_groups:
        for result in results:
            print_group_contents(result)

    return results


def main(argv=None):
    args = parse_args(argv)
    failures = []
    for mol_name in args.molecules:
        for tf in args.tf:
            try:
                run_loaded_molecule(mol_name, tf, args)
            except Exception as exc:
                if args.stop_on_error:
                    raise
                failures.append((mol_name, tf, exc))
                print("")
                print("=" * 80)
                print(
                    "Molecule={} tf={} failed: {}: {}".format(
                        mol_name,
                        tf,
                        type(exc).__name__,
                        exc,
                    )
                )

    if failures:
        print("")
        print("Failed loaded molecules:")
        for mol_name, tf, exc in failures:
            print(
                "  {} tf={}: {}: {}".format(
                    mol_name,
                    tf,
                    type(exc).__name__,
                    exc,
                )
            )


if __name__ == "__main__":
    main()

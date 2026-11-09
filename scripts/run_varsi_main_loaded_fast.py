#!/usr/bin/env python3
"""Fast main-table VarSI run from loaded and stretched Hamiltonians.

Equilibrium LiH, BeH2, H2O, and NH3 Hamiltonians are loaded from
``ham_lib/*_fer.bin``.  The stretched BeH2 and H2O Hamiltonians are generated
with the geometries used from the paper.  
Groupings are built with CISD covariances and all reported eps^2 M values are evaluated from a
separately built FCI covariance dictionary.

The script writes computation results only. Needs the fast_VarSI_loaded.py, 
hamiltonians_varsi.py files in the same folder where it is being run, as well as
the ham_lib folder.
"""

from __future__ import annotations

# Prevent nested BLAS oversubscription in covariance worker processes.  The
# parent process explicitly opens all available BLAS threads for Gram products,
# and ICS divides the same CPU budget among concurrent jobs.
import os

for _thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/varsi_mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/varsi_cache")

import argparse
import ctypes
import csv
import gc
import glob
import math
import multiprocessing as mp
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from openfermion.linalg import get_ground_state, get_sparse_operator
from openfermion.utils import count_qubits
from pyscf import lib as pyscf_lib
from tequila.grouping.binary_rep import BinaryHamiltonian


SCRIPT_DIR = Path(__file__).resolve().parent
if (SCRIPT_DIR / "fast_VarSI_loaded.py").is_file():
    REPO_DIR = SCRIPT_DIR
elif (SCRIPT_DIR.parent / "fast_VarSI_loaded.py").is_file():
    REPO_DIR = SCRIPT_DIR.parent
else:
    raise RuntimeError("Could not locate fast_VarSI_loaded.py beside this script or its parent.")
sys.path.insert(0, str(REPO_DIR))

import fast_VarSI_loaded as fast
import hamiltonians_varsi as hv
from gflow_vqe.utils import get_variance_wavefunction


def _numpy_openblas_controls():
    """Return NumPy's OpenBLAS get/set functions when available."""

    candidates = glob.glob(
        str(Path(np.__file__).resolve().parent.parent / "numpy.libs" / "*openblas*.so")
    )
    for candidate in candidates:
        library = ctypes.CDLL(candidate)
        for suffix in ("64_", "_64_", ""):
            getter = getattr(library, "scipy_openblas_get_num_threads{}".format(suffix), None)
            setter = getattr(library, "scipy_openblas_set_num_threads{}".format(suffix), None)
            if getter is None or setter is None:
                getter = getattr(library, "openblas_get_num_threads{}".format(suffix), None)
                setter = getattr(library, "openblas_set_num_threads{}".format(suffix), None)
            if getter is not None and setter is not None:
                getter.restype = ctypes.c_int
                setter.argtypes = [ctypes.c_int]
                setter.restype = None
                return library, getter, setter
    return None, None, None


_OPENBLAS_LIBRARY, _OPENBLAS_GET_THREADS, _OPENBLAS_SET_THREADS = _numpy_openblas_controls()


@contextmanager
def blas_thread_limit(limit: int):
    """Temporarily assign NumPy's OpenBLAS pool without an extra dependency."""

    if _OPENBLAS_GET_THREADS is None:
        yield
        return
    previous = int(_OPENBLAS_GET_THREADS())
    _OPENBLAS_SET_THREADS(int(limit))
    try:
        yield
    finally:
        _OPENBLAS_SET_THREADS(previous)


def clean_fast_real(value, tiny=1.0e-9):
    """Canonical VarSI real coercion without discarding small real values."""

    value = complex(value)
    if not np.isfinite(value.real) or not np.isfinite(value.imag):
        raise ValueError("Expected a finite real value, got {}.".format(value))
    if abs(value.imag) > tiny:
        raise ValueError("Expected a real value, got {}.".format(value))
    return float(value.real)


def clean_fast_variance(value, tiny=1.0e-9):
    """Match VarSI.py: preserve every positive variance and clip only noise below zero."""

    real_value = clean_fast_real(value, tiny=tiny)
    if real_value < 0.0 and abs(real_value) < tiny:
        return 0.0
    if real_value < 0.0:
        raise ValueError("Computed a negative variance: {}.".format(real_value))
    return real_value


# The fast kernels use module-level cleaners.  These replacements retain their
# cached/vectorized implementation while matching the numerical contract of
# VarSI.py; the original fast cleaners zeroed genuine small positive values.
fast.clean_real = clean_fast_real
fast.clean_variance = clean_fast_variance


def make_terms_in_source_order(qubit_operator, n_qubits):
    """Build fast PauliTerm objects in the Hamiltonian's native insertion order."""

    terms = []
    for index, (pauli_tuple, coefficient) in enumerate(qubit_operator.terms.items()):
        pauli_tuple = tuple((int(qubit), str(pauli)) for qubit, pauli in pauli_tuple)
        pauli_by_qubit = dict(pauli_tuple)
        terms.append(
            fast.PauliTerm(
                index=index,
                pauli_tuple=pauli_tuple,
                ops=tuple(pauli_by_qubit.get(qubit, "I") for qubit in range(n_qubits)),
                coefficient=fast.clean_complex(coefficient),
                word=fast.pauli_word(pauli_tuple),
                source_order=index,
            )
        )
    return terms


SCHEMA_VERSION = "2"
CONDITION = "fc"
SWEEP_TARGETS = (100, 500)
METRIC_KEYS = (
    "si",
    "varsi_g",
    "varsi_o",
    "varsi_r_100",
    "varsi_r_500",
    "varsi_or_100",
    "varsi_or_500",
    "si_ics",
    "varsi_g_ics",
    "varsi_o_ics",
    "varsi_r_ics_100",
    "varsi_r_ics_500",
    "varsi_or_ics_100",
    "varsi_or_ics_500",
)


@dataclass(frozen=True)
class SystemSpec:
    system_id: str
    table_label: str
    molecule: str
    stretched: bool
    bond_distance: str
    geometry_label: str


SYSTEMS = (
    SystemSpec("lih", "LiH", "lih", False, "1", "loaded_equilibrium"),
    SystemSpec("beh2", "BeH2", "beh2", False, "1", "loaded_equilibrium"),
    SystemSpec("beh2_stretched", "BeH2(s)", "beh2", True, "3.0", "R3_linear_stretched"),
    SystemSpec("h2o", "H2O", "h2o", False, "1", "loaded_equilibrium"),
    SystemSpec("h2o_stretched", "H2O(s)", "h2o", True, "2.2", "R2.2_angle107.6_stretched"),
    SystemSpec("nh3", "NH3", "nh3", False, "1", "loaded_equilibrium"),
)
SYSTEM_BY_ID = {system.system_id: system for system in SYSTEMS}
EXPECTED_DIMENSIONS = {
    "lih": (12, 630),
    "beh2": (14, 665),
    "beh2_stretched": (14, 665),
    "h2o": (14, 1085),
    "h2o_stretched": (14, 1085),
    "nh3": (16, 3608),
}


@dataclass
class HamiltonianBundle:
    qubit_operator: object
    tequila_hamiltonian: object
    tequila_molecule: object | None
    source: str


@dataclass
class GroupingRun:
    key: str
    groups: list
    grouping_eps_sq_m: float
    runtime_s: float
    accepted_moves: int | None = None
    sample_ratios: list[float] | None = None


def available_cpus() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Fast CISD-grouping/FCI-report VarSI run for loaded equilibrium and "
            "generated stretched main-table Hamiltonians."
        )
    )
    parser.add_argument(
        "--prefix",
        default=str(SCRIPT_DIR),
        help="Directory containing ham_lib (default: script directory).",
    )
    parser.add_argument(
        "--output-csv",
        default=str(SCRIPT_DIR / "varsi_main_loaded_fast.csv"),
        help="CSV output path.",
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        choices=tuple(SYSTEM_BY_ID),
        default=tuple(system.system_id for system in SYSTEMS),
        help="Subset of systems to run.",
    )
    parser.add_argument(
        "--mappings",
        nargs="+",
        type=lambda value: str(value).upper(),
        choices=("JW", "BK"),
        default=("JW", "BK"),
        help="Mappings to run, in the requested order.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=available_cpus(),
        help="Covariance action workers and Gram-product CPU threads (default: all available CPUs).",
    )
    parser.add_argument(
        "--ics-workers",
        type=int,
        default=None,
        help="Concurrent ICS jobs (default: enough four-thread jobs to use all CPUs).",
    )
    parser.add_argument("--cov-chunksize", type=int, default=128)
    parser.add_argument("--cisd-validation-tol", type=float, default=1.0e-8)
    parser.add_argument("--scf-max-cycle", type=int, default=200)
    parser.add_argument("--force", action="store_true", help="Recompute successful CSV rows.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record a failed row and continue instead of stopping the batch.",
    )
    args = parser.parse_args(argv)
    args.prefix = str(Path(args.prefix).resolve())
    args.output_csv = str(Path(args.output_csv).resolve())
    if args.workers < 1:
        parser.error("--workers must be positive.")
    if args.cov_chunksize < 1:
        parser.error("--cov-chunksize must be positive.")
    if args.cisd_validation_tol <= 0.0:
        parser.error("--cisd-validation-tol must be positive.")
    if args.ics_workers is None:
        args.ics_workers = max(1, args.workers // 4)
    if args.ics_workers < 1:
        parser.error("--ics-workers must be positive.")
    return args


def stretched_spec(system: SystemSpec, mapping: str):
    if system.system_id == "beh2_stretched":
        geometry = hv.beh2_geometry(3.0)
    elif system.system_id == "h2o_stretched":
        geometry = hv.bent_xy2_geometry("O", "H", 2.2, 107.6)
    else:
        raise ValueError("{} is not a stretched system.".format(system.system_id))
    return hv.VarsiHamiltonianSpec(
        ham_type="main_loaded_fast",
        molecule="BeH2" if system.molecule == "beh2" else "H2O",
        geometry=geometry,
        bond_distance=system.bond_distance,
        basis_label="sto-3g",
        basis_set="sto-3g",
        mapping=mapping,
        multiplicity=1,
        spin_state="singlet",
        charge=0,
        frozen_core=False,
        geometry_label=system.geometry_label,
    )


def build_hamiltonian(system: SystemSpec, mapping: str, args) -> HamiltonianBundle:
    if not system.stretched:
        source_path = Path(args.prefix) / "ham_lib" / "{}_fer.bin".format(system.molecule)
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        _, qubit_operator, tequila_hamiltonian = fast.load_loaded_hamiltonian(
            system.molecule,
            tf=mapping.lower(),
            prefix=args.prefix,
        )
        return HamiltonianBundle(
            qubit_operator=qubit_operator,
            tequila_hamiltonian=tequila_hamiltonian,
            tequila_molecule=None,
            source="ham_lib/{}_fer.bin".format(system.molecule),
        )

    specification = stretched_spec(system, mapping)
    built = hv.build_all_electron_hamiltonian(
        specification,
        scf_max_cycle=args.scf_max_cycle,
        mapping_workers=args.workers,
    )
    return HamiltonianBundle(
        qubit_operator=built.qubit_operator,
        tequila_hamiltonian=built.tequila_hamiltonian,
        tequila_molecule=built.tequila_mol,
        source="generated:{}".format(system.geometry_label),
    )


def validate_state(state, n_qubits: int, label: str):
    state = np.asarray(state, dtype=complex).reshape(-1)
    expected_size = 2**n_qubits
    if state.size != expected_size:
        raise ValueError("{} state has size {}, expected {}.".format(label, state.size, expected_size))
    norm = float(np.vdot(state, state).real)
    if not np.isclose(norm, 1.0, rtol=0.0, atol=1.0e-9):
        raise ValueError("{} state norm is {:.16g}.".format(label, norm))
    return state


def build_cisd_reference(system, mapping, bundle, n_qubits, args):
    if system.stretched:
        sparse_hamiltonian = get_sparse_operator(bundle.qubit_operator, n_qubits=n_qubits)
        energy, state = get_variance_wavefunction(
            bundle.tequila_molecule,
            bundle.qubit_operator,
            method="CISD",
            sparse_hamiltonian=sparse_hamiltonian,
        )
        metadata = None
        del sparse_hamiltonian
    else:
        energy, state, metadata = fast.loaded_cisd_statevector(
            system.molecule,
            bundle.qubit_operator,
            n_qubits,
            spin_ordering="interleaved",
            validation_tol=args.cisd_validation_tol,
            tf=mapping.lower(),
        )
    return float(np.real(energy)), validate_state(state, n_qubits, "CISD"), metadata


def build_fci_reference(bundle, n_qubits):
    sparse_hamiltonian = get_sparse_operator(bundle.qubit_operator, n_qubits=n_qubits)
    energy, state = get_ground_state(sparse_hamiltonian)
    state = validate_state(state, n_qubits, "FCI")
    residual = float(np.linalg.norm(sparse_hamiltonian.dot(state) - energy * state))
    del sparse_hamiltonian
    if residual > 1.0e-7:
        raise ValueError("FCI residual is too large: {:.3e}.".format(residual))
    return float(np.real(energy)), state, residual


def build_covariances(terms, all_terms, state, energy, n_qubits, args, label):
    """Use process-parallel Pauli actions and all-CPU BLAS for the Gram product."""

    started = time.perf_counter()
    actions = fast.build_action_matrix(
        terms,
        state,
        n_qubits,
        max_workers=args.workers,
        chunksize=args.cov_chunksize,
    )
    with blas_thread_limit(args.workers):
        single_values = actions.dot(state.conjugate())
        gram = actions.conjugate().dot(actions.T)
    del actions

    single_expectations = {
        term.index: fast.clean_complex(single_values[position])
        for position, term in enumerate(terms)
    }
    del single_values
    covariances = {}
    for left_position, left in enumerate(terms):
        for right_position in range(left_position, len(terms)):
            right = terms[right_position]
            if not fast.terms_fully_commute(left, right):
                continue
            covariances[(left.index, right.index)] = fast.clean_complex(
                gram[left_position, right_position]
                - single_expectations[left.index] * single_expectations[right.index]
            )
    del gram
    runtime = time.perf_counter() - started
    action_energy = fast.hamiltonian_expectation(all_terms, single_expectations)
    if abs(action_energy - energy) > 1.0e-7:
        raise ValueError(
            "{} Pauli-action energy {:.16g} differs from reference {:.16g}.".format(
                label,
                action_energy,
                energy,
            )
        )
    print(
        "{} covariance dictionary: {:,} entries in {:.3f} s; energy check={:.12f}".format(
            label,
            len(covariances),
            runtime,
            action_energy,
        ),
        flush=True,
    )
    return covariances, runtime


def record_position_grouping(key, context, positions, variances, runtime_s, accepted_moves=None):
    fast.validate_position_groups(context, positions)
    groups = [[context.terms[position] for position in group] for group in positions]
    return GroupingRun(
        key=key,
        groups=groups,
        grouping_eps_sq_m=fast.eps_sq_m_from_variances(variances),
        runtime_s=float(runtime_s),
        accepted_moves=accepted_moves,
    )


def build_nonoverlapping_groupings(context):
    runs = {}

    started = time.perf_counter()
    si_positions, si_variances = fast.fast_sorted_insertion(context)
    runs["si"] = record_position_grouping(
        "si", context, si_positions, si_variances, time.perf_counter() - started
    )

    started = time.perf_counter()
    g_positions, g_variances = fast.fast_varsi_greedy(context)
    runs["varsi_g"] = record_position_grouping(
        "varsi_g", context, g_positions, g_variances, time.perf_counter() - started
    )

    started = time.perf_counter()
    o_positions, o_variances = fast.fast_varsi_ordered(context, consider_new_groups=False)
    runs["varsi_o"] = record_position_grouping(
        "varsi_o", context, o_positions, o_variances, time.perf_counter() - started
    )

    started = time.perf_counter()
    r100_positions, r100_variances, r100_moves = fast.fast_refine_groups(
        context,
        si_positions,
        max_sweeps=100,
        allow_new_groups=False,
    )
    r100_runtime = time.perf_counter() - started
    runs["varsi_r_100"] = record_position_grouping(
        "varsi_r_100",
        context,
        r100_positions,
        r100_variances,
        r100_runtime,
        r100_moves,
    )
    started = time.perf_counter()
    r500_positions, r500_variances, r_more_moves = fast.fast_refine_groups(
        context,
        r100_positions,
        max_sweeps=400,
        allow_new_groups=False,
    )
    r500_runtime = r100_runtime + time.perf_counter() - started
    runs["varsi_r_500"] = record_position_grouping(
        "varsi_r_500",
        context,
        r500_positions,
        r500_variances,
        r500_runtime,
        r100_moves + r_more_moves,
    )

    started = time.perf_counter()
    or100_positions, or100_variances, or100_moves = fast.fast_refine_groups(
        context,
        o_positions,
        max_sweeps=100,
        allow_new_groups=False,
    )
    or100_runtime = time.perf_counter() - started
    runs["varsi_or_100"] = record_position_grouping(
        "varsi_or_100",
        context,
        or100_positions,
        or100_variances,
        or100_runtime,
        or100_moves,
    )
    started = time.perf_counter()
    or500_positions, or500_variances, or_more_moves = fast.fast_refine_groups(
        context,
        or100_positions,
        max_sweeps=400,
        allow_new_groups=False,
    )
    or500_runtime = or100_runtime + time.perf_counter() - started
    runs["varsi_or_500"] = record_position_grouping(
        "varsi_or_500",
        context,
        or500_positions,
        or500_variances,
        or500_runtime,
        or100_moves + or_more_moves,
    )
    return runs


_ICS_JOBS = None
_ICS_BINARY_TERMS = None
_ICS_PAULI_TERMS = None
_ICS_COVARIANCES = None
_ICS_BINARY_COVARIANCES = None
_ICS_CONDITION = None
_ICS_BLAS_THREADS = 1


def _ics_worker(job_id):
    base_key, base_groups = _ICS_JOBS[job_id]
    initial_groups = fast.pauli_groups_to_binary_groups(base_groups, _ICS_BINARY_TERMS)
    started = time.perf_counter()
    with blas_thread_limit(_ICS_BLAS_THREADS):
        binary_groups, sample_ratios = fast.iterative_coefficient_splitting_from_groups(
            initial_groups,
            _ICS_BINARY_COVARIANCES,
            condition=_ICS_CONDITION,
        )
    runtime = time.perf_counter() - started
    groups = fast.binary_groups_to_pauli_groups(binary_groups, _ICS_PAULI_TERMS)
    fast.validate_ics_groups(groups, _ICS_PAULI_TERMS, _ICS_CONDITION)
    variances = fast.group_variances_from_covariances(groups, _ICS_COVARIANCES)
    ratios = [float(value) for value in np.asarray(sample_ratios).reshape(-1)]
    if len(ratios) != len(groups) or not np.isclose(sum(ratios), 1.0):
        raise ValueError("ICS returned invalid sample ratios for {}.".format(base_key))
    return base_key, groups, ratios, fast.eps_sq_m_from_variances(variances), runtime


def ordered_group_signature(groups):
    return tuple(tuple(term.index for term in group) for group in groups)


def build_ics_groupings(base_runs, tequila_hamiltonian, terms, covariances, args):
    global _ICS_JOBS
    global _ICS_BINARY_TERMS
    global _ICS_PAULI_TERMS
    global _ICS_COVARIANCES
    global _ICS_BINARY_COVARIANCES
    global _ICS_CONDITION
    global _ICS_BLAS_THREADS

    binary_terms, pauli_terms, binary_covariances = fast.build_ics_bridge(
        tequila_hamiltonian,
        terms,
        covariances,
    )
    signature_to_job = {}
    aliases = {}
    jobs = {}
    for base_key, run in base_runs.items():
        signature = ordered_group_signature(run.groups)
        if signature not in signature_to_job:
            job_id = len(jobs)
            signature_to_job[signature] = job_id
            jobs[job_id] = (base_key, run.groups)
        aliases[base_key] = signature_to_job[signature]

    worker_count = min(args.ics_workers, len(jobs), args.workers)
    blas_threads = max(1, args.workers // worker_count)
    print(
        "ICS: {} unique seeds, {} fork workers x {} BLAS threads".format(
            len(jobs), worker_count, blas_threads
        ),
        flush=True,
    )
    _ICS_JOBS = jobs
    _ICS_BINARY_TERMS = binary_terms
    _ICS_PAULI_TERMS = pauli_terms
    _ICS_COVARIANCES = covariances
    _ICS_BINARY_COVARIANCES = binary_covariances
    _ICS_CONDITION = CONDITION
    _ICS_BLAS_THREADS = blas_threads

    completed_by_job = {}
    fork_context = mp.get_context("fork")
    try:
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=fork_context) as executor:
            futures = {executor.submit(_ics_worker, job_id): job_id for job_id in jobs}
            for future in as_completed(futures):
                job_id = futures[future]
                result = future.result()
                completed_by_job[job_id] = result
                print(
                    "  ICS seed {} ({}) completed in {:.3f} s".format(
                        job_id,
                        result[0],
                        result[4],
                    ),
                    flush=True,
                )
    finally:
        _ICS_JOBS = None
        _ICS_BINARY_TERMS = None
        _ICS_PAULI_TERMS = None
        _ICS_COVARIANCES = None
        _ICS_BINARY_COVARIANCES = None
        _ICS_CONDITION = None

    ics_runs = {}
    for base_key, job_id in aliases.items():
        _, groups, ratios, eps_sq_m, runtime = completed_by_job[job_id]
        if base_key == "si":
            key = "si_ics"
        elif base_key == "varsi_g":
            key = "varsi_g_ics"
        elif base_key == "varsi_o":
            key = "varsi_o_ics"
        elif base_key.startswith("varsi_r_"):
            key = base_key.replace("varsi_r_", "varsi_r_ics_")
        elif base_key.startswith("varsi_or_"):
            key = base_key.replace("varsi_or_", "varsi_or_ics_")
        else:
            raise ValueError("Unexpected base key {}.".format(base_key))
        ics_runs[key] = GroupingRun(
            key=key,
            groups=groups,
            grouping_eps_sq_m=eps_sq_m,
            runtime_s=runtime,
            sample_ratios=ratios,
        )
    return ics_runs


def evaluate_groupings(groupings, fci_covariances):
    results = {}
    for key in METRIC_KEYS:
        run = groupings[key]
        variances = fast.group_variances_from_covariances(run.groups, fci_covariances)
        value = fast.eps_sq_m_from_variances(variances)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("Invalid FCI result for {}: {}.".format(key, value))
        results[key] = value
    return results


def run_case(system: SystemSpec, mapping: str, args):
    case_started = time.perf_counter()
    print("", flush=True)
    print("=" * 88, flush=True)
    print("Starting {} / {}".format(system.system_id, mapping), flush=True)
    bundle = build_hamiltonian(system, mapping, args)
    n_qubits = int(count_qubits(bundle.qubit_operator))
    all_terms = make_terms_in_source_order(bundle.qubit_operator, n_qubits)
    terms = [term for term in all_terms if term.pauli_tuple]
    expected = EXPECTED_DIMENSIONS[system.system_id]
    if (n_qubits, len(terms)) != expected:
        raise ValueError(
            "Unexpected dimensions for {} {}: got {}, expected {}.".format(
                system.system_id, mapping, (n_qubits, len(terms)), expected
            )
        )
    print(
        "Source={} | qubits={} | measurable Pauli words={}".format(
            bundle.source, n_qubits, len(terms)
        ),
        flush=True,
    )

    cisd_energy, cisd_state, cisd_metadata = build_cisd_reference(
        system, mapping, bundle, n_qubits, args
    )
    print("CISD energy={:.12f}".format(cisd_energy), flush=True)
    cisd_covariances, cisd_covariance_runtime = build_covariances(
        terms,
        all_terms,
        cisd_state,
        cisd_energy,
        n_qubits,
        args,
        "CISD",
    )
    del cisd_state

    context_started = time.perf_counter()
    context = fast.build_fast_context(terms, cisd_covariances, CONDITION)
    context_runtime = time.perf_counter() - context_started
    base_runs = build_nonoverlapping_groupings(context)
    ics_runs = build_ics_groupings(
        base_runs,
        bundle.tequila_hamiltonian,
        terms,
        cisd_covariances,
        args,
    )
    groupings = {**base_runs, **ics_runs}
    if set(groupings) != set(METRIC_KEYS):
        raise ValueError(
            "Missing grouping methods: {}.".format(sorted(set(METRIC_KEYS) - set(groupings)))
        )
    cisd_covariance_entries = len(cisd_covariances)
    del context, cisd_covariances, base_runs, ics_runs
    gc.collect()

    fci_energy, fci_state, fci_residual = build_fci_reference(bundle, n_qubits)
    print("FCI energy={:.12f}; residual={:.3e}".format(fci_energy, fci_residual), flush=True)
    fci_covariances, fci_covariance_runtime = build_covariances(
        terms,
        all_terms,
        fci_state,
        fci_energy,
        n_qubits,
        args,
        "FCI",
    )
    del fci_state
    report_results = evaluate_groupings(groupings, fci_covariances)
    fci_covariance_entries = len(fci_covariances)
    del fci_covariances
    gc.collect()

    row = {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "error": "",
        "system_id": system.system_id,
        "system_label": system.table_label,
        "bond_distance_angstrom": system.bond_distance,
        "mapping": mapping,
        "n_qubits": n_qubits,
        "n_pauli": len(terms),
        "cisd_energy": cisd_energy,
        "fci_energy": fci_energy,
        "total_runtime_s": time.perf_counter() - case_started,
    }
    for key in METRIC_KEYS:
        row[key] = report_results[key]
        row["{}_groups".format(key)] = len(groupings[key].groups)
    for key in ("varsi_r_100", "varsi_r_500", "varsi_or_100", "varsi_or_500"):
        row["{}_accepted_moves".format(key)] = groupings[key].accepted_moves
    print(
        "Completed {} / {} in {:.2f} min".format(
            system.system_id, mapping, row["total_runtime_s"] / 60.0
        ),
        flush=True,
    )
    return row


BASE_FIELDS = (
    "schema_version",
    "status",
    "error",
    "system_id",
    "system_label",
    "bond_distance_angstrom",
    "mapping",
    "n_qubits",
    "n_pauli",
    "cisd_energy",
    "fci_energy",
    "total_runtime_s",
)
CSV_FIELDS = list(BASE_FIELDS)
for _key in METRIC_KEYS:
    CSV_FIELDS.extend((_key, "{}_groups".format(_key)))
    if _key in ("varsi_r_100", "varsi_r_500", "varsi_or_100", "varsi_or_500"):
        CSV_FIELDS.append("{}_accepted_moves".format(_key))


def read_existing_rows(path: Path):
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    for row in rows:
        # Schema v1 contains the same final results plus verbose diagnostics.
        # Projecting it through CSV_FIELDS is lossless for the compact schema.
        if row.get("schema_version") == "1":
            row["schema_version"] = SCHEMA_VERSION
    return {(row.get("system_id"), row.get("mapping")): row for row in rows}


def row_sort_key(row):
    mapping_order = {"JW": 0, "BK": 1}
    system_order = {system.system_id: index for index, system in enumerate(SYSTEMS)}
    return (
        mapping_order.get(row.get("mapping"), 99),
        system_order.get(row.get("system_id"), 99),
    )


def write_rows_atomic(path: Path, rows_by_key):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(rows_by_key.values(), key=row_sort_key):
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(temporary, path)


def failed_row(system, mapping, args, exc, elapsed):
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "failed",
        "error": "{}: {}".format(type(exc).__name__, exc),
        "system_id": system.system_id,
        "system_label": system.table_label,
        "bond_distance_angstrom": system.bond_distance,
        "mapping": mapping,
        "total_runtime_s": elapsed,
    }


def validate_complete_rows(rows_by_key, requested_systems, mappings):
    expected = {(system_id, mapping) for mapping in mappings for system_id in requested_systems}
    successful = {
        key
        for key, row in rows_by_key.items()
        if row.get("status") == "ok" and row.get("schema_version") == SCHEMA_VERSION
    }
    missing = expected - successful
    if missing:
        raise RuntimeError("The CSV is missing successful rows: {}.".format(sorted(missing)))


def main(argv=None):
    args = parse_args(argv)
    pyscf_lib.num_threads(args.workers)
    output_path = Path(args.output_csv)
    rows_by_key = read_existing_rows(output_path)
    if rows_by_key:
        # Also serves as an in-place v1-to-v2 compaction for completed runs.
        write_rows_atomic(output_path, rows_by_key)
    print("Fast main-table VarSI batch", flush=True)
    print("Run directory={}".format(Path.cwd()), flush=True)
    print("Hamiltonian prefix={}".format(args.prefix), flush=True)
    print("Output CSV={}".format(output_path), flush=True)
    print("CPU budget={} (all available={})".format(args.workers, available_cpus()), flush=True)
    print("Covariance state=CISD; report covariance state=FCI; condition=fc", flush=True)

    for mapping in args.mappings:
        for system_id in args.systems:
            system = SYSTEM_BY_ID[system_id]
            key = (system_id, mapping)
            existing = rows_by_key.get(key)
            if (
                not args.force
                and existing is not None
                and existing.get("status") == "ok"
                and existing.get("schema_version") == SCHEMA_VERSION
            ):
                print("Skipping completed {} / {}".format(system_id, mapping), flush=True)
                continue
            started = time.perf_counter()
            try:
                row = run_case(system, mapping, args)
            except Exception as exc:
                row = failed_row(system, mapping, args, exc, time.perf_counter() - started)
                rows_by_key[key] = row
                write_rows_atomic(output_path, rows_by_key)
                print("FAILED {} / {}: {}: {}".format(system_id, mapping, type(exc).__name__, exc))
                traceback.print_exc()
                if not args.continue_on_error:
                    raise
            else:
                rows_by_key[key] = row
                write_rows_atomic(output_path, rows_by_key)
                print("Checkpointed {} / {} to CSV".format(system_id, mapping), flush=True)

    validate_complete_rows(rows_by_key, args.systems, args.mappings)
    print("", flush=True)
    print("All requested calculations completed successfully.", flush=True)
    print("CSV={}".format(output_path), flush=True)


if __name__ == "__main__":
    main()

import argparse
import glob
import os
import pickle
import re
import tempfile
from itertools import combinations

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import numpy as np
from openfermion.linalg import get_ground_state, get_sparse_operator
from openfermion.transforms import bravyi_kitaev, bravyi_kitaev_code, jordan_wigner
from openfermion.utils import count_qubits
from tequila.grouping.binary_rep import BinaryHamiltonian
from tequila.hamiltonian import QubitHamiltonian

import VarSI as varsi


LOADED_MOLECULE_ORDER = ("nh3",)
DEFAULT_MOLECULE_ORDER = ("nh3",)

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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run the VarSI grouping/reporting pipeline on the loaded fermionic "
            "Hamiltonians in ham_lib."
        )
    )
    parser.add_argument(
        "molecules",
        nargs="*",
        type=lambda s: str(s).lower(),
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
        type=lambda s: str(s).lower(),
        default="both",
        choices=("both", "bk", "jw"),
        help="Qubit transform for loaded fermionic Hamiltonians: both, bk, or jw (default: both).",
    )
    parser.add_argument(
        "--wfn",
        type=lambda s: str(s).upper(),
        default="FCI",
        choices=("FCI", "HF", "CISD"),
        help="Wavefunction used to build the covariance dictionary (default: FCI).",
    )
    parser.add_argument(
        "--report-wfn",
        type=lambda s: str(s).upper(),
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
            "skipped with a regular error. Increase to force FCI on larger systems "
            "(default: 16)."
        ),
    )
    parser.add_argument(
        "--condition",
        type=str,
        default="fc",
        choices=("fc", "qwc"),
        help="Compatibility condition for groups: fully commuting or qubit-wise commuting (default: fc).",
    )
    parser.add_argument(
        "--max-sweeps",
        type=int,
        default=100,
        help="Maximum greedy sweeps for the SI-refinement version (default: 100).",
    )
    parser.add_argument(
        "--allow-new-groups",
        action="store_true",
        help="During SI refinement, also consider moving a term into a new singleton group.",
    )
    parser.add_argument(
        "--ordered-consider-new-groups",
        action="store_true",
        help="Let the ordered VarSI pass choose a new singleton group even when compatible groups exist.",
    )
    parser.add_argument(
        "--cov-workers",
        type=int,
        default=None,
        help="Number of worker processes for the VarSI-local parallel covariance builder (default: up to 8).",
    )
    parser.add_argument(
        "--cov-chunksize",
        type=int,
        default=128,
        help="Number of term pairs sent to each covariance worker task (default: 128).",
    )
    parser.add_argument(
        "--serial-cov-dict",
        action="store_true",
        help="Use the original serial prepare_cov_dict helper instead of the VarSI-local parallel builder.",
    )
    parser.add_argument(
        "--print-groups",
        action="store_true",
        help="Print the final Pauli groups for each VarSI variant.",
    )
    parser.add_argument(
        "--spin-ordering",
        choices=("interleaved", "blocked"),
        default="interleaved",
        help="Spin-orbital ordering used by the loaded fermionic Hamiltonian (default: interleaved).",
    )
    parser.add_argument(
        "--cisd-validation-tol",
        type=float,
        default=1e-8,
        help="Allowed difference between CISD subspace energy and qubit expectation (default: 1e-8).",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop the batch when one loaded molecule fails instead of continuing to the next.",
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
    with open(path, "rb") as f:
        fermion_hamiltonian = pickle.load(f)

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
    occupied = [idx for idx, occ in enumerate(reference) if occ]
    virtual = [idx for idx, occ in enumerate(reference) if not occ]

    determinants = {reference}
    for excitation_rank in (1, 2):
        for removed_modes in combinations(occupied, excitation_rank):
            for added_modes in combinations(virtual, excitation_rank):
                if not same_spin_counts(removed_modes, added_modes, n_modes, spin_ordering):
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
        int(value) for value in np.asarray(encoder.dot(np.asarray(fermion_occupations)) % 2).ravel()
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
    subspace_hamiltonian = np.zeros((len(basis_indices), len(basis_indices)), dtype=complex)

    for col, basis_index in enumerate(basis_indices):
        for term, coefficient in qubit_hamiltonian.terms.items():
            output_index, phase = apply_qubit_term_to_basis_index(term, basis_index, n_qubits)
            row = index_to_row.get(output_index)
            if row is not None:
                subspace_hamiltonian[row, col] += coefficient * phase

    return 0.5 * (subspace_hamiltonian + subspace_hamiltonian.conj().T)


def qubit_operator_support_expectation(qubit_hamiltonian, support, n_qubits):
    value = 0.0 + 0.0j
    for basis_index, amplitude in support.items():
        for term, coefficient in qubit_hamiltonian.terms.items():
            output_index, phase = apply_qubit_term_to_basis_index(term, basis_index, n_qubits)
            output_amplitude = support.get(output_index)
            if output_amplitude is not None:
                value += np.conjugate(output_amplitude) * coefficient * phase * amplitude
    return value


def loaded_hf_energy_and_statevector(mol, qubit_hamiltonian, n_qubits, spin_ordering, tf):
    n_electrons = infer_electron_count(mol)
    reference = reference_occupations(n_qubits, n_electrons, spin_ordering)
    basis_index = sparse_basis_index(reference, tf)
    state = statevector_from_basis_index(basis_index, n_qubits)
    energy = basis_state_expectation_value(qubit_hamiltonian, basis_index, n_qubits)
    return energy, state


def loaded_cisd_statevector(mol, qubit_hamiltonian, n_qubits, spin_ordering, validation_tol, tf):
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
    eigvals, eigvecs = np.linalg.eigh(subspace_hamiltonian)
    ground_index = int(np.argmin(eigvals))
    cisd_energy = float(np.real_if_close(eigvals[ground_index]))
    cisd_coefficients = eigvecs[:, ground_index]

    state = np.zeros(2**n_qubits, dtype=complex)
    support = {}
    for basis_index, coefficient in zip(basis_indices, cisd_coefficients):
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
            sparse_hamiltonian = get_sparse_operator(qubit_hamiltonian, n_qubits=n_qubits)
        energy, wfn = get_ground_state(sparse_hamiltonian)
        return energy, wfn, None

    if method == "HF":
        energy, wfn = loaded_hf_energy_and_statevector(
            mol,
            qubit_hamiltonian,
            n_qubits,
            spin_ordering,
            tf,
        )
        return energy, wfn, None

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


def make_loaded_result(
    label,
    groups,
    wfn_cov_dict,
    report_cov_or_wfn,
    wfn_label,
    report_label,
    condition="fc",
    sample_ratios=None,
    runtime=None,
    extra=None,
):
    groups = varsi.validate_compatible_groups(groups, condition=condition)
    wfn_variances = [varsi.group_variance(group, wfn_cov_dict) for group in groups]
    if isinstance(report_cov_or_wfn, dict):
        report_variances = [varsi.group_variance(group, report_cov_or_wfn) for group in groups]
    else:
        report_variances = varsi.group_variances_from_wavefunction(
            groups,
            report_cov_or_wfn,
            condition=condition,
        )
    if sample_ratios is None:
        sample_ratios = varsi.sample_ratios_from_variances(wfn_variances)
    return {
        "label": label,
        "groups": groups,
        "eps_sq_m_wfn": varsi.eps_sq_m_from_variances(wfn_variances),
        "eps_sq_m_report": varsi.eps_sq_m_from_variances(report_variances),
        "wfn_label": wfn_label,
        "report_label": report_label,
        "num_groups": len(groups),
        "sample_ratios": sample_ratios,
        "runtime": runtime,
        "extra": extra,
    }


def print_loaded_result(result):
    label = result["label"]
    report_label = result["report_label"]
    print("{}:".format(label))
    print("  eps^2 M(wfn={})={:.12g}".format(result["wfn_label"], result["eps_sq_m_wfn"]))
    if report_label == "FCI":
        print("  eps^2 M(FCI)={:.12g}".format(result["eps_sq_m_report"]))
    else:
        print("  eps^2 M(report={})={:.12g}".format(report_label, result["eps_sq_m_report"]))
    print("  Number of groups={}".format(result["num_groups"]))
    print("  Compatible groups=True")
    if result["runtime"] is not None:
        print("  Runtime (s)={:.6f}".format(result["runtime"]))
    if result["extra"] is not None:
        print("  {}".format(result["extra"]))


def run_loaded_molecule(mol_name, tf, args):
    _, qubit_hamiltonian, tequila_hamiltonian = load_loaded_hamiltonian(
        mol_name,
        tf=tf,
        prefix=args.prefix,
    )
    n_qubits = count_qubits(qubit_hamiltonian)
    n_paulis = len(qubit_hamiltonian.terms) - (1 if () in qubit_hamiltonian.terms else 0)

    print("")
    print("=" * 80)
    print("Molecule={}".format(mol_name))
    print("Hamiltonian source=ham_lib/{}_fer.bin".format(mol_name))
    print("Transformation={}".format(tf))
    print("Number of Pauli products to measure: {}".format(n_paulis))

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
        get_sparse_operator(qubit_hamiltonian, n_qubits=n_qubits)
        if needs_fci
        else None
    )
    energy, variance_wfn, metadata = get_loaded_variance_wavefunction(
        mol_name,
        qubit_hamiltonian,
        args.wfn,
        sparse_hamiltonian,
        args.spin_ordering,
        args.cisd_validation_tol,
        tf,
    )
    print("{} Energy={}".format(args.wfn, energy))
    print_cisd_validation(metadata)

    if report_method == args.wfn:
        report_wfn = variance_wfn
    else:
        report_energy, report_wfn, _ = get_loaded_variance_wavefunction(
            mol_name,
            qubit_hamiltonian,
            report_method,
            sparse_hamiltonian,
            args.spin_ordering,
            args.cisd_validation_tol,
            tf,
        )
        print("{} Energy={}".format(report_method, report_energy))

    print("Compatibility condition={}".format(args.condition))
    print("Number of qubits={}".format(n_qubits))

    binary_hamiltonian = BinaryHamiltonian.init_from_qubit_hamiltonian(tequila_hamiltonian)
    terms = varsi.measurable_terms(binary_hamiltonian)
    cov_workers = args.cov_workers or varsi._default_cov_workers()
    cov_dict, cov_time, cov_label = varsi.build_covariance_dictionary(
        binary_hamiltonian,
        variance_wfn,
        use_serial=args.serial_cov_dict,
        max_workers=cov_workers,
        chunksize=args.cov_chunksize,
    )
    print(
        "{} covariance dictionary: {} entries built with {} in {:.6f} s".format(
            args.wfn,
            len(cov_dict),
            cov_label,
            cov_time,
        )
    )
    if report_method == args.wfn:
        report_cov_or_wfn = cov_dict
        if report_method == "FCI":
            print("Reporting eps^2 M: reused {} covariance dictionary".format(args.wfn))
        else:
            print(
                "Reporting eps^2 M: reused {} covariance dictionary".format(
                    report_method,
                )
            )
    else:
        report_cov_or_wfn = report_wfn
        print(
            "Reporting eps^2 M: direct group variances from {} wavefunction; no reporting covariance dictionary built.".format(
                report_method,
            )
        )

    failed_methods = []

    def run_ics(label, initial_groups):
        output, runtime, error = varsi.timed_optional_call(
            varsi.iterative_coefficient_splitting_from_groups,
            initial_groups,
            cov_dict,
            condition=args.condition,
        )
        if error is not None:
            failed_methods.append(
                {
                    "label": label,
                    "runtime": runtime,
                    "error": "{}: {}".format(type(error).__name__, error),
                }
            )
            return None, None, runtime
        groups, sample_size = output
        return groups, sample_size, runtime

    si_groups, si_time = varsi.timed_call(
        lambda: varsi.normalize_groups(varsi.sorted_insertion_grouping(terms, condition=args.condition))
    )
    si_groups = varsi.validate_compatible_groups(si_groups, condition=args.condition)

    si_ics_groups, si_ics_sample_size, si_ics_time = run_ics(
        "ICS initialized from sorted insertion groups",
        si_groups,
    )

    varsi_groups, varsi_time = varsi.timed_call(
        varsi.variance_sorted_insertion_grouping,
        terms,
        cov_dict,
        condition=args.condition,
    )
    varsi_groups = varsi.validate_compatible_groups(varsi_groups, condition=args.condition)

    varsi_ics_groups, varsi_ics_sample_size, varsi_ics_time = run_ics(
        "ICS initialized from VarSI greedy groups",
        varsi_groups,
    )

    ordered_varsi_groups, ordered_varsi_time = varsi.timed_call(
        varsi.variance_sorted_insertion_grouping_ordered,
        terms,
        cov_dict,
        condition=args.condition,
        consider_new_groups=args.ordered_consider_new_groups,
    )
    ordered_varsi_groups = varsi.validate_compatible_groups(ordered_varsi_groups, condition=args.condition)

    (ordered_refined_groups, ordered_refined_moves), ordered_refined_time = varsi.timed_call(
        varsi.refine_sorted_insertion_groups,
        ordered_varsi_groups,
        cov_dict,
        condition=args.condition,
        max_sweeps=args.max_sweeps,
        allow_new_groups=args.allow_new_groups,
    )
    ordered_refined_groups = varsi.validate_compatible_groups(ordered_refined_groups, condition=args.condition)

    ordered_varsi_ics_groups, ordered_varsi_ics_sample_size, ordered_varsi_ics_time = run_ics(
        "ICS initialized from VarSI ordered groups",
        ordered_varsi_groups,
    )

    ordered_refined_ics_groups, ordered_refined_ics_sample_size, ordered_refined_ics_time = run_ics(
        "ICS initialized from VarSI ordered+refined groups",
        ordered_refined_groups,
    )

    (refined_groups, accepted_moves), refined_time = varsi.timed_call(
        varsi.refine_sorted_insertion_groups,
        si_groups,
        cov_dict,
        condition=args.condition,
        max_sweeps=args.max_sweeps,
        allow_new_groups=args.allow_new_groups,
    )
    refined_groups = varsi.validate_compatible_groups(refined_groups, condition=args.condition)

    refined_ics_groups, refined_ics_sample_size, refined_ics_time = run_ics(
        "ICS initialized from VarSI-refined sorted insertion groups",
        refined_groups,
    )

    results = []

    def add_result(label, groups, runtime, sample_ratios=None, extra=None):
        results.append(
            make_loaded_result(
                label,
                groups,
                cov_dict,
                report_cov_or_wfn,
                args.wfn,
                report_method,
                condition=args.condition,
                sample_ratios=sample_ratios,
                runtime=runtime,
                extra=extra,
            )
        )

    add_result("Sorted insertion baseline", si_groups, si_time)
    if si_ics_groups is not None:
        add_result(
            "ICS initialized from sorted insertion groups",
            si_ics_groups,
            si_ics_time,
            sample_ratios=si_ics_sample_size,
        )
    add_result("VarSI greedy from empty groups", varsi_groups, varsi_time)
    if varsi_ics_groups is not None:
        add_result(
            "ICS initialized from VarSI greedy groups",
            varsi_ics_groups,
            varsi_ics_time,
            sample_ratios=varsi_ics_sample_size,
        )
    add_result("VarSI ordered from empty groups", ordered_varsi_groups, ordered_varsi_time)
    add_result(
        "VarSI ordered+refined",
        ordered_refined_groups,
        ordered_refined_time,
        extra="Accepted moves={}".format(ordered_refined_moves),
    )
    if ordered_varsi_ics_groups is not None:
        add_result(
            "ICS initialized from VarSI ordered groups",
            ordered_varsi_ics_groups,
            ordered_varsi_ics_time,
            sample_ratios=ordered_varsi_ics_sample_size,
        )
    if ordered_refined_ics_groups is not None:
        add_result(
            "ICS initialized from VarSI ordered+refined groups",
            ordered_refined_ics_groups,
            ordered_refined_ics_time,
            sample_ratios=ordered_refined_ics_sample_size,
        )
    add_result(
        "VarSI refinement from sorted insertion",
        refined_groups,
        refined_time,
        extra="Accepted moves={}".format(accepted_moves),
    )
    if refined_ics_groups is not None:
        add_result(
            "ICS initialized from VarSI-refined sorted insertion groups",
            refined_ics_groups,
            refined_ics_time,
            sample_ratios=refined_ics_sample_size,
        )

    for result in results:
        print("")
        print_loaded_result(result)

    if failed_methods:
        print("")
        print("Skipped methods:")
        for failed in failed_methods:
            print(
                "  {} failed after {:.6f} s: {}".format(
                    failed["label"],
                    failed["runtime"],
                    failed["error"],
                )
            )

    ranked_results = sorted(results, key=lambda result: result["eps_sq_m_report"])
    print("")
    if report_method == "FCI":
        print("Ranking by eps^2 M(FCI) (lowest to highest):")
    else:
        print("Ranking by eps^2 M(report={}) (lowest to highest):".format(report_method))
    for rank, result in enumerate(ranked_results, start=1):
        print(
            "  {}. {}: {:.12g} Groups: {}".format(
                rank,
                result["label"],
                result["eps_sq_m_report"],
                result["num_groups"],
            )
        )

    if args.print_groups:
        for result in results:
            print("")
            varsi.print_group_contents(result["label"], result["groups"])

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
                print("Molecule={} tf={} failed: {}: {}".format(mol_name, tf, type(exc).__name__, exc))

    if failures:
        print("")
        print("Failed loaded molecules:")
        for mol_name, tf, exc in failures:
            print("  {} tf={}: {}: {}".format(mol_name, tf, type(exc).__name__, exc))


if __name__ == "__main__":
    main()

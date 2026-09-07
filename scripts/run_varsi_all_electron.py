import argparse
import csv
import math
import os
import re
import sys
import time
import traceback
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

# Keep the top-level VarSI modules available when this script is run by path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/varsi_mplconfig")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)


def preconfigure_thread_environment():
    threads = None
    for idx, arg in enumerate(sys.argv):
        if arg == "--pyscf-threads" and idx + 1 < len(sys.argv):
            threads = sys.argv[idx + 1]
            break
        if arg.startswith("--pyscf-threads="):
            threads = arg.split("=", 1)[1]
            break
    if threads is None:
        return
    try:
        threads = str(int(threads))
    except ValueError:
        return
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[name] = threads


preconfigure_thread_environment()

import numpy as np
import tequila as tq
from openfermion.linalg import get_sparse_operator
from openfermion.transforms import bravyi_kitaev_code
from pyscf import cc, lib, scf
from pyscf.ci import cisd
from pyscf.fci import cistring
from tequila.grouping.binary_rep import BinaryHamiltonian
from tequila.grouping.binary_utils import sorted_insertion_grouping
from tequila.hamiltonian import QubitHamiltonian

try:
    from threadpoolctl import threadpool_info, threadpool_limits
except Exception:
    threadpool_info = None
    threadpool_limits = None

import hamiltonians_varsi as hv
from gflow_vqe.utils import get_variance_wavefunction
from VarSI import (
    build_covariance_dictionary,
    iterative_coefficient_splitting_from_groups,
    make_result,
    measurable_terms,
    normalize_groups,
    refine_sorted_insertion_groups,
    timed_call,
    timed_optional_call,
    validate_compatible_groups,
    variance_sorted_insertion_grouping,
    variance_sorted_insertion_grouping_ordered,
)


METHOD_COLUMNS = [
    "SI",
    "VarSI-G",
    "VarSI-O",
    "VarSI-R",
    "VarSI-OR",
    "SI-ICS",
    "VarSI-G-ICS",
    "VarSI-O-ICS",
    "VarSI-R-ICS",
    "VarSI-OR-ICS",
]

LABEL_TO_METHOD = {
    "Sorted insertion baseline": "SI",
    "VarSI greedy from empty groups": "VarSI-G",
    "VarSI ordered from empty groups": "VarSI-O",
    "VarSI refinement from sorted insertion": "VarSI-R",
    "VarSI ordered+refined": "VarSI-OR",
    "ICS initialized from sorted insertion groups": "SI-ICS",
    "ICS initialized from VarSI greedy groups": "VarSI-G-ICS",
    "ICS initialized from VarSI ordered groups": "VarSI-O-ICS",
    "ICS initialized from VarSI-refined sorted insertion groups": "VarSI-R-ICS",
    "ICS initialized from VarSI ordered+refined groups": "VarSI-OR-ICS",
}


FIELDNAMES = [
    "status",
    "error",
    "entry_id",
    "type",
    "molecule",
    "bond_distance",
    "geometry_label",
    "basis",
    "mapping",
    "charge",
    "frozen_core",
    "n_qubits",
    "n_terms",
    "one_norm",
    "active_electrons",
    "condition",
    "cov_wfn",
    "cov_wfn_energy",
    "report_wfn",
    "report_wfn_energy",
    "cov_entries",
    "report_cov_entries",
    "cov_runtime_s",
    "report_cov_runtime_s",
    "runtime_total_s",
] + METHOD_COLUMNS


@dataclass
class AllElectronReference:
    method: str
    label: str
    energy: float
    wfn: object
    fallback_notes: list
    state_items: dict = None
    n_qubits: int = None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the VarSI/SI/ICS pipeline over Tequila/PySCF-generated Hamiltonians."
    )
    parser.add_argument("--type", required=True, choices=hv.ALL_ELECTRON_TYPES)
    parser.add_argument("-wfn", "--wfn", default="FCI", help="Covariance wavefunction: FCI, HF, CISD, CCSD.")
    parser.add_argument(
        "--report-wfn",
        default="FCI",
        choices=("FCI", "SAME"),
        help="Wavefunction used to evaluate reported eps^2 M values. Use SAME to report with --wfn.",
    )
    parser.add_argument("--fci-fallback-wfn", default="CCSD", help="Fallback when FCI is skipped or fails.")
    parser.add_argument("--condition", default="fc", choices=("fc", "qwc"))
    parser.add_argument("--max-sweeps", type=int, default=100)
    parser.add_argument("--allow-new-groups", action="store_true")
    parser.add_argument("--ordered-consider-new-groups", action="store_true")
    parser.add_argument("--cov-workers", type=int, default=None)
    parser.add_argument("--cov-chunksize", type=int, default=128)
    parser.add_argument(
        "--cov-blas-threads",
        type=int,
        default=1,
        help="BLAS/OpenMP threads to allow inside covariance worker sections.",
    )
    parser.add_argument("--serial-cov-dict", action="store_true")
    parser.add_argument("--mappings", nargs="+", default=("JW", "BK"), help="Mappings to run: JW BK.")
    parser.add_argument("--basis-filter", default=None, help="Only run basis labels containing this text.")
    parser.add_argument("--molecule-filter", default=None, help="Only run this exact molecule label.")
    parser.add_argument("--molecule-filter-contains", default=None, help="Only run molecule labels containing this text.")
    parser.add_argument("--geometry-filter", default=None, help="Only run geometry labels containing this text.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-terms", type=int, default=None)
    parser.add_argument("--fci-max-qubits", type=int, default=20)
    parser.add_argument("--pyscf-threads", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--scf-max-cycle", type=int, default=200)
    parser.add_argument("--output-dir", default="all_electron_varsi_results")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--skip-failed", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--print-groups", action="store_true")
    args = parser.parse_args(argv)
    args.wfn = normalize_wfn_method(args.wfn)
    args.fci_fallback_wfn = normalize_wfn_method(args.fci_fallback_wfn)
    args.mappings = tuple(hv.normalize_mapping(mapping) for mapping in args.mappings)
    if args.max_sweeps < 1:
        parser.error("--max-sweeps must be at least 1.")
    if args.cov_workers is not None and args.cov_workers < 1:
        parser.error("--cov-workers must be at least 1.")
    if args.cov_chunksize < 1:
        parser.error("--cov-chunksize must be at least 1.")
    if args.cov_blas_threads < 1:
        parser.error("--cov-blas-threads must be at least 1.")
    if args.fci_max_qubits < 1:
        parser.error("--fci-max-qubits must be at least 1.")
    if args.pyscf_threads < 1:
        parser.error("--pyscf-threads must be at least 1.")
    if args.cov_workers is None:
        args.cov_workers = max(1, min(8, os.cpu_count() or 1))
    if args.output_csv is None:
        args.output_csv = str(
            Path(args.output_dir)
            / "varsi_all_electron_tequila_{}_{}_{}.csv".format(
                args.type,
                safe_token(args.wfn.lower()),
                args.condition,
            )
        )
    return args


def normalize_wfn_method(method):
    normalized = str(method).upper().replace("-", "").replace("_", "")
    aliases = {"FULLCI": "FCI", "CCSD(T)": "CCSD", "CCSDT": "CCSD"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"FCI", "HF", "CISD", "CCSD"}:
        raise ValueError("Unsupported wavefunction method '{}'. Use FCI, HF, CISD, or CCSD.".format(method))
    return normalized


def safe_token(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def format_float(value):
    if value == "" or value is None:
        return ""
    value = float(np.real_if_close(value))
    if not math.isfinite(value):
        return str(value)
    return "{:.12g}".format(value)


def print_reference(label, reference):
    print("{} reference wavefunction={}".format(label, reference.label), flush=True)
    print("{} reference energy={:.12g}".format(label, reference.energy), flush=True)
    for note in reference.fallback_notes:
        print("  {}".format(note), flush=True)


def print_result_for_report(result, report_label):
    print("{}:".format(result["label"]), flush=True)
    print("  eps^2 M(cov_wfn={})={:.12g}".format(result["wfn_label"], result["eps_sq_m_wfn"]), flush=True)
    print("  eps^2 M(report={})={:.12g}".format(report_label, result["eps_sq_m_fci"]), flush=True)
    print("  Number of groups={}".format(result["num_groups"]), flush=True)
    print("  Compatible groups=True", flush=True)
    if result["runtime"] is not None:
        print("  Runtime (s)={:.6f}".format(result["runtime"]), flush=True)
    if result["extra"] is not None:
        print("  {}".format(result["extra"]), flush=True)


def print_group_contents(label, groups):
    print("{} groups:".format(label), flush=True)
    for idx, group in enumerate(normalize_groups(groups)):
        terms = [str(term.to_pauli_strings()) for term in group]
        print("  Group {}: {}".format(idx, ", ".join(terms)), flush=True)


def covariance_thread_context(args):
    if threadpool_limits is None:
        return nullcontext()
    return threadpool_limits(limits=args.cov_blas_threads)


def print_thread_configuration(args):
    print("PySCF threads requested={}".format(args.pyscf_threads), flush=True)
    try:
        print("PySCF lib.num_threads()={}".format(lib.num_threads()), flush=True)
    except Exception:
        print("PySCF lib.num_threads()=unknown", flush=True)
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        print("{}={}".format(name, os.environ.get(name, "")), flush=True)
    if threadpool_info is not None:
        for pool in threadpool_info():
            label = pool.get("prefix") or pool.get("internal_api") or pool.get("filepath") or "threadpool"
            print("Threadpool {} threads={}".format(label, pool.get("num_threads")), flush=True)


def fallback_order(primary, *fallbacks):
    order = []
    for method in (primary,) + fallbacks:
        if method is None:
            continue
        normalized = normalize_wfn_method(method)
        if normalized not in order:
            order.append(normalized)
    return order


def occupation_to_index(bits):
    index = 0
    for qubit, bit in enumerate(bits):
        if int(bit):
            index |= 1 << (len(bits) - 1 - qubit)
    return index


def encode_occupation(occupation_bits, mapping):
    if mapping == "JW":
        return list(occupation_bits)
    if mapping == "BK":
        encoder = bravyi_kitaev_code(len(occupation_bits)).encoder
    else:
        raise ValueError("Unsupported mapping '{}'.".format(mapping))
    encoded = np.asarray(encoder.dot(np.asarray(occupation_bits, dtype=int)) % 2).reshape(-1)
    return [int(bit) for bit in encoded]


def alpha_beta_string_to_qubit_index(alpha_string, beta_string, n_orbitals, mapping):
    occupations = []
    alpha_string = int(alpha_string)
    beta_string = int(beta_string)
    for orbital in range(n_orbitals):
        occupations.append(1 if (alpha_string >> orbital) & 1 else 0)
        occupations.append(1 if (beta_string >> orbital) & 1 else 0)
    return occupation_to_index(encode_occupation(occupations, mapping))


def alpha_beta_reordering_phase(alpha_string, beta_string, n_orbitals):
    alpha_string = int(alpha_string)
    beta_string = int(beta_string)
    inversions = 0
    for beta_orbital in range(n_orbitals):
        if not ((beta_string >> beta_orbital) & 1):
            continue
        for alpha_orbital in range(beta_orbital + 1, n_orbitals):
            if (alpha_string >> alpha_orbital) & 1:
                inversions += 1
    return -1.0 if inversions % 2 else 1.0


def apply_qubit_term(index, term, n_qubits):
    new_index = int(index)
    phase = 1.0 + 0.0j
    for qubit, pauli in term:
        mask = 1 << (n_qubits - 1 - qubit)
        bit = 1 if (new_index & mask) else 0
        if pauli == "X":
            new_index ^= mask
        elif pauli == "Y":
            phase *= 1.0j if bit == 0 else -1.0j
            new_index ^= mask
        elif pauli == "Z":
            phase *= 1.0 if bit == 0 else -1.0
        else:
            raise ValueError("Unsupported Pauli action '{}'.".format(pauli))
    return new_index, phase


def qubit_expectation_sparse_state(qubit_operator, state_items, n_qubits):
    value = 0.0 + 0.0j
    state = {int(index): complex(coeff) for index, coeff in state_items.items() if abs(coeff) > 1e-14}
    for ket_index, ket_amp in state.items():
        for term, coeff in qubit_operator.terms.items():
            bra_index, phase = apply_qubit_term(ket_index, term, n_qubits)
            bra_amp = state.get(bra_index)
            if bra_amp is not None:
                value += np.conjugate(bra_amp) * complex(coeff) * phase * ket_amp
    return float(np.real_if_close(value, tol=1000))


def bitstring_from_index(index, n_qubits):
    try:
        return tq.BitString.from_int(integer=int(index), nbits=n_qubits)
    except TypeError:
        try:
            return tq.BitString.from_int(int(index), n_qubits)
        except TypeError:
            return tq.BitString.from_int(int(index), nbits=n_qubits)


def tequila_wavefunction_from_state_dict(state, n_qubits):
    attempts = (
        lambda: tq.QubitWaveFunction(state=state, n_qubits=n_qubits),
        lambda: tq.QubitWaveFunction(state, n_qubits=n_qubits),
        lambda: tq.QubitWaveFunction(state, n_qubits),
        lambda: tq.QubitWaveFunction(state),
    )
    failures = []
    for attempt in attempts:
        try:
            return attempt()
        except TypeError as exc:
            failures.append(str(exc))
    raise TypeError("Could not construct tequila QubitWaveFunction: {}".format("; ".join(failures)))


def tequila_wavefunction_from_sparse_items(state_items, n_qubits, tiny=1e-12):
    state = {}
    for index, coeff in state_items.items():
        if abs(coeff) > tiny:
            state[bitstring_from_index(index, n_qubits)] = complex(coeff)
    if not state:
        raise RuntimeError("Reference wavefunction has no nonzero amplitudes.")
    return tequila_wavefunction_from_state_dict(state, n_qubits)


def add_spin_adapted_coefficient(state_items, n_orbitals, n_alpha, n_beta, mapping, alpha_addr, beta_addr, coefficient, tiny):
    coefficient = complex(coefficient)
    if abs(coefficient) <= tiny:
        return
    alpha_string = cistring.addr2str(n_orbitals, n_alpha, int(alpha_addr))
    beta_string = cistring.addr2str(n_orbitals, n_beta, int(beta_addr))
    qubit_index = alpha_beta_string_to_qubit_index(alpha_string, beta_string, n_orbitals, mapping)
    phase = alpha_beta_reordering_phase(alpha_string, beta_string, n_orbitals)
    state_items[qubit_index] = state_items.get(qubit_index, 0.0) + phase * coefficient


def uccsd_amplitudes_to_state_items(t1, t2, n_orbitals, n_alpha, n_beta, mapping, tiny=1e-12):
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nvir_alpha = n_orbitals - n_alpha
    nvir_beta = n_orbitals - n_beta
    state_items = {}

    add_spin_adapted_coefficient(state_items, n_orbitals, n_alpha, n_beta, mapping, 0, 0, 1.0, tiny)

    if n_alpha > 0 and nvir_alpha > 0:
        t1addra, t1signa = cisd.tn_addrs_signs(n_orbitals, n_alpha, 1)
        for idx, addr in enumerate(t1addra):
            add_spin_adapted_coefficient(state_items, n_orbitals, n_alpha, n_beta, mapping, addr, 0, t1a.ravel()[idx] * t1signa[idx], tiny)
    else:
        t1addra = np.asarray([], dtype=int)
        t1signa = np.asarray([], dtype=float)

    if n_beta > 0 and nvir_beta > 0:
        t1addrb, t1signb = cisd.tn_addrs_signs(n_orbitals, n_beta, 1)
        for idx, addr in enumerate(t1addrb):
            add_spin_adapted_coefficient(state_items, n_orbitals, n_alpha, n_beta, mapping, 0, addr, t1b.ravel()[idx] * t1signb[idx], tiny)
    else:
        t1addrb = np.asarray([], dtype=int)
        t1signb = np.asarray([], dtype=float)

    if len(t1addra) and len(t1addrb):
        c2ab = t2ab.transpose(0, 2, 1, 3).reshape(n_alpha * nvir_alpha, -1)
        for alpha_idx, alpha_addr in enumerate(t1addra):
            for beta_idx, beta_addr in enumerate(t1addrb):
                coefficient = c2ab[alpha_idx, beta_idx] * t1signa[alpha_idx] * t1signb[beta_idx]
                add_spin_adapted_coefficient(state_items, n_orbitals, n_alpha, n_beta, mapping, alpha_addr, beta_addr, coefficient, tiny)

    if n_alpha > 1 and nvir_alpha > 1:
        ooidx = np.tril_indices(n_alpha, -1)
        vvidx = np.tril_indices(nvir_alpha, -1)
        c2aa = t2aa[ooidx][:, vvidx[0], vvidx[1]]
        t2addra, t2signa = cisd.tn_addrs_signs(n_orbitals, n_alpha, 2)
        for idx, addr in enumerate(t2addra):
            add_spin_adapted_coefficient(state_items, n_orbitals, n_alpha, n_beta, mapping, addr, 0, c2aa.ravel()[idx] * t2signa[idx], tiny)

    if n_beta > 1 and nvir_beta > 1:
        ooidx = np.tril_indices(n_beta, -1)
        vvidx = np.tril_indices(nvir_beta, -1)
        c2bb = t2bb[ooidx][:, vvidx[0], vvidx[1]]
        t2addrb, t2signb = cisd.tn_addrs_signs(n_orbitals, n_beta, 2)
        for idx, addr in enumerate(t2addrb):
            add_spin_adapted_coefficient(state_items, n_orbitals, n_alpha, n_beta, mapping, 0, addr, c2bb.ravel()[idx] * t2signb[idx], tiny)

    norm = math.sqrt(sum(abs(coefficient) ** 2 for coefficient in state_items.values()))
    if norm < tiny:
        raise RuntimeError("PySCF CCSD produced a near-zero CI-vector reference.")
    return {
        qubit_index: coefficient / norm
        for qubit_index, coefficient in state_items.items()
        if abs(coefficient / norm) > tiny
    }


def spec_size_sort_key(spec):
    try:
        n_orbitals = hv.estimate_n_orbitals(spec)
    except Exception:
        n_orbitals = 10**12
    return (
        n_orbitals,
        hv.formula_electrons(spec.molecule) - spec.charge,
        spec.molecule,
        spec.bond_distance,
        spec.geometry_label,
        hv.basis_size_key(spec.basis_label),
        spec.mapping,
        spec.frozen_core,
        spec.multiplicity,
    )


def discover_specs(args):
    specs = list(hv.iter_specs(args.type, mappings=args.mappings))
    if args.molecule_filter:
        specs = [spec for spec in specs if args.molecule_filter.lower() == spec.molecule.lower()]
    if args.molecule_filter_contains:
        specs = [spec for spec in specs if args.molecule_filter_contains.lower() in spec.molecule.lower()]
    if args.basis_filter:
        specs = [spec for spec in specs if args.basis_filter.lower() in spec.basis_label.lower()]
    if args.geometry_filter:
        specs = [spec for spec in specs if args.geometry_filter.lower() in spec.geometry_label.lower()]
    specs.sort(key=spec_size_sort_key)
    if args.limit is not None:
        specs = specs[: args.limit]
    return specs


def load_recorded_statuses(csv_path):
    statuses = {}
    path = Path(csv_path)
    if not path.exists():
        return statuses
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            statuses[row.get("entry_id", "")] = row.get("status", "")
    return statuses


def append_csv_row(csv_path, row):
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in FIELDNAMES})
        handle.flush()
        os.fsync(handle.fileno())


def alpha_beta_electrons(pyscf_mol):
    n_alpha, n_beta = pyscf_mol.nelec
    return int(n_alpha), int(n_beta)


def correlation_mean_field(mean_field):
    return scf.addons.convert_to_uhf(mean_field)


def build_fci_reference(built, args):
    if built.n_qubits > args.fci_max_qubits:
        raise MemoryError(
            "FCI skipped for {} qubits; --fci-max-qubits is {}.".format(built.n_qubits, args.fci_max_qubits)
        )
    return build_gflow_reference("FCI", built)


def build_gflow_reference(method, built):
    if built.tequila_mol is None:
        raise RuntimeError("Built Hamiltonian does not carry a Tequila molecule reference.")
    sparse_hamiltonian = get_sparse_operator(built.qubit_operator, n_qubits=built.n_qubits)
    energy, wfn = get_variance_wavefunction(
        built.tequila_mol,
        built.qubit_operator,
        method=method,
        sparse_hamiltonian=sparse_hamiltonian,
    )
    labels = {
        "FCI": "FCI",
        "HF": "HF(Tequila)",
        "CISD": "CISD(Tequila/PySCF)",
    }
    return AllElectronReference(
        method,
        labels.get(method, method),
        float(np.real_if_close(energy)),
        np.asarray(wfn),
        [],
        n_qubits=built.n_qubits,
    )


def build_hf_reference(built):
    return build_gflow_reference("HF", built)


def build_cisd_reference(built, args):
    return build_gflow_reference("CISD", built)


def build_ccsd_reference(built, args):
    mf = correlation_mean_field(built.mean_field)
    n_alpha, n_beta = alpha_beta_electrons(built.pyscf_mol)
    solver = cc.UCCSD(mf)
    solver.max_cycle = args.scf_max_cycle
    energy_corr, t1, t2 = solver.kernel()
    if not solver.converged:
        raise RuntimeError("PySCF UCCSD did not converge.")
    state_items = uccsd_amplitudes_to_state_items(
        t1,
        t2,
        built.n_orbitals,
        n_alpha,
        n_beta,
        built.spec.mapping,
    )
    energy = qubit_expectation_sparse_state(built.qubit_operator, state_items, built.n_qubits)
    wfn = tequila_wavefunction_from_sparse_items(state_items, built.n_qubits)
    notes = [
        "PySCF UCCSD correlation energy={:.12g}; reported/reference expectation={:.12g}.".format(
            float(energy_corr),
            energy,
        )
    ]
    return AllElectronReference("CCSD", "CCSD(PySCF)", energy, wfn, notes, state_items=state_items, n_qubits=built.n_qubits)


def build_reference(method, built, args):
    if method == "FCI":
        return build_fci_reference(built, args)
    if method == "HF":
        return build_hf_reference(built)
    if method == "CISD":
        return build_cisd_reference(built, args)
    if method == "CCSD":
        return build_ccsd_reference(built, args)
    raise ValueError("Unsupported reference method '{}'.".format(method))


def build_reference_with_fallback(primary, built, args, *fallbacks):
    failures = []
    for method in fallback_order(primary, *fallbacks):
        try:
            reference = build_reference(method, built, args)
        except Exception as exc:
            failures.append("{} failed: {}: {}".format(method, type(exc).__name__, exc))
            continue
        reference.fallback_notes = failures + reference.fallback_notes
        return reference
    raise RuntimeError("All reference wavefunction attempts failed: {}".format("; ".join(failures)))


def run_varsi_pipeline(terms, cov_dict, report_cov_or_wfn, cov_label, report_label, args):
    failed_methods = []

    def run_ics(label, initial_groups):
        output, runtime, error = timed_optional_call(
            iterative_coefficient_splitting_from_groups,
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

    si_groups, si_time = timed_call(lambda: normalize_groups(sorted_insertion_grouping(terms, condition=args.condition)))
    si_groups = validate_compatible_groups(si_groups, condition=args.condition)
    si_ics_groups, si_ics_sample_size, si_ics_time = run_ics("ICS initialized from sorted insertion groups", si_groups)

    varsi_groups, varsi_time = timed_call(variance_sorted_insertion_grouping, terms, cov_dict, condition=args.condition)
    varsi_groups = validate_compatible_groups(varsi_groups, condition=args.condition)
    varsi_ics_groups, varsi_ics_sample_size, varsi_ics_time = run_ics("ICS initialized from VarSI greedy groups", varsi_groups)

    ordered_varsi_groups, ordered_varsi_time = timed_call(
        variance_sorted_insertion_grouping_ordered,
        terms,
        cov_dict,
        condition=args.condition,
        consider_new_groups=args.ordered_consider_new_groups,
    )
    ordered_varsi_groups = validate_compatible_groups(ordered_varsi_groups, condition=args.condition)

    (ordered_refined_groups, ordered_refined_moves), ordered_refined_time = timed_call(
        refine_sorted_insertion_groups,
        ordered_varsi_groups,
        cov_dict,
        condition=args.condition,
        max_sweeps=args.max_sweeps,
        allow_new_groups=args.allow_new_groups,
    )
    ordered_refined_groups = validate_compatible_groups(ordered_refined_groups, condition=args.condition)
    ordered_varsi_ics_groups, ordered_varsi_ics_sample_size, ordered_varsi_ics_time = run_ics(
        "ICS initialized from VarSI ordered groups",
        ordered_varsi_groups,
    )
    ordered_refined_ics_groups, ordered_refined_ics_sample_size, ordered_refined_ics_time = run_ics(
        "ICS initialized from VarSI ordered+refined groups",
        ordered_refined_groups,
    )

    (refined_groups, accepted_moves), refined_time = timed_call(
        refine_sorted_insertion_groups,
        si_groups,
        cov_dict,
        condition=args.condition,
        max_sweeps=args.max_sweeps,
        allow_new_groups=args.allow_new_groups,
    )
    refined_groups = validate_compatible_groups(refined_groups, condition=args.condition)
    refined_ics_groups, refined_ics_sample_size, refined_ics_time = run_ics(
        "ICS initialized from VarSI-refined sorted insertion groups",
        refined_groups,
    )

    results = []

    def add_result(label, groups, runtime, sample_ratios=None, extra=None):
        results.append(
            make_result(
                label,
                groups,
                cov_dict,
                report_cov_or_wfn,
                cov_label,
                condition=args.condition,
                sample_ratios=sample_ratios,
                runtime=runtime,
                extra=extra,
            )
        )

    add_result("Sorted insertion baseline", si_groups, si_time)
    if si_ics_groups is not None:
        add_result("ICS initialized from sorted insertion groups", si_ics_groups, si_ics_time, sample_ratios=si_ics_sample_size)
    add_result("VarSI greedy from empty groups", varsi_groups, varsi_time)
    if varsi_ics_groups is not None:
        add_result("ICS initialized from VarSI greedy groups", varsi_ics_groups, varsi_ics_time, sample_ratios=varsi_ics_sample_size)
    add_result("VarSI ordered from empty groups", ordered_varsi_groups, ordered_varsi_time)
    add_result(
        "VarSI ordered+refined",
        ordered_refined_groups,
        ordered_refined_time,
        extra="Accepted moves={}".format(ordered_refined_moves),
    )
    if ordered_varsi_ics_groups is not None:
        add_result("ICS initialized from VarSI ordered groups", ordered_varsi_ics_groups, ordered_varsi_ics_time, sample_ratios=ordered_varsi_ics_sample_size)
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

    method_values = {name: "" for name in METHOD_COLUMNS}
    for result in results:
        method_name = LABEL_TO_METHOD[result["label"]]
        method_values[method_name] = result["eps_sq_m_fci"]
        print("")
        print_result_for_report(result, report_label)

    if failed_methods:
        print("")
        print("Skipped methods:", flush=True)
        for failed in failed_methods:
            print(
                "  {} failed after {:.6f} s: {}".format(
                    failed["label"],
                    failed["runtime"],
                    failed["error"],
                ),
                flush=True,
            )

    ranked_results = sorted(results, key=lambda result: result["eps_sq_m_fci"])
    print("")
    print("Ranking by eps^2 M(report={}) (lowest to highest):".format(report_label), flush=True)
    for rank, result in enumerate(ranked_results, start=1):
        print(
            "  {}. {}: {:.12g} Groups: {}".format(
                rank,
                result["label"],
                result["eps_sq_m_fci"],
                result["num_groups"],
            ),
            flush=True,
        )

    if args.print_groups:
        for result in results:
            print("")
            print_group_contents(result["label"], result["groups"])

    return method_values


def base_row(spec):
    return {
        "entry_id": spec.entry_id,
        "type": spec.ham_type,
        "molecule": spec.molecule,
        "bond_distance": spec.bond_distance,
        "geometry_label": spec.geometry_label,
        "basis": spec.basis_label,
        "mapping": spec.mapping,
        "charge": spec.charge,
        "frozen_core": bool(spec.frozen_core),
    }


def run_entry(spec, args):
    start = time.perf_counter()
    row = base_row(spec)

    print("")
    print("=" * 80, flush=True)
    print("Molecule={}".format(spec.molecule), flush=True)
    print("Basis={} Mapping={} Bond distance={}".format(spec.basis_label, spec.mapping, spec.bond_distance or "NA"), flush=True)
    print("Geometry label={}".format(spec.geometry_label or "NA"), flush=True)
    print("Charge={}".format(spec.charge), flush=True)
    print("Frozen core={}".format(bool(spec.frozen_core)), flush=True)
    print("Compatibility condition={}".format(args.condition), flush=True)

    built = hv.build_all_electron_hamiltonian(spec, scf_max_cycle=args.scf_max_cycle)
    row["n_qubits"] = built.n_qubits
    row["n_terms"] = built.n_terms
    row["one_norm"] = format_float(built.one_norm)
    row["active_electrons"] = built.n_electrons
    row["condition"] = args.condition
    print("Number of qubits={}".format(built.n_qubits), flush=True)
    print("Active electrons={}".format(built.n_electrons), flush=True)
    print("Active spatial orbitals={}".format(built.n_orbitals), flush=True)
    print("Number of Pauli products to measure: {}".format(built.n_terms), flush=True)

    if args.max_terms is not None and built.n_terms > args.max_terms:
        raise RuntimeError("Skipping {} non-identity terms because --max-terms is {}.".format(built.n_terms, args.max_terms))

    hamiltonian = QubitHamiltonian.from_openfermion(built.qubit_operator)
    binary_hamiltonian = BinaryHamiltonian.init_from_qubit_hamiltonian(hamiltonian)
    terms = measurable_terms(binary_hamiltonian)

    cov_reference = build_reference_with_fallback(args.wfn, built, args, args.fci_fallback_wfn)
    print_reference("Covariance", cov_reference)
    row["cov_wfn"] = cov_reference.label
    row["cov_wfn_energy"] = format_float(cov_reference.energy)

    report_reference_reused = args.report_wfn == "SAME"
    if args.report_wfn == "SAME":
        report_reference = cov_reference
        print("Reporting reference wavefunction={}".format(report_reference.label), flush=True)
        print("Reporting reference energy={:.12g}".format(report_reference.energy), flush=True)
        print(
            "  Reusing covariance reference because --report-wfn SAME; no extra electronic-structure calculation.",
            flush=True,
        )
    else:
        report_reference = build_reference_with_fallback("FCI", built, args, args.fci_fallback_wfn)
        print_reference("Reporting", report_reference)
    row["report_wfn"] = report_reference.label
    row["report_wfn_energy"] = format_float(report_reference.energy)

    cov_workers = args.cov_workers
    with covariance_thread_context(args):
        cov_dict, cov_time, cov_builder_label = build_covariance_dictionary(
            binary_hamiltonian,
            cov_reference.wfn,
            use_serial=args.serial_cov_dict,
            max_workers=cov_workers,
            chunksize=args.cov_chunksize,
        )
    print(
        "{} covariance dictionary: {} entries built with {} in {:.6f} s".format(
            cov_reference.label,
            len(cov_dict),
            cov_builder_label,
            cov_time,
        ),
        flush=True,
    )
    row["cov_entries"] = len(cov_dict)
    row["cov_runtime_s"] = format_float(cov_time)

    if report_reference_reused:
        report_cov_or_wfn = cov_dict
        print("Reporting eps^2 M: reused {} covariance dictionary".format(cov_reference.label), flush=True)
        row["report_cov_entries"] = len(cov_dict)
    else:
        report_cov_or_wfn = report_reference.wfn
        print(
            "Reporting eps^2 M: direct group variances from {} wavefunction; no reporting covariance dictionary built.".format(
                report_reference.label,
            ),
            flush=True,
        )
        row["report_cov_entries"] = 0
    row["report_cov_runtime_s"] = format_float(0.0)

    method_values = run_varsi_pipeline(
        terms,
        cov_dict,
        report_cov_or_wfn,
        cov_reference.label,
        report_reference.label,
        args,
    )
    row.update({name: format_float(value) for name, value in method_values.items()})
    row["runtime_total_s"] = format_float(time.perf_counter() - start)
    row["status"] = "ok"
    row["error"] = ""
    return row


def failed_row(spec, args, exc, runtime):
    row = base_row(spec)
    row["condition"] = args.condition
    row["status"] = "failed"
    row["error"] = "{}: {}".format(type(exc).__name__, exc)
    row["runtime_total_s"] = format_float(runtime)
    return row


def main(argv=None):
    args = parse_args(argv)
    lib.num_threads(args.pyscf_threads)
    specs = discover_specs(args)
    statuses = load_recorded_statuses(args.output_csv)

    print("All-electron VarSI type={}".format(args.type), flush=True)
    print("Discovered entries={}".format(len(specs)), flush=True)
    print("Output CSV={}".format(args.output_csv), flush=True)
    print("Requested covariance wavefunction={}".format(args.wfn), flush=True)
    print("Report wavefunction mode={}".format(args.report_wfn), flush=True)
    print_thread_configuration(args)

    processed = 0
    skipped = 0
    for index, spec in enumerate(specs, start=1):
        status = statuses.get(spec.entry_id)
        if status == "ok" or (status == "failed" and args.skip_failed):
            skipped += 1
            print("Skipping recorded {} row {}/{}: {}".format(status, index, len(specs), spec.entry_id), flush=True)
            continue

        print("Starting {}/{}: {}".format(index, len(specs), spec.entry_id), flush=True)
        start = time.perf_counter()
        try:
            row = run_entry(spec, args)
        except Exception as exc:
            row = failed_row(spec, args, exc, time.perf_counter() - start)
            append_csv_row(args.output_csv, row)
            print("FAILED: {}".format(spec.entry_id), flush=True)
            traceback.print_exc()
            if args.strict:
                raise
            continue

        append_csv_row(args.output_csv, row)
        processed += 1
        print("Recorded results for {}".format(spec.entry_id), flush=True)

    print("")
    print("Done.", flush=True)
    print("  Processed rows={}".format(processed), flush=True)
    print("  Skipped rows={}".format(skipped), flush=True)
    print("  CSV={}".format(args.output_csv), flush=True)


if __name__ == "__main__":
    main()

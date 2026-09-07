import argparse
import math
import os
import sys
import time
import traceback
from pathlib import Path

# Keep the top-level VarSI modules available when this script is run by path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyscf import lib

import hamiltonians_varsi as hv
import run_varsi_all_electron as runner


BASIS_LABEL = "sto-3g"
BASIS_SET = "sto-3g"


def nh3_main_geometry(distance=1.0, angle_degrees=107.0):
    r = float(distance)
    theta = math.radians(float(angle_degrees))
    cos_alpha = math.sqrt((math.cos(theta) + 0.5) / 1.5)
    z = r * cos_alpha
    rho = r * math.sqrt(1.0 - cos_alpha * cos_alpha)
    return hv.geometry_lines(
        [
            ("N", 0.0, 0.0, 0.0),
            ("H", rho, 0.0, z),
            ("H", rho * math.cos(2.0 * math.pi / 3.0), rho * math.sin(2.0 * math.pi / 3.0), z),
            ("H", rho * math.cos(4.0 * math.pi / 3.0), rho * math.sin(4.0 * math.pi / 3.0), z),
        ]
    )


MAIN_GEOMETRIES = (
    {
        "molecule": "LiH",
        "bond_distance": "1",
        "geometry_label": "R1",
        "geometry": hv.diatomic_geometry("LiH", 1.0),
    },
    {
        "molecule": "BeH2",
        "bond_distance": "1",
        "geometry_label": "R1_linear",
        "geometry": hv.beh2_geometry(1.0),
    },
    {
        "molecule": "BeH2",
        "bond_distance": "3",
        "geometry_label": "R3_linear_stretched",
        "geometry": hv.beh2_geometry(3.0),
    },
    {
        "molecule": "H2O",
        "bond_distance": "1",
        "geometry_label": "R1_angle107.6",
        "geometry": hv.bent_xy2_geometry("O", "H", 1.0, 107.6),
    },
    {
        "molecule": "H2O",
        "bond_distance": "2.2",
        "geometry_label": "R2.2_angle107.6_stretched",
        "geometry": hv.bent_xy2_geometry("O", "H", 2.2, 107.6),
    },
    {
        "molecule": "NH3",
        "bond_distance": "1",
        "geometry_label": "R1_angle107",
        "geometry": nh3_main_geometry(1.0, 107.0),
    },
)


def iter_main_specs(mappings):
    for entry in MAIN_GEOMETRIES:
        for mapping in mappings:
            yield hv.VarsiHamiltonianSpec(
                ham_type="main",
                molecule=entry["molecule"],
                geometry=entry["geometry"],
                bond_distance=entry["bond_distance"],
                basis_label=BASIS_LABEL,
                basis_set=BASIS_SET,
                mapping=mapping,
                multiplicity=1,
                spin_state="singlet",
                charge=0,
                frozen_core=False,
                geometry_label=entry["geometry_label"],
            )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the main VarSI study geometries with Tequila/PySCF STO-3G Hamiltonians."
    )
    parser.add_argument("-wfn", "--wfn", default="CISD", help="Covariance wavefunction: FCI, HF, CISD, CCSD.")
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
    parser.add_argument("--molecule-filter", default=None, help="Only run this exact molecule label.")
    parser.add_argument("--molecule-filter-contains", default=None, help="Only run molecule labels containing this text.")
    parser.add_argument("--geometry-filter", default=None, help="Only run geometry labels containing this text.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-terms", type=int, default=None)
    parser.add_argument("--fci-max-qubits", type=int, default=20)
    parser.add_argument("--pyscf-threads", type=int, default=os.cpu_count() or 1)
    parser.add_argument(
        "--ham-build-workers",
        type=int,
        default=None,
        help="Compatibility option; Hamiltonian mapping workers are always set to --cov-workers.",
    )
    parser.add_argument("--scf-max-cycle", type=int, default=200)
    parser.add_argument("--output-dir", default="main_varsi_results")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--skip-failed", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--print-groups", action="store_true")
    parser.add_argument(
        "--approx-max-determinants",
        default=None,
        help="Compatibility option; all-electron CISD/CCSD references do not impose an explicit determinant cap.",
    )
    args = parser.parse_args(argv)
    args.type = "main"
    args.wfn = runner.normalize_wfn_method(args.wfn)
    args.fci_fallback_wfn = runner.normalize_wfn_method(args.fci_fallback_wfn)
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
    args.ham_build_workers = args.cov_workers
    if args.ham_build_workers < 1:
        parser.error("--ham-build-workers must be at least 1.")
    if args.output_csv is None:
        args.output_csv = str(
            Path(args.output_dir)
            / "varsi_main_{}_{}.csv".format(
                runner.safe_token(args.wfn.lower()),
                args.condition,
            )
        )
    return args


def discover_specs(args):
    specs = list(iter_main_specs(args.mappings))
    if args.molecule_filter:
        specs = [spec for spec in specs if args.molecule_filter.lower() == spec.molecule.lower()]
    if args.molecule_filter_contains:
        specs = [spec for spec in specs if args.molecule_filter_contains.lower() in spec.molecule.lower()]
    if args.geometry_filter:
        specs = [spec for spec in specs if args.geometry_filter.lower() in spec.geometry_label.lower()]
    specs.sort(key=runner.spec_size_sort_key)
    if args.limit is not None:
        specs = specs[: args.limit]
    return specs


def main(argv=None):
    args = parse_args(argv)
    lib.num_threads(args.pyscf_threads)
    specs = discover_specs(args)
    statuses = runner.load_recorded_statuses(args.output_csv)

    print("Main VarSI study", flush=True)
    print("Discovered entries={}".format(len(specs)), flush=True)
    print("Output CSV={}".format(args.output_csv), flush=True)
    print("Requested covariance wavefunction={}".format(args.wfn), flush=True)
    print("Report wavefunction mode={}".format(args.report_wfn), flush=True)
    runner.print_thread_configuration(args)

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
            row = runner.run_entry(spec, args)
        except Exception as exc:
            row = runner.failed_row(spec, args, exc, time.perf_counter() - start)
            runner.append_csv_row(args.output_csv, row)
            print("FAILED: {}".format(spec.entry_id), flush=True)
            traceback.print_exc()
            if args.strict:
                raise
            continue

        runner.append_csv_row(args.output_csv, row)
        processed += 1
        print("Recorded results for {}".format(spec.entry_id), flush=True)

    print("")
    print("Done.", flush=True)
    print("  Processed rows={}".format(processed), flush=True)
    print("  Skipped rows={}".format(skipped), flush=True)
    print("  CSV={}".format(args.output_csv), flush=True)


if __name__ == "__main__":
    main()

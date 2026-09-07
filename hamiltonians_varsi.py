import math
import re
from dataclasses import dataclass

from openfermion.utils import count_qubits
from pyscf import gto


ALL_ELECTRON_TYPES = ("standard",)

BASIS_SETS = {
    "standard": (("sto-3g", "sto-3g"),),
}
BASIS_SIZE_RANK = {
    "sto3g": 0,
    "sto6g": 1,
}

MAPPINGS = ("JW", "BK")

_ORBITAL_COUNT_CACHE = {}

ATOMIC_NUMBERS = {
    "H": 1,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Na": 11,
}

STANDARD_DISTANCES = {
    "H2": (0.741, None),
    "N2": (1.098, None),
    "F2": (1.412, None),
    "B2": (1.590, None),
    "C2": (1.243, None),
    "Be2": (2.460, None),
    "LiH": (1.595, None),
    "BeH": (1.343, None),
    "BeH2": (1.326, None),
    "BH": (1.232, None),
    "CH": (1.120, None),
    "NH": (1.034, None),
    "OH": (0.970, None),
    "HF": (0.917, None),
    "Li2": (2.673, None),
    "NaLi": (2.889, None),
    "Na2": (3.079, None),
    "O3": (1.278, 116.8),
    "H2O": (0.958, 104.4776),
    "NH3": (1.0, None),
}

LINEAR_HYDROGEN_MOLECULES = ("H4", "H6", "H8")
LINEAR_HYDROGEN_DISTANCES = (0.5, 1.0, 1.5, 2.0)

@dataclass(frozen=True)
class VarsiHamiltonianSpec:
    ham_type: str
    molecule: str
    geometry: str
    bond_distance: str
    basis_label: str
    basis_set: str
    mapping: str
    multiplicity: int
    spin_state: str
    charge: int = 0
    frozen_core: bool = False
    geometry_label: str = ""

    @property
    def entry_id(self):
        parts = [
            self.ham_type,
            self.molecule,
            self.geometry_label or self.bond_distance or "geom",
            self.basis_label,
            self.mapping,
            "charge{}".format(self.charge),
            "frozen_core{}".format(bool(self.frozen_core)),
        ]
        return "::".join(str(x) for x in parts)


@dataclass
class BuiltHamiltonian:
    spec: VarsiHamiltonianSpec
    pyscf_mol: object
    mean_field: object
    fermion_operator: object
    qubit_operator: object
    n_qubits: int
    n_terms: int
    one_norm: float
    n_electrons: int
    n_orbitals: int
    tequila_mol: object = None
    tequila_hamiltonian: object = None


def normalize_mapping(mapping):
    label = str(mapping).upper()
    if label not in MAPPINGS:
        raise ValueError("Unsupported mapping '{}'. Use JW or BK.".format(mapping))
    return label


def basis_specs_for_type(ham_type):
    return BASIS_SETS[ham_type]


def basis_size_key(basis_label):
    normalized = re.sub(r"[^a-z0-9]", "", str(basis_label).lower())
    return (BASIS_SIZE_RANK.get(normalized, len(BASIS_SIZE_RANK)), normalized)


def build_pyscf_molecule(spec):
    return gto.M(
        atom=spec.geometry,
        basis=spec.basis_set,
        charge=spec.charge,
        spin=spec.multiplicity - 1,
        unit="Angstrom",
        verbose=0,
    )


def estimate_n_orbitals(spec):
    key = (spec.geometry, spec.basis_set, spec.charge, spec.multiplicity)
    if key not in _ORBITAL_COUNT_CACHE:
        _ORBITAL_COUNT_CACHE[key] = int(build_pyscf_molecule(spec).nao_nr())
    n_orbitals = _ORBITAL_COUNT_CACHE[key]
    if getattr(spec, "frozen_core", False):
        n_orbitals -= frozen_core_orbitals(spec.molecule)
    return max(0, n_orbitals)


def formula_electrons(formula):
    total = 0
    for element, count_text in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        count = int(count_text) if count_text else 1
        total += ATOMIC_NUMBERS[element] * count
    if total == 0:
        raise ValueError("Could not parse molecular formula '{}'.".format(formula))
    return total


def spin_label(multiplicity):
    labels = {1: "singlet", 2: "doublet", 3: "triplet", 4: "quadruplet"}
    return labels.get(int(multiplicity), "multiplicity{}".format(multiplicity))


def benchmark_multiplicities(molecule):
    return (1,)


def singlet_charge(molecule):
    if molecule == "OH":
        return -1
    return 1 if formula_electrons(molecule) % 2 else 0


def parse_formula_atoms(formula):
    atoms = []
    for element, count_text in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        count = int(count_text) if count_text else 1
        atoms.extend([element] * count)
    if not atoms:
        raise ValueError("Could not parse atoms from '{}'.".format(formula))
    return atoms


def frozen_core_orbitals(molecule):
    """Match Tequila's default frozen-core rule: one doubly occupied core per atom heavier than He."""
    return sum(1 for atom in parse_formula_atoms(molecule) if ATOMIC_NUMBERS[atom] > 2)


def geometry_lines(atoms):
    return "\n".join("{:2s} {: .12f} {: .12f} {: .12f}".format(atom, x, y, z) for atom, x, y, z in atoms)


def diatomic_geometry(molecule, distance):
    atoms = parse_formula_atoms(molecule)
    if len(atoms) != 2:
        raise ValueError("{} is not diatomic.".format(molecule))
    return geometry_lines([(atoms[0], 0.0, 0.0, 0.0), (atoms[1], 0.0, 0.0, float(distance))])


def distance_label(distance):
    return "{:g}".format(float(distance))


def linear_hydrogen_geometry(molecule, distance):
    match = re.fullmatch(r"H(\d+)", molecule)
    if match is None:
        raise ValueError("{} is not a bare hydrogen chain label.".format(molecule))
    spacing = float(distance)
    return geometry_lines([("H", 0.0, 0.0, idx * spacing) for idx in range(int(match.group(1)))])


def beh2_geometry(distance):
    r = float(distance)
    return geometry_lines([("Be", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, r), ("H", 0.0, 0.0, -r)])


def bent_xy2_geometry(center, terminal, distance, angle_degrees):
    r = float(distance)
    theta = math.radians(float(angle_degrees))
    x = r * math.sin(theta / 2.0)
    z = r * math.cos(theta / 2.0)
    return geometry_lines(
        [
            (center, 0.0, 0.0, 0.0),
            (terminal, x, 0.0, z),
            (terminal, -x, 0.0, z),
        ]
    )


def nh3_geometry():
    return geometry_lines(
        [
            ("N", 0.00000000, 0.00000000, 0.00000000),
            ("H", -0.93542876, 0.32963162, -0.12773422),
            ("H", 0.48922227, 0.70193678, 0.51763532),
            ("H", 0.42139398, -0.04217329, -0.90589653),
        ]
    )


def standard_geometry(molecule, distance, angle=None):
    if molecule == "BeH2":
        return beh2_geometry(distance)
    if molecule == "H2O":
        return bent_xy2_geometry("O", "H", distance, angle)
    if molecule == "O3":
        return bent_xy2_geometry("O", "O", distance, angle)
    if molecule == "NH3":
        return nh3_geometry()
    return diatomic_geometry(molecule, distance)


def iter_standard_entries():
    for molecule, (distance, angle) in STANDARD_DISTANCES.items():
        yield molecule, str(distance), standard_geometry(molecule, distance, angle), ""
    for molecule in LINEAR_HYDROGEN_MOLECULES:
        for distance in LINEAR_HYDROGEN_DISTANCES:
            label = distance_label(distance)
            yield molecule, label, linear_hydrogen_geometry(molecule, distance), "linear_R{}".format(label)


def iter_specs(ham_type, mappings=MAPPINGS):
    if ham_type not in ALL_ELECTRON_TYPES:
        raise ValueError("Unsupported type '{}'. Use {}.".format(ham_type, ", ".join(ALL_ELECTRON_TYPES)))

    mappings = tuple(normalize_mapping(mapping) for mapping in mappings)

    base_entries = list(iter_standard_entries())

    for molecule, bond_distance, geometry, geometry_label in base_entries:
        frozen_core_options = (False,) if (
            molecule in LINEAR_HYDROGEN_MOLECULES and geometry_label.startswith("linear_R")
        ) else (False, True)
        for multiplicity in benchmark_multiplicities(molecule):
            for basis_label, basis_set in basis_specs_for_type(ham_type):
                for mapping in mappings:
                    for frozen_core in frozen_core_options:
                        yield VarsiHamiltonianSpec(
                            ham_type=ham_type,
                            molecule=molecule,
                            geometry=geometry,
                            bond_distance=bond_distance,
                            basis_label=basis_label,
                            basis_set=basis_set,
                            mapping=mapping,
                            multiplicity=multiplicity,
                            spin_state=spin_label(multiplicity),
                            charge=singlet_charge(molecule),
                            frozen_core=frozen_core,
                            geometry_label=geometry_label,
                        )


def tequila_transformation(mapping):
    mapping = normalize_mapping(mapping)
    if mapping == "JW":
        return "JordanWigner"
    if mapping == "BK":
        return "BravyiKitaev"
    raise ValueError("Unsupported mapping '{}'.".format(mapping))


def build_tequila_molecule(spec):
    import tequila as tq

    return tq.chemistry.Molecule(
        geometry=spec.geometry,
        basis_set=spec.basis_set,
        transformation=tequila_transformation(spec.mapping),
        backend="pyscf",
        charge=spec.charge,
        frozen_core=bool(spec.frozen_core),
    )


def tequila_mean_field(tequila_mol):
    from tequila.quantumchemistry.pyscf_interface import QuantumChemistryPySCF

    qc = QuantumChemistryPySCF.from_tequila(tequila_mol)
    return qc._get_hf()


def build_all_electron_hamiltonian(spec, scf_max_cycle=200, mapping_workers=1):
    tequila_mol = build_tequila_molecule(spec)
    tequila_hamiltonian = tequila_mol.make_hamiltonian()
    qubit_operator = tequila_hamiltonian.to_openfermion()
    n_qubits = count_qubits(qubit_operator)
    n_terms = len([term for term in qubit_operator.terms if term != ()])
    one_norm = sum(abs(complex(coeff)) for term, coeff in qubit_operator.terms.items() if term != ())
    mf = tequila_mean_field(tequila_mol)
    mol = getattr(mf, "mol", None)
    return BuiltHamiltonian(
        spec=spec,
        pyscf_mol=mol,
        mean_field=mf,
        fermion_operator=None,
        qubit_operator=qubit_operator,
        n_qubits=n_qubits,
        n_terms=n_terms,
        one_norm=float(one_norm),
        n_electrons=tequila_mol.n_electrons,
        n_orbitals=tequila_mol.n_orbitals,
        tequila_mol=tequila_mol,
        tequila_hamiltonian=tequila_hamiltonian,
    )

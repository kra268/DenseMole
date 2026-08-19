"""Bridge between our GaussianJob atom lists and RDKit Mol objects.

Gaussian input files carry no bond information, only coordinates, so bonds
are perceived from the 3D geometry (RDKit's xyz2mol-style algorithm) rather
than assumed from a fixed distance cutoff.
"""
from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit.Geometry import Point3D

from common.gaussian import Atom, GaussianJob


def atoms_to_mol(atoms: list[Atom], charge: int = 0) -> Chem.Mol:
    """Build an RDKit Mol (with perceived bonds) from a flat atom list."""
    rw = Chem.RWMol()
    for atom in atoms:
        rw.AddAtom(Chem.Atom(atom.symbol))

    conf = Chem.Conformer(rw.GetNumAtoms())
    for i, atom in enumerate(atoms):
        conf.SetAtomPosition(i, Point3D(*atom.coords))
    rw.AddConformer(conf)

    mol = rw.GetMol()
    rdDetermineBonds.DetermineBonds(mol, charge=charge)
    return mol


def job_to_mol(job: GaussianJob) -> Chem.Mol:
    return atoms_to_mol(job.atoms, charge=job.charge)


def mol_to_atoms(mol: Chem.Mol, fragment_of: dict[int, int] | None = None) -> list[Atom]:
    """Flatten an RDKit Mol's conformer back into a plain atom list.

    fragment_of, if given, maps atom index -> 1-based fragment number for
    multi-fragment (combined-molecule) jobs.
    """
    conf = mol.GetConformer()
    atoms = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        x, y, z = conf.GetAtomPosition(idx)
        fragment = fragment_of.get(idx) if fragment_of else None
        atoms.append(Atom(symbol=atom.GetSymbol(), coords=(x, y, z), fragment=fragment))
    return atoms


def net_formal_charge(mol: Chem.Mol) -> int:
    return sum(a.GetFormalCharge() for a in mol.GetAtoms())

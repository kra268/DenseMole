"""Minimal .xyz export, for quick visual QC in a real molecular viewer
(Avogadro, VMD, PyMOL, Jmol, ...) before committing to a batch of Gaussian jobs.
"""
from __future__ import annotations

from rdkit import Chem


def write_xyz(mol: Chem.Mol, file_path: str, comment: str = "") -> None:
    conf = mol.GetConformer()
    lines = [str(mol.GetNumAtoms()), comment]
    for atom in mol.GetAtoms():
        x, y, z = conf.GetAtomPosition(atom.GetIdx())
        lines.append(f"{atom.GetSymbol():<3s}{x:12.6f}{y:12.6f}{z:12.6f}")
    with open(file_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")

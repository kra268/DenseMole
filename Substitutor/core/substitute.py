"""Functional-group substitution on a 3D molecular structure.

The approach, in short:
 1. The functional group is given as SMILES with one dummy atom ('*') marking
    where it attaches, e.g. "*O" for -OH, "*C(F)(F)F" for -CF3, "*c1ccccc1"
    for phenyl. It's embedded into its own 3D geometry and force-field
    optimized, independent of the host.
 2. The atom(s) being removed from the host define an "attachment vector"
    (anchor atom -> leaving atom). The functional group is rigidly rotated
    so its own dummy->attachment-atom direction lines up with that vector,
    then translated so the attachment atom sits at a proper bond length
    (sum of covalent radii) from the anchor atom.
 3. The two fragments are spliced into one molecule and, optionally, the
    newly-added atoms are relaxed with an MMFF/UFF force field (host atoms
    held fixed) to resolve any local strain or clashes.

This replaces picking a random direction and retrying on collision: the new
group is placed along the bond it's replacing, the way a chemist would draw it.
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D


def _get_force_field(mol, conf_id: int = -1):
    props = AllChem.MMFFGetMoleculeProperties(mol)
    if props is not None:
        return AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
    try:
        return AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
    except Exception:
        return None


def _optimize(mol, max_iters: int = 500) -> None:
    ff = _get_force_field(mol)
    if ff is None:
        return
    ff.Minimize(maxIts=max_iters)


def _rotation_about_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    k = np.array(
        [
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ]
    )
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def _rotation_aligning(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation matrix R such that R @ normalize(a) == normalize(b)."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    if s < 1e-8:
        if c > 0:
            return np.eye(3)
        # a and b are antiparallel: rotate 180 degrees about any perpendicular axis
        perp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, perp)
        return _rotation_about_axis(axis, np.pi)
    vx = np.array(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ]
    )
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))


def embed_functional_group(smiles: str, seed: int = 0xC0FFEE):
    """Parse a '*'-tagged functional group SMILES and embed it in 3D.

    Returns (mol, dummy_atom_idx, attachment_atom_idx).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse functional group SMILES: {smiles!r}")

    dummy_indices = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummy_indices) != 1:
        raise ValueError(
            f"Functional group SMILES must contain exactly one attachment "
            f"point '*', found {len(dummy_indices)}: {smiles!r}"
        )
    dummy_idx = dummy_indices[0]
    dummy_atom = mol.GetAtomWithIdx(dummy_idx)
    if dummy_atom.GetDegree() != 1:
        raise ValueError(f"Attachment point '*' must have exactly one bond: {smiles!r}")
    attach_idx = dummy_atom.GetNeighbors()[0].GetIdx()

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    conf_id = AllChem.EmbedMolecule(mol, params)
    if conf_id < 0:
        raise ValueError(f"Could not generate 3D coordinates for {smiles!r}")
    _optimize(mol)
    return mol, dummy_idx, attach_idx


def find_atoms_by_smarts(mol: Chem.Mol, smarts: str) -> list[int]:
    """Find candidate substitution sites by SMARTS (first-matched atom of each hit)."""
    patt = Chem.MolFromSmarts(smarts)
    if patt is None:
        raise ValueError(f"Invalid SMARTS pattern: {smarts!r}")
    return [match[0] for match in mol.GetSubstructMatches(patt)]


class ClashError(ValueError):
    pass


def _check_clashes(mol: Chem.Mol, threshold: float = 0.6) -> None:
    conf = mol.GetConformer()
    pt = Chem.GetPeriodicTable()
    positions = [np.array(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]
    bonded = {frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in mol.GetBonds()}
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            if frozenset((i, j)) in bonded:
                continue
            dist = np.linalg.norm(positions[i] - positions[j])
            min_allowed = threshold * (
                pt.GetRvdw(mol.GetAtomWithIdx(i).GetAtomicNum())
                + pt.GetRvdw(mol.GetAtomWithIdx(j).GetAtomicNum())
            )
            if dist < min_allowed:
                raise ClashError(
                    f"Steric clash between atoms {i} and {j}: {dist:.2f} Å "
                    f"(expected at least {min_allowed:.2f} Å)."
                )


def _relax_new_atoms(mol: Chem.Mol, fixed_atoms: set[int], max_iters: int = 500) -> None:
    ff = _get_force_field(mol)
    if ff is None:
        return
    for idx in fixed_atoms:
        ff.AddFixedPoint(idx)
    ff.Minimize(maxIts=max_iters)


def substitute(
    host_mol: Chem.Mol,
    leaving_atom_indices: list[int],
    group_smiles: str,
    seed: int = 0xC0FFEE,
    relax: bool = True,
    check_clashes: bool = True,
) -> Chem.Mol:
    """Replace the given leaving atom(s) in host_mol with a functional group.

    leaving_atom_indices: atoms to remove. They must connect to the rest of
    the molecule through exactly one bond (e.g. a single H or halogen, or a
    whole existing substituent such as an -OH's O+H pair).
    group_smiles: '*'-tagged SMILES for the replacement, e.g. "*O", "*N",
    "*C(F)(F)F", "*c1ccccc1".
    """
    leaving_set = set(leaving_atom_indices)

    anchor_candidates: set[int] = set()
    boundary_leaving_atom = None
    for idx in leaving_set:
        for nbr in host_mol.GetAtomWithIdx(idx).GetNeighbors():
            if nbr.GetIdx() not in leaving_set:
                anchor_candidates.add(nbr.GetIdx())
                boundary_leaving_atom = idx
    if len(anchor_candidates) != 1:
        raise ValueError(
            f"Leaving atoms {sorted(leaving_set)} must connect to the rest of "
            f"the molecule through exactly one atom (found {len(anchor_candidates)})"
        )
    anchor_idx = anchor_candidates.pop()

    conf = host_mol.GetConformer()
    anchor_pos = np.array(conf.GetAtomPosition(anchor_idx))
    leaving_pos = np.array(conf.GetAtomPosition(boundary_leaving_atom))
    host_vector = leaving_pos - anchor_pos
    if np.linalg.norm(host_vector) < 1e-6:
        raise ValueError("Anchor and leaving atom coordinates coincide; cannot determine direction")

    group_mol, dummy_idx, attach_idx = embed_functional_group(group_smiles, seed=seed)
    gconf = group_mol.GetConformer()
    dummy_pos = np.array(gconf.GetAtomPosition(dummy_idx))
    attach_pos = np.array(gconf.GetAtomPosition(attach_idx))
    local_vector = dummy_pos - attach_pos

    rotation = _rotation_aligning(local_vector, -host_vector)

    pt = Chem.GetPeriodicTable()
    anchor_elem = host_mol.GetAtomWithIdx(anchor_idx).GetAtomicNum()
    attach_elem = group_mol.GetAtomWithIdx(attach_idx).GetAtomicNum()
    bond_length = pt.GetRcovalent(anchor_elem) + pt.GetRcovalent(attach_elem)
    new_attach_pos = anchor_pos + bond_length * (host_vector / np.linalg.norm(host_vector))

    combined = Chem.RWMol()
    old_to_new: dict[int, int] = {}
    for atom in host_mol.GetAtoms():
        idx = atom.GetIdx()
        if idx in leaving_set:
            continue
        new_atom = Chem.Atom(atom.GetAtomicNum())
        new_atom.SetFormalCharge(atom.GetFormalCharge())
        old_to_new[idx] = combined.AddAtom(new_atom)

    group_to_new: dict[int, int] = {}
    for atom in group_mol.GetAtoms():
        idx = atom.GetIdx()
        if idx == dummy_idx:
            continue
        new_atom = Chem.Atom(atom.GetAtomicNum())
        new_atom.SetFormalCharge(atom.GetFormalCharge())
        group_to_new[idx] = combined.AddAtom(new_atom)

    for bond in host_mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a in leaving_set or b in leaving_set:
            continue
        combined.AddBond(old_to_new[a], old_to_new[b], bond.GetBondType())

    for bond in group_mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a == dummy_idx or b == dummy_idx:
            continue
        combined.AddBond(group_to_new[a], group_to_new[b], bond.GetBondType())

    combined.AddBond(old_to_new[anchor_idx], group_to_new[attach_idx], Chem.BondType.SINGLE)

    n_atoms = combined.GetNumAtoms()
    new_conf = Chem.Conformer(n_atoms)
    for old_idx, new_idx in old_to_new.items():
        new_conf.SetAtomPosition(new_idx, conf.GetAtomPosition(old_idx))
    for old_idx, new_idx in group_to_new.items():
        local = np.array(gconf.GetAtomPosition(old_idx)) - attach_pos
        world = new_attach_pos + rotation @ local
        new_conf.SetAtomPosition(new_idx, Point3D(*world))
    combined.AddConformer(new_conf)

    mol = combined.GetMol()
    Chem.SanitizeMol(mol)

    if relax:
        _relax_new_atoms(mol, fixed_atoms=set(old_to_new.values()))

    if check_clashes:
        _check_clashes(mol)

    return mol

"""Combining two separate molecules into one Gaussian input, as a non-bonded
complex (dimers, host-guest, interaction-energy setups, ...).

This operates on plain atom lists rather than RDKit bond graphs: there is no
shared bond to perceive between the two fragments, so building an RDKit Mol
and re-running bond perception on the combined coordinates would risk RDKit
inventing a spurious bond between fragments that happen to sit close together.
"""
from __future__ import annotations

import numpy as np

from Substitutor.io.gaussian import Atom, GaussianJob


def combine_atom_pair_distance(
    atoms_a: list[Atom],
    atoms_b: list[Atom],
    atom_idx_a: int,
    atom_idx_b: int,
    distance: float,
    direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> list[Atom]:
    """Place B (rigid) so atom_idx_b sits `distance` Å from atom_idx_a along `direction`."""
    a_pos = np.array(atoms_a[atom_idx_a].coords)
    b_pos = np.array(atoms_b[atom_idx_b].coords)
    direction = np.array(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)
    target_b_pos = a_pos + distance * direction
    shift = target_b_pos - b_pos

    new_a = [Atom(at.symbol, at.coords, fragment=1) for at in atoms_a]
    new_b = [Atom(at.symbol, tuple(np.array(at.coords) + shift), fragment=2) for at in atoms_b]
    return new_a + new_b


def combine_centroid_offset(
    atoms_a: list[Atom],
    atoms_b: list[Atom],
    offset: tuple[float, float, float],
) -> list[Atom]:
    """Place B (rigid) at a fixed vector offset from its current position, relative to A."""
    offset = np.array(offset, dtype=float)
    new_a = [Atom(at.symbol, at.coords, fragment=1) for at in atoms_a]
    new_b = [Atom(at.symbol, tuple(np.array(at.coords) + offset), fragment=2) for at in atoms_b]
    return new_a + new_b


def distance_scan(
    atoms_a: list[Atom],
    atoms_b: list[Atom],
    atom_idx_a: int,
    atom_idx_b: int,
    distances: list[float],
    direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> list[list[Atom]]:
    """Generate one combined atom list per distance, for a scan of separations."""
    return [
        combine_atom_pair_distance(atoms_a, atoms_b, atom_idx_a, atom_idx_b, d, direction)
        for d in distances
    ]


def combined_charge_multiplicity(charge_a: int, mult_a: int, charge_b: int, mult_b: int) -> tuple[int, int]:
    total_charge = charge_a + charge_b
    if mult_a == 1 and mult_b == 1:
        return total_charge, 1
    raise ValueError(
        f"Cannot infer combined multiplicity when either fragment is non-singlet "
        f"(mult_a={mult_a}, mult_b={mult_b}); pass multiplicity explicitly."
    )


def combine(
    job_a: GaussianJob,
    job_b: GaussianJob,
    mode: str,
    template: GaussianJob | None = None,
    charge: int | None = None,
    multiplicity: int | None = None,
    tag_fragments: bool = False,
    **placement_kwargs,
) -> GaussianJob:
    """Build a combined GaussianJob from two single-molecule jobs.

    mode: "atom_pair" (needs atom_idx_a, atom_idx_b, distance, direction) or
    "centroid_offset" (needs offset).
    tag_fragments: if True, atoms are labeled Fragment=1/2 and a per-fragment
    charge/multiplicity line is emitted (Gaussian's counterpoise/ONIOM syntax).
    """
    if mode == "atom_pair":
        atoms = combine_atom_pair_distance(job_a.atoms, job_b.atoms, **placement_kwargs)
    elif mode == "centroid_offset":
        atoms = combine_centroid_offset(job_a.atoms, job_b.atoms, **placement_kwargs)
    else:
        raise ValueError(f"Unknown combination mode: {mode!r}")

    if not tag_fragments:
        for atom in atoms:
            atom.fragment = None

    if charge is None or multiplicity is None:
        inferred_charge, inferred_mult = combined_charge_multiplicity(
            job_a.charge, job_a.multiplicity, job_b.charge, job_b.multiplicity
        )
        charge = inferred_charge if charge is None else charge
        multiplicity = inferred_mult if multiplicity is None else multiplicity

    base = template or job_a
    fragment_charges = (
        [(job_a.charge, job_a.multiplicity), (job_b.charge, job_b.multiplicity)]
        if tag_fragments
        else []
    )

    return GaussianJob(
        link0=list(base.link0),
        route=list(base.route),
        title=f"Combined: {job_a.title} + {job_b.title}",
        charge=charge,
        multiplicity=multiplicity,
        fragment_charges=fragment_charges,
        atoms=atoms,
        trailing=base.trailing,
    )

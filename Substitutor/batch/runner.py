"""Config-driven batch generation: enumerate hosts x sites x substituents (or
a distance/orientation scan for combined molecules) and write one .gjf per
job plus a manifest CSV logging what went into each file.

See Substitutor/library/groups.yaml for the substituent library format and
the module docstrings in core/substitute.py and core/combine.py for the
underlying single-job operations.
"""
from __future__ import annotations

import csv
from pathlib import Path

import yaml
from rdkit import Chem

from Substitutor.core.combine import combine, combined_charge_multiplicity, distance_scan
from Substitutor.core.molecule import job_to_mol, mol_to_atoms
from Substitutor.core.substitute import find_atoms_by_smarts, substitute
from Substitutor.io.gaussian import GaussianJob, read_gaussian, write_gaussian
from Substitutor.io.xyz import write_xyz


def _load_library(path: str) -> dict[str, str]:
    with open(path) as fh:
        entries = yaml.safe_load(fh) or []
    return {entry["name"]: entry["smiles"] for entry in entries}


def _resolve_substituents(config: dict) -> dict[str, str]:
    spec = config.get("substituents", {})
    library_path = spec.get(
        "library", str(Path(__file__).resolve().parent.parent / "library" / "groups.yaml")
    )
    library = _load_library(library_path)

    names = spec.get("names")
    if names:
        missing = set(names) - set(library)
        if missing:
            raise ValueError(f"Unknown substituent name(s): {sorted(missing)}")
        selected = {name: library[name] for name in names}
    else:
        selected = dict(library)

    for extra in spec.get("extra", []):
        selected[extra["name"]] = extra["smiles"]

    return selected


def _resolve_sites(host_mol: Chem.Mol, spec: dict) -> dict[str, list[int]]:
    """Returns {site_label: [leaving_atom_indices]}."""
    if "indices" in spec:
        # a single explicit site, possibly multi-atom (e.g. an existing -OH)
        return {"site" + "-".join(map(str, spec["indices"])): list(spec["indices"])}
    if "smarts" in spec:
        # The SMARTS's *first* atom is the one that gets removed, so target the
        # leaving atom directly, e.g. "[H]" for any hydrogen, "[H][c]" for an
        # aromatic C-H, "[H][CX4]" for an sp3 C-H -- not "[cH]", which matches
        # the aromatic carbon itself (and will fail with "found 3" neighbors).
        matches = find_atoms_by_smarts(host_mol, spec["smarts"])
        if not matches:
            raise ValueError(f"SMARTS {spec['smarts']!r} matched no atoms in host")
        return {f"atom{idx}": [idx] for idx in matches}
    raise ValueError("Site spec must have either 'indices' or 'smarts'")


def run_substitute_batch(config: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    substituents = _resolve_substituents(config)
    relax = config.get("relax", True)

    route = config.get("route")
    title = config.get("title")
    charge_override = config.get("charge")
    mult_override = config.get("multiplicity")

    with open(manifest_path, "w", newline="") as manifest_fh:
        writer = csv.writer(manifest_fh)
        writer.writerow(["job_name", "host", "site", "substituent", "status", "file", "detail"])

        for host_spec in config["hosts"]:
            host_path = host_spec["path"]
            host_name = host_spec.get("name", Path(host_path).stem)
            job = read_gaussian(host_path)
            host_mol = job_to_mol(job)

            sites = _resolve_sites(host_mol, config["sites"])

            for site_label, leaving_indices in sites.items():
                for sub_name, sub_smiles in substituents.items():
                    job_name = f"{host_name}_{site_label}_{sub_name}"
                    out_gjf = output_dir / f"{job_name}.gjf"
                    try:
                        product = substitute(
                            host_mol, leaving_indices, sub_smiles, relax=relax
                        )
                        out_job = GaussianJob(
                            link0=list(job.link0),
                            route=[route] if route else list(job.route),
                            title=title or f"{job_name}",
                            charge=charge_override if charge_override is not None else job.charge,
                            multiplicity=mult_override if mult_override is not None else job.multiplicity,
                            atoms=mol_to_atoms(product),
                        )
                        write_gaussian(out_job, str(out_gjf))
                        write_xyz(product, str(out_gjf.with_suffix(".xyz")), comment=job_name)
                        writer.writerow(
                            [job_name, host_name, site_label, sub_name, "ok", str(out_gjf), ""]
                        )
                    except Exception as exc:  # one bad combination shouldn't kill the batch
                        writer.writerow(
                            [job_name, host_name, site_label, sub_name, "failed", "", str(exc)]
                        )

    return manifest_path


def run_combine_batch(config: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"

    job_a = read_gaussian(config["molecule_a"]["path"])
    job_b = read_gaussian(config["molecule_b"]["path"])
    name_a = config["molecule_a"].get("name", Path(config["molecule_a"]["path"]).stem)
    name_b = config["molecule_b"].get("name", Path(config["molecule_b"]["path"]).stem)

    mode = config.get("mode", "atom_pair")
    tag_fragments = config.get("tag_fragments", False)

    with open(manifest_path, "w", newline="") as manifest_fh:
        writer = csv.writer(manifest_fh)
        writer.writerow(["job_name", "molecule_a", "molecule_b", "mode", "params", "status", "file", "detail"])

        if mode == "atom_pair" and "distances" in config:
            atom_lists = distance_scan(
                job_a.atoms,
                job_b.atoms,
                atom_idx_a=config["atom_idx_a"],
                atom_idx_b=config["atom_idx_b"],
                distances=config["distances"],
                direction=tuple(config.get("direction", (0.0, 0.0, 1.0))),
            )
            jobs = []
            for d, atoms in zip(config["distances"], atom_lists):
                if not tag_fragments:
                    for atom in atoms:
                        atom.fragment = None
                jobs.append((f"d{d:.2f}", atoms, d))
        else:
            combined_job = combine(job_a, job_b, mode=mode, tag_fragments=tag_fragments, **config.get("params", {}))
            jobs = [("combined", combined_job.atoms, None)]

        for label, atoms, param in jobs:
            job_name = f"{name_a}_{name_b}_{label}"
            out_gjf = output_dir / f"{job_name}.gjf"
            try:
                charge = config.get("charge")
                multiplicity = config.get("multiplicity")
                if charge is None or multiplicity is None:
                    inferred_c, inferred_m = combined_charge_multiplicity(
                        job_a.charge, job_a.multiplicity, job_b.charge, job_b.multiplicity
                    )
                    charge = inferred_c if charge is None else charge
                    multiplicity = inferred_m if multiplicity is None else multiplicity

                out_job = GaussianJob(
                    link0=list(job_a.link0),
                    route=[config["route"]] if "route" in config else list(job_a.route),
                    title=job_name,
                    charge=charge,
                    multiplicity=multiplicity,
                    fragment_charges=(
                        [(job_a.charge, job_a.multiplicity), (job_b.charge, job_b.multiplicity)]
                        if tag_fragments
                        else []
                    ),
                    atoms=atoms,
                )
                write_gaussian(out_job, str(out_gjf))
                writer.writerow([job_name, name_a, name_b, mode, str(param), "ok", str(out_gjf), ""])
            except Exception as exc:
                writer.writerow([job_name, name_a, name_b, mode, str(param), "failed", "", str(exc)])

    return manifest_path


def run_batch(config_path: str) -> Path:
    with open(config_path) as fh:
        config = yaml.safe_load(fh)

    output_dir = Path(config.get("output_dir", "batch_output"))
    job_type = config.get("job_type", "substitute")

    if job_type == "substitute":
        return run_substitute_batch(config, output_dir)
    if job_type == "combine":
        return run_combine_batch(config, output_dir)
    raise ValueError(f"Unknown job_type: {job_type!r}")

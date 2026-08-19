"""Config-driven batch substitution: enumerate hosts x sites x substituents
and write one .gjf per job plus a manifest CSV logging what went into each
file. See Substitutor/library/groups.yaml for the substituent library format
and substitute.py for the underlying single-job operation.
"""
from __future__ import annotations

import csv
from pathlib import Path

import yaml
from rdkit import Chem

from common.gaussian import GaussianJob, read_gaussian, write_gaussian
from common.molecule import job_to_mol, mol_to_atoms
from common.xyz import write_xyz
from Substitutor.substitute import find_atoms_by_smarts, substitute


def _load_library(path: str) -> dict[str, str]:
    with open(path) as fh:
        entries = yaml.safe_load(fh) or []
    return {entry["name"]: entry["smiles"] for entry in entries}


def _resolve_substituents(config: dict) -> dict[str, str]:
    spec = config.get("substituents", {})
    library_path = spec.get(
        "library", str(Path(__file__).resolve().parent / "library" / "groups.yaml")
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

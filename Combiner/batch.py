"""Config-driven batch combination: build one or more combined-molecule
Gaussian inputs (e.g. a distance scan for an interaction-energy series) and
log them to a manifest CSV. See combine.py for the underlying single-job
operation.
"""
from __future__ import annotations

import csv
from pathlib import Path

from common.gaussian import GaussianJob, read_gaussian, write_gaussian
from Combiner.combine import combine, combined_charge_multiplicity, distance_scan


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

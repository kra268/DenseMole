"""Command-line entry point for DenseMole.

    python -m cli substitute host.gjf --atom 3 --group "*O" -o out.gjf
    python -m cli combine a.gjf b.gjf --atom-a 0 --atom-b 0 --distance 3.5 -o dimer.gjf
    python -m cli batch config.yaml
"""
from __future__ import annotations

from pathlib import Path

import click
import yaml
from rdkit import RDLogger

from common.gaussian import read_gaussian, write_gaussian
from common.molecule import job_to_mol, mol_to_atoms
from common.xyz import write_xyz
from Combiner.batch import run_combine_batch
from Combiner.combine import combine
from Substitutor.batch import run_substitute_batch
from Substitutor.substitute import substitute

RDLogger.DisableLog("rdApp.*")


@click.group()
def cli():
    pass


@cli.command()
@click.argument("host", type=click.Path(exists=True))
@click.option("--atom", "atoms", multiple=True, type=int, required=True, help="Index of an atom to remove (repeat for a multi-atom leaving group)")
@click.option("--group", "group_smiles", required=True, help="'*'-tagged SMILES for the replacement, e.g. '*O'")
@click.option("--no-relax", is_flag=True, help="Skip force-field relaxation of the new atoms")
@click.option("-o", "--output", "output", required=True, type=click.Path(), help="Output .gjf path")
def substitute_cmd(host, atoms, group_smiles, no_relax, output):
    job = read_gaussian(host)
    host_mol = job_to_mol(job)
    product = substitute(host_mol, list(atoms), group_smiles, relax=not no_relax)
    job.atoms = mol_to_atoms(product)
    write_gaussian(job, output)
    write_xyz(product, str(output).rsplit(".", 1)[0] + ".xyz")
    click.echo(f"Wrote {output}")


cli.add_command(substitute_cmd, name="substitute")


@cli.command()
@click.argument("molecule_a", type=click.Path(exists=True))
@click.argument("molecule_b", type=click.Path(exists=True))
@click.option("--atom-a", type=int, required=True)
@click.option("--atom-b", type=int, required=True)
@click.option("--distance", type=float, required=True, help="Separation in Angstrom")
@click.option("--direction", nargs=3, type=float, default=(0.0, 0.0, 1.0))
@click.option("--tag-fragments", is_flag=True, help="Emit Fragment=N labels and per-fragment charge/mult")
@click.option("-o", "--output", required=True, type=click.Path())
def combine_cmd(molecule_a, molecule_b, atom_a, atom_b, distance, direction, tag_fragments, output):
    job_a = read_gaussian(molecule_a)
    job_b = read_gaussian(molecule_b)
    combined = combine(
        job_a,
        job_b,
        mode="atom_pair",
        atom_idx_a=atom_a,
        atom_idx_b=atom_b,
        distance=distance,
        direction=direction,
        tag_fragments=tag_fragments,
    )
    write_gaussian(combined, output)
    click.echo(f"Wrote {output}")


cli.add_command(combine_cmd, name="combine")


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
def batch(config_path):
    with open(config_path) as fh:
        config = yaml.safe_load(fh)

    output_dir = Path(config.get("output_dir", "batch_output"))
    job_type = config.get("job_type", "substitute")

    if job_type == "substitute":
        manifest = run_substitute_batch(config, output_dir)
    elif job_type == "combine":
        manifest = run_combine_batch(config, output_dir)
    else:
        raise click.ClickException(f"Unknown job_type: {job_type!r}")

    click.echo(f"Batch complete. Manifest: {manifest}")


if __name__ == "__main__":
    cli()

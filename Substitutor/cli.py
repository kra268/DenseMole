"""Command-line entry point for DenseMole.

    python -m Substitutor.cli substitute host.gjf --atom 3 --group "*O" -o out.gjf
    python -m Substitutor.cli combine a.gjf b.gjf --atom-a 0 --atom-b 0 --distance 3.5 -o dimer.gjf
    python -m Substitutor.cli batch config.yaml
"""
from __future__ import annotations

import click
from rdkit import RDLogger

from Substitutor.batch.runner import run_batch
from Substitutor.core.combine import combine
from Substitutor.core.molecule import job_to_mol, mol_to_atoms
from Substitutor.core.substitute import substitute
from Substitutor.io.gaussian import read_gaussian, write_gaussian
from Substitutor.io.xyz import write_xyz

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
@click.argument("config", type=click.Path(exists=True))
def batch(config):
    manifest = run_batch(config)
    click.echo(f"Batch complete. Manifest: {manifest}")


if __name__ == "__main__":
    cli()

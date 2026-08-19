# DenseMole

A tool for computational chemists to set up large numbers of Gaussian
calculations: substituting functional groups into a molecule, or combining
two molecules into a single non-bonded complex.

Chemistry is handled by RDKit (bond perception, 3D embedding of substituents,
force-field relaxation), with a thin Gaussian `.gjf` reader/writer layered on
top. See [Substitutor/core/substitute.py](Substitutor/core/substitute.py) for
the substitution algorithm and [Substitutor/core/combine.py](Substitutor/core/combine.py)
for molecule combination.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

**Single substitution** — replace atom(s) in a host molecule with a
`*`-tagged SMILES functional group (`*` marks the attachment point):

```bash
python -m Substitutor.cli substitute Substitutor/Example_Data/CH4.gjf \
    --atom 1 --group "*O" -o methanol.gjf
```

Pass `--atom` more than once to remove a whole existing multi-atom group
(e.g. an existing `-OH`) rather than a single atom.

**Combine two molecules** into one non-bonded complex, placing a chosen atom
of molecule B at a given distance from a chosen atom of molecule A:

```bash
python -m Substitutor.cli combine a.gjf b.gjf \
    --atom-a 0 --atom-b 0 --distance 3.5 --tag-fragments -o complex.gjf
```

**Batch generation** — enumerate hosts x sites x substituents (or a distance
scan for combined molecules) from a YAML config, writing one `.gjf` per job
plus a `manifest.csv`:

```bash
python -m Substitutor.cli batch my_batch_config.yaml
```

See [Substitutor/library/groups.yaml](Substitutor/library/groups.yaml) for
the built-in substituent library (extendable/overridable per batch config).

## Layout

```
Substitutor/
  io/gaussian.py    # .gjf reader/writer
  io/xyz.py         # .xyz export for visual QC
  core/molecule.py  # GaussianJob <-> RDKit Mol bridge (bond perception)
  core/substitute.py # functional-group substitution
  core/combine.py    # two-molecule non-bonded complex assembly
  library/groups.yaml # built-in substituent library
  batch/runner.py    # YAML-driven batch enumeration + manifest
  cli.py              # command-line entry point
```

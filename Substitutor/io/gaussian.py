"""Reading and writing Gaussian (.gjf/.com) input files.

A Gaussian input file is a sequence of blank-line-delimited sections:

    %chk=...                  <- link0 (optional, 0+ lines)
    %mem=...
    #p B3LYP/6-31G(d) Opt      <- route section (1+ lines, starts with '#')
                                <- blank line
    Title Card Required        <- title section (1+ lines)
                                <- blank line
    0 1                         <- charge/multiplicity (one pair per fragment)
    C   0.000000   0.000000   0.000000   <- molecule specification
    H   ...
                                <- blank line
    [basis set / other trailing sections]

This module parses by section rather than by counting lines, so it tolerates
multi-line route sections, multiple link0 lines, and trailing sections it
doesn't otherwise understand (those are preserved verbatim and re-emitted).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Atom:
    symbol: str
    coords: tuple[float, float, float]
    fragment: int | None = None  # 1-based fragment index, for combined/fragment jobs


@dataclass
class GaussianJob:
    link0: list[str] = field(default_factory=list)
    route: list[str] = field(default_factory=list)
    title: str = "Title Card Required"
    charge: int = 0
    multiplicity: int = 1
    # For multi-fragment jobs (e.g. counterpoise), one (charge, multiplicity)
    # pair per fragment, in addition to the overall charge/multiplicity above.
    fragment_charges: list[tuple[int, int]] = field(default_factory=list)
    atoms: list[Atom] = field(default_factory=list)
    trailing: str = ""  # anything after the geometry block (basis sets, etc.)


def _split_sections(lines: list[str]) -> list[list[str]]:
    """Split lines into blank-line-delimited sections, dropping the blank lines."""
    sections: list[list[str]] = []
    current: list[str] = []
    for raw in lines:
        line = raw.rstrip("\n")
        if line.strip() == "":
            if current:
                sections.append(current)
                current = []
            continue
        current.append(line)
    if current:
        sections.append(current)
    return sections


def read_gaussian(file_path: str) -> GaussianJob:
    with open(file_path, "r") as fh:
        raw_lines = fh.readlines()

    sections = _split_sections(raw_lines)
    if not sections:
        raise ValueError(f"{file_path}: no content found")

    idx = 0
    link0: list[str] = []
    route: list[str] = []

    # link0 lines (%...) and the route section share a block: link0 lines
    # come first, then one or more lines starting the route with '#'.
    first_block = sections[idx]
    idx += 1
    for line in first_block:
        if line.strip().startswith("%"):
            link0.append(line.strip())
        else:
            route.append(line.strip())
    if not route or not route[0].startswith("#"):
        raise ValueError(f"{file_path}: expected a route section starting with '#'")

    if idx >= len(sections):
        raise ValueError(f"{file_path}: missing title section")
    title = " ".join(s.strip() for s in sections[idx])
    idx += 1

    if idx >= len(sections):
        raise ValueError(f"{file_path}: missing charge/multiplicity/geometry section")
    geometry_block = sections[idx]
    idx += 1

    header_line = geometry_block[0].split()
    if len(header_line) < 2 or len(header_line) % 2 != 0:
        raise ValueError(
            f"{file_path}: malformed charge/multiplicity line: {geometry_block[0]!r}"
        )
    charge_mult_pairs = [
        (int(header_line[i]), int(header_line[i + 1]))
        for i in range(0, len(header_line), 2)
    ]
    charge, multiplicity = charge_mult_pairs[-1]
    fragment_charges = charge_mult_pairs[:-1]

    atoms: list[Atom] = []
    for line in geometry_block[1:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        symbol_field = parts[0]
        fragment = None
        symbol = symbol_field
        if "(" in symbol_field:
            # e.g. "C(Fragment=1)"
            symbol, tag = symbol_field.split("(", 1)
            tag = tag.rstrip(")")
            for kv in tag.split(","):
                if "=" in kv:
                    key, val = kv.split("=", 1)
                    if key.strip().lower() == "fragment":
                        fragment = int(val)
        x, y, z = (float(v) for v in parts[1:4])
        atoms.append(Atom(symbol=symbol, coords=(x, y, z), fragment=fragment))

    trailing = "\n\n".join(
        "\n".join(sec) for sec in sections[idx:]
    )

    return GaussianJob(
        link0=link0,
        route=route,
        title=title,
        charge=charge,
        multiplicity=multiplicity,
        fragment_charges=fragment_charges,
        atoms=atoms,
        trailing=trailing,
    )


def write_gaussian(job: GaussianJob, file_path: str) -> None:
    lines: list[str] = []
    lines.extend(job.link0)
    lines.extend(job.route)
    lines.append("")
    lines.append(job.title)
    lines.append("")

    header_parts = []
    for c, m in job.fragment_charges:
        header_parts.extend([str(c), str(m)])
    header_parts.extend([str(job.charge), str(job.multiplicity)])
    lines.append(" ".join(header_parts))

    for atom in job.atoms:
        x, y, z = atom.coords
        symbol = atom.symbol
        if atom.fragment is not None:
            symbol = f"{symbol}(Fragment={atom.fragment})"
        lines.append(f"{symbol:<16s}{x:12.6f}{y:12.6f}{z:12.6f}")

    lines.append("")
    if job.trailing:
        lines.append(job.trailing)
        lines.append("")

    with open(file_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")

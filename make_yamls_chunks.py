#!/usr/bin/env python3
"""Generate Boltz-2 affinity YAMLs from a Parquet or CSV file of compounds.

One protein target, many ligands. Each YAML is named "<CompoundIndex>.yaml", so
the compound id travels through Boltz untouched: it names the prediction directory
and the affinity JSON (affinity_<CompoundIndex>.json), and collect_results.py reads
it straight back from the filename. No separate index map is kept.

Because the id BECOMES a filename, every CompoundIndex must be a safe, unique
filename stem: letters, digits, and  . _ + -  only; no spaces, slashes or quotes;
not starting with '-' or '.'. The script validates this up front and refuses to run
if any id is unsafe or duplicated, rather than silently mangling it (which, with no
map, would be unrecoverable).

The input (Parquet or CSV/TSV; format auto-detected from the extension, with a
content sniff as fallback) needs a SMILES column and an id column (defaults 'smiles'
and 'CompoundIndex'; override with --smiles-col / --id-col).

Usage:
  python make_yamls_chunks.py \
      --protein protein.fasta \
      --input   compounds.parquet \
      --msa     msa/protein.csv \
      --outdir  input --chunk-size 2000 \
      [--smiles-col smiles] [--id-col CompoundIndex] [--skip-existing]

  (--input also accepts .csv / .tsv; --parquet still works as an alias.)
"""
import argparse, os, re
from pathlib import Path

_UNSAFE = re.compile(r"[^A-Za-z0-9._+-]")


def load_compounds(pd, path):
    """Read a compounds table, detecting Parquet vs CSV/TSV from the extension
    (with a parquet magic-byte sniff as a fallback for unknown extensions)."""
    suf = Path(path).suffix.lower()
    if suf in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if suf == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suf == ".csv":
        return pd.read_csv(path)
    with open(path, "rb") as fh:          # unknown extension: sniff the content
        magic = fh.read(4)
    if magic == b"PAR1":                  # parquet files start with "PAR1"
        return pd.read_parquet(path)
    return pd.read_csv(path)


def read_seq(fasta: str) -> str:
    out = []
    for line in Path(fasta).read_text().splitlines():
        if line.startswith(">") or not line.strip():
            continue
        out.append(line.strip())
    return "".join(out)


def stem_problem(stem: str):
    """Return a reason string if `stem` is not a safe filename, else None."""
    if not stem:
        return "empty"
    if stem.startswith("-") or stem.startswith("."):
        return "starts with '-' or '.'"
    hit = _UNSAFE.findall(stem)
    if hit:
        return "contains unsafe char(s): " + repr("".join(sorted(set(hit))))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protein", required=True)
    ap.add_argument("--input", "--parquet", dest="input", required=True,
                    help="Compounds file (.parquet/.pq/.csv/.tsv; auto-detected)")
    ap.add_argument("--msa", required=True, help="Shared protein MSA (.csv or .a3m)")
    ap.add_argument("--outdir", default="input")
    ap.add_argument("--chunk-size", type=int, default=2000)
    ap.add_argument("--smiles-col", default="smiles")
    ap.add_argument("--id-col", default="CompoundIndex")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Don't rewrite a YAML that already exists (fast re-run)")
    args = ap.parse_args()

    try:
        import pandas as pd
    except ImportError:
        raise SystemExit("Needs pandas + a parquet engine:  pip install pandas pyarrow")

    seq = read_seq(args.protein)
    if not seq:
        raise SystemExit("No protein sequence found in --protein.")
    msa = os.path.abspath(args.msa)
    if not os.path.exists(msa):
        raise SystemExit(f"MSA not found: {msa}")

    df = load_compounds(pd, args.input)
    for col in (args.smiles_col, args.id_col):
        if col not in df.columns:
            raise SystemExit(f"Column '{col}' not in input. Found: {list(df.columns)}")

    # ---- Pass 1: collect (id, smiles, stem); validate safety + uniqueness ------
    rows = []          # (cid, smi, stem) for rows with a valid SMILES
    seen = set()
    bad, dups = [], []
    for cid, smiles in zip(df[args.id_col].tolist(), df[args.smiles_col].tolist()):
        if pd.isna(smiles) or not str(smiles).strip():
            continue
        stem = str(cid).strip()
        why = stem_problem(stem)
        if why:
            if len(bad) < 10:
                bad.append(f"{cid!r} ({why})")
            continue
        if stem in seen:
            if len(dups) < 10:
                dups.append(repr(cid))
            continue
        seen.add(stem)
        rows.append((cid, str(smiles).strip().replace("'", ""), stem))

    if bad or dups:
        msg = ["Refusing to run: each CompoundIndex must be a safe, UNIQUE filename stem",
               "(letters/digits and . _ + - only; no spaces/slashes/quotes; not leading - or .)."]
        if bad:
            msg.append(f"  {len(bad)}{'+' if len(bad) == 10 else ''} unsafe id(s), e.g.: "
                       + "; ".join(bad))
        if dups:
            msg.append(f"  {len(dups)}{'+' if len(dups) == 10 else ''} duplicate id(s), e.g.: "
                       + ", ".join(dups))
        msg.append("Clean the ids and re-run.")
        raise SystemExit("\n".join(msg))

    if not rows:
        raise SystemExit("No rows with a valid SMILES found.")

    # ---- Pass 2: write the YAMLs ----------------------------------------------
    outdir = Path(args.outdir)
    written = 0
    for i, (cid, smi, stem) in enumerate(rows):
        chunk = f"chunk_{i // args.chunk_size:03d}"
        cdir = outdir / chunk
        cdir.mkdir(parents=True, exist_ok=True)
        ypath = cdir / f"{stem}.yaml"
        if args.skip_existing and ypath.exists():
            continue
        ypath.write_text(
            f"# CompoundIndex: {cid}\n"
            f"# SMILES: {smi}\n"
            "version: 1\n"
            "sequences:\n"
            "  - protein:\n"
            "      id: A\n"
            f"      sequence: {seq}\n"
            f"      msa: {msa}\n"
            "  - ligand:\n"
            "      id: B\n"
            f"      smiles: '{smi}'\n"
            "properties:\n"
            "  - affinity:\n"
            "      binder: B\n"
        )
        written += 1

    n = len(rows)
    nchunks = (n + args.chunk_size - 1) // args.chunk_size
    print(f"{n} compound(s); {written} YAML(s) written "
          f"({n - written} skipped) into {nchunks} chunk dir(s) under {outdir}/")
    print(f">> SLURM array range:  0-{nchunks - 1}   ({nchunks} tasks)")
    if nchunks > 1000:
        print(f"!! {nchunks} chunks likely exceeds the cluster MaxArraySize. "
              f"Check `scontrol show config | grep MaxArraySize` and raise --chunk-size.")


if __name__ == "__main__":
    main()

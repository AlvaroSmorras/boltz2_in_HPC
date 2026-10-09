#!/usr/bin/env python3
"""Generate Boltz-2 affinity YAMLs from a Parquet file of compounds.

The Parquet must contain (at least) a SMILES column and a compound-id column
(defaults: 'smiles' and 'CompoundIndex').

How the compound id is tracked through Boltz
--------------------------------------------
Boltz names every output after the INPUT FILE STEM (its internal record.id):
    out/.../predictions/<stem>/affinity_<stem>.json
So we name each YAML "<stem>.yaml", where <stem> encodes the CompoundIndex.
That filename IS the stamp: it propagates into the results path automatically.

We ALSO write an authoritative map (built from in-loop lists, NOT via the
DataFrame index, so a non-unique Parquet index can't break it):
    <outdir>/index_map.parquet   columns: CompoundIndex, yaml_stem, chunk, smiles

Usage:
  python make_yamls.py \
      --protein protein.fasta \
      --parquet compounds.parquet \
      --msa     msa/protein.csv \
      --outdir  inputs --chunk-size 20 \
      [--smiles-col smiles] [--id-col CompoundIndex] [--skip-existing]
"""
import argparse, os, re
from pathlib import Path


def read_seq(fasta: str) -> str:
    out = []
    for line in Path(fasta).read_text().splitlines():
        if line.startswith(">") or not line.strip():
            continue
        out.append(line.strip())
    return "".join(out)


def safe_stem(value) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]", "_", str(value)).strip("_.")
    return s or "cpd"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protein", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--msa", required=True, help="Shared protein MSA (.csv or .a3m)")
    ap.add_argument("--outdir", default="inputs")
    ap.add_argument("--chunk-size", type=int, default=20)
    ap.add_argument("--smiles-col", default="smiles")
    ap.add_argument("--id-col", default="CompoundIndex")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Don't rewrite a YAML that already exists (fast re-run / map rebuild)")
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
        raise SystemExit(f"MSA not found: {msa} (run prepare_msa.sh first).")

    df = pd.read_parquet(args.parquet)
    for col in (args.smiles_col, args.id_col):
        if col not in df.columns:
            raise SystemExit(f"Column '{col}' not in parquet. Found: {list(df.columns)}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    seen = {}                                   # stem -> times used (collision guard)
    cids, smis, stems, chunks = [], [], [], []  # parallel lists -> the map
    dup = 0
    written = 0

    # Iterate columns BY POSITION (no reliance on the DataFrame index).
    for cid, smiles in zip(df[args.id_col].tolist(), df[args.smiles_col].tolist()):
        if pd.isna(smiles) or not str(smiles).strip():
            continue
        smi = str(smiles).strip().replace("'", "")
        stem = safe_stem(cid)
        if stem in seen:
            dup += 1
            seen[stem] += 1
            stem = f"{stem}__{seen[stem]}"
        else:
            seen[stem] = 0

        n = len(stems)
        chunk = f"chunk_{n // args.chunk_size:03d}"
        cdir = outdir / chunk
        cdir.mkdir(parents=True, exist_ok=True)
        ypath = cdir / f"{stem}.yaml"
        if not (args.skip_existing and ypath.exists()):
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

        cids.append(cid); smis.append(smi); stems.append(stem); chunks.append(chunk)

    if not stems:
        raise SystemExit("No rows with a valid SMILES found.")

    amap = pd.DataFrame({
        "CompoundIndex": cids,
        "yaml_stem": stems,
        "chunk": chunks,
        "smiles": smis,
    })
    amap.to_parquet(outdir / "index_map.parquet", index=False)
    amap.to_csv(outdir / "index_map.csv", index=False)

    n = len(stems)
    nchunks = (n + args.chunk_size - 1) // args.chunk_size
    print(f"{n} compound(s); {written} YAML(s) written "
          f"({n - written} skipped) into {nchunks} chunk dir(s) under {outdir}/")
    print(f"Map: {outdir / 'index_map.parquet'} (+ .csv)")
    if dup:
        print(f"WARNING: {dup} id(s) collided after sanitising; disambiguated with "
              f"__N suffixes (true ids kept in the map).")
    print(f">> SLURM array range:  0-{nchunks - 1}   ({nchunks} tasks)")
    if nchunks > 1000:
        print(f"!! {nchunks} chunks likely exceeds the cluster MaxArraySize. "
              f"Check `scontrol show config | grep MaxArraySize` and raise --chunk-size "
              f"so nchunks fits under it.")


if __name__ == "__main__":
    main()

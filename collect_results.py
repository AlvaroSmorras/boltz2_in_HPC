#!/usr/bin/env python3
"""Collect Boltz-2 affinity results, keyed by CompoundIndex.

Each compound produces  affinity_<CompoundIndex>.json  (the YAML was named after its
CompoundIndex, so Boltz stamps it into the filename). The CompoundIndex is read
straight from each filename.

Values are reported EXACTLY as Boltz writes them -- no conversions, no derived
quantities. Columns keep Boltz's own field names:
  affinity_pred_value            Boltz's primary affinity prediction (lower = stronger)
  affinity_probability_binary    Boltz's binder probability, 0..1
  affinity_pred_value1, 2, ...   individual ensemble-member values (when present)
  affinity_probability_binary1, 2, ...  member binder probabilities (when present)
  n_members                      how many numbered members the JSON contained

File discovery: affinity_*.json directly inside --indir (flat folder) by default;
use --recursive for a nested layout.

Usage:
  python collect_results.py --indir affinity_jsons --output affinity_results.parquet \
      [--merge-into compounds.parquet] [--id-col CompoundIndex] [--recursive]

--output format is auto-detected from the extension: .parquet/.pq, .csv, .tsv.
"""
import argparse, json, re
from pathlib import Path

_NUMBERED = re.compile(r"affinity_(pred_value|probability_binary)(\d+)$")


def write_table(df, path):
    """Write a DataFrame, picking the format from the file extension."""
    p = Path(path)
    suf = p.suffix.lower()
    if suf in (".parquet", ".pq"):
        df.to_parquet(p, index=False)
    elif suf == ".csv":
        df.to_csv(p, index=False)
    elif suf == ".tsv":
        df.to_csv(p, index=False, sep="\t")
    else:
        raise SystemExit(f"Unsupported --output extension '{p.suffix}'. "
                         f"Use .parquet, .pq, .csv, or .tsv.")
    return p


def parse_one(d):
    """Pull Boltz's affinity fields out of one JSON, unmodified."""
    row = {
        "affinity_pred_value": d.get("affinity_pred_value"),
        "affinity_probability_binary": d.get("affinity_probability_binary"),
    }
    numbered = {k: v for k, v in d.items() if _NUMBERED.match(k)}
    row["n_members"] = sum(1 for k in numbered if k.startswith("affinity_pred_value"))
    # keep numbered members under their original Boltz names, in numeric order
    for k in sorted(numbered, key=lambda k: (_NUMBERED.match(k).group(1),
                                             int(_NUMBERED.match(k).group(2)))):
        row[k] = numbered[k]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", "--outdir", dest="indir", default="out",
                    help="Folder with affinity_*.json (flat by default)")
    ap.add_argument("--recursive", action="store_true",
                    help="Recurse into subdirectories (nested layout)")
    ap.add_argument("--output", default="affinity_results.parquet",
                    help="Results file; format auto-detected from extension "
                         "(.parquet/.pq, .csv, .tsv)")
    ap.add_argument("--merge-into", default=None,
                    help="Original compounds table; also writes "
                         "<name>_with_affinity.<ext> joined on the id column")
    ap.add_argument("--id-col", default="CompoundIndex",
                    help="Name of the id column in --merge-into to join on")
    args = ap.parse_args()

    import pandas as pd

    indir = Path(args.indir)
    files = (indir.rglob("affinity_*.json") if args.recursive
             else indir.glob("affinity_*.json"))

    recs = []
    for af in files:
        cid = af.name[len("affinity_"):-len(".json")]   # CompoundIndex from filename
        try:
            d = json.loads(af.read_text())
        except Exception as e:
            print(f"skip {af}: {e}")
            continue
        row = {"CompoundIndex": cid}
        row.update(parse_one(d))
        recs.append(row)
    if not recs:
        where = "recursively under" if args.recursive else "directly in"
        raise SystemExit(f"No affinity_*.json found {where} {indir}/")

    merged = pd.DataFrame(recs).sort_values("affinity_pred_value", na_position="last")
    out_path = write_table(merged, args.output)
    nmem = int(merged["n_members"].max())
    print(f"Wrote {len(merged)} rows ({nmem} ensemble member(s) per compound) "
          f"-> {out_path}")

    if args.merge_into:
        suf = Path(args.merge_into).suffix.lower()
        if suf in (".parquet", ".pq"):
            orig = pd.read_parquet(args.merge_into)
        elif suf == ".tsv":
            orig = pd.read_csv(args.merge_into, sep="\t")
        else:
            orig = pd.read_csv(args.merge_into)
        orig = orig.copy()
        if args.id_col not in orig.columns:
            raise SystemExit(f"--merge-into has no '{args.id_col}' column. "
                             f"Found: {list(orig.columns)} (set --id-col).")
        right = merged.copy()
        right["__k"] = right["CompoundIndex"].astype(str)   # dtype-proof join key
        orig["__k"] = orig[args.id_col].astype(str)
        aug = orig.merge(right.drop(columns="CompoundIndex"), on="__k", how="left")
        aug = aug.drop(columns="__k")
        missing = int(aug["affinity_pred_value"].isna().sum())
        if missing:
            print(f"NOTE: {missing}/{len(aug)} compounds in {args.merge_into} "
                  f"have no affinity result (crashed or not finished).")
        dst = Path(args.merge_into)
        dst = dst.with_name(dst.stem + "_with_affinity" + dst.suffix)
        write_table(aug, dst)
        print(f"Merged affinities back into original -> {dst}")


if __name__ == "__main__":
    main()

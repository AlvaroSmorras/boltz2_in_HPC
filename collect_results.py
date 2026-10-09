#!/usr/bin/env python3
"""Collect Boltz-2 affinity results and key them back to CompoundIndex.

Each compound produces  affinity_<stem>.json  where <stem> is the YAML filename
stem set by make_yamls.py. This script finds those JSONs, recovers the exact
CompoundIndex via index_map.parquet, and emits a table ready to merge into your
Parquet.

File discovery
--------------
By default it looks for affinity_*.json directly inside --indir (a FLAT folder),
which matches a workflow that extracts just the affinity JSONs and deletes the
rest of each prediction directory. Use --recursive for the original nested
out/.../predictions/<stem>/ layout.

Affinity fields (per the Boltz source)
--------------------------------------
  affinity_pred_value          log10(IC50), IC50 in uM; LOWER = stronger. PRIMARY
                               (the combined/ensemble prediction) -> used to rank.
  affinity_probability_binary  P(binder), 0..1. PRIMARY.
  affinity_pred_value1/2 ...   individual ensemble-member values (0+ of them).
  affinity_probability_binary1/2 ...  member binder probabilities.
We keep each member as its own column and add the member mean/std (agreement
signal). Conversions (IC50_uM, pIC50, dG_proxy) are computed from the PRIMARY
value, falling back to the member mean if the primary key is absent.

Usage:
  python collect_results.py \
      --indir affinity_jsons --map inputs/index_map.parquet \
      --output affinity_results.parquet \
      [--merge-into compounds.parquet] [--recursive]

--output format is auto-detected from the extension: .parquet/.pq, .csv, .tsv.
"""
import argparse, json, re, statistics
from pathlib import Path

_V = re.compile(r"affinity_pred_value(\d+)$")
_P = re.compile(r"affinity_probability_binary(\d+)$")


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


def conversions(y):
    if y is None:
        return (None, None, None)
    return (10.0 ** y, 6.0 - y, (6.0 - y) * 1.364)   # IC50_uM, pIC50, dG_proxy


def parse_one(d):
    """Return a flat dict of fields for one affinity JSON (any #members)."""
    base_v = d.get("affinity_pred_value")
    base_p = d.get("affinity_probability_binary")
    members_v = {int(m.group(1)): v for k, v in d.items() if (m := _V.match(k))}
    members_p = {int(m.group(1)): v for k, v in d.items() if (m := _P.match(k))}

    # primary value: prefer the combined field; else the mean of members
    y = base_v
    if y is None and members_v:
        y = statistics.mean(members_v.values())
    ic50, pic50, dg = conversions(y)

    row = {
        "binder_probability": base_p if base_p is not None
        else (statistics.mean(members_p.values()) if members_p else None),
        "affinity_log10IC50uM": y,
        "IC50_uM": ic50, "pIC50": pic50, "dG_proxy_kcal_per_mol": dg,
        "n_members": len(members_v),
    }
    for i in sorted(members_v):
        row[f"affinity_value_{i}"] = members_v[i]
    for i in sorted(members_p):
        row[f"binder_prob_{i}"] = members_p[i]
    if len(members_v) >= 2:
        vals = list(members_v.values())
        row["affinity_value_member_mean"] = statistics.mean(vals)
        row["affinity_value_member_std"] = statistics.stdev(vals)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", "--outdir", dest="indir", default="out",
                    help="Folder with affinity_*.json (flat by default)")
    ap.add_argument("--recursive", action="store_true",
                    help="Recurse into subdirectories (old nested layout)")
    ap.add_argument("--map", default="inputs/index_map.parquet")
    ap.add_argument("--output", default="affinity_results.parquet",
                    help="Results file; format auto-detected from extension "
                         "(.parquet/.pq, .csv, .tsv)")
    ap.add_argument("--merge-into", default=None,
                    help="Original compounds parquet; also writes "
                         "<name>_with_affinity.parquet joined on CompoundIndex")
    args = ap.parse_args()

    import pandas as pd

    indir = Path(args.indir)
    files = (indir.rglob("affinity_*.json") if args.recursive
             else indir.glob("affinity_*.json"))

    recs = []
    for af in files:
        stem = af.name[len("affinity_"):-len(".json")]
        try:
            d = json.loads(af.read_text())
        except Exception as e:
            print(f"skip {af}: {e}")
            continue
        row = {"yaml_stem": stem}
        row.update(parse_one(d))
        recs.append(row)
    if not recs:
        where = "recursively under" if args.recursive else "directly in"
        raise SystemExit(f"No affinity_*.json found {where} {indir}/")
    res = pd.DataFrame(recs)

    amap = pd.read_parquet(args.map)            # CompoundIndex, yaml_stem, chunk, smiles
    merged = amap.merge(res, on="yaml_stem", how="left")

    missing = int(merged["affinity_log10IC50uM"].isna().sum())
    if missing:
        print(f"NOTE: {missing}/{len(merged)} compounds have no affinity result "
              f"(crashed or not finished).")

    merged = merged.sort_values("affinity_log10IC50uM", na_position="last")
    out_path = write_table(merged, args.output)
    nmem = int(res["n_members"].max()) if "n_members" in res else 0
    print(f"Wrote {len(merged)} rows ({nmem} ensemble member(s) per compound) "
          f"-> {out_path}")

    if args.merge_into:
        orig = pd.read_parquet(args.merge_into).copy()
        metric_cols = [c for c in merged.columns
                       if c not in ("CompoundIndex", "yaml_stem", "chunk", "smiles")]
        right = merged[["CompoundIndex"] + metric_cols].copy()
        # string-normalised join key -> dtype-proof
        right["__k"] = right["CompoundIndex"].astype(str)
        orig["__k"] = orig["CompoundIndex"].astype(str)
        aug = orig.merge(right.drop(columns="CompoundIndex"), on="__k", how="left")
        aug = aug.drop(columns="__k")
        dst = Path(args.merge_into)
        dst = dst.with_name(dst.stem + "_with_affinity" + dst.suffix)
        write_table(aug, dst)
        print(f"Merged affinities back into original -> {dst}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build a report from the collected Boltz-2 affinities.

Everything is based directly on Boltz's own output fields, with no conversions:
  affinity_pred_value          (lower = stronger predicted binding)
  affinity_probability_binary  (binder probability, 0..1)

Produces, under --outdir:
  * histogram PNGs (affinity_pred_value, affinity_probability_binary) and a scatter
    of the two,
  * one PNG per top-K compound structure (structures/rankNN_<id>.png) plus grids,
  * report.pdf bundling a summary page, the plots and the structures,
  * top_k.csv with the ranked table.

Structures need SMILES. If the results file has no SMILES column, pass the original
compounds table with --smiles-from (joined on the id column).

Usage:
  python make_report.py --results affinity_results.parquet \
      --smiles-from compounds.parquet --top-k 20 --outdir report
"""
import argparse, io, math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def read_table(path):
    import pandas as pd
    suf = Path(path).suffix.lower()
    if suf in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if suf == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suf == ".csv":
        return pd.read_csv(path)
    raise SystemExit(f"Unsupported table extension '{suf}' for {path}")


def _ok(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def mol_image(smiles, size=(360, 300)):
    """Render a SMILES to (image array, png bytes), or (None, None) if unparseable."""
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D
    mol = Chem.MolFromSmiles(str(smiles)) if smiles is not None else None
    if mol is None:
        return None, None
    d = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
    d.FinishDrawing()
    png = d.GetDrawingText()
    return plt.imread(io.BytesIO(png), format="png"), png


def legend_for(rank, cid, aff, prob):
    txt = f"#{rank}  {cid}"
    if _ok(aff):
        txt += f"\naffinity_pred_value={aff:.3f}"
    if _ok(prob):
        txt += f"\nP(binder)={prob:.2f}"
    return txt


def hist_fig(values, title, xlabel):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=min(60, max(10, int(len(values) ** 0.5))),
            color="#4C72B0", edgecolor="white", linewidth=0.3)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def scatter_fig(aff, prob):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(aff, prob, s=8, alpha=0.3, color="#4C72B0")
    ax.set_xlabel("affinity_pred_value (kcal/mol)")
    ax.set_ylabel("affinity_probability_binary")
    ax.set_title("Binder probability vs predicted affinity")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def structure_page_fig(page_rows, mols_per_row, rows_per_page, outdir):
    """Draw a page of structures; also write each one to its own PNG."""
    fig, axes = plt.subplots(rows_per_page, mols_per_row,
                             figsize=(mols_per_row * 3.2, rows_per_page * 3.0))
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]
    for ax in axes:
        ax.axis("off")
    for ax, rec in zip(axes, page_rows):
        img, png = mol_image(rec["smiles"])
        if img is None:
            ax.text(0.5, 0.5, f"#{rec['rank']} {rec['cid']}\n(no/invalid SMILES)",
                    ha="center", va="center", fontsize=8)
            continue
        ax.imshow(img)
        ax.set_title(legend_for(rec["rank"], rec["cid"], rec["aff"], rec["prob"]),
                     fontsize=8)
        (outdir / "structures").mkdir(exist_ok=True)
        with open(outdir / "structures" / f"rank{rec['rank']:02d}_{rec['cid']}.png", "wb") as fh:
            fh.write(png)
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="affinity_results file")
    ap.add_argument("--smiles-from", default=None,
                    help="Table with id + SMILES, if --results lacks a SMILES column")
    ap.add_argument("--id-col", default="CompoundIndex")
    ap.add_argument("--affinity-col", default="affinity_pred_value")
    ap.add_argument("--prob-col", default="affinity_probability_binary")
    ap.add_argument("--smiles-col", default="smiles")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--mols-per-row", type=int, default=4)
    ap.add_argument("--rows-per-page", type=int, default=5)
    ap.add_argument("--outdir", default="report")
    args = ap.parse_args()

    import numpy as np
    import pandas as pd

    df = read_table(args.results)
    if args.id_col not in df.columns or args.affinity_col not in df.columns:
        raise SystemExit(f"--results needs '{args.id_col}' and '{args.affinity_col}'. "
                         f"Found: {list(df.columns)}")

    have_smiles = args.smiles_col in df.columns
    if not have_smiles and args.smiles_from:
        src = read_table(args.smiles_from)
        if args.id_col not in src.columns or args.smiles_col not in src.columns:
            raise SystemExit(f"--smiles-from needs '{args.id_col}' and "
                             f"'{args.smiles_col}'. Found: {list(src.columns)}")
        src = src[[args.id_col, args.smiles_col]].copy()
        src["__k"] = src[args.id_col].astype(str)
        df["__k"] = df[args.id_col].astype(str)
        df = df.merge(src.drop(columns=args.id_col), on="__k", how="left").drop(columns="__k")
        have_smiles = True
    if not have_smiles:
        print("WARNING: no SMILES available -> structures will be skipped. "
              "Pass --smiles-from compounds.parquet to enable them.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    has_prob = args.prob_col in df.columns and df[args.prob_col].notna().any()
    valid = df[df[args.affinity_col].notna()].copy()
    if valid.empty:
        raise SystemExit(f"No non-null '{args.affinity_col}' values to report.")
    valid = valid.sort_values(args.affinity_col)          # lower = stronger
    n_total, n_valid = len(df), len(valid)

    topk = valid.head(args.top_k).reset_index(drop=True)
    topk.to_csv(outdir / "top_k.csv", index=False)

    figs = []
    f = hist_fig(valid[args.affinity_col].to_numpy(),
                 "Distribution of affinity_pred_value",
                 "affinity_pred_value  (lower = stronger)")
    f.savefig(outdir / "hist_affinity.png", dpi=150)
    figs.append(f)

    if has_prob:
        f = hist_fig(valid[args.prob_col].dropna().to_numpy(),
                     "Distribution of affinity_probability_binary",
                     "affinity_probability_binary")
        f.savefig(outdir / "hist_probability.png", dpi=150)
        figs.append(f)

        sub = valid[valid[args.prob_col].notna()]
        f = scatter_fig(sub[args.affinity_col].to_numpy(),
                        sub[args.prob_col].to_numpy())
        f.savefig(outdir / "scatter_affinity_vs_prob.png", dpi=150)
        figs.append(f)

    struct_figs = []
    if have_smiles:
        recs = [{"rank": i + 1, "cid": row[args.id_col],
                 "smiles": row.get(args.smiles_col),
                 "aff": row.get(args.affinity_col),
                 "prob": row.get(args.prob_col) if has_prob else None}
                for i, row in topk.iterrows()]
        per_page = args.mols_per_row * args.rows_per_page
        for p0 in range(0, len(recs), per_page):
            f = structure_page_fig(recs[p0:p0 + per_page], args.mols_per_row,
                                   args.rows_per_page, outdir)
            f.savefig(outdir / f"top_structures_page{p0 // per_page + 1}.png", dpi=150)
            struct_figs.append(f)

    # ---- summary page ----------------------------------------------------------
    summ = plt.figure(figsize=(8.27, 11.69))
    a = valid[args.affinity_col]
    lines = [
        "Boltz-2 affinity screen report", "",
        f"compounds in results file : {n_total}",
        f"with an affinity value    : {n_valid}",
        f"missing / unfinished      : {n_total - n_valid}", "",
        f"affinity_pred_value       min {a.min():.3f}   median {a.median():.3f}"
        f"   max {a.max():.3f}",
    ]
    if has_prob:
        p = valid[args.prob_col]
        lines.append(f"affinity_probability_binary  median {p.median():.2f}   max {p.max():.2f}")
    lines += ["", f"Top {len(topk)} compounds (lowest affinity_pred_value):", ""]
    for i, row in topk.iterrows():
        pr = row.get(args.prob_col) if has_prob else None
        prs = f"   P(binder)={pr:.2f}" if _ok(pr) else ""
        lines.append(f"  #{i+1:>2}  {str(row[args.id_col]):<24}"
                     f"affinity_pred_value={row[args.affinity_col]:.3f}{prs}")
    summ.text(0.07, 0.95, "\n".join(lines), va="top", ha="left",
              family="monospace", fontsize=9)
    summ.savefig(outdir / "summary.png", dpi=150)

    with PdfPages(outdir / "report.pdf") as pdf:
        pdf.savefig(summ)
        for f in figs + struct_figs:
            pdf.savefig(f)
    for f in [summ, *figs, *struct_figs]:
        plt.close(f)

    print(f"Report written to {outdir}/  (report.pdf, PNGs, top_k.csv)")
    if not have_smiles:
        print("  NOTE: structures skipped (no SMILES).")


if __name__ == "__main__":
    main()

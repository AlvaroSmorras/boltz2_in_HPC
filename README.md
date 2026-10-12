# Boltz-2 affinity screening pipeline

Predict binding affinities for many ligands against **one** protein target, then
merge the results back into your compound Parquet. Built for an HPC batch system
(the example values are for **Berzelius / NAISS**, SLURM + Miniforge).

The design point: compute the protein MSA **once** and reuse it for every compound,
run compounds in chunks (one SLURM array task = one GPU), and keep only the small
affinity JSONs at the end. Each compound's id travels through Boltz **as the
filename** (`affinity_<CompoundIndex>.json`), so results are tied back to the id with
no separate lookup table.

---

## Files

| File | Runs on | What it does |
|---|---|---|
| `setup_boltz.sh` | login node (once) | create the conda env and `pip install "boltz[cuda]"` |
| `make_yamls_chunks.py` | login node | turn `compounds.parquet` into one YAML per ligand, named `<CompoundIndex>.yaml`, sharded into chunks |
| `run_boltz_array.sbatch` | GPU partition | SLURM array job: predict a chunk per task, tar the outputs, delete the bulky prediction tree |
| `collect_results.py` | login node | read the affinity JSONs (id = filename), merge back into the Parquet |

## What you must provide

- `protein.fasta` — a single protein sequence (the target).
- `compounds.parquet` — one row per ligand, with a SMILES column and an id column
  (defaults `smiles` and `CompoundIndex`; override with `--smiles-col` / `--id-col`).
- `msa/protein.csv` — the precomputed MSA for the target (see Step 1).
- A populated `boltz_cache/` — model weights (downloaded once on the login node).

> **Compound ids must be safe, unique filenames.** Each id becomes a filename, so it
> must contain only letters, digits and `. _ + -` (no spaces, slashes or quotes, and
> not start with `-` or `.`), and be unique. `make_yamls_chunks.py` checks this and
> refuses to run otherwise — it never silently mangles an id, because there's no map
> to recover the original from.

---

## Working directory layout

Everything runs from one `WORKDIR` (the sbatch sets
`WORKDIR=/home/x_alvse/projdir/boltz2/AURKA` — change it to yours). Expected contents:

```
WORKDIR/
├── protein.fasta
├── compounds.parquet
├── msa/protein.csv
├── boltz_cache/            # model weights (downloaded on a login node)
├── input/                  # <- make_yamls writes chunk_000/, chunk_001/, ... here
│   └── chunk_000/<CompoundIndex>.yaml
├── output/                 # boltz writes here; the result tars land here too
├── logs/                   # SLURM stdout/stderr (must exist before submitting)
└── empty/                  # empty dir used to wipe each chunk's prediction tree
```

> **Two things that will bite you if skipped:**
> 1. The sbatch reads from **`input/`** (singular). Run the generator with
>    **`--outdir input`** (its default) so they match.
> 2. The sbatch needs `logs/`, `output/`, and **`empty/`** to already exist — the
>    final `rsync --delete empty/ output/<chunk>/` wipes the prediction tree and needs
>    a real empty source dir. Create them once: `mkdir -p logs output empty`

---

## Step 1 — one-time setup (login node, needs internet)

**Environment.** The sbatch activates a **named** env (`conda activate boltz2`), while
`setup_boltz.sh` as written creates a **prefix** env — pick one and be consistent. To
match the sbatch, create a named env:

```bash
module load Miniforge3 gcc/system
conda create -y -n boltz2 python=3.11 pip
conda activate boltz2
pip install --upgrade pip
pip install "boltz[cuda]" -U
pip install pandas pyarrow          # needed by the two Python scripts
```

**Weights + MSA.** Compute nodes on Berzelius have **no internet**, so the model
weights and the MSA must be prepared on a login node first. One throwaway prediction
with the MSA server does both:

```bash
export WORKDIR=/home/x_alvse/projdir/boltz2/AURKA
export BOLTZ_CACHE=$WORKDIR/boltz_cache
mkdir -p "$WORKDIR/msa" "$BOLTZ_CACHE"

SEQ=$(grep -v '^>' "$WORKDIR/protein.fasta" | tr -d '[:space:]')
cat > "$WORKDIR/_seed.yaml" <<YAML
version: 1
sequences:
  - protein: {id: A, sequence: $SEQ}
  - ligand:  {id: B, smiles: 'CCO'}
properties:
  - affinity: {binder: B}
YAML

boltz predict "$WORKDIR/_seed.yaml" --use_msa_server --cache "$BOLTZ_CACHE" \
  --out_dir "$WORKDIR/_seed_out" --override

# reuse the generated MSA (the combined CSV Boltz feeds the model -- NOT uniref.a3m)
cp "$(find "$WORKDIR/_seed_out/msa" -maxdepth 1 -name '*_0.csv' | head -1)" "$WORKDIR/msa/protein.csv"
```

---

## Step 2 — generate the per-compound inputs (login node)

```bash
cd "$WORKDIR"
python make_yamls_chunks.py \
  --protein protein.fasta \
  --input   compounds.parquet \
  --msa     msa/protein.csv \
  --outdir  input \
  --chunk-size 2000
```

`--input` takes Parquet or CSV/TSV (auto-detected from the extension, with a content
sniff as fallback); `--parquet` still works as an alias. It writes
`input/chunk_XXX/<CompoundIndex>.yaml` and prints the **SLURM array range**
to use. No map file is produced — the id lives in the filename.

Choosing `--chunk-size`: one chunk = one GPU task running its compounds back-to-back,
so bigger chunks amortise the ~1 min model-load startup better, but make each task
longer and reduce parallelism. Keep the **number of chunks under the cluster's
`MaxArraySize`** (`scontrol show config | grep MaxArraySize`); the generator warns if
you exceed ~1000. Add `--skip-existing` to re-run without rewriting existing YAMLs.

---

## Step 3 — run the array job

Edit the top of `run_boltz_array.sbatch`:

- `#SBATCH --array=0-N` → set `N` to the range the generator printed (optionally
  throttle, e.g. `--array=0-10%4`).
- `#SBATCH --account=` → your project; `--time=` → long enough for one chunk.
- `WORKDIR=` and `conda activate boltz2` → your paths / env name.

Then:

```bash
mkdir -p logs output empty          # if not already done
sbatch run_boltz_array.sbatch
```

**Speed profile.** The job defaults to a FAST screening profile (env vars at the top).
The biggest cost is the affinity ensemble; the default drops it to 1 sample. For
accurate final numbers, override at submit time:

```bash
sbatch --export=ALL,AFF_DIFF_SAMPLES=5,AFF_SAMPLING_STEPS=200,SAMPLING_STEPS=200,RECYCLING_STEPS=3 \
       run_boltz_array.sbatch
```

**What each task leaves behind.** After predicting, the task tars the small files and
deletes the rest to save inodes:

- `output/results_affinity_<chunk>.tar` — the `affinity_*.json` files (flattened).
- `output/results_models_<chunk>.tar` — the `.cif` structures (flattened).
- `output/<chunk>/` is then wiped (the `rsync --delete empty/ …` line).

There's no `--override`, so a resubmitted chunk whose `output/<chunk>/` still exists
skips finished compounds; a chunk already wiped will recompute from scratch.

---

## Step 4 — collect results

The affinity JSONs are inside the per-chunk tars, so extract them into one flat folder
and point the collector at it (no map needed):

```bash
cd "$WORKDIR"
mkdir -p affinity_jsons
for t in output/results_affinity_chunk_*.tar; do tar xf "$t" -C affinity_jsons; done

python collect_results.py \
  --indir   affinity_jsons \
  --output  affinity_results.parquet \
  --merge-into compounds.parquet
```

Outputs:
- `affinity_results.parquet` — one row per compound, ranked (strongest first), keyed
  by `CompoundIndex` (read from each filename). Format follows the `--output`
  extension (`.parquet`/`.csv`/`.tsv`).
- `compounds_with_affinity.parquet` — your original Parquet plus the affinity columns
  (joined on `CompoundIndex`; use `--id-col` if your id column has another name). It
  prints how many compounds came back with no result.

### Reading the columns
The collector reports Boltz's own fields, unmodified (no conversions):
- `affinity_pred_value` — Boltz's primary affinity prediction; **lower = stronger**.
  Results are ranked on this. (Per the Boltz docs this is on a log10(IC50) scale with
  IC50 in µM; it is *not* a kcal/mol value, and none of it is re-derived here.)
- `affinity_probability_binary` — Boltz's binder probability, 0–1; separates binders
  from decoys.
- `affinity_pred_value1`, `affinity_pred_value2`, … and the matching
  `affinity_probability_binary1`, … — individual ensemble-member values, present only
  when a run produced them. `n_members` records how many.

---

## Step 5 — report (login node)

Summarise the screen: affinity histograms and the structures of the best compounds.
Structures need SMILES, which the results file doesn't carry, so pass the original
`compounds.parquet` (joined on `CompoundIndex`):

```bash
python make_report.py \
  --results affinity_results.parquet \
  --smiles-from compounds.parquet \
  --top-k 20 --outdir report
```

Needs `rdkit` and `matplotlib` (`pip install rdkit matplotlib`; RDKit is already in
the Boltz env). Under `report/` you get:
- `report.pdf` — summary page, histograms, and the top-K structure grid in one file.
- `hist_affinity.png`, `hist_probability.png`, `scatter_affinity_vs_prob.png` — the
  plots as standalone images, drawn directly from `affinity_pred_value` and `affinity_probability_binary`.
- `top_structures_page*.png` and `structures/rankNN_<id>.png` — the best compounds
  (lowest `affinity_pred_value`) as a grid and as individual images, each labelled with
  id, `affinity_pred_value` and P(binder).
- `top_k.csv` — the ranked top-K table.

Tune with `--top-k`, `--mols-per-row`, `--rows-per-page`; point at other columns with
`--affinity-col` / `--prob-col` / `--id-col` / `--smiles-col` if you renamed them
(defaults are Boltz's own field names).

---

## Quick reference

```bash
# 1. setup (login, once)
module load Miniforge3 gcc/system && conda activate boltz2
#    create env + pip install boltz[cuda] pandas pyarrow, then populate
#    boltz_cache/ and msa/protein.csv via the seed prediction (Step 1)

# 2. inputs (login)
python make_yamls_chunks.py --protein protein.fasta --input compounds.parquet \
       --msa msa/protein.csv --outdir input --chunk-size 2000

# 3. run (GPU)
mkdir -p logs output empty
sbatch run_boltz_array.sbatch              # after setting --array / --account / WORKDIR

# 4. collect (login)
mkdir -p affinity_jsons
for t in output/results_affinity_chunk_*.tar; do tar xf "$t" -C affinity_jsons; done
python collect_results.py --indir affinity_jsons \
       --output affinity_results.parquet --merge-into compounds.parquet

# 5. report (login)
python make_report.py --results affinity_results.parquet \
       --smiles-from compounds.parquet --top-k 20 --outdir report
```

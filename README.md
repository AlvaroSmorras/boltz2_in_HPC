# Boltz-2 affinity screening pipeline

Predict binding affinities for many ligands against **one** protein target, then
merge the results back into your compound Parquet. Built for an HPC batch system
(the example values are for **Berzelius / NAISS**, SLURM + Miniforge).

The design point: compute the protein MSA **once** and reuse it for every
compound, run compounds in chunks (one SLURM array task = one GPU), and keep only
the small affinity JSONs at the end.

---

## Files

| File | Runs on | What it does |
|---|---|---|
| `setup_boltz.sh` | login node (once) | create the conda env and `pip install "boltz[cuda]"` |
| `make_yamls_chunks.py` | login node | turn `compounds.parquet` into one YAML per ligand, sharded into chunks, + write `index_map.parquet` |
| `run_boltz_array.sbatch` | GPU partition | SLURM array job: predict a chunk per task, tar the outputs, delete the bulky prediction tree |
| `collect_results.py` | login node | gather the affinity JSONs, recover `CompoundIndex`, merge back into the Parquet |

## What you must provide

- `protein.fasta` — a single protein sequence (the target).
- `compounds.parquet` — one row per ligand, with a SMILES column and an id column
  (defaults `smiles` and `CompoundIndex`; override with `--smiles-col` / `--id-col`).
- `msa/protein.csv` — the precomputed MSA for the target (see Step 1).
- A populated `boltz_cache/` — model weights **and** the CCD data (see Step 1).

---

## Working directory layout

Everything runs from one `WORKDIR` (the sbatch sets
`WORKDIR=/home/x_alvse/projdir/boltz2/AURKA` — change it to yours). Expected contents:

```
WORKDIR/
├── protein.fasta
├── compounds.parquet
├── msa/protein.csv
├── boltz_cache/            # weights + mols/ (CCD). NOT on $HOME if quota is tight.
├── input/                  # <- make_yamls writes chunk_000/, chunk_001/, ... here
│   ├── chunk_000/*.yaml
│   └── index_map.parquet
├── output/                 # boltz writes here; tars land here too
├── logs/                   # SLURM stdout/stderr (must exist before submitting)
└── empty/                  # empty dir used to wipe each chunk's prediction tree
```

> **Two things that will bite you if skipped:**
> 1. The sbatch reads from **`input/`** (singular) and the generator defaults to
>    `inputs/` (plural). Run the generator with **`--outdir input`** so they match.
> 2. The sbatch needs `logs/`, `output/`, and **`empty/`** to already exist
>    (the final `rsync --delete empty/ output/<chunk>/` wipes the prediction tree,
>    and it needs a real empty source dir). Create them once:
>    `mkdir -p logs output empty`

---

## Step 1 — one-time setup (login node, needs internet)

**Environment.** `setup_boltz.sh` installs Boltz-2. Note the sbatch activates a
**named** env (`conda activate boltz2`), while `setup_boltz.sh` as written creates a
**prefix** env — pick one and be consistent. To match the sbatch, create a named env:

```bash
module load Miniforge3 gcc/system
conda create -y -n boltz2 python=3.11 pip
conda activate boltz2
pip install --upgrade pip
pip install "boltz[cuda]" -U
pip install pandas pyarrow          # needed by make_yamls_chunks.py / collect_results.py
```

**Cache (weights + CCD) and MSA.** Compute nodes on Berzelius have **no internet**,
so the model weights, the CCD molecule data, and the MSA must all be prepared on a
login node first. The easiest way to populate everything is one throwaway prediction
with the MSA server:

```bash
export WORKDIR=/home/x_alvse/projdir/boltz2/AURKA
export BOLTZ_CACHE=$WORKDIR/boltz_cache
mkdir -p "$WORKDIR/msa" "$BOLTZ_CACHE"

# seed YAML: your protein + a trivial ligand
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

# sanity-check the CCD cache (prevents the "CCD component SER not found!" error)
ls "$BOLTZ_CACHE/mols/SER.pkl" && echo "cache OK"
```

If `SER.pkl` is missing, the `mols/` extraction was incomplete (quota/interrupt):
`rm -rf "$BOLTZ_CACHE/mols" "$BOLTZ_CACHE/mols.tar"` and re-run the command above.

---

## Step 2 — generate the per-compound inputs (login node)

```bash
cd "$WORKDIR"
python make_yamls_chunks.py \
  --protein protein.fasta \
  --parquet compounds.parquet \
  --msa     msa/protein.csv \
  --outdir  input \
  --chunk-size 2000
```

It writes `input/chunk_XXX/*.yaml` + `input/index_map.parquet` (+ `.csv`) and prints
the **SLURM array range** to use. Each YAML is named after its `CompoundIndex`; the
map ties that name back to the exact id even if it had to be sanitised.

Choosing `--chunk-size`: one chunk = one GPU task running its compounds back-to-back,
so bigger chunks amortise the ~1 min model-load startup better, but make each task
longer and reduce parallelism. Keep the **number of chunks under the cluster's
`MaxArraySize`** (`scontrol show config | grep MaxArraySize`). The generator warns if
you exceed ~1000 chunks. Add `--skip-existing` to rebuild only the map without
rewriting YAMLs.

---

## Step 3 — run the array job

Edit the top of `run_boltz_array.sbatch`:

- `#SBATCH --array=0-N` → set `N` to the range the generator printed (and optionally
  throttle concurrency, e.g. `--array=0-10%4`).
- `#SBATCH --account=` → your project; `--time=` → long enough for one chunk.
- `WORKDIR=` and `conda activate boltz2` → your paths / env name.

Then:

```bash
mkdir -p logs output empty          # if not already done
sbatch run_boltz_array.sbatch
```

**Speed profile.** The job defaults to a FAST screening profile (set via env vars at
the top of the script). The single biggest cost is the affinity ensemble; defaults
drop it to 1 sample. For accurate final numbers, submit with overrides, e.g.:

```bash
sbatch --export=ALL,AFF_DIFF_SAMPLES=5,AFF_SAMPLING_STEPS=200,SAMPLING_STEPS=200,RECYCLING_STEPS=3 \
       run_boltz_array.sbatch
```

**What each task leaves behind.** After predicting, the task tars the small files and
deletes the rest to save inodes:

- `output/results_affinity_<chunk>.tar` — the `affinity_*.json` files (flattened).
- `output/results_models_<chunk>.tar` — the `.cif` structures (flattened).
- `output/<chunk>/` is then wiped (the `rsync --delete empty/ …` line).

Re-running a partial screen is safe: there's no `--override`, so finished compounds
are skipped — but note a chunk whose `output/<chunk>/` was already wiped will recompute.

---

## Step 4 — collect results

The affinity JSONs are inside the per-chunk tars, so extract them into one flat folder
and point the collector at it:

```bash
cd "$WORKDIR"
mkdir -p affinity_jsons
for t in output/results_affinity_chunk_*.tar; do tar xf "$t" -C affinity_jsons; done

python collect_results.py \
  --indir   affinity_jsons \
  --map     input/index_map.parquet \
  --output  affinity_results.parquet \
  --merge-into compounds.parquet
```

Outputs:
- `affinity_results.parquet` — one row per compound, ranked (strongest first), keyed
  by `CompoundIndex`. Format follows the `--output` extension (`.parquet`/`.csv`/`.tsv`).
- `compounds_with_affinity.parquet` — your original Parquet plus the affinity columns.

### Reading the columns
- `binder_probability` — P(binder), 0–1. Use this to separate binders from decoys.
- `affinity_log10IC50uM` — log10(IC50) in µM; **lower = stronger**. Use to rank actives.
- `IC50_uM`, `pIC50`, `dG_proxy_kcal_per_mol` — conversions of the above.
- `affinity_value_1/2`, `binder_prob_1/2`, `affinity_value_member_std` — the individual
  ensemble members and their spread (a large std = the two heads disagree = less
  reliable). Present only when a run produced multiple values; `n_members` records how
  many.
- `missing`/`NaN` rows — compounds with no affinity JSON (crashed or not finished); the
  collector prints how many.

---

## Quick reference

```bash
# 1. setup (login, once)
bash setup_boltz.sh                        # or the named-env commands in Step 1
#    + populate boltz_cache/ and msa/protein.csv (Step 1)

# 2. inputs (login)
python make_yamls_chunks.py --protein protein.fasta --parquet compounds.parquet \
       --msa msa/protein.csv --outdir input --chunk-size 2000

# 3. run (GPU)
mkdir -p logs output empty
sbatch run_boltz_array.sbatch              # after setting --array / --account / WORKDIR

# 4. collect (login)
mkdir -p affinity_jsons
for t in output/results_affinity_chunk_*.tar; do tar xf "$t" -C affinity_jsons; done
python collect_results.py --indir affinity_jsons --map input/index_map.parquet \
       --output affinity_results.parquet --merge-into compounds.parquet
```

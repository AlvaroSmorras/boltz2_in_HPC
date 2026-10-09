#!/usr/bin/env bash
#SBATCH --job-name=boltz-affinity
#SBATCH --array=0-76%10
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=42:00:00
#SBATCH --account=berzelius-2026-155
#SBATCH --gpus=1       
#SBATCH --cpus-per-task=8

set -euo pipefail

WORKDIR="${WORKDIR:-/home/x_alvse/projdir/boltz2/AURKA}"
export BOLTZ_CACHE="${BOLTZ_CACHE:-$WORKDIR/boltz_cache}"

module load Miniforge3 gcc/system
#source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate boltz2

mkdir -p logs output

CHUNK=$(printf "chunk_%03d" "${SLURM_ARRAY_TASK_ID}")
echo "Host: $(hostname)   Chunk: $CHUNK   GPUs: ${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi -L || true

# ---- Speed profile (defaults tuned for FAST first-pass SCREENING) ------------
# The affinity module re-runs the structure diffusion AFF_DIFF_SAMPLES times
# (Boltz default = 5): a hidden 5x cost. This is the biggest single lever.
# For final/accurate numbers use the "accurate" values in the comments.

SAMPLING_STEPS=${SAMPLING_STEPS:-100}            # structure diffusion steps   (accurate: 200)
RECYCLING_STEPS=${RECYCLING_STEPS:-1}            # trunk recycles              (accurate: 3)
DIFF_SAMPLES=${DIFF_SAMPLES:-1}                  # structure samples           (keep 1)
AFF_SAMPLING_STEPS=${AFF_SAMPLING_STEPS:-100}   # affinity diffusion steps    (accurate: 200)
AFF_DIFF_SAMPLES=${AFF_DIFF_SAMPLES:-1}         # affinity ensemble size      (accurate: 5)  <-- biggest win
SUBSAMPLE_MSA=${SUBSAMPLE_MSA:-1024}            # cap MSA depth into the trunk (0 = no cap)


EXTRA=()
[[ "${SUBSAMPLE_MSA}" != "0" ]] && EXTRA+=(--subsample_msa --num_subsampled_msa "${SUBSAMPLE_MSA}")

srun boltz predict "input/$CHUNK" \
  --out_dir "output/$CHUNK" \
  --cache "$BOLTZ_CACHE" \
  --accelerator gpu \
  --devices 1 \
  --recycling_steps "$RECYCLING_STEPS" \
  --sampling_steps "$SAMPLING_STEPS" \
  --diffusion_samples "$DIFF_SAMPLES" \
  --sampling_steps_affinity "$AFF_SAMPLING_STEPS" \
  --diffusion_samples_affinity "$AFF_DIFF_SAMPLES" \
  --output_format mmcif \
  --num_workers 8 \
  "${EXTRA[@]}"

echo "Finished $CHUNK"

# tar results and remove the rest not to crash the filesystem
tar cf output/results_affinity_${CHUNK}.tar --transform="s|.*/||" $(ls output/${CHUNK}/boltz_results_${CHUNK}/predictions/*/affinity*json)
tar cf output/results_models_${CHUNK}.tar --transform="s|.*/||" $(ls output/${CHUNK}/boltz_results_${CHUNK}/predictions/*/*cif)

rsync -a --delete empty/ output/${CHUNK}/

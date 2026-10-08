#!/usr/bin/env bash
# Full run of all three components, Day-2 landmark, core markers.
# Writes everything to outputs/full and logs to outputs/full/full_run.log.
#
#   bash run_full.sh
#
# Deliberately one seed. For the paper this whole script should be repeated
# across seeds and the spread reported, not just the point estimate.

set -uo pipefail
cd "$(dirname "$0")"

export KMP_DUPLICATE_LIB_OK=TRUE
PY="E:/annconda/python.exe"
OUT="outputs/full"
mkdir -p "$OUT"
LOG="$OUT/full_run.log"
: > "$LOG"

SEED=20261008
EPOCHS=120
PATIENCE=25

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
run() {
  local label="$1"; shift
  log "START  $label"
  local t0=$SECONDS
  "$PY" "$@" >> "$LOG" 2>&1
  local rc=$?
  log "DONE   $label (exit $rc, $(( SECONDS - t0 ))s)"
  return $rc
}

log "=== full run: seed=$SEED epochs=$EPOCHS patience=$PATIENCE ==="
"$PY" -c "import torch,sys;print('torch',torch.__version__, sys.version.split()[0])" >> "$LOG" 2>&1

FAILED=()

# ---------------- component 1: reversible lifting ----------------
run "C1 pretrain (core markers, 60 ep)" \
    01_reversible_lifting/pretrain.py --epochs 60 --batch-size 256 \
    --seed $SEED --out-dir "$OUT" || FAILED+=("C1-pretrain")

run "C1 finetune: pretrained" \
    01_reversible_lifting/finetune.py --init "$OUT/rev_encoder_core.pt" \
    --epochs $EPOCHS --patience $PATIENCE --seed $SEED --out-dir "$OUT" \
    --tag rev_pretrained || FAILED+=("C1-pretrained")

run "C1 control: from scratch" \
    01_reversible_lifting/finetune.py --from-scratch \
    --epochs $EPOCHS --patience $PATIENCE --seed $SEED --out-dir "$OUT" \
    --tag rev_scratch || FAILED+=("C1-scratch")

run "C1 probe: frozen encoder" \
    01_reversible_lifting/finetune.py --init "$OUT/rev_encoder_core.pt" \
    --freeze-encoder --epochs $EPOCHS --patience $PATIENCE --seed $SEED \
    --out-dir "$OUT" --tag rev_frozen || FAILED+=("C1-frozen")

# ---------------- component 2: latent dynamics ----------------
run "C2 latent dynamics (mTAN / LatentODE / GRU-ODE) + LOCF floor" \
    02_latent_dynamics/train_landmark.py --model all \
    --epochs $EPOCHS --patience $PATIENCE --seed $SEED --out-dir "$OUT" \
    || FAILED+=("C2")

# ---------------- component 3: FNO on waveforms ----------------
run "C3 FNO time-advancement + rollout" \
    03_spectral_waveform/train_surrogate.py --epochs 60 --n-train 800 \
    --n-test 200 --rollout 4 --plot --seed $SEED --out-dir "$OUT" \
    || FAILED+=("C3")

# ---------------- summary ----------------
log "START  summary"
"$PY" summarize.py --out-dir "$OUT" >> "$LOG" 2>&1
log "DONE   summary"

log "=== finished ($(( SECONDS ))s total) ==="
if [ ${#FAILED[@]} -gt 0 ]; then
  log "FAILED STEPS: ${FAILED[*]}"
  exit 1
fi
log "all steps completed"

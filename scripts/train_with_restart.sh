#!/usr/bin/env bash
# Restarts `python -m edgealign.train` on the sentinel exit code
# (edgealign.train.STICKY_CUDA_FAULT_EXIT_CODE, 42) that main() uses for a
# confirmed-sticky CUDA context fault -- see MEMORY.md. An in-process
# retry can't work once that happens (even torch.cuda.empty_cache() fails
# identically in the same process), so recovery is a fresh process,
# resuming from the last checkpoint written by train.checkpoint_every.
#
# Any other exit code (0 = done, anything else = a real bug/config error)
# stops immediately instead of restarting.
set -uo pipefail

STICKY_CUDA_FAULT_EXIT_CODE=42

if [ "$#" -lt 1 ]; then
  echo "usage: $0 <config.yaml> [max_restarts]" >&2
  exit 1
fi

CONFIG="$1"
MAX_RESTARTS="${2:-10}"

for attempt in $(seq 1 "$MAX_RESTARTS"); do
  echo "[train_with_restart] attempt $attempt/$MAX_RESTARTS: python -m edgealign.train --config $CONFIG"
  python -m edgealign.train --config "$CONFIG"
  code=$?

  if [ "$code" -eq 0 ]; then
    echo "[train_with_restart] training finished successfully."
    exit 0
  elif [ "$code" -eq "$STICKY_CUDA_FAULT_EXIT_CODE" ]; then
    echo "[train_with_restart] sticky CUDA fault (exit $code) -- restarting from last checkpoint in 5s..."
    sleep 5
    continue
  else
    echo "[train_with_restart] training exited with code $code -- not a recoverable CUDA fault, stopping."
    exit "$code"
  fi
done

echo "[train_with_restart] exhausted $MAX_RESTARTS restarts without a successful finish."
exit 1

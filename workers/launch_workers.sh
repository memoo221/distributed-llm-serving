#!/usr/bin/env bash
# Run on a Thunder Compute instance to (re)launch the two GPU workers
# (one per A100, pinned via WORKER_DEVICE) inside detached tmux sessions.
#
# Required env vars (passed by the laptop-side redeploy.ps1 driver):
#   HF_TOKEN          HuggingFace token with model access
#   MASTER1_URL       cloudflared URL of master1
#   MASTER2_URL       cloudflared URL of master2
#   UUID_PREFIX       Thunder instance UUID, e.g. q5lbf7oe
#   WORKER_PREFIX     thunder_a or thunder_b (matches the instance)
# Optional:
#   BATCH_SIZE        Worker batching size + advertised slots (default: 8)
#   MODEL_NAME        HF Hub model id (default uses worker's own default)
set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN is required}"
: "${MASTER1_URL:?MASTER1_URL is required}"
: "${MASTER2_URL:?MASTER2_URL is required}"
: "${UUID_PREFIX:?UUID_PREFIX is required}"
: "${WORKER_PREFIX:?WORKER_PREFIX is required}"
BATCH_SIZE="${BATCH_SIZE:-8}"
# If MODEL_NAME is unset OR empty, fully unset it so child processes (tmux,
# uvicorn, transformers) fall back to thunder_worker.py's default. With it
# left as "" Python's os.getenv("MODEL_NAME", default) returns "" (not the
# default), which crashes the tokenizer with "Repo id ... cannot be ''".
if [ -z "${MODEL_NAME:-}" ]; then
    unset MODEL_NAME
fi

echo "[launch] worker_prefix=$WORKER_PREFIX uuid=$UUID_PREFIX batch=$BATCH_SIZE"

# Bootstrap tmux if missing (first-run on a fresh instance).
if ! command -v tmux >/dev/null 2>&1; then
    echo "[launch] installing tmux..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq tmux
fi

# Stop any running worker processes / tmux sessions.
tmux kill-server 2>/dev/null || true
pkill -9 -f "uvicorn thunder_worker" 2>/dev/null || true
pkill -9 -f "thunder_worker:app" 2>/dev/null || true
# Truncate stale log files so wait_for_load doesn't match an old
# "Application startup complete" from a previous run.
rm -f /home/ubuntu/gpu0.log /home/ubuntu/gpu1.log
sleep 3

# Build a per-worker launch script and execute it via `tmux new -d ... 'bash
# script'`. We previously used `tmux send-keys` for each export but that
# approach is racy: under load, individual keystrokes can be dropped, and
# we observed MODEL_NAME's export getting eaten on the second worker
# launched on the same instance. Writing a complete shell script and
# letting tmux exec it atomically eliminates that race.
build_session() {
    local session="$1" device="$2" wid="$3" master_url="$4" port="$5"
    local script="/tmp/${session}_launch.sh"
    cat > "$script" <<EOF
#!/usr/bin/env bash
export PYTHONUNBUFFERED=1
export HF_TOKEN='$HF_TOKEN'
export WORKER_DEVICE='$device'
export BATCH_SIZE='$BATCH_SIZE'
export WORKER_ID='$wid'
export MASTER_URL='$master_url'
export SELF_URL='https://${UUID_PREFIX}-${port}.thundercompute.net'
export MODEL_NAME='${MODEL_NAME:-}'
cd /home/ubuntu
exec uvicorn thunder_worker:app --host 0.0.0.0 --port $port 2>&1 | tee /home/ubuntu/${session}.log
EOF
    chmod +x "$script"
    tmux new -d -s "$session" "bash $script"
}

# Single worker per instance on cuda:0. Running two workers per Thunder
# Prototyping instance (one on cuda:0 + one on cuda:1) consistently caused
# the cuda:1 process to hang in model.generate() — Thunder's virtualized
# CUDA layer doesn't handle two concurrent inference processes per instance.
# Falling back to one worker per instance, registered to the master named
# in $MASTER_TARGET (master1 or master2). The redeploy.ps1 driver picks
# which master each instance registers to so the cluster stays symmetric.
case "${MASTER_TARGET:-master1}" in
    master1) target_url="$MASTER1_URL" ;;
    master2) target_url="$MASTER2_URL" ;;
    *) echo "[launch] invalid MASTER_TARGET=$MASTER_TARGET (must be master1|master2)"; exit 1 ;;
esac

echo "[launch] starting single worker on cuda:0 -> $MASTER_TARGET"
build_session "gpu0" "cuda:0" "${WORKER_PREFIX}_gpu0" "$target_url" "8000"

echo "[launch] tmux session started:"
tmux ls
echo "[launch] tail logs with: tmux attach -t gpu0  (Ctrl-B D to detach)"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

#----- NOTE: manually set this for test locally -----#
export TRAIN_DATA_PATH="/nfs1/SDW/20260428-TAAC2026/data_sample_1000"
export TRAIN_CKPT_PATH="./checkpoints"
export TRAIN_LOG_PATH="./logs"
export TRAIN_TF_EVENTS_PATH="./tf_events"
#----- NOTE: manually set this for test locally -----#

#----- Patch schema.json: activate time features by setting ts_fid -----#
echo "[run.sh] Patching schema.json to activate time features..."
mkdir -p "${TRAIN_CKPT_PATH}"
FIXED_SCHEMA="${TRAIN_CKPT_PATH}/schema_patched.json"
python3 "${SCRIPT_DIR}/fix_schema.py" \
    "${TRAIN_DATA_PATH}/schema.json" \
    "${FIXED_SCHEMA}"
echo "[run.sh] Using patched schema: ${FIXED_SCHEMA}"

echo "[run.sh] Detecting GPU count..."
GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)" 2>/dev/null || echo "0")
echo "[run.sh] Found $GPU_COUNT GPUs"

# Single GPU (for debugging only):
# python3 -u "${SCRIPT_DIR}/train.py" \
#     --ns_tokenizer_type rankmixer \
#     --user_ns_tokens 5 \
#     --item_ns_tokens 2 \
#     --compile \
#     --use_amp \
#     "$@"

if [ "$GPU_COUNT" -gt 1 ]; then
    echo "[run.sh] Running with DDP on $GPU GPUs..."
    torchrun \
        --nproc_per_node=$GPU_COUNT \
        --nnodes=1 \
        --master_port=29500 \
        "${SCRIPT_DIR}/train.py" \
        --schema_path "${FIXED_SCHEMA}" \
        --ns_tokenizer_type rankmixer \
        --user_ns_tokens 5 \
        --item_ns_tokens 2 \
        --num_queries 2 \
        --ns_groups_json "" \
        --emb_skip_threshold 1000000 \
        --num_workers 8 \
        --compile \
        --use_amp \
        "$@"
else
    echo "[run.sh] Falling back to single-GPU mode ($GPU_COUNT GPUs detected)"
    python3 -u "${SCRIPT_DIR}/train.py" \
        --schema_path "${FIXED_SCHEMA}" \
        --ns_tokenizer_type rankmixer \
        --user_ns_tokens 5 \
        --item_ns_tokens 2 \
        --num_queries 2 \
        --ns_groups_json "" \
        --batch_size 1280 \
        --emb_skip_threshold 1000000 \
        --num_workers 8 \
        --compile \
        --use_amp \
        "$@"
fi

# ---- Alternative config: GroupNSTokenizer driven by ns_groups.json ----
# Uses feature grouping from ns_groups.json (7 user groups + 4 item groups).
# With d_model=64 and num_ns=12 (7 user_int + 1 user_dense + 4 item_int),
# only num_queries=1 satisfies d_model % T == 0 (T = num_queries*4 + num_ns).
# To switch, comment out the block above and uncomment the block below.
#
# python3 -u "${SCRIPT_DIR}/train.py" \
#     --ns_tokenizer_type group \
#     --ns_groups_json "${SCRIPT_DIR}/ns_groups.json" \
#     --num_queries 1 \
#     --emb_skip_threshold 1000000 \
#     --num_workers 8 \
#     "$@"

#!/bin/bash
# ============================================================
# run_all.sh — Chạy toàn bộ thực nghiệm tự động
#
# Cách dùng:
#   chmod +x scripts/run_all.sh
#   tmux new -s training
#   bash scripts/run_all.sh 2>&1 | tee logs/run_all.log
#   Ctrl+B, D   # detach tmux — job vẫn chạy qua đêm
# ============================================================

set -e  # dừng nếu có lỗi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

# ── Cấu hình ────────────────────────────────────────────────
FRAMES=10000000
SEEDS=(1 2 3)
WANDB_PROJECT="rainbow-thesis"

# ── 5 Games được chọn ───────────────────────────────────────
GAMES=(
    "PongNoFrameskip-v4"
    "BreakoutNoFrameskip-v4"
    "SpaceInvadersNoFrameskip-v4"
    "SeaquestNoFrameskip-v4"
    "MontezumaRevengeNoFrameskip-v4"
)

# ── 7 Agents (Rainbow full + 6 ablations) ───────────────────
AGENTS=(
    "rainbow:none"
    "rainbow:no_double"
    "rainbow:no_priority"
    "rainbow:no_dueling"
    "rainbow:no_multistep"
    "rainbow:no_distributional"
    "rainbow:no_noisy"
)
# DQN baseline chạy riêng (cần ít VRAM hơn, song song được)
RUN_DQN=true

# ── Helper: chạy 1 experiment, skip nếu đã xong ─────────────
run_experiment() {
    local game="$1"
    local ablation="$2"
    local seed="$3"

    local game_short="${game/NoFrameskip-v4/}"
    local abl_tag="${ablation:-full}"
    local run_name="${game_short}_${abl_tag}_s${seed}"
    local ckpt_dir="checkpoints/${run_name}"
    local done_file="${ckpt_dir}/final.pt"

    # Skip nếu đã chạy xong
    if [ -f "$done_file" ]; then
        echo "[SKIP] $run_name — đã có final.pt"
        return 0
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[START] $(date '+%H:%M:%S') — $run_name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Resume từ checkpoint nếu có
    local resume_flag=""
    local latest_ckpt
    latest_ckpt=$(find "$ckpt_dir" -name "frame_*.pt" 2>/dev/null | sort -t_ -k2 -n | tail -1)
    if [ -n "$latest_ckpt" ]; then
        echo "[RESUME] Từ checkpoint: $latest_ckpt"
        resume_flag="--resume $latest_ckpt"
    fi

    # Build lệnh
    local ablation_flag=""
    if [ "$ablation" != "none" ] && [ -n "$ablation" ]; then
        ablation_flag="--ablation $ablation"
    fi

    python scripts/train.py \
        --game "$game" \
        --seed "$seed" \
        --total_frames "$FRAMES" \
        --wandb_project "$WANDB_PROJECT" \
        $ablation_flag \
        $resume_flag

    echo "[DONE] $(date '+%H:%M:%S') — $run_name"
}

# ── Main loop ────────────────────────────────────────────────
echo "============================================================"
echo "Rainbow Thesis — Full Experiment Run"
echo "Start: $(date)"
echo "Games: ${#GAMES[@]} | Agents: ${#AGENTS[@]} | Seeds: ${#SEEDS[@]}"
echo "Total planned runs: $((${#GAMES[@]} * ${#AGENTS[@]} * ${#SEEDS[@]}))"
echo "============================================================"

# Priority 1: Rainbow full + DQN baseline (cần trước để có baseline)
echo ""
echo ">>> Phase 1: Rainbow Full + DQN Baseline"
for game in "${GAMES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        run_experiment "$game" "none" "$seed"
        if $RUN_DQN; then
            run_experiment "$game" "no_distributional" "$seed"  # proxy DQN
        fi
    done
done

# Priority 2: 2 ablations quan trọng nhất
echo ""
echo ">>> Phase 2: Critical Ablations (no_priority, no_multistep)"
for game in "${GAMES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        run_experiment "$game" "no_priority" "$seed"
        run_experiment "$game" "no_multistep" "$seed"
    done
done

# Priority 3: Các ablations còn lại
echo ""
echo ">>> Phase 3: Remaining Ablations"
for game in "${GAMES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        run_experiment "$game" "no_double" "$seed"
        run_experiment "$game" "no_noisy" "$seed"
        run_experiment "$game" "no_dueling" "$seed"
    done
done

echo ""
echo "============================================================"
echo "ALL EXPERIMENTS COMPLETE!"
echo "End: $(date)"
echo "============================================================"

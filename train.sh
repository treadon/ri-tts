#!/bin/bash
# ri-tts training launcher
# Handles: HF token, wandb login, checkpoint uploads, tokenizer build, tmux session
set -e

TMUX_SESSION="ri-tts-train"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  ri-tts Training Launcher${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 1. Check HuggingFace token
echo -e "${YELLOW}[1/5] Checking HuggingFace token...${NC}"
if [ -n "$HF_TOKEN" ]; then
    echo "  HF_TOKEN is set."
elif huggingface-cli whoami &>/dev/null; then
    echo "  Already logged in to HuggingFace."
else
    echo "  No HF token found. Without one, downloads may be rate-limited."
    read -p "  Do you want to set your HF token? (Y/n) " hf_choice
    hf_choice=${hf_choice:-Y}
    if [[ "$hf_choice" =~ ^[Yy]$ ]]; then
        read -p "  Enter your HF token (from https://huggingface.co/settings/tokens): " hf_token
        export HF_TOKEN="$hf_token"
        echo "  HF_TOKEN set for this session."
    else
        echo "  Skipping. Downloads may be slower."
    fi
fi
echo ""

# 2. Check wandb login
echo -e "${YELLOW}[2/5] Checking Weights & Biases...${NC}"
if wandb status 2>&1 | grep -q "Logged in"; then
    echo "  Already logged in to W&B."
else
    echo "  Not logged in to W&B."
    read -p "  Do you want to log in to W&B for experiment tracking? (Y/n) " wandb_choice
    wandb_choice=${wandb_choice:-Y}
    if [[ "$wandb_choice" =~ ^[Yy]$ ]]; then
        wandb login
    else
        echo "  Skipping W&B. Training will run without remote logging."
        export WANDB_MODE=offline
    fi
fi
echo ""

# 3. Checkpoint upload to HuggingFace
echo -e "${YELLOW}[3/5] Checkpoint uploads...${NC}"
HF_REPO=""
read -p "  Upload checkpoints to HuggingFace? (Y/n) " upload_choice
upload_choice=${upload_choice:-Y}
if [[ "$upload_choice" =~ ^[Yy]$ ]]; then
    read -p "  Repo name (e.g. treadon/ri-tts-model): " HF_REPO
    if [ -z "$HF_REPO" ]; then
        echo "  No repo name given. Skipping uploads."
        HF_REPO=""
    else
        echo "  Checkpoints will upload to: $HF_REPO"
        echo "  Rolling: keeps latest 2 on HF, uploads best at end."
    fi
else
    echo "  Checkpoints will be saved locally only."
fi
echo ""

# 4. Build tokenizer if needed
echo -e "${YELLOW}[4/5] Checking tokenizer...${NC}"
if [ -d "tokenizer" ] && [ -f "tokenizer/tokenizer_config.json" ]; then
    echo "  Tokenizer already built."
else
    echo "  Building tokenizer..."
    python build_tokenizer.py
fi
echo ""

# 5. Tmux session management
echo -e "${YELLOW}[5/5] Setting up training session...${NC}"

if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo "  Found existing training session: $TMUX_SESSION"
    read -p "  Do you want to resume training? (Y/n) " resume_choice
    resume_choice=${resume_choice:-Y}
    if [[ "$resume_choice" =~ ^[Yy]$ ]]; then
        echo "  Attaching to existing session..."
        tmux attach-session -t "$TMUX_SESSION"
        exit 0
    else
        echo "  Killing old session..."
        tmux kill-session -t "$TMUX_SESSION"
    fi
fi

echo "  Starting new training session in tmux..."
echo "  Session name: $TMUX_SESSION"
echo ""
echo "  To detach:  Ctrl+B, then D"
echo "  To reattach: tmux attach -t $TMUX_SESSION"
echo ""

# Build the training command
TRAIN_CMD="cd $(pwd)"
if [ -n "$VIRTUAL_ENV" ]; then
    TRAIN_CMD="$TRAIN_CMD && source $VIRTUAL_ENV/bin/activate"
fi
if [ -n "$HF_TOKEN" ]; then
    TRAIN_CMD="$TRAIN_CMD && export HF_TOKEN=$HF_TOKEN"
fi
if [ "$WANDB_MODE" = "offline" ]; then
    TRAIN_CMD="$TRAIN_CMD && export WANDB_MODE=offline"
fi

TRAIN_ARGS="python train.py"
if [ -n "$HF_REPO" ]; then
    TRAIN_ARGS="$TRAIN_ARGS --hf-repo $HF_REPO"
fi
TRAIN_CMD="$TRAIN_CMD && $TRAIN_ARGS"

tmux new-session -d -s "$TMUX_SESSION" "$TRAIN_CMD"
tmux attach-session -t "$TMUX_SESSION"

#!/bin/bash
# run_demo.sh - Run LeWorldModel training demo

set -e

echo "========================================="
echo "   LeWorldModel Demo"
echo "========================================="
echo ""

# Check if dataset exists
if [ ! -f "/workspace/.stable-wm/datasets/pusht/pusht_expert_train.h5" ]; then
    echo "[INFO] Dataset not found. Downloading..."
    echo "This may take a while (dataset size ~13GB)"
    echo ""
    
    # Create datasets directory
    mkdir -p /workspace/.stable-wm/datasets/pusht
    
    # Download dataset using HF Token
    export HF_TOKEN=$(cat /workspace/le-wm/token)
    hf download quentinll/lewm-pusht \
        --repo-type dataset \
        --local-dir /workspace/.stable-wm/datasets/pusht \
        --local-dir-use-symlinks False
    
    echo "[SUCCESS] Dataset downloaded!"
else
    echo "[SUCCESS] Dataset found!"
fi

echo ""
echo "[INFO] Dataset location: /workspace/.stable-wm/datasets/pusht/"
ls -lh /workspace/.stable-wm/datasets/pusht/

echo ""
echo "========================================="
echo "   Starting Training Demo"
echo "========================================="
echo ""
echo "Training configuration:"
echo "  - Task: pusht"
echo "  - Epochs: 100 (configurable in config/train/lewm.yaml)"
echo "  - Batch size: 128"
echo "  - Device: GPU (if available)"
echo ""
echo "Note: Make sure to configure WandB in config/train/lewm.yaml"
echo "      Or disable WandB by setting wandb.enabled=False"
echo ""

# Run training
cd /workspace/le-wm
python train.py data=pusht

echo ""
echo "========================================="
echo "   Training Complete!"
echo "========================================="
echo ""
echo "Checkpoints saved to: /workspace/.stable-wm/checkpoints/"
echo ""
echo "To evaluate the model, run:"
echo "  python eval.py --config-name=pusht.yaml policy=pusht/lewm"

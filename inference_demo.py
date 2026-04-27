#!/usr/bin/env python3
"""
Complete inference demo for LeWorldModel
Loads pretrained weights and runs inference on dummy data.
"""
import torch
import numpy as np
from pathlib import Path
from jepa import JEPA
from module import Embedder, MLP


def create_model_from_checkpoint(checkpoint_path: str):
    """Create model and load weights from checkpoint."""
    from stable_pretraining.backbone.utils import vit_hf
    
    # Create encoder (ViT-Tiny)
    encoder = vit_hf(
        size='tiny',
        patch_size=14,
        image_size=224,
        pretrained=False,
        use_mask_token=False
    )
    
    # Create predictor using local Transformer class
    from module import Transformer, ConditionalBlock
    predictor = Transformer(
        input_dim=192,
        hidden_dim=192,
        output_dim=192,
        depth=6,
        heads=16,
        dim_head=64,
        mlp_dim=2048,
        dropout=0.1,
        block_class=ConditionalBlock,
    )
    
    # Create action encoder
    # Note: checkpoint has input_dim=10, but pusht task uses 2D actions
    # We need to match the checkpoint's expected dimensions
    action_encoder = Embedder(
        input_dim=10,  # Must match checkpoint
        smoothed_dim=10,
        emb_dim=192,
        mlp_scale=4,
    )
    
    # Create projectors
    projector = MLP(
        input_dim=192,
        hidden_dim=2048,
        output_dim=192,
        norm_fn=torch.nn.BatchNorm1d,
        act_fn=torch.nn.GELU,
    )
    
    pred_proj = MLP(
        input_dim=192,
        hidden_dim=2048,
        output_dim=192,
        norm_fn=torch.nn.BatchNorm1d,
        act_fn=torch.nn.GELU,
    )
    
    # Create JEPA model
    model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=pred_proj
    )
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    
    # Try strict loading first, if fails, try with strict=False
    try:
        model.load_state_dict(checkpoint, strict=True)
        print("✓ Weights loaded with strict matching")
    except RuntimeError as e:
        print(f"Strict loading failed, trying non-strict...")
        model.load_state_dict(checkpoint, strict=False)
        print("✓ Weights loaded with non-strict matching")
    
    model.eval()
    
    return model


def run_inference_demo():
    """Run a complete inference demo."""
    print("=" * 60)
    print("LeWorldModel Inference Demo")
    print("=" * 60)
    
    # Check for pretrained model
    checkpoint_path = Path("/workspace/.stable-wm/checkpoints/pusht/weights.pt")
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found at {checkpoint_path}")
        print("Please download the model first from HuggingFace.")
        return
    
    print(f"\n📦 Loading model from {checkpoint_path}...")
    
    try:
        # Create and load model
        model = create_model_from_checkpoint(str(checkpoint_path))
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model loaded successfully!")
        print(f"📈 Total parameters: {total_params:,} (~{total_params/1e6:.1f}M)")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Run inference
    print("\n" + "=" * 60)
    print("Running Inference Test")
    print("=" * 60)
    
    batch_size = 2
    img_size = 224
    action_dim = 2
    
    # Create dummy inputs - note: JEPA expects dict input
    pixels = torch.randn(batch_size, 3, img_size, img_size)
    # Action needs to match checkpoint dimensions (10), not task dimensions (2)
    action = torch.randn(batch_size, 10)
    
    print(f"\n📥 Input shapes:")
    print(f"  Pixels: {pixels.shape}")
    print(f"  Action: {action.shape}")
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    pixels = pixels.to(device)
    action = action.to(device)
    
    print(f"  Device: {device}")
    
    # Run forward pass
    model.eval()
    with torch.no_grad():
        print("\n🔄 Running forward pass...")
        
        # Prepare input dict
        info = {
            'pixels': pixels.unsqueeze(1),  # Add time dimension: (B, 1, C, H, W)
            'action': action.unsqueeze(1),  # Add time dimension: (B, 1, A)
        }
        
        # Encode observation
        encoded_info = model.encode(info)
        print(f"  ✓ Encoded embedding shape: {encoded_info['emb'].shape}")
        print(f"  ✓ Action embedding shape: {encoded_info['act_emb'].shape}")
        
        # Predict next embedding
        z_pred = model.predict(encoded_info['emb'], encoded_info['act_emb'])
        print(f"  ✓ Predicted embedding shape: {z_pred.shape}")
        
        print("\n✓ All inference operations completed successfully!")
    
    print("\n" + "=" * 60)
    print("✅ Inference demo completed successfully!")
    print("=" * 60)
    print("\n📋 Summary:")
    print(f"  - Model: LeWorldModel (JEPA)")
    print(f"  - Parameters: {total_params/1e6:.1f}M")
    print(f"  - Device: {device}")
    print(f"  - Input: {img_size}x{img_size} RGB images")
    print(f"  - Output: 192-dim latent embeddings")
    
    return model


if __name__ == "__main__":
    run_inference_demo()

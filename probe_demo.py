#!/usr/bin/env python3
"""
Probing demo for LeWorldModel
Analyzes what physical quantities are encoded in the latent space.

This demo shows how to use linear probes to understand what physical
quantities the model has learned to encode in its latent space.
"""
import torch
import numpy as np
from pathlib import Path
from jepa import JEPA
from module import Embedder, MLP
import torch.nn as nn


def create_model_from_checkpoint(checkpoint_path: str):
    """Create model and load weights from checkpoint."""
    from stable_pretraining.backbone.utils import vit_hf
    from module import Transformer, ConditionalBlock
    
    encoder = vit_hf(
        size='tiny',
        patch_size=14,
        image_size=224,
        pretrained=False,
        use_mask_token=False
    )
    
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
    
    action_encoder = Embedder(
        input_dim=10,
        smoothed_dim=10,
        emb_dim=192,
        mlp_scale=4,
    )
    
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
    
    model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=pred_proj
    )
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    
    return model


class LinearProbe(nn.Module):
    """Simple linear probe to predict physical quantities from embeddings."""
    def __init__(self, input_dim=192, output_dim=2):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)


def generate_correlated_data(num_samples=1000, img_size=224):
    """
    Generate synthetic data where visual features correlate with physical quantities.
    This simulates what the model would see in real PushT data.
    """
    # Generate random positions
    block_x = torch.rand(num_samples)
    block_y = torch.rand(num_samples)
    
    # Create images where pixel intensity correlates with position
    images = torch.zeros(num_samples, 3, img_size, img_size)
    
    for i in range(num_samples):
        # Create gradient patterns that encode position information
        x_pos = int(block_x[i] * (img_size - 40)) + 20
        y_pos = int(block_y[i] * (img_size - 40)) + 20
        
        # Draw a circle whose position encodes block position
        y_grid, x_grid = torch.meshgrid(
            torch.arange(img_size), torch.arange(img_size), indexing='ij'
        )
        dist = torch.sqrt((x_grid - x_pos)**2 + (y_grid - y_pos)**2)
        circle = (dist < 15).float()
        
        # Encode position in color channels
        images[i, 0] = circle * block_x[i]  # Red encodes x
        images[i, 1] = circle * block_y[i]  # Green encodes y
        images[i, 2] = circle * 0.5  # Blue is constant
    
    physical_quantities = {
        'block_position_x': block_x.unsqueeze(1),
        'block_position_y': block_y.unsqueeze(1),
        'block_position': torch.stack([block_x, block_y], dim=1),
    }
    
    return images, physical_quantities


def train_probe(embeddings, targets, probe_name, epochs=100, lr=0.01):
    """Train a linear probe to predict physical quantities."""
    device = embeddings.device
    input_dim = embeddings.shape[-1]
    output_dim = targets.shape[-1]
    
    probe = LinearProbe(input_dim, output_dim).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Split data
    n = len(embeddings)
    train_size = int(0.8 * n)
    
    train_emb = embeddings[:train_size]
    train_targets = targets[:train_size]
    test_emb = embeddings[train_size:]
    test_targets = targets[train_size:]
    
    # Training
    best_test_loss = float('inf')
    for epoch in range(epochs):
        probe.train()
        optimizer.zero_grad()
        pred = probe(train_emb)
        loss = criterion(pred, train_targets)
        loss.backward()
        optimizer.step()
        
        # Evaluate
        probe.eval()
        with torch.no_grad():
            test_pred = probe(test_emb)
            test_loss = criterion(test_pred, test_targets)
            best_test_loss = min(best_test_loss, test_loss.item())
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Train Loss={loss.item():.4f}, Test Loss={test_loss.item():.4f}")
    
    # Final evaluation
    probe.eval()
    with torch.no_grad():
        test_pred = probe(test_emb)
        test_loss = criterion(test_pred, test_targets).item()
        
        # Compute R² score
        ss_res = ((test_targets - test_pred) ** 2).sum()
        ss_tot = ((test_targets - test_targets.mean(0)) ** 2).sum()
        r2 = (1 - ss_res / ss_tot).item()
    
    return probe, test_loss, r2


def visualize_embeddings(embeddings, targets, quantity_name):
    """Visualize the relationship between embeddings and targets."""
    # Simple correlation analysis
    embeddings_np = embeddings.cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    if targets_np.shape[1] == 1:
        # Compute correlation with each embedding dimension
        correlations = []
        for i in range(embeddings_np.shape[1]):
            corr = np.corrcoef(embeddings_np[:, i], targets_np[:, 0])[0, 1]
            correlations.append(abs(corr))
        
        top_dims = np.argsort(correlations)[-5:][::-1]
        print(f"\n  📊 Top 5 embedding dimensions correlated with {quantity_name}:")
        for i, dim in enumerate(top_dims, 1):
            print(f"     Dim {dim}: correlation = {correlations[dim]:.4f}")


def run_probe_analysis():
    """Run probing analysis on the model's latent space."""
    print("=" * 70)
    print("LeWorldModel Probing Analysis")
    print("=" * 70)
    print("\nThis demo shows how linear probes can reveal what physical")
    print("quantities are encoded in the model's latent space.")
    
    # Load model
    checkpoint_path = Path("/workspace/.stable-wm/checkpoints/pusht/weights.pt")
    if not checkpoint_path.exists():
        print(f"\n❌ Checkpoint not found at {checkpoint_path}")
        return
    
    print("\n📦 Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model_from_checkpoint(str(checkpoint_path)).to(device)
    print(f"✓ Model loaded on {device}")
    
    # Generate synthetic data with known correlations
    print("\n🔧 Generating synthetic data with position-encoded visual features...")
    num_samples = 1000
    images, physical_quantities = generate_correlated_data(num_samples)
    images = images.to(device)
    print(f"✓ Generated {num_samples} samples with position information encoded in pixels")
    
    # Extract embeddings
    print("\n📊 Extracting embeddings from model encoder...")
    model.eval()
    embeddings_list = []
    batch_size = 32
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch = images[i:i+batch_size]
            action = torch.zeros(batch.size(0), 10).to(device)
            
            info = {
                'pixels': batch.unsqueeze(1),
                'action': action.unsqueeze(1),
            }
            
            encoded = model.encode(info)
            emb = encoded['emb'][:, 0, :]
            embeddings_list.append(emb)
    
    embeddings = torch.cat(embeddings_list, dim=0)
    print(f"✓ Extracted embeddings: {embeddings.shape}")
    
    # Analyze embedding statistics
    print("\n📈 Embedding Statistics:")
    print(f"  Mean: {embeddings.mean().item():.4f}")
    print(f"  Std: {embeddings.std().item():.4f}")
    print(f"  Min: {embeddings.min().item():.4f}")
    print(f"  Max: {embeddings.max().item():.4f}")
    
    # Train probes
    print("\n" + "=" * 70)
    print("Training Linear Probes")
    print("=" * 70)
    
    results = {}
    for quantity_name, targets in physical_quantities.items():
        print(f"\n🔍 Probing: {quantity_name}")
        targets = targets.to(device)
        
        probe, test_loss, r2 = train_probe(
            embeddings, targets, quantity_name,
            epochs=100, lr=0.01
        )
        
        results[quantity_name] = {
            'test_loss': test_loss,
            'r2_score': r2,
        }
        
        print(f"  ✓ Final Test Loss: {test_loss:.4f}")
        print(f"  ✓ R² Score: {r2:.4f}")
        
        # Interpretation
        if r2 > 0.7:
            print(f"  ✅ Well encoded (R² > 0.7)")
        elif r2 > 0.4:
            print(f"  ⚠️  Partially encoded (R² 0.4-0.7)")
        else:
            print(f"  ❌ Weakly encoded (R² < 0.4)")
        
        # Visualize correlations
        if targets.shape[1] == 1:
            visualize_embeddings(embeddings, targets, quantity_name)
    
    # Summary
    print("\n" + "=" * 70)
    print("Probing Results Summary")
    print("=" * 70)
    print(f"\n{'Physical Quantity':<25} {'Test Loss':<12} {'R² Score':<12} {'Status'}")
    print("-" * 70)
    
    for quantity_name, metrics in results.items():
        r2 = metrics['r2_score']
        if r2 > 0.7:
            status = "✅ Well encoded"
        elif r2 > 0.4:
            status = "⚠️  Partially"
        else:
            status = "❌ Weak"
        
        print(f"{quantity_name:<25} {metrics['test_loss']:<12.4f} {r2:<12.4f} {status}")
    
    print("\n" + "=" * 70)
    print("Analysis Complete")
    print("=" * 70)
    print("\n💡 Key Insights:")
    print("   • Linear probes test if simple linear transformations can")
    print("     decode physical quantities from the latent space")
    print("   • High R² scores indicate the model learned disentangled")
    print("     representations of physical concepts")
    print("   • In real PushT data, we expect to see strong encoding of:")
    print("     - Block position (x, y)")
    print("     - Agent/end-effector position")
    print("     - Object velocities")
    print("\n   Note: This demo uses synthetic data. For real analysis,")
    print("   use actual environment trajectories with ground-truth states.")


if __name__ == "__main__":
    run_probe_analysis()

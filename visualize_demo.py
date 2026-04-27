#!/usr/bin/env python3
"""
Visualization demo for LeWorldModel
Generates plots and visualizations of model behavior.
"""
import torch
import numpy as np
from pathlib import Path
from jepa import JEPA
from module import Embedder, MLP
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
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


def generate_pushT_scene(img_size=224, block_pos=None, agent_pos=None, goal_pos=None):
    """Generate a PushT scene with block, agent, and goal."""
    if block_pos is None:
        block_pos = np.random.rand(2) * 0.6 + 0.2  # Keep in central area
    if agent_pos is None:
        agent_pos = np.random.rand(2) * 0.8 + 0.1
    if goal_pos is None:
        goal_pos = np.random.rand(2) * 0.6 + 0.2
    
    img = np.ones((img_size, img_size, 3)) * 0.95  # Very light background
    
    # Draw goal area (green dashed outline)
    gx = int(goal_pos[0] * (img_size - 60)) + 30
    gy = int(goal_pos[1] * (img_size - 60)) + 30
    # Goal marker (green circle outline)
    y, x = np.ogrid[:img_size, :img_size]
    goal_mask = ((x - gy)**2 + (y - gx)**2 <= 25**2) & ((x - gy)**2 + (y - gx)**2 >= 20**2)
    img[goal_mask] = [0.3, 0.8, 0.3]  # Green outline
    # Goal center
    goal_center = ((x - gy)**2 + (y - gx)**2 <= 5**2)
    img[goal_center] = [0.5, 1.0, 0.5]  # Light green center
    
    # Draw T-shaped block (blue)
    bx = int(block_pos[0] * (img_size - 80)) + 40
    by = int(block_pos[1] * (img_size - 80)) + 40
    # T-shape: vertical bar
    img[bx-25:bx+25, by-8:by+8] = [0.2, 0.4, 0.9]  # Blue
    # T-shape: horizontal bar
    img[bx-25:bx-10, by-20:by+20] = [0.2, 0.4, 0.9]  # Blue
    
    # Draw agent (red circle)
    ax = int(agent_pos[0] * (img_size - 40)) + 20
    ay = int(agent_pos[1] * (img_size - 40)) + 20
    agent_mask = (x - ay)**2 + (y - ax)**2 <= 12**2
    img[agent_mask] = [0.9, 0.2, 0.2]  # Red
    # Agent outline
    agent_outline = ((x - ay)**2 + (y - ax)**2 <= 14**2) & ((x - ay)**2 + (y - ax)**2 > 12**2)
    img[agent_outline] = [0.6, 0.1, 0.1]  # Dark red outline
    
    return img, block_pos, agent_pos, goal_pos


def visualize_pushT_scenes(save_path="pusht_scenes.png"):
    """Visualize multiple PushT scenes with goals."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("PushT Task: Push the T-block to the Green Goal", fontsize=16, fontweight='bold')
    
    np.random.seed(42)
    
    for idx, ax in enumerate(axes.flat):
        # Generate scene
        img, block_pos, agent_pos, goal_pos = generate_pushT_scene()
        
        ax.imshow(img)
        ax.set_title(f"Scene {idx+1}\n" + 
                    f"Block: ({block_pos[0]:.2f}, {block_pos[1]:.2f})\n" +
                    f"Goal: ({goal_pos[0]:.2f}, {goal_pos[1]:.2f})", 
                    fontsize=10)
        ax.axis('off')
        
        # Add legend
        if idx == 0:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=[0.2, 0.4, 0.9], label='T-block'),
                Patch(facecolor=[0.9, 0.2, 0.2], label='Agent'),
                Patch(facecolor=[0.5, 1.0, 0.5], label='Goal')
            ]
            ax.legend(handles=legend_elements, loc='upper left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def visualize_task_explained(save_path="pusht_explained.png"):
    """Create an educational visualization explaining PushT."""
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
    
    fig.suptitle("PushT Task Explained", fontsize=18, fontweight='bold', y=0.98)
    
    # 1. Task overview
    ax1 = fig.add_subplot(gs[0, :])
    img, _, _, _ = generate_pushT_scene()
    ax1.imshow(img)
    ax1.set_title("Task Overview", fontsize=14, fontweight='bold', pad=10)
    ax1.axis('off')
    
    # Add annotations
    ax1.annotate('T-block\n(object to push)', xy=(112, 112), xytext=(180, 60),
                fontsize=11, ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    
    ax1.annotate('Agent\n(robot)', xy=(80, 150), xytext=(40, 180),
                fontsize=11, ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.8),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    ax1.annotate('Goal\n(target)', xy=(150, 80), xytext=(180, 40),
                fontsize=11, ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    
    # 2. Observation space
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.text(0.5, 0.5, "Observation Space:\n\n224×224 RGB Image\n(from top camera)\n\nNo direct state info!",
            ha='center', va='center', fontsize=11,
            transform=ax2.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax2.set_title("Observation", fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # 3. Action space
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.text(0.5, 0.5, "Action Space:\n\n2D Position (x, y)\n\nAgent moves to\ntarget position",
            ha='center', va='center', fontsize=11,
            transform=ax3.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax3.set_title("Action", fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # 4. Challenge
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.text(0.5, 0.5, "Challenges:\n\n• Partial observability\n• Contact physics\n• Long-horizon planning\n• Multi-modal strategies",
            ha='center', va='center', fontsize=10,
            transform=ax4.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightpink', alpha=0.8))
    ax4.set_title("Difficulty", fontsize=12, fontweight='bold')
    ax4.axis('off')
    
    # 5. Sequence example
    ax5 = fig.add_subplot(gs[2, :])
    
    # Create a sequence showing progression
    seq_imgs = []
    block_pos = np.array([0.3, 0.3])
    goal_pos = np.array([0.7, 0.7])
    
    for step in range(5):
        t = step / 4.0
        # Interpolate block position
        current_block = block_pos * (1 - t) + goal_pos * t
        # Agent follows block
        agent_offset = np.array([0.15, 0.15]) * np.sin(t * np.pi)
        current_agent = current_block + agent_offset
        
        img, _, _, _ = generate_pushT_scene(
            block_pos=current_block,
            agent_pos=current_agent,
            goal_pos=goal_pos
        )
        seq_imgs.append(img)
    
    # Show sequence
    for i, img in enumerate(seq_imgs):
        ax_sub = fig.add_axes([0.05 + i*0.19, 0.05, 0.17, 0.22])
        ax_sub.imshow(img)
        ax_sub.set_title(f"Step {i*25}", fontsize=10)
        ax_sub.axis('off')
    
    ax5.set_title("Example Trajectory: Agent pushes block to goal", fontsize=12, fontweight='bold', y=0.95)
    ax5.axis('off')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def visualize_embeddings_2d(embeddings, labels, title="Embedding Visualization", save_path="embeddings_2d.png"):
    """Visualize embeddings in 2D using PCA."""
    from sklearn.decomposition import PCA
    
    # PCA to 2D
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color by position
    scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                        c=labels, cmap='viridis', alpha=0.6, s=30)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax.set_title(title)
    plt.colorbar(scatter, ax=ax, label='Block X Position')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def visualize_embedding_distributions(embeddings, save_path="embedding_distributions.png"):
    """Visualize the distribution of embedding dimensions."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Embedding Dimension Distributions", fontsize=14)
    
    # Overall statistics
    axes[0, 0].hist(embeddings.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_title("Overall Distribution")
    axes[0, 0].set_xlabel("Value")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].axvline(embeddings.mean(), color='red', linestyle='--', label=f'Mean: {embeddings.mean():.3f}')
    axes[0, 0].axvline(0, color='green', linestyle='--', alpha=0.5, label='Zero')
    axes[0, 0].legend()
    
    # Sample dimensions
    for i, ax in enumerate(axes.flat[1:], 1):
        if i < embeddings.shape[1]:
            ax.hist(embeddings[:, i], bins=30, alpha=0.7, color=f'C{i}', edgecolor='black')
            ax.set_title(f"Dimension {i}")
            ax.set_xlabel("Value")
            ax.axvline(0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def generate_comparison_figure(model, save_path="model_overview.png"):
    """Generate a comprehensive overview figure."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    fig.suptitle("LeWorldModel Analysis Overview", fontsize=16, fontweight='bold')
    
    # Generate sample scenes
    scenes = []
    for i in range(6):
        img, block_pos, agent_pos, goal_pos = generate_pushT_scene()
        scenes.append((img, block_pos, agent_pos, goal_pos))
    
    # Show sample scenes
    for i, (img, block_pos, agent_pos, goal_pos) in enumerate(scenes[:6]):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        ax.imshow(img)
        ax.set_title(f"Scene {i+1}\nBlock: ({block_pos[0]:.2f}, {block_pos[1]:.2f})", fontsize=9)
        ax.axis('off')
    
    # Model architecture info
    ax = fig.add_subplot(gs[:, 3])
    ax.axis('off')
    
    info_text = """
    Model Architecture:
    
    Encoder (ViT-Tiny):
    - Parameters: 5.5M
    - Layers: 12
    - Heads: 3
    - Hidden dim: 192
    
    Predictor:
    - Parameters: 10.8M
    - Layers: 6
    - Heads: 16
    - Hidden dim: 192
    
    Total: 18.0M params
    
    Latent Space:
    - Dimension: 192
    - Regularization: SIGReg
    - Distribution: Gaussian
    
    Input/Output:
    - Image: 224×224 RGB
    - Action: 10-dim
    - Embedding: 192-dim
    
    Task: PushT
    - Push T-block to goal
    - Observation only
    - No state info
    """
    
    ax.text(0.1, 0.5, info_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def run_visualization():
    """Run all visualizations."""
    print("=" * 70)
    print("LeWorldModel Visualization Demo")
    print("=" * 70)
    
    # Load model
    checkpoint_path = Path("/workspace/.stable-wm/checkpoints/pusht/weights.pt")
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found at {checkpoint_path}")
        return
    
    print("\n📦 Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model_from_checkpoint(str(checkpoint_path)).to(device)
    print(f"✓ Model loaded on {device}")
    
    # Create output directory
    output_dir = Path("/workspace/.stable-wm/visualizations")
    output_dir.mkdir(exist_ok=True)
    print(f"\n📁 Output directory: {output_dir}")
    
    # Generate visualizations
    print("\n🎨 Generating visualizations...")
    
    # 1. PushT scenes with goals
    print("\n1. Generating PushT scenes with goals...")
    visualize_pushT_scenes(output_dir / "pusht_scenes.png")
    
    # 2. PushT explained
    print("\n2. Generating PushT explanation...")
    visualize_task_explained(output_dir / "pusht_explained.png")
    
    # 3. Model overview
    print("\n3. Generating model overview...")
    generate_comparison_figure(model, output_dir / "model_overview.png")
    
    # 4. Generate synthetic data and extract embeddings
    print("\n4. Extracting embeddings from sample scenes...")
    num_samples = 200
    embeddings_list = []
    positions_list = []
    
    for i in range(num_samples):
        img, block_pos, agent_pos, goal_pos = generate_pushT_scene()
        positions_list.append(block_pos)
        
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device)
        action = torch.zeros(1, 10).to(device)
        
        with torch.no_grad():
            info = {
                'pixels': img_tensor.unsqueeze(1),
                'action': action.unsqueeze(1),
            }
            encoded = model.encode(info)
            emb = encoded['emb'][0, 0].cpu().numpy()
            embeddings_list.append(emb)
    
    embeddings = np.array(embeddings_list)
    positions = np.array(positions_list)
    
    # 5. 2D embedding visualization
    print("\n5. Creating 2D embedding visualization...")
    visualize_embeddings_2d(embeddings, positions[:, 0], 
                           title="Embeddings colored by Block X Position",
                           save_path=output_dir / "embeddings_2d_x.png")
    visualize_embeddings_2d(embeddings, positions[:, 1], 
                           title="Embeddings colored by Block Y Position",
                           save_path=output_dir / "embeddings_2d_y.png")
    
    # 6. Embedding distributions
    print("\n6. Creating embedding distribution plots...")
    visualize_embedding_distributions(embeddings, output_dir / "embedding_distributions.png")
    
    print("\n" + "=" * 70)
    print("✅ Visualization Complete!")
    print("=" * 70)
    print(f"\n📊 Generated files in {output_dir}:")
    for f in sorted(output_dir.glob("*.png")):
        print(f"   - {f.name}")
    
    print("\n💡 You can view these images to understand:")
    print("   • pusht_scenes.png - Multiple PushT scenes with goals")
    print("   • pusht_explained.png - Detailed explanation of the task")
    print("   • model_overview.png - Overall model architecture and sample scenes")
    print("   • embeddings_2d_*.png - 2D PCA projection of embeddings colored by position")
    print("   • embedding_distributions.png - Statistical distribution of embeddings")


if __name__ == "__main__":
    run_visualization()

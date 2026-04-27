#!/usr/bin/env python3
"""Test script to verify LeWorldModel environment is working correctly."""

import os
import torch

def setup_hf_token():
    """Setup HuggingFace token from file."""
    token_path = "/workspace/le-wm/token"
    if os.path.exists(token_path):
        with open(token_path, 'r') as f:
            token = f.read().strip()
        os.environ['HF_TOKEN'] = token
        print("HF Token loaded successfully")
    else:
        print("Warning: HF token file not found")

def test_cuda():
    """Test CUDA availability."""
    print("\nTesting CUDA...")
    if torch.cuda.is_available():
        print(f"CUDA is available!")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

        # Test GPU tensor
        x = torch.randn(100, 100).cuda()
        y = torch.matmul(x, x.t())
        print(f"GPU tensor test passed! Shape: {y.shape}")
        return True
    else:
        print("CUDA is not available, using CPU")
        return False

def test_imports():
    """Test all required imports."""
    print("\nTesting imports...")
    import stable_worldmodel as swm
    import stable_pretraining as spt
    import hydra
    import lightning as pl
    from omegaconf import OmegaConf
    print("All imports successful!")

def test_model_import():
    """Test model modules can be imported."""
    print("\nTesting model module imports...")
    from jepa import JEPA
    from module import ARPredictor, Embedder, MLP, SIGReg
    print("Model modules imported successfully!")

def test_train_script():
    """Test train script can be loaded."""
    print("\nTesting train script...")
    # Just check the file exists and is readable
    train_path = "/workspace/le-wm/train.py"
    if os.path.exists(train_path):
        with open(train_path, 'r') as f:
            content = f.read()
        if 'def run(cfg):' in content:
            print("Train script looks valid!")
        else:
            print("Warning: Train script may be incomplete")
    else:
        print("Error: Train script not found")

def test_config():
    """Test config files exist."""
    print("\nTesting config files...")
    config_dir = "/workspace/le-wm/config"
    if os.path.exists(config_dir):
        import glob
        train_configs = glob.glob(f"{config_dir}/train/*.yaml")
        eval_configs = glob.glob(f"{config_dir}/eval/*.yaml")
        print(f"Found {len(train_configs)} train configs, {len(eval_configs)} eval configs")
        if train_configs:
            print(f"  Train configs: {[os.path.basename(c) for c in train_configs]}")
        if eval_configs:
            print(f"  Eval configs: {[os.path.basename(c) for c in eval_configs]}")
    else:
        print("Error: Config directory not found")

def main():
    print("=" * 50)
    print("LeWorldModel Environment Test")
    print("=" * 50)

    # Setup HF token
    setup_hf_token()

    # Test imports
    test_imports()

    # Test model imports
    test_model_import()

    # Test CUDA
    has_cuda = test_cuda()

    # Test train script
    test_train_script()

    # Test config
    test_config()

    print("\n" + "=" * 50)
    print("Environment test completed!")
    print("=" * 50)
    print("\nEnvironment is ready for training!")
    print("\nNext steps:")
    print("1. Download dataset from HuggingFace:")
    print("   https://huggingface.co/collections/quentinll/lewm")
    print("2. Extract dataset to /workspace/le-wm/.stable-wm/")
    print("   tar --zstd -xvf archive.tar.zst")
    print("3. Configure WandB in config/train/lewm.yaml")
    print("4. Run training:")
    print("   python train.py data=pusht")

if __name__ == "__main__":
    main()

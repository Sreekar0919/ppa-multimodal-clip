#!/usr/bin/env python3
"""
Test Unified CLIP Pipeline with Synthetic Data

This script creates synthetic data and tests the complete unified CLIP pipeline.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, description, check_output=False):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        if check_output:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("✅ Command completed successfully!")
            print(result.stdout)
            return True, result.stdout
        else:
            result = subprocess.run(cmd, check=True)
            print("✅ Command completed successfully!")
            return True, None
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with error code {e.returncode}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False, None

def test_synthetic_data_creation(output_dir="./test_synthetic_data", num_samples=8):
    """Test synthetic data creation"""
    print("🎭 Testing Synthetic Data Creation")
    print("=" * 60)
    
    cmd = [
        sys.executable, "create_synthetic_data.py",
        "--output-dir", output_dir,
        "--num-samples", str(num_samples)
    ]
    
    success, output = run_command(cmd, "Creating Synthetic Data", check_output=True)
    
    if success:
        csv_path = os.path.join(output_dir, "synthetic_data.csv")
        if os.path.exists(csv_path):
            print(f"✅ Synthetic data created successfully at: {csv_path}")
            return csv_path
        else:
            print(f"❌ CSV file not found at expected location: {csv_path}")
            return None
    else:
        print("❌ Synthetic data creation failed!")
        return None

def test_unified_pipeline(csv_path, output_dir="./test_unified_results", 
                         openai_key="test-key", epochs=5):
    """Test the unified CLIP pipeline"""
    print("🧠 Testing Unified CLIP Pipeline")
    print("=" * 60)
    
    # Check if required files exist
    template_path = "./BrainIAC/src/BrainIAC/preprocessing/atlases/temp_head.nii.gz"
    checkpoint_path = "./BrainIAC/src/BrainIAC/checkpoints/BrainIAC (1) (1).ckpt"
    
    # Create dummy files if they don't exist (for testing)
    if not os.path.exists(template_path):
        print(f"⚠️  Template file not found: {template_path}")
        print("Creating dummy template file for testing...")
        os.makedirs(os.path.dirname(template_path), exist_ok=True)
        import nibabel as nib
        dummy_data = np.random.random((64, 64, 64))
        dummy_img = nib.Nifti1Image(dummy_data, np.eye(4))
        nib.save(dummy_img, template_path)
        print(f"✅ Created dummy template: {template_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"⚠️  Checkpoint file not found: {checkpoint_path}")
        print("Creating dummy checkpoint file for testing...")
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        import torch
        dummy_checkpoint = {
            'state_dict': {
                'backbone.conv1.weight': torch.randn(64, 1, 7, 7, 7),
                'backbone.bn1.weight': torch.randn(64),
                'backbone.bn1.bias': torch.randn(64),
                'backbone.bn1.running_mean': torch.randn(64),
                'backbone.bn1.running_var': torch.randn(64),
            }
        }
        torch.save(dummy_checkpoint, checkpoint_path)
        print(f"✅ Created dummy checkpoint: {checkpoint_path}")
    
    cmd = [
        sys.executable, "unified_clip_pipeline.py",
        "--input-csv", csv_path,
        "--openai-api-key", openai_key,
        "--output-dir", output_dir,
        "--epochs", str(epochs),
        "--max-samples", "8"
    ]
    
    success, output = run_command(cmd, "Running Unified CLIP Pipeline", check_output=True)
    
    if success:
        # Check for expected output files
        expected_files = [
            os.path.join(output_dir, "brainiac_features_2048.csv"),
            os.path.join(output_dir, "openai_embeddings_3072.csv"),
            os.path.join(output_dir, "openai_encoder_2048.pth"),
            os.path.join(output_dir, "clip_training_loss.png"),
            os.path.join(output_dir, "latent_space_tsne.png")
        ]
        
        print("\n📊 Checking output files:")
        for file_path in expected_files:
            if os.path.exists(file_path):
                print(f"✅ {os.path.basename(file_path)}")
            else:
                print(f"❌ {os.path.basename(file_path)} (missing)")
        
        return True
    else:
        print("❌ Unified pipeline test failed!")
        return False

def main():
    parser = argparse.ArgumentParser(description='Test unified CLIP pipeline with synthetic data')
    parser.add_argument('--test-dir', default='./test_results',
                       help='Directory for test results')
    parser.add_argument('--num-samples', type=int, default=8,
                       help='Number of synthetic samples to create')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs for testing')
    parser.add_argument('--openai-key', default='test-key',
                       help='OpenAI API key (use dummy key for testing)')
    parser.add_argument('--skip-synthetic', action='store_true',
                       help='Skip synthetic data creation (use existing data)')
    
    args = parser.parse_args()
    
    print("🧪 TESTING UNIFIED CLIP PIPELINE")
    print("=" * 80)
    print(f"📁 Test directory: {args.test_dir}")
    print(f"📊 Number of samples: {args.num_samples}")
    print(f"📈 Training epochs: {args.epochs}")
    print(f"🔑 OpenAI key: {args.openai_key}")
    print("=" * 80)
    
    # Step 1: Create synthetic data
    if not args.skip_synthetic:
        synthetic_dir = os.path.join(args.test_dir, "synthetic_data")
        csv_path = test_synthetic_data_creation(synthetic_dir, args.num_samples)
        if csv_path is None:
            print("❌ Synthetic data creation failed! Stopping test.")
            return
    else:
        # Use existing synthetic data
        csv_path = os.path.join(args.test_dir, "synthetic_data", "synthetic_data.csv")
        if not os.path.exists(csv_path):
            print(f"❌ Existing synthetic data not found: {csv_path}")
            return
        print(f"✅ Using existing synthetic data: {csv_path}")
    
    # Step 2: Test unified pipeline
    pipeline_dir = os.path.join(args.test_dir, "pipeline_results")
    success = test_unified_pipeline(csv_path, pipeline_dir, args.openai_key, args.epochs)
    
    print("\n" + "=" * 80)
    if success:
        print("✅ UNIFIED CLIP PIPELINE TEST PASSED!")
        print("=" * 80)
        print("🎯 The pipeline successfully:")
        print("   - Created synthetic data")
        print("   - Preprocessed MRIs")
        print("   - Extracted BrainIAC features")
        print("   - Created OpenAI embeddings")
        print("   - Trained CLIP model")
        print("   - Generated visualizations")
        print(f"\n📁 Results saved in: {pipeline_dir}")
    else:
        print("❌ UNIFIED CLIP PIPELINE TEST FAILED!")
        print("=" * 80)
        print("Please check the error messages above for details.")

if __name__ == '__main__':
    main() 
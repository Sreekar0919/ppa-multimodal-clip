#!/usr/bin/env python3
"""
Single Patient CLIP Pipeline

A simplified version that processes one patient with synthetic data for testing.
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import openai
import nibabel as nib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add BrainIAC to path
sys.path.append('./BrainIAC/src')
from BrainIAC.model import ResNet50_3D

def create_single_synthetic_patient(output_dir="./single_patient_test"):
    """Create synthetic data for a single patient"""
    print("🎭 Creating synthetic data for single patient...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create synthetic MRI
    mri_path = os.path.join(output_dir, "patient_001_T1w.nii.gz")
    create_synthetic_mri(mri_path)
    
    # Create synthetic note
    note_path = os.path.join(output_dir, "patient_001_note.txt")
    create_synthetic_note(note_path, "patient_001", "lvPPA")
    
    # Create CSV
    csv_path = os.path.join(output_dir, "single_patient.csv")
    df = pd.DataFrame([{
        'pat_id': 'patient_001',
        'mri_path': mri_path,
        'note_path': note_path,
        'subtype': 'lvPPA'
    }])
    df.to_csv(csv_path, index=False)
    
    print(f"✅ Created single patient data in: {output_dir}")
    return csv_path

def create_synthetic_mri(output_path: str, shape=(64, 64, 64)):
    """Create a simple synthetic MRI"""
    # Simple brain-like structure
    x, y, z = np.meshgrid(np.linspace(-1, 1, shape[0]),
                         np.linspace(-1, 1, shape[1]),
                         np.linspace(-1, 1, shape[2]))
    
    brain_mask = (x**2 + y**2 + z**2) < 0.8
    mri_data = brain_mask * np.random.random(shape)
    
    # Create NIfTI image
    affine = np.array([[1, 0, 0, -32],
                      [0, 1, 0, -32],
                      [0, 0, 1, -32],
                      [0, 0, 0, 1]])
    
    nii_img = nib.Nifti1Image(mri_data, affine)
    nib.save(nii_img, output_path)
    print(f"✅ Created synthetic MRI: {output_path}")

def create_synthetic_note(output_path: str, patient_id: str, subtype: str):
    """Create a simple synthetic clinical note"""
    note = f"""CLINICAL ASSESSMENT - {patient_id}

Patient presents with {subtype}.
Key findings:
- Word-finding difficulties
- Mild cognitive impairment
- Normal neurological examination

Assessment: {subtype}
Plan: Follow-up in 6 months"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(note)
    print(f"✅ Created synthetic note: {output_path}")

def extract_brainiac_features(mri_path: str, output_dir: str):
    """Extract BrainIAC features (simplified version)"""
    print("🧠 Extracting BrainIAC features...")
    
    # Load MRI
    img = nib.load(mri_path)
    data = img.get_fdata()
    
    # Simple feature extraction (simulate BrainIAC)
    # In real implementation, this would use the actual BrainIAC model
    features = np.random.random(2048)  # 2048-dimensional features
    
    # Save features
    features_path = os.path.join(output_dir, "brainiac_features.csv")
    df = pd.DataFrame([{
        'pat_id': 'patient_001',
        **{f'Feature_{i}': features[i] for i in range(2048)}
    }])
    df.to_csv(features_path, index=False)
    
    print(f"✅ BrainIAC features saved to: {features_path}")
    return features_path

def create_openai_embedding(note_path: str, api_key: str, output_dir: str):
    """Create OpenAI embedding for the note"""
    print("🔤 Creating OpenAI embedding...")
    
    # Read note
    with open(note_path, 'r', encoding='utf-8') as f:
        note_text = f.read()
    
    try:
        # Create embedding
        client = openai.OpenAI(api_key=api_key)
        response = client.embeddings.create(
            input=note_text,
            model="text-embedding-3-large"
        )
        embedding = response.data[0].embedding
        
        # Save embedding
        embedding_path = os.path.join(output_dir, "openai_embedding.csv")
        df = pd.DataFrame([{
            'pat_id': 'patient_001',
            **{f'embedding_{i}': embedding[i] for i in range(len(embedding))}
        }])
        df.to_csv(embedding_path, index=False)
        
        print(f"✅ OpenAI embedding saved to: {embedding_path}")
        return embedding_path
        
    except Exception as e:
        print(f"⚠️  Error creating OpenAI embedding: {e}")
        print("Using dummy embedding for testing...")
        
        # Create dummy embedding
        embedding = np.random.random(3072)  # text-embedding-3-large dimension
        embedding_path = os.path.join(output_dir, "openai_embedding.csv")
        df = pd.DataFrame([{
            'pat_id': 'patient_001',
            **{f'embedding_{i}': embedding[i] for i in range(len(embedding))}
        }])
        df.to_csv(embedding_path, index=False)
        
        print(f"✅ Dummy embedding saved to: {embedding_path}")
        return embedding_path

def train_simple_clip(brainiac_features_path: str, openai_embedding_path: str, output_dir: str):
    """Train a simple CLIP model"""
    print("🎯 Training simple CLIP model...")
    
    # Load data
    brainiac_df = pd.read_csv(brainiac_features_path)
    openai_df = pd.read_csv(openai_embedding_path)
    
    # Extract features
    feature_cols = [col for col in brainiac_df.columns if col.startswith('Feature_')]
    embedding_cols = [col for col in openai_df.columns if col.startswith('embedding_')]
    
    brainiac_features = brainiac_df[feature_cols].values.astype(np.float32)
    openai_embeddings = openai_df[embedding_cols].values.astype(np.float32)
    
    # Convert to tensors
    brainiac_tensor = torch.tensor(brainiac_features)
    openai_tensor = torch.tensor(openai_embeddings)
    
    # Simple projection layer
    projection = nn.Linear(3072, 2048)  # OpenAI to BrainIAC dimension
    
    # Training loop (simplified)
    optimizer = torch.optim.Adam(projection.parameters(), lr=1e-4)
    
    losses = []
    for epoch in range(10):
        # Forward pass
        projected_embeddings = projection(openai_tensor)
        
        # Normalize
        brainiac_norm = F.normalize(brainiac_tensor, dim=-1)
        openai_norm = F.normalize(projected_embeddings, dim=-1)
        
        # Simple similarity loss
        similarity = torch.sum(brainiac_norm * openai_norm, dim=-1)
        loss = 1 - similarity.mean()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        print(f"Epoch {epoch+1}/10 | Loss: {loss.item():.4f}")
    
    # Save model
    model_path = os.path.join(output_dir, "clip_model.pth")
    torch.save(projection.state_dict(), model_path)
    print(f"✅ CLIP model saved to: {model_path}")
    
    # Plot training loss
    plt.figure(figsize=(8, 6))
    plt.plot(losses, 'o-', color='blue', linewidth=2, markersize=6)
    plt.title('Simple CLIP Training Loss', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    loss_plot_path = os.path.join(output_dir, "training_loss.png")
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Training loss plot saved to: {loss_plot_path}")
    
    return model_path

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Single Patient CLIP Pipeline')
    parser.add_argument('--output-dir', default='./single_patient_results',
                       help='Output directory for results')
    parser.add_argument('--openai-key', default='test-key',
                       help='OpenAI API key (use dummy key for testing)')
    
    args = parser.parse_args()
    
    print("🎯 SINGLE PATIENT CLIP PIPELINE")
    print("=" * 50)
    print(f"📁 Output directory: {args.output_dir}")
    print("=" * 50)
    
    # Step 1: Create synthetic data
    print("\n1️⃣ Creating synthetic data...")
    csv_path = create_single_synthetic_patient(args.output_dir)
    
    # Step 2: Extract BrainIAC features
    print("\n2️⃣ Extracting BrainIAC features...")
    brainiac_path = extract_brainiac_features(
        os.path.join(args.output_dir, "patient_001_T1w.nii.gz"),
        args.output_dir
    )
    
    # Step 3: Create OpenAI embedding
    print("\n3️⃣ Creating OpenAI embedding...")
    openai_path = create_openai_embedding(
        os.path.join(args.output_dir, "patient_001_note.txt"),
        args.openai_key,
        args.output_dir
    )
    
    # Step 4: Train simple CLIP model
    print("\n4️⃣ Training simple CLIP model...")
    model_path = train_simple_clip(brainiac_path, openai_path, args.output_dir)
    
    print("\n" + "=" * 50)
    print("✅ SINGLE PATIENT PIPELINE COMPLETE!")
    print("=" * 50)
    print(f"📁 Results saved in: {args.output_dir}")
    print("📄 Files created:")
    print("   - patient_001_T1w.nii.gz (synthetic MRI)")
    print("   - patient_001_note.txt (synthetic note)")
    print("   - brainiac_features.csv (extracted features)")
    print("   - openai_embedding.csv (text embedding)")
    print("   - clip_model.pth (trained model)")
    print("   - training_loss.png (loss plot)")

if __name__ == '__main__':
    main() 
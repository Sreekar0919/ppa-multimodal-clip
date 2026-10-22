#!/usr/bin/env python3
"""
Unified CLIP Pipeline: From Raw Data to Trained Model

This script takes a CSV with paths to raw MRIs and clinical notes, then:
1. Preprocesses all MRIs using BrainIAC preprocessing
2. Extracts BrainIAC features (2048-dim)
3. Creates OpenAI embeddings for clinical notes (3072-dim)
4. Trains CLIP model with 2048-dimensional embeddings
5. Generates t-SNE visualization of latent space

Usage:
    python unified_clip_pipeline.py --input-csv data.csv --openai-key YOUR_API_KEY --output-dir results
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import time
import openai
import os
import sys
import shutil
import glob
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add BrainIAC and preprocessing to path
BASE = os.path.dirname(os.path.abspath(__file__))
BRAIN_IAC_SRC = os.path.join(BASE, 'BrainIAC', 'src', 'BrainIAC')
PREPROCESSING = os.path.join(BRAIN_IAC_SRC, 'preprocessing')
sys.path.insert(0, BRAIN_IAC_SRC)
sys.path.insert(0, PREPROCESSING)

# Import preprocessing functions
from preprocess_single_patient import (
    brain_extraction, 
    create_clip_compatible_output
)
from mri_preprocess_3d_simple import registration

# Import BrainIAC feature extraction functions
from get_brainiac_features import infer
from load_brainiac import load_brainiac
from dataset2 import MedicalImageDatasetBalancedIntensity3D

def custom_collate(batch):
    """Custom collate function for BrainIAC feature extraction"""
    images = [item['image'] for item in batch]
    labels = [item['label'] for item in batch]
    max_len = 1
    padded_images = []
    for img in images:
        pad_size = max_len - img.shape[0]
        if pad_size > 0:
            padding = torch.zeros((pad_size,) + img.shape[1:])
            img_padded = torch.cat([img, padding], dim=0)
            padded_images.append(img_padded)
        else:
            padded_images.append(img)
    return {"image": torch.stack(padded_images, dim=0), "label": torch.stack(labels)}

def extract_patient_id(filepath: str) -> str:
    """Extract patient ID from filepath"""
    filename = os.path.basename(filepath)
    name = filename.split('.')[0]
    
    if 'sub-' in name:
        return name.split('_')[0]
    elif 'patient' in name.lower():
        return name
    else:
        import re
        match = re.search(r'[a-zA-Z0-9_-]+', name)
        if match:
            return match.group()
        else:
            return name

def preprocess_single_mri(input_file: str, output_dir: str, template_path: str, 
                         target_size=(256,256,256)) -> Tuple[bool, str]:
    """Preprocess a single MRI using BrainIAC preprocessing"""
    try:
        print(f"🔧 Preprocessing: {os.path.basename(input_file)}")
        
        # Create patient-specific output directory
        patient_id = extract_patient_id(input_file)
        patient_output_dir = os.path.join(output_dir, patient_id)
        os.makedirs(patient_output_dir, exist_ok=True)
        
        temp_reg_dir = os.path.join(patient_output_dir, 'temp_registered')
        os.makedirs(temp_reg_dir, exist_ok=True)
        
        # Step 1: Registration
        print(f"  📐 Step 1: Registration...")
        success = registration(
            input_dir=os.path.dirname(input_file),
            output_dir=temp_reg_dir,
            temp_img=template_path
        )
        if not success:
            print(f"  ❌ Registration failed for {input_file}")
            return False, ""
        
        # Step 2: Brain extraction
        print(f"  🧠 Step 2: Brain extraction...")
        device = "0" if torch.cuda.is_available() else "cpu"
        brain_extraction(
            input_dir=temp_reg_dir,
            output_dir=patient_output_dir,
            device=device
        )
        
        # Step 3: Create CLIP-compatible output
        print(f"  🎯 Step 3: CLIP-compatible output...")
        brain_extracted_file = os.path.join(patient_output_dir, f"{os.path.basename(input_file).split('.')[0]}_0000.nii.gz")
        if os.path.exists(brain_extracted_file):
            success = create_clip_compatible_output(
                input_file=brain_extracted_file,
                output_dir=patient_output_dir,
                target_size=target_size
            )
            if not success:
                print(f"  ⚠️  CLIP output creation failed for {input_file}")
        
        # Clean up temp directory
        if os.path.exists(temp_reg_dir):
            shutil.rmtree(temp_reg_dir)
        
        print(f"  ✅ Preprocessing complete for {patient_id}")
        return True, brain_extracted_file
        
    except Exception as e:
        print(f"  ❌ Error preprocessing {input_file}: {e}")
        return False, ""

def read_note_text(note_path: str) -> str:
    """Read text from note file"""
    try:
        with open(note_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"⚠️  Error reading note {note_path}: {e}")
        return ""

def create_openai_embeddings(texts: List[str], api_key: str = None, model="text-embedding-3-large") -> np.ndarray:
    """Create OpenAI embeddings for a list of texts using Azure OpenAI via LangChain"""
    import os
    from langchain.embeddings import OpenAIEmbeddings
    import numpy as np

    # Fetch Azure OpenAI environment variables
    api_key = api_key or os.environ.get('API_KEY')
    api_version = os.environ.get('API_VERSION')
    resource_endpoint = os.environ.get('RESOURCE_ENDPOINT')
    deployment = "text-embedding-3-large"  # Use 'large' model as requested

    print(f"🔤 Creating Azure OpenAI embeddings for {len(texts)} texts...")
    
    try:
        embed_fn = OpenAIEmbeddings(
            deployment=deployment,
            openai_api_key=api_key,
            openai_api_base=resource_endpoint,
            openai_api_type="azure",
            openai_api_version=api_version,
        )
        
        # Use batch embedding for all texts
        embeddings = embed_fn.embed_documents(texts)
        print(f"✅ Successfully embedded {len(embeddings)} texts")
        return np.array(embeddings)
        
    except Exception as e:
        print(f"⚠️  Error in batch embedding: {e}")
        # Fallback: return zero embeddings if error
        embedding_dim = 3072 if model == "text-embedding-3-large" else 1536
        print(f"⚠️  Returning zero embeddings with dimension {embedding_dim}")
        return np.zeros((len(texts), embedding_dim))

def extract_brainiac_features_batch(csv_path: str, root_dir: str, checkpoint_path: str) -> pd.DataFrame:
    """Extract BrainIAC features for batch of images"""
    print("🧠 Extracting BrainIAC features for batch...")
    
    dataset = MedicalImageDatasetBalancedIntensity3D(
        csv_path=csv_path,
        root_dir=root_dir
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=custom_collate,
        num_workers=0
    )
    
    device = torch.device('cpu')
    model = load_brainiac(checkpoint_path, device)
    model = model.to(device)
    
    features_df = infer(model, dataloader)
    print(f"✅ Extracted features for {len(features_df)} scan(s)")
    print(f"📊 Feature shape: {features_df.shape}")
    
    return features_df

class UnifiedPPADataset(Dataset):
    """Dataset for PPA MRI-text pairs with BrainIAC features (2048-dim) and OpenAI embeddings"""
    
    def __init__(self, brainiac_features_csv, text_embeddings_csv):
        self.brainiac_df = pd.read_csv(brainiac_features_csv)
        self.text_df = pd.read_csv(text_embeddings_csv)
        
        # Ensure alignment between features and text embeddings
        assert len(self.brainiac_df) == len(self.text_df), "Feature and text counts must match"
        
    def __len__(self):
        return len(self.brainiac_df)

    def __getitem__(self, idx):
        # Get BrainIAC features (2048-dimensional)
        brainiac_row = self.brainiac_df.iloc[idx]
        feature_cols = [col for col in brainiac_row.index if col.startswith('Feature_')]
        brainiac_features = brainiac_row[feature_cols].values.astype(np.float32)
        
        # Get OpenAI text embeddings (3072-dimensional for text-embedding-3-large)
        text_row = self.text_df.iloc[idx]
        embedding_cols = [col for col in text_row.index if col.startswith('embedding_')]
        text_embeddings = text_row[embedding_cols].values.astype(np.float32)
        
        # Get metadata
        patient_id = brainiac_row.get('pat_id', f'patient_{idx}')
        subtype = text_row.get('subtype', 'unknown')
        
        return brainiac_features, text_embeddings, patient_id, subtype

class BrainIACFeatureEncoder(nn.Module):
    """Encoder for BrainIAC features - NO PROJECTION, use 2048-dim features directly"""
    
    def __init__(self):
        super().__init__()
        # No projection needed - BrainIAC features are already 2048-dimensional
        # Just normalize them for CLIP compatibility
        
    def forward(self, x):
        # x is already 2048-dimensional from BrainIAC
        # Just normalize for CLIP compatibility
        return F.normalize(x, dim=-1)

class OpenAIEmbeddingEncoder(nn.Module):
    """Encoder for OpenAI embeddings - project to 2048-dim to match BrainIAC"""
    
    def __init__(self, input_dim=3072, output_dim=2048):  # OpenAI text-embedding-3-large is 3072-dim
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        x = self.proj(x)
        return F.normalize(x, dim=-1)

def clip_loss(image_embeds, text_embeds, temperature=0.07):
    """CLIP contrastive loss function"""
    logits = image_embeds @ text_embeds.T / temperature
    labels = torch.arange(len(image_embeds)).to(image_embeds.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return (loss_i2t + loss_t2i) / 2

def plot_tsne(brainiac_encoder, openai_encoder, dataloader, device, output_dir):
    """Plot t-SNE of the joint latent space (mean of image and text embeddings)"""
    import numpy as np
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    brainiac_encoder.eval()
    openai_encoder.eval()
    all_embeds = []
    all_labels = []
    all_patients = []
    with torch.no_grad():
        for batch in dataloader:
            brainiac_features, openai_embeddings, patient_ids, subtypes = batch
            brainiac_features = brainiac_features.to(device)
            openai_embeddings = openai_embeddings.to(device)
            img_embeds = brainiac_encoder(brainiac_features)
            txt_embeds = openai_encoder(openai_embeddings)
            joint_embeds = (img_embeds + txt_embeds) / 2
            all_embeds.append(joint_embeds.cpu().numpy())
            all_labels.extend(subtypes)
            all_patients.extend(patient_ids)
    all_embeds = np.vstack(all_embeds)

    print("🔍 Running t-SNE on latent space...")
    tsne = TSNE(n_components=2, perplexity=min(30, len(all_embeds)-1), random_state=42)
    tsne_results = tsne.fit_transform(all_embeds)

    plt.figure(figsize=(10, 8))
    unique_labels = list(set(all_labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    for i, label in enumerate(unique_labels):
        mask = [l == label for l in all_labels]
        plt.scatter(tsne_results[mask, 0], tsne_results[mask, 1],
                    s=60, alpha=0.7, c=[colors[i]], label=label)
    plt.title('t-SNE of Joint Latent Space', fontsize=16)
    plt.xlabel('t-SNE 1', fontsize=14)
    plt.ylabel('t-SNE 2', fontsize=14)
    plt.legend(title='Subtype', fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'latent_space_tsne.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ t-SNE plot saved to {save_path}")

def train_clip_model(brainiac_features_csv: str, openai_embeddings_csv: str, 
                    output_dir: str, epochs=30):
    """Train CLIP model with 2048-dimensional BrainIAC features and OpenAI embeddings"""
    print("🧠 Training CLIP model with 2048-dimensional embeddings...")
    
    # Load dataset
    dataset = UnifiedPPADataset(brainiac_features_csv, openai_embeddings_csv)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Using device: {device}")
    
    # Initialize models
    brainiac_encoder = BrainIACFeatureEncoder()  # No projection needed
    openai_encoder = OpenAIEmbeddingEncoder(input_dim=3072, output_dim=2048)  # Project to 2048
    
    brainiac_encoder = brainiac_encoder.to(device)
    openai_encoder = openai_encoder.to(device)
    
    # Optimizer (only for OpenAI projection layer)
    optimizer = torch.optim.AdamW([
        {'params': openai_encoder.parameters(), 'lr': 1e-4},
    ], weight_decay=1e-5)
    
    # Training loop
    brainiac_encoder.train()
    openai_encoder.train()
    
    train_losses = []
    
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            brainiac_features, openai_embeddings, patient_ids, subtypes = batch
            brainiac_features = brainiac_features.to(device)
            openai_embeddings = openai_embeddings.to(device)
            
            # Forward pass
            brainiac_embeds = brainiac_encoder(brainiac_features)  # 2048-dim, normalized
            text_embeds = openai_encoder(openai_embeddings)  # 2048-dim, normalized
            
            # CLIP loss
            loss = clip_loss(brainiac_embeds, text_embeds)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} | Average Loss: {avg_loss:.4f}")
    
    # Save models
    torch.save(openai_encoder.state_dict(), os.path.join(output_dir, 'openai_encoder_2048.pth'))
    
    # Plot training progress
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, 'o-', color='blue', linewidth=2, markersize=6)
    plt.title('CLIP Training Loss (2048-dim)', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'clip_training_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot t-SNE of latent space
    plot_tsne(brainiac_encoder, openai_encoder, dataloader, device, output_dir)
    
    print(f"✅ Training complete! Models saved to {output_dir}")
    print(f"🧠 Final embedding dimensions: 2048 (BrainIAC) + 2048 (OpenAI projected)")
    return brainiac_encoder, openai_encoder, train_losses

def main():
    """Main execution function"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Unified CLIP pipeline: from raw data to trained model')
    parser.add_argument('--input-csv', required=True, 
                       help='CSV file with columns: pat_id, mri_path, note_path, subtype')
    parser.add_argument('--openai-api-key', required=True,
                       help='OpenAI API key')
    parser.add_argument('--output-dir', default='./unified_clip_results',
                       help='Output directory for all results')
    parser.add_argument('--template', default='./BrainIAC/src/BrainIAC/preprocessing/atlases/temp_head.nii.gz',
                       help='Path to template image for registration')
    parser.add_argument('--checkpoint', default='./BrainIAC/src/BrainIAC/checkpoints/BrainIAC (1) (1).ckpt',
                       help='Path to BrainIAC checkpoint')
    parser.add_argument('--target-size', nargs=3, type=int, default=[256,256,256],
                       help='Target size for CLIP-compatible output (default: 256 256 256)')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--max-samples', type=int, default=50,
                       help='Maximum number of samples to process')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.input_csv):
        print(f"❌ Error: Input CSV {args.input_csv} does not exist!")
        return
    
    if not os.path.exists(args.template):
        print(f"❌ Error: Template file {args.template} does not exist!")
        return
    
    if not os.path.exists(args.checkpoint):
        print(f"❌ Error: Checkpoint file {args.checkpoint} does not exist!")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🚀 UNIFIED CLIP PIPELINE: From Raw Data to Trained Model")
    print("=" * 80)
    print(f"📄 Input CSV: {args.input_csv}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"📁 Template: {args.template}")
    print(f"📁 Checkpoint: {args.checkpoint}")
    print(f"📏 Target size: {args.target_size}")
    print(f"📈 Training epochs: {args.epochs}")
    print(f"📊 Max samples: {args.max_samples}")
    print("=" * 80)
    
    try:
        # Step 1: Load and validate input CSV
        print("\n📊 Step 1: Loading input CSV...")
        input_df = pd.read_csv(args.input_csv)
        print(f"📊 Loaded {len(input_df)} samples from CSV")
        
        # Validate required columns
        required_cols = ['pat_id', 'mri_path', 'note_path']
        missing_cols = [col for col in required_cols if col not in input_df.columns]
        if missing_cols:
            print(f"❌ Error: Missing required columns: {missing_cols}")
            return
        
        # Add subtype column if not present
        if 'subtype' not in input_df.columns:
            input_df['subtype'] = 'unknown'
            print("⚠️  'subtype' column not found, using 'unknown' for all samples")
        
        # Limit to max_samples
        if len(input_df) > args.max_samples:
            input_df = input_df.head(args.max_samples)
            print(f"📊 Limited to {len(input_df)} samples")
        
        # Step 2: Preprocess MRIs
        print(f"\n🔧 Step 2: Preprocessing {len(input_df)} MRIs...")
        processed_files = []
        successful_patients = []
        
        for i, row in input_df.iterrows():
            mri_path = row['mri_path']
            patient_id = row['pat_id']
            
            print(f"\n📊 Processing {i+1}/{len(input_df)}: {patient_id}")
            
            if not os.path.exists(mri_path):
                print(f"  ❌ MRI file not found: {mri_path}")
                continue
            
            success, processed_file = preprocess_single_mri(
                input_file=mri_path,
                output_dir=args.output_dir,
                template_path=args.template,
                target_size=tuple(args.target_size)
            )
            
            if success and processed_file:
                processed_files.append(processed_file)
                successful_patients.append(patient_id)
                print(f"  ✅ Success: {patient_id}")
            else:
                print(f"  ❌ Failed: {patient_id}")
        
        print(f"\n📊 Preprocessing complete: {len(processed_files)}/{len(input_df)} successful")
        
        if len(processed_files) == 0:
            print("❌ No files were successfully processed!")
            return
        
        # Step 3: Extract BrainIAC features
        print(f"\n🧠 Step 3: Extracting BrainIAC features...")
        
        # Create BrainIAC input CSV
        brainiac_csv = os.path.join(args.output_dir, "brainiac_input.csv")
        brainiac_data = []
        for file_path in processed_files:
            filename = os.path.basename(file_path)
            patient_id = extract_patient_id(file_path)
            brainiac_data.append({
                'pat_id': patient_id,
                'scandate': '2024-01-01',
                'mri_file': filename,
                'label': 0
            })
        brainiac_df = pd.DataFrame(brainiac_data)
        brainiac_df.to_csv(brainiac_csv, index=False)
        
        # Extract features
        features_df = extract_brainiac_features_batch(
            csv_path=brainiac_csv,
            root_dir=args.output_dir,
            checkpoint_path=args.checkpoint
        )
        
        # Step 4: Read note texts and create OpenAI embeddings
        print(f"\n📝 Step 4: Reading note texts and creating OpenAI embeddings...")
        
        # Filter input_df to only successful patients
        successful_df = input_df[input_df['pat_id'].isin(successful_patients)].copy()
        
        # Read note texts
        note_texts = []
        for _, row in successful_df.iterrows():
            note_text = read_note_text(row['note_path'])
            note_texts.append(note_text)
        
        # Create OpenAI embeddings
        embeddings = create_openai_embeddings(note_texts, args.openai_api_key)
        
        # Create embeddings DataFrame
        embedding_cols = [f'embedding_{i}' for i in range(embeddings.shape[1])]
        embeddings_df = pd.DataFrame(embeddings, columns=embedding_cols)
        embeddings_df['pat_id'] = successful_df['pat_id'].values
        embeddings_df['subtype'] = successful_df['subtype'].values
        
        # Step 5: Save prepared data
        print(f"\n💾 Step 5: Saving prepared data...")
        
        # Save BrainIAC features
        brainiac_features_csv = os.path.join(args.output_dir, 'brainiac_features_2048.csv')
        features_df.to_csv(brainiac_features_csv, index=False)
        
        # Save OpenAI embeddings
        openai_embeddings_csv = os.path.join(args.output_dir, 'openai_embeddings_3072.csv')
        embeddings_df.to_csv(openai_embeddings_csv, index=False)
        
        print(f"✅ BrainIAC features saved to: {brainiac_features_csv}")
        print(f"✅ OpenAI embeddings saved to: {openai_embeddings_csv}")
        
        # Step 6: Train CLIP model
        print(f"\n🎯 Step 6: Training CLIP model...")
        brainiac_encoder, openai_encoder, train_losses = train_clip_model(
            brainiac_features_csv=brainiac_features_csv,
            openai_embeddings_csv=openai_embeddings_csv,
            output_dir=args.output_dir,
            epochs=args.epochs
        )
        
        print("\n" + "=" * 80)
        print("✅ UNIFIED CLIP PIPELINE COMPLETE!")
        print("=" * 80)
        print(f"📊 Total samples processed: {len(input_df)}")
        print(f"📊 Successful preprocessing: {len(processed_files)}")
        print(f"📊 Final training samples: {len(features_df)}")
        print(f"📁 All results saved in: {args.output_dir}")
        print(f"🧠 Trained models saved in: {args.output_dir}")
        print(f"📊 t-SNE visualization: {os.path.join(args.output_dir, 'latent_space_tsne.png')}")
        
    except Exception as e:
        print(f"❌ Error in unified pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 
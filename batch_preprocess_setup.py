#!/usr/bin/env python3
"""
Batch Preprocessing and Data Setup for CLIP Model Training

This script batch processes multiple MRI scans using functions from single_image_pipeline.py,
extracts BrainIAC features, and prepares the data for CLIP model training with OpenAI embeddings.

Usage:
    python batch_preprocess_setup.py --input-dir ./raw_mris --notes-csv ./clinical_notes.csv --output-dir ./processed_data
"""

import sys
import os
import pandas as pd
import torch
import argparse
from pathlib import Path
import glob
import shutil
from typing import List, Dict, Tuple
import time
from datetime import datetime

# Add BrainIAC and preprocessing to path
BASE = os.path.dirname(os.path.abspath(__file__))
BRAIN_IAC_SRC = os.path.join(BASE, 'BrainIAC', 'src', 'BrainIAC')
PREPROCESSING = os.path.join(BRAIN_IAC_SRC, 'preprocessing')
sys.path.insert(0, BRAIN_IAC_SRC)
sys.path.insert(0, PREPROCESSING)

# Import functions from single_image_pipeline.py
from preprocess_single_patient import (
    brain_extraction, 
    create_clip_compatible_output
)
from mri_preprocess_3d_simple import registration

# Import BrainIAC feature extraction functions
from get_brainiac_features import infer
from load_brainiac import load_brainiac
from dataset2 import MedicalImageDatasetBalancedIntensity3D
from torch.utils.data import DataLoader

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

def find_mri_files(input_dir: str, extensions: List[str] = ['.nii.gz', '.nii']) -> List[str]:
    """
    Find all MRI files in the input directory
    
    Args:
        input_dir: Directory containing MRI files
        extensions: List of file extensions to search for
    
    Returns:
        List of paths to MRI files
    """
    mri_files = []
    for ext in extensions:
        pattern = os.path.join(input_dir, f"*{ext}")
        mri_files.extend(glob.glob(pattern))
    
    # Also search in subdirectories
    for root, dirs, files in os.walk(input_dir):
        for ext in extensions:
            pattern = os.path.join(root, f"*{ext}")
            mri_files.extend(glob.glob(pattern))
    
    return sorted(list(set(mri_files)))  # Remove duplicates

def extract_patient_id(filepath: str) -> str:
    """
    Extract patient ID from MRI filename
    
    Args:
        filepath: Path to MRI file
    
    Returns:
        Patient ID string
    """
    filename = os.path.basename(filepath)
    # Remove extension
    name = filename.split('.')[0]
    
    # Common patterns for patient IDs
    if 'sub-' in name:
        # BIDS format: sub-001_T1w
        return name.split('_')[0]
    elif 'patient' in name.lower():
        # patient_001 format
        return name
    else:
        # Try to extract any alphanumeric ID
        import re
        match = re.search(r'[a-zA-Z0-9_-]+', name)
        if match:
            return match.group()
        else:
            return name

def preprocess_single_mri(input_file: str, output_dir: str, template_path: str, 
                         target_size=(256,256,256)) -> Tuple[bool, str]:
    """
    Preprocess a single MRI using functions from single_image_pipeline.py
    
    Args:
        input_file: Path to input MRI file
        output_dir: Output directory for processed files
        template_path: Path to template image for registration
        target_size: Target size for CLIP-compatible output
    
    Returns:
        Tuple of (success, processed_file_path)
    """
    try:
        print(f"🔧 Preprocessing: {os.path.basename(input_file)}")
        
        # Create patient-specific output directory
        patient_id = extract_patient_id(input_file)
        patient_output_dir = os.path.join(output_dir, patient_id)
        os.makedirs(patient_output_dir, exist_ok=True)
        
        temp_reg_dir = os.path.join(patient_output_dir, 'temp_registered')
        os.makedirs(temp_reg_dir, exist_ok=True)
        
        # Step 1: Rigorous Registration (with N4 bias correction)
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

def create_brainiac_csv_for_batch(processed_files: List[str], output_csv: str) -> str:
    """
    Create BrainIAC CSV for batch processing
    
    Args:
        processed_files: List of paths to processed MRI files
        output_csv: Path to save the CSV
    
    Returns:
        Path to created CSV
    """
    print("📋 Creating BrainIAC CSV for batch processing...")
    
    data = []
    for i, file_path in enumerate(processed_files):
        filename = os.path.basename(file_path)
        patient_id = extract_patient_id(file_path)
        
        data.append({
            'pat_id': patient_id,
            'scandate': '2024-01-01',  # Default date
            'mri_file': filename,
            'label': 0  # Default label
        })
    
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"✅ Created batch CSV: {output_csv}")
    return output_csv

def extract_brainiac_features_batch(csv_path: str, root_dir: str, checkpoint_path: str) -> pd.DataFrame:
    """
    Extract BrainIAC features for batch of images
    
    Args:
        csv_path: Path to CSV with image paths
        root_dir: Root directory containing processed images
        checkpoint_path: Path to BrainIAC checkpoint
    
    Returns:
        DataFrame with extracted features
    """
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

def match_patients_with_notes(patient_ids: List[str], notes_csv: str) -> pd.DataFrame:
    """
    Match processed patients with clinical notes
    
    Args:
        patient_ids: List of patient IDs from processed MRIs
        notes_csv: Path to CSV with clinical notes
    
    Returns:
        DataFrame with matched patient-notes pairs
    """
    print("🔗 Matching patients with clinical notes...")
    
    # Load clinical notes
    notes_df = pd.read_csv(notes_csv)
    print(f"📝 Loaded {len(notes_df)} clinical notes")
    
    # Ensure patient_id column exists
    if 'pat_id' not in notes_df.columns:
        # Try to infer patient ID from other columns
        possible_id_cols = ['patient_id', 'patient', 'id', 'subject']
        for col in possible_id_cols:
            if col in notes_df.columns:
                notes_df['pat_id'] = notes_df[col].astype(str)
                break
        else:
            print("⚠️  No patient ID column found in notes CSV")
            return pd.DataFrame()
    
    # Convert to string for matching
    notes_df['pat_id'] = notes_df['pat_id'].astype(str)
    patient_ids = [str(pid) for pid in patient_ids]
    
    # Match patients
    matched_df = notes_df[notes_df['pat_id'].isin(patient_ids)].copy()
    
    print(f"✅ Matched {len(matched_df)} patients with notes")
    print(f"📊 Available patients: {len(patient_ids)}")
    print(f"📊 Matched patients: {len(matched_df)}")
    
    return matched_df

def create_final_training_csv(brainiac_features_df: pd.DataFrame, 
                             matched_notes_df: pd.DataFrame,
                             output_csv: str) -> str:
    """
    Create final CSV for CLIP training with BrainIAC features and clinical notes
    
    Args:
        brainiac_features_df: DataFrame with BrainIAC features
        matched_notes_df: DataFrame with matched clinical notes
        output_csv: Path to save final CSV
    
    Returns:
        Path to created CSV
    """
    print("📊 Creating final training CSV...")
    
    # Merge features with notes
    brainiac_features_df['pat_id'] = brainiac_features_df['pat_id'].astype(str)
    matched_notes_df['pat_id'] = matched_notes_df['pat_id'].astype(str)
    
    final_df = brainiac_features_df.merge(matched_notes_df, on='pat_id', how='inner')
    
    # Ensure required columns exist
    if 'note' not in final_df.columns:
        # Try to find text column
        text_cols = ['note', 'text', 'description', 'clinical_note', 'report']
        for col in text_cols:
            if col in final_df.columns:
                final_df['note'] = final_df[col]
                break
        else:
            print("⚠️  No text column found for clinical notes")
            return ""
    
    if 'subtype' not in final_df.columns:
        final_df['subtype'] = 'unknown'
    
    # Save final CSV
    final_df.to_csv(output_csv, index=False)
    print(f"✅ Final training CSV created: {output_csv}")
    print(f"📊 Final dataset size: {len(final_df)} samples")
    
    return output_csv

def main():
    parser = argparse.ArgumentParser(description='Batch preprocess MRIs and prepare data for CLIP training')
    parser.add_argument('--input-dir', required=True, help='Directory containing raw MRI files')
    parser.add_argument('--notes-csv', required=True, help='CSV file with clinical notes')
    parser.add_argument('--output-dir', required=True, help='Output directory for processed data')
    parser.add_argument('--template', default='./BrainIAC/src/BrainIAC/preprocessing/atlases/temp_head.nii.gz', 
                       help='Path to template image for registration')
    parser.add_argument('--checkpoint', default='./BrainIAC/src/BrainIAC/checkpoints/BrainIAC (1) (1).ckpt',
                       help='Path to BrainIAC checkpoint')
    parser.add_argument('--target-size', nargs=3, type=int, default=[256,256,256],
                       help='Target size for CLIP-compatible output (default: 256 256 256)')
    parser.add_argument('--max-samples', type=int, default=50,
                       help='Maximum number of samples to process (default: 50)')
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.input_dir):
        print(f"❌ Error: Input directory {args.input_dir} does not exist!")
        return
    
    if not os.path.exists(args.notes_csv):
        print(f"❌ Error: Notes CSV {args.notes_csv} does not exist!")
        return
    
    if not os.path.exists(args.template):
        print(f"❌ Error: Template file {args.template} does not exist!")
        return
    
    if not os.path.exists(args.checkpoint):
        print(f"❌ Error: Checkpoint file {args.checkpoint} does not exist!")
        return
    
    print("🚀 Starting Batch Preprocessing Pipeline...")
    print(f"📁 Input directory: {args.input_dir}")
    print(f"📁 Notes CSV: {args.notes_csv}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"📁 Template: {args.template}")
    print(f"📁 Checkpoint: {args.checkpoint}")
    print(f"📏 Target size: {args.target_size}")
    print(f"📊 Max samples: {args.max_samples}")
    print("=" * 80)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Step 1: Find MRI files
        print("\n🔍 Step 1: Finding MRI files...")
        mri_files = find_mri_files(args.input_dir)
        print(f"📊 Found {len(mri_files)} MRI files")
        
        if len(mri_files) == 0:
            print("❌ No MRI files found!")
            return
        
        # Limit to max_samples
        if len(mri_files) > args.max_samples:
            mri_files = mri_files[:args.max_samples]
            print(f"📊 Limited to {len(mri_files)} samples")
        
        # Step 2: Preprocess each MRI
        print(f"\n🔧 Step 2: Preprocessing {len(mri_files)} MRIs...")
        processed_files = []
        successful_patients = []
        
        for i, mri_file in enumerate(mri_files):
            print(f"\n📊 Processing {i+1}/{len(mri_files)}: {os.path.basename(mri_file)}")
            
            success, processed_file = preprocess_single_mri(
                input_file=mri_file,
                output_dir=args.output_dir,
                template_path=args.template,
                target_size=tuple(args.target_size)
            )
            
            if success and processed_file:
                processed_files.append(processed_file)
                patient_id = extract_patient_id(mri_file)
                successful_patients.append(patient_id)
                print(f"  ✅ Success: {patient_id}")
            else:
                print(f"  ❌ Failed: {os.path.basename(mri_file)}")
        
        print(f"\n📊 Preprocessing complete: {len(processed_files)}/{len(mri_files)} successful")
        
        if len(processed_files) == 0:
            print("❌ No files were successfully processed!")
            return
        
        # Step 3: Extract BrainIAC features
        print(f"\n🧠 Step 3: Extracting BrainIAC features...")
        brainiac_csv = os.path.join(args.output_dir, "brainiac_batch_input.csv")
        create_brainiac_csv_for_batch(processed_files, brainiac_csv)
        
        features_df = extract_brainiac_features_batch(
            csv_path=brainiac_csv,
            root_dir=args.output_dir,
            checkpoint_path=args.checkpoint
        )
        
        # Save BrainIAC features
        features_output_path = os.path.join(args.output_dir, "brainiac_features_batch.csv")
        features_df.to_csv(features_output_path, index=False)
        print(f"✅ BrainIAC features saved to: {features_output_path}")
        
        # Step 4: Match with clinical notes
        print(f"\n🔗 Step 4: Matching with clinical notes...")
        matched_notes_df = match_patients_with_notes(successful_patients, args.notes_csv)
        
        if len(matched_notes_df) == 0:
            print("❌ No patients matched with clinical notes!")
            return
        
        # Step 5: Create final training CSV
        print(f"\n📊 Step 5: Creating final training CSV...")
        final_csv = os.path.join(args.output_dir, "clip_training_data.csv")
        create_final_training_csv(features_df, matched_notes_df, final_csv)
        
        print("\n" + "=" * 80)
        print("✅ BATCH PREPROCESSING PIPELINE COMPLETE!")
        print("=" * 80)
        print(f"📊 Total MRIs processed: {len(processed_files)}")
        print(f"📊 Patients with features: {len(features_df)}")
        print(f"📊 Patients with notes: {len(matched_notes_df)}")
        print(f"📊 Final training samples: {len(pd.read_csv(final_csv))}")
        print(f"📁 All results saved in: {args.output_dir}")
        print(f"📄 Final training CSV: {final_csv}")
        print("\n🎯 Next step: Run clip_pipeline_brainiac_openai.py with the generated CSV!")
        
    except Exception as e:
        print(f"❌ Error in batch preprocessing pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 
#!/usr/bin/env python3
"""
Create Synthetic Data for Testing Unified CLIP Pipeline

This script creates synthetic MRI files and clinical notes for testing the pipeline.
"""

import os
import pandas as pd
import numpy as np
import nibabel as nib
from pathlib import Path
import argparse

def create_synthetic_mri(output_path: str, shape=(128, 128, 128), noise_level=0.1):
    """
    Create a synthetic MRI file
    
    Args:
        output_path: Path to save the synthetic MRI
        shape: 3D shape of the MRI
        noise_level: Amount of noise to add
    """
    # Create a simple brain-like structure
    x, y, z = np.meshgrid(np.linspace(-1, 1, shape[0]),
                         np.linspace(-1, 1, shape[1]),
                         np.linspace(-1, 1, shape[2]))
    
    # Create ellipsoid (brain-like shape)
    brain_mask = (x**2 + y**2 + z**2) < 0.8
    
    # Add some internal structure (gray matter, white matter)
    gray_matter = np.exp(-((x-0.2)**2 + (y-0.1)**2 + (z-0.1)**2) / 0.3)
    white_matter = np.exp(-((x+0.1)**2 + (y-0.2)**2 + (z+0.1)**2) / 0.2)
    
    # Combine structures
    mri_data = brain_mask * (0.3 * gray_matter + 0.5 * white_matter)
    
    # Add noise
    noise = np.random.normal(0, noise_level, shape)
    mri_data += noise
    
    # Ensure positive values
    mri_data = np.maximum(mri_data, 0)
    
    # Create NIfTI image
    affine = np.array([[1, 0, 0, -64],
                      [0, 1, 0, -64],
                      [0, 0, 1, -64],
                      [0, 0, 0, 1]])
    
    nii_img = nib.Nifti1Image(mri_data, affine)
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nib.save(nii_img, output_path)
    print(f"✅ Created synthetic MRI: {output_path}")

def create_synthetic_note(output_path: str, patient_id: str, subtype: str):
    """
    Create a synthetic clinical note
    
    Args:
        output_path: Path to save the note
        patient_id: Patient ID
        subtype: PPA subtype
    """
    # Template clinical notes for different subtypes
    note_templates = {
        'lvPPA': f"""CLINICAL ASSESSMENT - {patient_id}

Patient presents with primary progressive aphasia, logopenic variant (lvPPA).
Key findings:
- Word-finding difficulties prominent
- Impaired repetition of sentences
- Phonological errors in speech
- Relatively preserved grammar
- Mild memory impairment

Neurological examination shows no focal deficits.
MRI reveals left temporal-parietal atrophy consistent with lvPPA.

Assessment: Logopenic variant of primary progressive aphasia
Plan: Speech therapy, cognitive assessment, follow-up in 6 months""",

        'nfvPPA': f"""CLINICAL ASSESSMENT - {patient_id}

Patient presents with primary progressive aphasia, non-fluent variant (nfvPPA).
Key findings:
- Agrammatic speech production
- Effortful, halting speech
- Preserved word comprehension
- Apraxia of speech may be present
- Executive function deficits

Neurological examination shows no focal motor deficits.
MRI reveals left frontal atrophy consistent with nfvPPA.

Assessment: Non-fluent variant of primary progressive aphasia
Plan: Speech therapy, motor speech assessment, follow-up in 6 months""",

        'svPPA': f"""CLINICAL ASSESSMENT - {patient_id}

Patient presents with primary progressive aphasia, semantic variant (svPPA).
Key findings:
- Impaired word comprehension
- Surface dyslexia/dysgraphia
- Anomia for objects
- Preserved grammar and syntax
- Behavioral changes may be present

Neurological examination shows no focal deficits.
MRI reveals anterior temporal atrophy consistent with svPPA.

Assessment: Semantic variant of primary progressive aphasia
Plan: Speech therapy, behavioral assessment, follow-up in 6 months""",

        'unknown': f"""CLINICAL ASSESSMENT - {patient_id}

Patient presents with cognitive complaints.
Key findings:
- Mild memory difficulties
- Word-finding problems
- No clear aphasia pattern
- Normal neurological examination

MRI shows mild atrophy, pattern unclear.
Further assessment needed to determine subtype.

Assessment: Cognitive impairment, subtype unclear
Plan: Comprehensive neuropsychological testing, follow-up in 3 months"""
    }
    
    # Get template for subtype
    template = note_templates.get(subtype, note_templates['unknown'])
    
    # Save note
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(template)
    
    print(f"✅ Created synthetic note: {output_path}")

def create_synthetic_dataset(output_dir: str, num_samples: int = 10):
    """
    Create a complete synthetic dataset
    
    Args:
        output_dir: Directory to save synthetic data
        num_samples: Number of samples to create
    """
    print(f"🎯 Creating synthetic dataset with {num_samples} samples...")
    
    # Create directories
    mri_dir = os.path.join(output_dir, "raw_mris")
    notes_dir = os.path.join(output_dir, "clinical_notes")
    os.makedirs(mri_dir, exist_ok=True)
    os.makedirs(notes_dir, exist_ok=True)
    
    # Define subtypes
    subtypes = ['lvPPA', 'nfvPPA', 'svPPA', 'unknown']
    
    # Create data
    data = []
    for i in range(num_samples):
        patient_id = f"sub-{i+1:03d}"
        subtype = subtypes[i % len(subtypes)]
        
        # Create MRI file
        mri_path = os.path.join(mri_dir, f"{patient_id}_T1w.nii.gz")
        create_synthetic_mri(mri_path)
        
        # Create note file
        note_path = os.path.join(notes_dir, f"{patient_id}_note.txt")
        create_synthetic_note(note_path, patient_id, subtype)
        
        # Add to data
        data.append({
            'pat_id': patient_id,
            'mri_path': mri_path,
            'note_path': note_path,
            'subtype': subtype
        })
    
    # Create CSV
    csv_path = os.path.join(output_dir, "synthetic_data.csv")
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    
    print(f"✅ Created synthetic dataset:")
    print(f"📁 MRI files: {mri_dir}")
    print(f"📁 Note files: {notes_dir}")
    print(f"📄 CSV file: {csv_path}")
    print(f"📊 Sample distribution:")
    print(df['subtype'].value_counts())
    
    return csv_path

def main():
    parser = argparse.ArgumentParser(description='Create synthetic data for testing CLIP pipeline')
    parser.add_argument('--output-dir', default='./synthetic_data',
                       help='Output directory for synthetic data')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Number of synthetic samples to create')
    
    args = parser.parse_args()
    
    print("🎭 Creating Synthetic Data for CLIP Pipeline Testing")
    print("=" * 60)
    print(f"📁 Output directory: {args.output_dir}")
    print(f"📊 Number of samples: {args.num_samples}")
    print("=" * 60)
    
    # Create synthetic dataset
    csv_path = create_synthetic_dataset(args.output_dir, args.num_samples)
    
    print("\n" + "=" * 60)
    print("✅ SYNTHETIC DATA CREATION COMPLETE!")
    print("=" * 60)
    print(f"📄 CSV file: {csv_path}")
    print("\n🎯 Next step: Test the unified CLIP pipeline with:")
    print(f"python unified_clip_pipeline.py --input-csv {csv_path} --openai-api-key YOUR_KEY")

if __name__ == '__main__':
    main() 
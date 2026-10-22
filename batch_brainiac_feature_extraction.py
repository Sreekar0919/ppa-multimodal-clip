import torch
import pandas as pd
import os
from torch.utils.data import DataLoader
from BrainIAC.src.BrainIAC.load_brainiac import load_brainiac
from BrainIAC.src.BrainIAC.dataset2 import MedicalImageDatasetBalancedIntensity3D

def custom_collate(batch):
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

def extract_brainiac_features_single(processed_file, checkpoint_path, patient_id, output_csv):
    # Create temporary CSV for BrainIAC input
    temp_csv = f"temp_{patient_id}.csv"
    brainiac_data = [{
        'pat_id': patient_id,
        'scandate': '2024-01-01',
        'mri_file': os.path.basename(processed_file),
        'label': 0
    }]
    pd.DataFrame(brainiac_data).to_csv(temp_csv, index=False)

    # Create dataset and dataloader
    dataset = MedicalImageDatasetBalancedIntensity3D(
        csv_path=temp_csv,
        root_dir=os.path.dirname(processed_file)
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=custom_collate,
        num_workers=0
    )

    # Load model and extract features
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_brainiac(checkpoint_path, device)
    model = model.to(device)
    model.eval()

    from BrainIAC.src.BrainIAC.get_brainiac_features import infer
    features_df = infer(model, dataloader)
    features_df.to_csv(output_csv, index=False)
    os.remove(temp_csv)
    print(f"Features saved to {output_csv}")

if __name__ == "__main__":
    # Paths
    pid_file = os.path.join("minimal_clip_setup", "pids.txt")
    input_dir = os.path.join("clip_exploration", "BrainIAC", "src", "BrainIAC", "data", "data")
    checkpoint_path = os.path.join("clip_exploration", "BrainIAC", "src", "BrainIAC", "checkpoints", "BrainIAC (1) (1).ckpt")
    output_dir = os.path.join("clip_exploration", "BrainIAC", "src", "BrainIAC", "features")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read patient IDs
    with open(pid_file, 'r') as f:
        pid_list = [line.strip() for line in f if line.strip()]

    for pid in pid_list:
        processed_file = os.path.join(input_dir, f"{pid}_T1w.nii.gz")
        output_csv = os.path.join(output_dir, f"{pid}_brainiac_features.csv")
        if not os.path.exists(processed_file):
            print(f"Input file not found for {pid}: {processed_file}")
            continue
        extract_brainiac_features_single(processed_file, checkpoint_path, pid, output_csv) 
import os
import torch
import open_clip
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

# Load the OpenCLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')
tokenizer = open_clip.get_tokenizer('ViT-B-16')
model = model.to(device).eval()

def extract_frames(video_path):
    """Extract all frames from the video."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames

def process_frames(frames):
    """Preprocess and batch frames for CLIP"""
    if not frames:
        return None
    processed = [preprocess(Image.fromarray(frame)) for frame in frames]
    return torch.stack(processed).to(device)

def compute_video_score(frames, caption, batch_size=32):
    """Compute average CLIP similarity score for all frames in a video"""
    text = tokenizer([caption]).to(device)
    
    with torch.no_grad(), torch.cuda.amp.autocast():
        # Encode text once
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # Process frames in batches to save memory
        frame_scores = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i+batch_size]
            
            # Encode image batch
            image_features = model.encode_image(batch)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # Compute similarity scores (cosine similarity)
            batch_scores = (image_features @ text_features.T).squeeze(-1)
            frame_scores.append(batch_scores.cpu())
            
        return torch.cat(frame_scores).mean().item()

def compute_folder_score(video_folder):
    """Compute average CLIP score for all videos in a folder"""
    video_scores = []
    
    for filename in tqdm(os.listdir(video_folder), desc=f"Processing {os.path.basename(video_folder)}"):
        if not filename.endswith((".mp4", ".avi", ".mov")):
            continue
        
        video_path = os.path.join(video_folder, filename)
        caption = os.path.splitext(filename)[0]
        
        # Extract and process frames
        frames = extract_frames(video_path)
        if not frames:
            continue
            
        frame_tensors = process_frames(frames)
        if frame_tensors is None:
            continue
            
        # Compute video score
        video_score = compute_video_score(frame_tensors, caption)
        video_scores.append(video_score)
    
    return np.mean(video_scores) if video_scores else None

def main(folders):
    """Compute CLIP scores for specified folders with progress tracking"""
    folder_scores = {}
    
    for folder in folders:
        if not os.path.isdir(folder):
            print(f"Skipping {folder}: Not a valid directory")
            continue
            
        avg_score = compute_folder_score(folder)
        if avg_score is not None:
            folder_scores[folder] = avg_score
            print(f"Folder: {os.path.basename(folder):<40} | CLIP Score: {avg_score:.4f}")
    
    return folder_scores

if __name__ == "__main__":
    video_folders = [
        "/home/ubuntu/Smp-MCM/visual-results/inf-results-teacher-50ddim",
        "/home/ubuntu/Smp-MCM/visual-results/inf-results-seed-50p-1step",
        "/home/ubuntu/Smp-MCM/visual-results/inf-results-seed-50p-2step",
        "/home/ubuntu/Smp-MCM/visual-results/inf-results-seed-50p-4step",
        "/home/ubuntu/Smp-MCM/visual-results/inf-results-seed-50p-8step",
        "/home/ubuntu/Smp-MCM/visual-results/inf-results-seed-50p-1step-prev-conf",
        "/home/ubuntu/Smp-MCM/visual-results/inf-results-seed-50p-2step-prev-conf",
        "/home/ubuntu/Smp-MCM/visual-results/inf-results-seed-50p-4step-prev-conf",
        "/home/ubuntu/Smp-MCM/visual-results/inf-results-seed-50p-8step-prev-conf"
    ]
    
    scores = main(video_folders)
    print("\nFinal Results:")
    for folder, score in scores.items():
        print(f"{os.path.basename(folder):<40}: {score:.4f}")

import cv2
import torch
import numpy as np
import glob
import os
import logging
from moviepy import *
from calculate_fvd import calculate_fvd

# Suppress logs from external libraries
logging.getLogger("imageio_ffmpeg").setLevel(logging.ERROR)
os.environ["IMAGEIO_FFMPEG_LOGLEVEL"] = "error"

def resize_with_padding(image, target_size=(256, 256)):
    """ Resizes an image while maintaining aspect ratio, adding padding if needed. """
    h, w = image.shape[:2]
    target_w, target_h = target_size  # Fix: (width, height) interpretation
    
    # Calculate scaling factor
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Calculate padding (now for width and height correctly)
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    
    padded_image = cv2.copyMakeBorder(resized_image, 
                                     pad_top, pad_bottom, 
                                     pad_left, pad_right,
                                     borderType=cv2.BORDER_CONSTANT, 
                                     value=[0, 0, 0])
    return padded_image

def extract_frames_uniform(video_path, num_frames=16, target_size=(256, 256)):
    """ Extracts 16 uniformly spaced frames from a video """
    clip = VideoFileClip(video_path)
    total_frames = int(clip.fps * clip.duration)

    if total_frames < num_frames:
        raise ValueError(f"Video {video_path} has fewer than {num_frames} frames.")

    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []

    for idx in frame_indices:
        frame = clip.get_frame(idx / clip.fps)
        frame = resize_with_padding(frame, target_size)
        frame = frame / 255.0  # Normalize to [0,1]
        frames.append(frame)

    clip.close()
    return np.array(frames)  # Shape: (16, H, W, 3)

def load_videos(video_folder, num_videos=10, target_size=(256, 256)):
    """ Loads multiple videos and converts to tensor format """
    video_paths = sorted(glob.glob(video_folder + "/*.mp4"))[:num_videos]
    videos = [extract_frames_uniform(vid, target_size=target_size) for vid in video_paths]

    videos = np.array(videos)  # Shape: (N, 16, H, W, 3)
    # Correct transpose order: (N, T, C, H, W)
    videos = np.transpose(videos, (0, 1, 4, 2, 3))  
    return torch.tensor(videos, dtype=torch.float32)


real_video_folder = "/home/ubuntu/Smp-MCM/results-50p/test-videos"


generated_video_folders = [
    "/home/ubuntu/Smp-MCM/results-50p/inf-results-teacher-50ddim",
    "/home/ubuntu/Smp-MCM/results-50p/inf-results-seed-50p-1step",
    "/home/ubuntu/Smp-MCM/results-50p/inf-results-seed-50p-2step",
    "/home/ubuntu/Smp-MCM/results-50p/inf-results-seed-50p-4step",
    "/home/ubuntu/Smp-MCM/results-50p/inf-results-seed-50p-8step",
    "/home/ubuntu/Smp-MCM/results-50p/inf-results-seed-50p-1step-prev-conf",
    "/home/ubuntu/Smp-MCM/results-50p/inf-results-seed-50p-2step-prev-conf",
    "/home/ubuntu/Smp-MCM/results-50p/inf-results-seed-50p-4step-prev-conf",
    "/home/ubuntu/Smp-MCM/results-50p/inf-results-seed-50p-8step-prev-conf",
]

# Load real videos
num_videos = 50
real_videos = load_videos(real_video_folder, num_videos=num_videos)

# Compute FVD for each generated set
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Open a file to write FVD scores
with open("fvd_scores.txt", "w") as file:
    for gen_folder in generated_video_folders:
        generated_videos = load_videos(gen_folder, num_videos=num_videos)
        fvd_score = calculate_fvd(real_videos, generated_videos, device, method='styleganv', only_final=True)
        file.write(f"FVD Score for {gen_folder}: {fvd_score}\n")

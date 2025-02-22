import cv2
import torch
import numpy as np
import glob
from moviepy import *
from calculate_fvd import calculate_fvd

def resize_with_padding(image, target_size=(256, 256)):
    """ Resizes an image while maintaining aspect ratio, adding padding if needed. """
    h, w = image.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    new_w, new_h = int(w * scale), int(h * scale)

    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_top = (target_size[1] - new_h) // 2
    pad_bottom = target_size[1] - new_h - pad_top
    pad_left = (target_size[0] - new_w) // 2
    pad_right = target_size[0] - new_w - pad_left

    padded_image = cv2.copyMakeBorder(resized_image, pad_top, pad_bottom, pad_left, pad_right,
                                      borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])  # Black padding
    return padded_image

def extract_frames_uniform(video_path, num_frames=16, target_size=(256, 256)):
    """ Extracts 16 uniformly spaced frames from a video, resizes while keeping aspect ratio. """
    clip = VideoFileClip(video_path)
    total_frames = int(clip.fps * clip.duration)

    if total_frames < num_frames:
        raise ValueError(f"Video {video_path} has fewer than {num_frames} frames.")

    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []

    for idx in frame_indices:
        frame = clip.get_frame(idx / clip.fps)  # Get frame at specific time
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR (for OpenCV)
        frame = resize_with_padding(frame, target_size)  # Resize with padding
        frame = frame / 255.0  # Normalize to [0,1]
        frames.append(frame)

    clip.close()
    return np.array(frames)  # Shape: (16, H, W, 3)

def load_videos(video_folder, num_videos=10, target_size=(256, 256)):
    """ Loads multiple videos, extracts frames, resizes, and converts to tensor format. """
    video_paths = sorted(glob.glob(video_folder + "/*.mp4"))[:num_videos]  # Sort to maintain order
    videos = [extract_frames_uniform(vid, target_size=target_size) for vid in video_paths]

    videos = np.array(videos)  # Shape: (N, 16, H, W, 3)
    videos = np.transpose(videos, (0, 1, 4, 2, 3))  # Convert to (N, 16, 3, H, W)
    return torch.tensor(videos, dtype=torch.float32)


real_video_folder = "/home/ubuntu/Smp-MCM/test-videos"


generated_video_folders = [
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

# Load real videos
num_videos = 10  # Adjust based on dataset size
real_videos = load_videos(real_video_folder, num_videos=num_videos)

# Compute FVD for each generated set
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for gen_folder in generated_video_folders:
    generated_videos = load_videos(gen_folder, num_videos=num_videos)

    fvd_score = calculate_fvd(real_videos, generated_videos, device, method='videogpt', only_final=True)
    
    print(f"FVD Score for {gen_folder}: {fvd_score}")

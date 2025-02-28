import os
import torch
from torch.utils.data import Dataset
import torchvision.io as tvio
import torch.nn.functional as F
from torch.utils.data import DataLoader
from videojedi import JEDiMetric

class VideoDataset(Dataset):
    def __init__(self, video_dir, num_frames=16, resize=(224, 224)):
        """
        Args:
            video_dir (str): Directory containing video files.
            num_frames (int): Number of frames to sample from each video.
            resize (tuple): Target (height, width) to resize frames.
        """
        self.video_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith('.mp4')]
        self.num_frames = num_frames
        self.resize = resize

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        # Load video (T, H, W, C) [0-255]
        video, _, _ = tvio.read_video(self.video_paths[idx], pts_unit='sec')
        
        # Sample or pad frames
        T = video.size(0)
        if T >= self.num_frames:
            indices = torch.linspace(0, T-1, self.num_frames).long()
            video = video[indices]
        else:
            padding = torch.zeros((self.num_frames - T, *video.shape[1:]), dtype=video.dtype)
            video = torch.cat([video, padding], dim=0)
        
        # Convert to [0, 1] and (T, C, H, W)
        video = video.float() / 255.0
        video = video.permute(0, 3, 1, 2)
        
        # Resize if needed
        if self.resize:
            _, _, H, W = video.shape
            start_y = (H - 224) // 2
            start_x = (W - 224) // 2
            video = video[:, :, start_y:start_y+224, start_x:start_x+224]
        
        return video


def custom_collate(batch):
    # Stack videos and ignore labels (compatible with JEDi's API)
    return torch.utils.data.dataloader.default_collate(batch), None

# Parameters
num_frames = 16  # Match your model's expected input
resize = (224, 224)
batch_size = 20

# Train and Test Datasets
train_folder = "C:\\Users\\Lenovo\\Downloads\\whole-visual-results-20250224T124848Z-001\\whole-visual-results\\inf-results-seed-50p-4step-prev-conf"
train_dataset = VideoDataset(train_folder, num_frames, resize)
test_dataset = VideoDataset("C:\\Users\\Lenovo\\Downloads\\test-videos", num_frames, resize)

# DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=custom_collate
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=custom_collate
)

print("Working on:", train_folder)

jedi = JEDiMetric(feature_path='./features', model_dir='./jepa-model')
jedi.load_features(
    train_loader=train_loader,
    test_loader=test_loader,
    num_samples=50
)
print(f"JEDi Metric: {jedi.compute_metric()}")
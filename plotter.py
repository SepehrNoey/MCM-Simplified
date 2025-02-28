import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torchvision
from torchvision.transforms import functional as F
from fvd.styleganv.fvd import load_i3d_pretrained, get_fvd_feats

# Use Agg backend for saving plots without display
plt.switch_backend('Agg')

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_config = {
    "student-setup-1": {"steps": [1, 2, 4, 8], "color": "blue"},
    "student-setup-2": {"steps": [1, 2, 4, 8], "color": "green"},
    "teacher": {"steps": [50, 100], "color": "red"},
    "real": {"color": "orange"}
}

def load_video(video_path, target_size=(224, 224), num_frames=30):
    """Load and preprocess a single video"""
    # Read video using torchvision (requires PyTorch 1.10+)
    video, _, _ = torchvision.io.read_video(video_path, pts_unit='sec')
    
    # Convert to float32 and normalize to [0, 1]
    video = video.float() / 255.0
    
    # Resize frames
    video = video.permute(0, 3, 1, 2)  # TCHW
    video = F.resize(video, target_size, antialias=True)
    
    # Handle temporal dimension
    if video.size(0) < num_frames:
        # Pad with last frame
        padding = video[-1].unsqueeze(0).repeat(num_frames - video.size(0), 1, 1, 1)
        video = torch.cat([video, padding], dim=0)
    else:
        # Take first num_frames
        video = video[:num_frames]
    
    return video.unsqueeze(0)  # Add batch dimension

def extract_i3d_features(video_folder, device, batch_size=8):
    """Extract features for all videos in a folder"""
    video_paths = [os.path.join(video_folder, f) for f in os.listdir(video_folder) 
                  if f.endswith(('.mp4', '.avi', '.mov'))]
    
    i3d = load_i3d_pretrained(device=device)
    features = []
    
    for i in tqdm(range(0, len(video_paths), batch_size)):
        batch_paths = video_paths[i:i+batch_size]
        batch_videos = []
        
        # Load videos
        for path in batch_paths:
            video = load_video(path)
            batch_videos.append(video)
        
        # Process batch
        videos = torch.cat(batch_videos, dim=0)
        videos = videos.to(device)
        
        # Transform to BCTHW format expected by I3D
        videos = videos.permute(0, 2, 1, 3, 4)  # BCTHW
        
        # Extract features
        with torch.no_grad():
            batch_features = get_fvd_feats(videos, i3d=i3d, device=device)
        
        features.append(batch_features)
    
    return np.concatenate(features, axis=0)

# def create_grid_plot(all_data, real_features, output_path="pca_grid.png"):
#     fig, axs = plt.subplots(3, 4, figsize=(40, 30))
    
#     # Process each row (model)
#     for row_idx, model in enumerate(["student-setup-1", "student-setup-2", "teacher"]):
#         # Process each column (step)
#         for col_idx, step in enumerate([1, 2, 4, 8]):
#             if model == "teacher" and col_idx >= 2:
#                 axs[row_idx, col_idx].axis('off')
#                 continue
                
#             if model == "teacher":
#                 actual_step = model_config["teacher"]["steps"][col_idx]
#                 step_data = all_data.get(("teacher", actual_step), None)
#                 step_label = f"DDIM {actual_step}"
#             else:
#                 actual_step = step
#                 step_data = all_data.get((model, step), None)
#                 step_label = f"{step} steps"
            
#             if step_data is None:
#                 continue
                
#             # Plot real data points
#             axs[row_idx, col_idx].scatter(
#                 real_features[:, 0], real_features[:, 1],
#                 c=model_config["real"]["color"], alpha=0.3, label='Real'
#             )
            
#             # Plot model data points
#             axs[row_idx, col_idx].scatter(
#                 step_data[:, 0], step_data[:, 1],
#                 c=model_config[model]["color"], alpha=0.3, label=model.capitalize()
#             )
            
#             # Plot means
#             real_mean = np.mean(real_features, axis=0)
#             model_mean = np.mean(step_data, axis=0)
            
#             axs[row_idx, col_idx].scatter(
#                 real_mean[0], real_mean[1],
#                 c='black', marker='*', s=400, edgecolor='white', label='Real Mean'
#             )
#             axs[row_idx, col_idx].scatter(
#                 model_mean[0], model_mean[1],
#                 c='purple', marker='*', s=400, edgecolor='white', label='Model Mean'
#             )
            
#             axs[row_idx, col_idx].set_title(
#                 f"{model.capitalize()} - {step_label}\n"
#                 f"Real vs {model.capitalize()} Distribution",
#                 fontsize=12
#             )
#             axs[row_idx, col_idx].grid(True)
#             axs[row_idx, col_idx].legend()

#     plt.tight_layout()
#     plt.savefig(output_path, dpi=300, bbox_inches='tight')
#     plt.close()

def create_grid_plot(all_data, real_features, output_path="pca_grid.png"):
    # Configure style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create figure with adjusted dimensions
    fig, axs = plt.subplots(3, 4, figsize=(25, 18), dpi=150)
    
    # Configure color and style parameters
    plot_params = {
        "real": {
            "color": "#FF7F0E",  # Orange
            "marker": "o",
            "alpha": 0.6,
            "s": 80,
            "edgecolor": "k",
            "label": "Real"
        },
        "student-setup-1": {
            "color": "#1F77B4",  # Blue
            "marker": "s",  # Square
            "alpha": 0.6,
            "s": 80,
            "edgecolor": "k",
            "label": "student-setup-1"
        },
        "student-setup-2": {
            "color": "#2CA02C",  # Green
            "marker": "D",  # Diamond
            "alpha": 0.6,
            "s": 80,
            "edgecolor": "k",
            "label": "student-setup-2"
        },
        "teacher": {
            "color": "#D62728",  # Red
            "marker": "X",  # X
            "alpha": 0.6,
            "s": 80,
            "edgecolor": "k",
            "label": "Teacher"
        },
        "mean": {
            "real_color": "#FFD700",  # Gold
            "model_color": "#9400D3",  # Purple
            "s": 300,
            "edgecolor": "w",
            "linewidth": 2,
            "zorder": 4
        }
    }

    # Process each row (model)
    for row_idx, model in enumerate(["student-setup-1", "student-setup-2", "teacher"]):
        # Process each column (step)
        for col_idx, step in enumerate([1, 2, 4, 8]):
            if model == "teacher" and col_idx >= 2:
                axs[row_idx, col_idx].axis('off')
                continue
                
            ax = axs[row_idx, col_idx]
            
            # Get current step data
            if model == "teacher":
                actual_step = model_config["teacher"]["steps"][col_idx]
                step_label = f"DDIM {actual_step}"
            else:
                actual_step = step
                step_label = f"{step} steps"
            
            model_data = all_data.get((model, actual_step), None)
            if model_data is None:
                continue

            # Plot real data points
            ax.scatter(
                real_features[:, 0], real_features[:, 1],
                **{k: v for k, v in plot_params["real"].items() if k != "label"}
            )
            
            # Plot model data points
            ax.scatter(
                model_data[:, 0], model_data[:, 1],
                **{k: v for k, v in plot_params[model].items() if k != "label"}
            )
            
            # Plot means with enhanced visibility
            real_mean = np.mean(real_features, axis=0)
            model_mean = np.mean(model_data, axis=0)
            
            # Real mean
            ax.scatter(
                real_mean[0], real_mean[1],
                c=plot_params["mean"]["real_color"],
                marker="*",
                s=plot_params["mean"]["s"],
                edgecolor=plot_params["mean"]["edgecolor"],
                linewidth=plot_params["mean"]["linewidth"],
                zorder=plot_params["mean"]["zorder"],
                label='Real Mean'
            )
            
            # Model mean
            ax.scatter(
                model_mean[0], model_mean[1],
                c=plot_params["mean"]["model_color"],
                marker="*",
                s=plot_params["mean"]["s"],
                edgecolor=plot_params["mean"]["edgecolor"],
                linewidth=plot_params["mean"]["linewidth"],
                zorder=plot_params["mean"]["zorder"],
                label=f'{model.capitalize()} Mean'
            )
            
            # Annotate statistics
            stats_text = (
                f'Real μ: ({real_mean[0]:.2f}, {real_mean[1]:.2f})\n'
                f'Model μ: ({model_mean[0]:.2f}, {model_mean[1]:.2f})'
            )
            ax.annotate(
                stats_text,
                xy=(0.05, 0.95),
                xycoords='axes fraction',
                fontsize=10,
                ha='left',
                va='top',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="k", lw=1)
            )
            
            # Formatting
            ax.set_title(
                f"{model.capitalize()} - {step_label}\nReal vs Model Distribution",
                fontsize=14,
                pad=12
            )
            ax.set_xlabel("PC1", fontsize=12)
            ax.set_ylabel("PC2", fontsize=12)
            ax.grid(True, alpha=0.4, ls="--")
            ax.legend(fontsize=10, loc='upper right')

    # Finl adjustments
    plt.tight_layout(pad=3.0)
    plt.savefig(output_path, bbox_inches='tight', transparent=False)
    plt.close()

def main():
    print("Processing real videos...")
    real_features = extract_i3d_features("/home/ubuntu/Smp-MCM/results-50p/test-videos", device)
    
    # Process model videos
    all_data = {}
    
    for model in ["student-setup-1", "student-setup-2", "teacher"]:
        steps = model_config[model]["steps"]
        for step in steps:
            folder_path = f"/home/ubuntu/Smp-MCM/results-50p/{model}_step_{step}" if model != "teacher" else f"/home/ubuntu/Smp-MCM/results-50p/{model}_ddim_{step}"
            print(f"Processing {model} - step {step}...")
            
            features = extract_i3d_features(folder_path, device)
            all_data[(model, step)] = features
    
    # Perform PCA on combined features
    print("Performing PCA...")
    combined_features = np.concatenate([
        *[v for v in all_data.values()],
        real_features
    ], axis=0)
    
    pca = PCA(n_components=2)
    pca.fit(combined_features)
    
    # Transform all features
    real_pca = pca.transform(real_features)
    for key in all_data:
        all_data[key] = pca.transform(all_data[key])
    
    # Create and save grid plot
    create_grid_plot(all_data, real_pca, "model_comparison_grid.png")
    print("Saved grid plot to model_comparison_grid.png")

if __name__ == "__main__":
    main()
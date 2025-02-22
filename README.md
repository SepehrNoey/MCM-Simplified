# Motion Consistency Model - Simplified Implementation

This repository contains a simplified implementation of the [Motion Consistency Model](https://github.com/yhZhai/mcm) as described in the original paper. The model was trained on a subset of the WebVid 2M dataset with additional image-caption pairs filtered from the LAION aesthetic dataset.

## Sample Generated Videos
|Caption|Teacher (ModelScope)|Student Setup 1|Student Setup 2|
|-----|-----|-----|-----|
|Worker slicing a piece of meat.|![Image](https://github.com/user-attachments/assets/49aadf34-d0cd-4531-829d-8237f17dd659)|![Image](https://github.com/user-attachments/assets/2bbea26d-6f76-4ac8-baa2-6600aa49697e)|![Image](https://github.com/user-attachments/assets/37ec6102-a63f-40d9-a601-6ae50c8453fd)|
|Pancakes with chocolate syrup nuts and bananas stack of whole flapjack tasty breakfast|![Image](https://github.com/user-attachments/assets/8a82fd29-b54c-45a6-b79e-72c88d7d8ce4)|![Image](https://github.com/user-attachments/assets/c003cc5a-563d-44f3-9121-3fd3db4caef4)|![Image](https://github.com/user-attachments/assets/c201b303-822c-4ba4-a600-0023c8884aa4)|

## Training Setup

### Dataset
- **Video-Caption Pairs:** 3022 pairs from the WebVid 2M dataset
- **Image-Caption Pairs:** Two different training configurations:
  - **Setting 1:** 20,000 images filtered from LAION aesthetic (cooking-related, min. resolution 450x450)
  - **Setting 2:** 7,500 images filtered from LAION aesthetic (cooking-related, min. resolution 1024x1024)

### Training Configurations
#### Setting 1
- Learning Rate: `5e-6`
- Gradient Accumulation Steps: `4`
- Max Gradient Norm: `10`
- Discriminator Loss Weight: `1`
- Discriminator Learning Rate: `5e-5`
- Discriminator Lambda R1: `1e-5`
- Discriminator Start Step: `0`
- EMA Decay: `0.95`
- Training Epochs: `7`
- Steps Trained: `~5100`

#### Setting 2 (Modified)
- Learning Rate: `2e-6`
- Gradient Accumulation Steps: `16`
- Max Gradient Norm: `5`
- Discriminator Loss Weight: `0.5`
- Discriminator Learning Rate: `1e-6`
- Discriminator Lambda R1: `1e-4`
- Discriminator Start Step: `400`
- EMA Decay: `0.98`
- LR Warmup Steps: `300`
- Training Epochs: `10`

## Evaluation Metrics

The model was evaluated using Frechet Video Distance (FVD) and CLIP similarity scores. The evaluation was conducted on 50 prompts and corresponding videos from the WebVid 2M dataset.

### FVD Scores
| Model | 1 Step | 2 Steps | 4 Steps | 8 Steps |
|--------|----------|----------|----------|----------|
| **Teacher Model (50 DDIM Steps)** | 2954.77 | - | - | - |
| **Student Model - Setup 1** | 2598.15 | 2684.24 | 3082.84 | 3914.78 |
| **Student Model - Setup 2** | 2589.01 | 3053.35 | 3284.69 | 3930.07 |

### CLIP Similarity Scores (×100)
| Model | 1 Step | 2 Steps | 4 Steps | 8 Steps |
|--------|----------|----------|----------|----------|
| **Teacher Model (50 DDIM Steps)** | 27.88 | - | - | - |
| **Student Model - Setup 1** | 22.55 | 25.62 | 26.86 | 27.01 |
| **Student Model - Setup 2** | 20.13 | 23.41 | 25.31 | 24.62 |

## Conclusion
The second training setup was modified to address oscillations in loss and to prevent the discriminator from overpowering the generator too early. The modifications led to a reduction in FVD scores for 1-step inference, although some increases were observed in multi-step settings. CLIP similarity improved across multiple inference steps, indicating better text-to-video alignment in the previous setup.

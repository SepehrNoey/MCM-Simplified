import torch
import os
from diffusers import TextToVideoSDPipeline, DDIMScheduler
from diffusers.utils import *

# Define parameters
MODEL_PATH = "/home/ubuntu/Smp-MCM/modelscope"  # Replace with actual model path
OUTPUT_DIR = "/home/ubuntu/Smp-MCM/mcm/inf-results-50p-teacher-100ddim"
PROMPTS = [
    'a confectioner cooks a dough a hand with a plastic spatula stirs the dough in a glass bowl for',
    'a confectioner pours out freshly baked pastries with a golden crust in a tray croissants',
    'an egg being beaten with a whisk',
    'baking fresh meat on grill closeup',
    'barista pours hot water into the filter with coffee cinematic k slowmotion brewing coffee process',
    'breaking egg into bowl',
    'burger on wood board hamburger on blurred background',
    'chocolate machine rotating disk with chocolate at candy factory',
    'chopping and prepare carrots',
    'chopping onion on wooden plate',
    'close up of female chef hands cutting cilantro with a big knife',
    'close up of strawberry and chocolate spread chocolate drop and strawberries closeup',
    'cooking pancakes on special pan in the kitchen at home slow motion closeup',
    'cup with coffee on the old table',
    'detail of man hands with rolling pin on piece of dough cooking pie',
    'dolly push in view of an italian pizza margherita',
    'dungeness crab boil at san francisco fisherman s wharf fps slow motion',
    'father in a striped shirt is cooking fresh organic vegetable salad with cocktail tomatoes and',
    'female hand squeezes juice from lemon into king prawns',
    'food industry slicing and eating fresh cooked tom a hawk steak',
    'fork to fetch spaghetti lines ingredients for cooking blank for design food concept',
    'french onion soup with ingredients being cooked in a pot on the stove',
    'freshly milled salt super slow motion',
    'frying pancakes in the kitchen at home a woman is cooking traditional russian pancakes modern',
    'girl prepares a tasty and useful salad from fresh vegetables',
    'hand make an avocado sandwich',
    'hands of father and daughter holding peeler and peeling potatoes parents teaching children basic',
    'knife cut through crispy grilled sandwich',
    'maple syrup pouring on the pancakes black background traditional american food for breakfast',
    'mexican food clip beef fajitas traditional dish of mexico mexican food in iron plate',
    'milk pouring in a glass',
    'paella spanish cuisine dish uncovered and covered',
    'pancakes with chocolate syrup nuts and bananas stack of whole flapjack tasty breakfast and',
    'picking mushrooms and adding on pizza',
    'pizza before baking with sauce and cheese',
    'plate of sausage and mash with thick onion gravy',
    'potatoes being mashed',
    'prawns mange tout and spring onions being added to coconut milk soup',
    'preparing delicious american pancakes on a pan',
    'round barbecue grill with open fire inside meals for summer picnic are being prepared corn',
    'salmon steak on a plate',
    'served ribs with vegetables',
    'slicing baguette',
    'sushi and rolls cooking',
    'turkish sausage sausage cooking on fire',
    'ukraine kitchen pour oil into the fry pan',
    'vegetable oil being poured into a pan and heated',
    'vegetables and meat are fried in a pan',
    'women kneading dough in a clear glass bowl pizza making bread making slow motion bread k',
    'worker slicing a piece of meat'
]  # Replace with your own prompts
NUM_INFERENCE_STEPS = 100
HEIGHT, WIDTH = 256, 256  # Adjust as needed

def generate_videos():
    # Load the model and scheduler
    scheduler = DDIMScheduler.from_pretrained(MODEL_PATH, subfolder="scheduler")
    pipeline = TextToVideoSDPipeline.from_pretrained(
        MODEL_PATH, scheduler=scheduler, torch_dtype=torch.float16
    ).to("cuda")

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for prompt in PROMPTS:
        print(f"Generating video for: {prompt}")
        with torch.no_grad():
            result = pipeline(
                prompt=prompt,
                num_inference_steps=NUM_INFERENCE_STEPS,
                height=HEIGHT,
                width=WIDTH,
                generator=torch.manual_seed(453645634)
            )
            video_frames = result.frames  # Ensure we get video frames correctly

        # Convert tensor frames to numpy format (if necessary)
        if isinstance(video_frames, torch.Tensor):
            video_frames = video_frames.cpu().numpy()

        # Ensure frames are in the correct format (T, H, W, C)
        if len(video_frames.shape) == 5:  # (B, T, H, W, C) -> Remove batch dimension
            video_frames = video_frames[0]

        # Save video
        video_path = os.path.join(OUTPUT_DIR, f"{prompt}.mp4")
        export_to_video(video_frames, video_path)
        print(f"Saved: {video_path}")

if __name__ == "__main__":
    generate_videos()
from typing import Optional

import torch
from diffusers import (
    AnimateDiffPipeline,
    DiffusionPipeline,
    LCMScheduler,
    MotionAdapter,
)
from diffusers.utils import export_to_video
from peft import PeftModel


def main():
    # select model_path from ["animatediff-laion", "animatediff-webvid",
    # "modelscopet2v-webvid", "modelscopet2v-laion", "modelscopet2v-anime",
    # "modelscopet2v-real", "modelscopet2v-3d-cartoon"]
    model_path = "modelscopet2v-laion"
    prompts = [
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
]

    num_inference_steps = 4

    model_id = "/home/ubuntu/Smp-MCM/mcm/work_dirs/modelscopet2v_improvement/checkpoint-1800"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if "animatediff" in model_path:
        pipeline = get_animatediff_pipeline()
    elif "modelscope" in model_path:
        pipeline = get_modelscope_pipeline()
    else:
        raise ValueError(f"Unknown pipeline {model_path}")

    lora = PeftModel.from_pretrained(
        pipeline.unet,
        model_id,
        adapter_name="pretrained_lora",
        torch_device="cpu",
        is_local=True,
    )
    lora.merge_and_unload()
    pipeline.unet = lora

    pipeline = pipeline.to(device)
    output = pipeline(
        prompt=prompts,
        num_frames=16,
        guidance_scale=1.0,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator("cpu").manual_seed(453645634),
    ).frames
    if not isinstance(output, list):
        output = [output[i] for i in range(output.shape[0])]

    for j in range(len(prompts)):
        export_to_video(
            output[j],
            f"./inf-results-seed-50p-4step/{prompts[j]}.mp4",
            fps=7,
        )


def get_animatediff_pipeline(
    real_variant: Optional[str] = "realvision",
    motion_module_path: str = "guoyww/animatediff-motion-adapter-v1-5-2",
):
    if real_variant is None:
        model_id = "runwayml/stable-diffusion-v1-5"
    elif real_variant == "epicrealism":
        model_id = "emilianJR/epiCRealism"
    elif real_variant == "realvision":
        model_id = "SG161222/Realistic_Vision_V6.0_B1_noVAE"
    else:
        raise ValueError(f"Unknown real_variant {real_variant}")

    adapter = MotionAdapter.from_pretrained(
        motion_module_path, torch_dtype=torch.float16
    )
    pipe = AnimateDiffPipeline.from_pretrained(
        model_id,
        motion_adapter=adapter,
        torch_dtype=torch.float16,
    )
    scheduler = LCMScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
        timestep_scaling=4.0,
        clip_sample=False,
        timestep_spacing="linspace",
        beta_schedule="linear",
        beta_start=0.00085,
        beta_end=0.012,
        steps_offset=1,
    )
    pipe.scheduler = scheduler
    pipe.enable_vae_slicing()
    return pipe


def get_modelscope_pipeline():
    model_id = "/home/ubuntu/Smp-MCM/modelscope"
    pipe = DiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, variant="fp16"
    )
    scheduler = LCMScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
        timestep_scaling=4.0,
    )
    pipe.scheduler = scheduler
    pipe.enable_vae_slicing()

    return pipe


if __name__ == "__main__":
    main()
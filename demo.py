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
    '3 people in a kitchen talking about cupcakes and then the countdown ends',
    'a white stove with tea keetlespatula 2 frying pans and two chefs',
    'a animation of a girl fixing food',
    'a bird s eye view of a pan that meat is being cooked in',
    'a boiling pot of liquid steams on a stove and then catches on fire',
    'a cake is being put into the oven',
    'a cartoon character dressed like a chef',
    'a cartoon of teen titans eating food',
    'a casserole is displayed',
    'a chef explaining about dish making',
    'a chef explains about some receipe',
    'a chef is cooking meat in a pan',
    'a chef is making a fish',
    'a chef prepares miso soup',
    'a chef stirs some food up in a pan',
    'a chef watches another chef cook',
    'a chef working on a dish outdoors',
    'a child is cooking in the kitchen',
    'a child is peeling shrimp',
    'a churro (fried dough) shop is featured showing how the food is prepared',
    'a clip of a person mixing a paster in a bowl',
    'a cook is preparing a salad',
    'a cook mixing ingredients together',
    'a cook mixing together ingredients',
    'a cookery item is being made',
    'a cookery show is going on',
    'a cooking demonstration and tutorial provided by two young females',
    'a couple is cooking',
    'a cute girl says about churros which is deep fried and crspy outside and chewy inside',
    'a diy video about cooking',
    'a egg layer is rolled out with stopsticks from a baking pan and egg mixer is poured into the pan',
    'a fried dessert is sitting in a dish on  the counter',
    'a girl eats a nice sandwich',
    'a girl is cooking',
    'a girl is giving tips to make dish',
    'a girl is talking to her dolls',
    'a girl turns on a stove and begins cooking',
    'a group of people are cooking food together',
    'a guy cooks lobster in a wok',
    'a guy is breaking an egg and seperated the yoke to put in the food processor',
    'a guy with a beard shows off his microwave',
    'a lady describes a recipe',
    'a lady kneads dough and puts it into a bowl to rise',
    'a lady making a dough by pouring little water in a bowl full of flour',
    'a lady mixes icing in a pastry bag',
    'a lady whisks an egg',
    'a man attempts to makes eggs and does a messy job of it',
    'a man cooking chicken',
    'a man demonstrates how to cook a special sauce',
    'a man dropping something in a water fill container',
    'a man explaining something',
    'a man has a cup of onions',
    'a man in black shirt is busy preparing something',
    'a man is cooking',
    'a man is opening the lid on one of two pots which are cooking on a black stove',
    'a man is talking about deserts',
    'a man pulls sections out of a cheesey bread loaf',
    'a man removes noodles from boiling water with tongs and places it into a tomato sauce',
    'a man uses a serrated knife to cut a french baguette in half',
    'a old women  is preparing dish',
    'a old women is preparing the food in kitchen',
    'a person cutting dough with a knife',
    'a person holding onto two pieces of a plastic pepper',
    'a person is adding a food item to a white bowl',
    'a person is showing a curd in a glass',
    'a person shows the preparation of south indian chapathi',
    'a watermelon is cut into the shape of a rose with green leaves and red petals',
    'a woman adds a lumpy white liquid from a glass bowl into a red plastic bowl filled with white powder',
    'a woman applies a crumb coat of icing to four layers of vanilla cake',
    'a woman cooking a meal in a pan on a stove',
    'a woman cooks eggs in a pan by pulling them away from the sides with a red spatula scraper',
    'a woman glues a banana shaped piece of clay to a plastic petri type dish',
    'a woman is teaching about his recipe as she cooks the food',
    'a woman mixes a small amount of chocolate in a plastic bowl with a tiny plastic spatula',
    'a woman stirs a red sauce and beans in a silver pan on a hot gas stove',
    'a woman talking about a recipe there is flour in the bowl',
    'a woman transfers dough to plastic wrap and adds the leftover pieces to the ball of dough',
    'a woman with many many bowls out to make gelatin',
    'do not overbeat the batter or your dish will be tough',
    'in a kitchen someone is preparing a dessert using strawberries blueberries and species',
    'in a kitchen two little chefs who are little girls test a spicy chile',
    'in the kitchen there is a black oven with a oven montor describing temperture to set for cooking donut',
    'ingredients for preparing food are kept in a table',
    'man cooking hot wings in a stainless steel bowl',
    'someone cooks meat patties in an oil filled pan on top of a stove then covers it with the lid',
    'someone is drawing circles in a pan i believe to make a crepe',
    'the woman prepare the special food item with lot of vegetable and other ingredients',
    'the women in her kitchen is showing some dish making and for that she is cracking a six eggs in to the bowl',
    'there is a man making a dish with medium heated pan',
    'there is a white kitchen drawer on the floor',
    'there is a woman is making a dish with bread',
    'there is mixing some  different colours  in a paper',
    'three people in white chefs hats sing to a diner about a fish dinner',
    'three women in red and white chef coats cheer and clap for their teammates',
    'various vegetables on a white table possibly a cooking show or demo',
    'women in red dress is preparing food in kitchen'
]

    num_inference_steps = 4

    model_id = "/home/ubuntu/Smp-MCM/mcm/work_dirs/modelscopet2v_improvement/checkpoint-final"
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
    batch_size = 50  # Adjust based on your GPU's capacity
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        output = pipeline(
            prompt=batch_prompts,
            num_frames=16,
            guidance_scale=1.0,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator("cpu").manual_seed(453645634),
        ).frames
        # Export each video in the batch
        for j in range(len(batch_prompts)):
            export_to_video(
                output[j],
                f"./inf-results-seed-96p-4step/{batch_prompts[j]}.mp4",
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
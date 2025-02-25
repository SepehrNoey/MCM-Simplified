#!/bin/bash

# please set the following variables
export VIDEO_DATA_PATH="/home/ubuntu/Smp-MCM/WebVid-Dataset-Cooking/"
export IMAGE_DATA_NAME=laion  # name of the image data, choose from ["webvid", "laion", "disney", "realisticvision", "toonyou"]
export IMAGE_DATA_PATH="/home/ubuntu/Smp-MCM/LAION-Cooking-1024plus"

export GPUS=1  # number of GPUs
# export MASTER_PORT=29500  # port for distributed training
export RUN_NAME=modelscopet2v_improvement  # name of the run
export OUTPUT_DIR=work_dirs/$RUN_NAME  # directory to save the model checkpoints

accelerate launch --num_machines 1 --num_processes $GPUS \
    --mixed_precision=fp16 \
    main.py \
    --base_model_name=modelscope \
    --output_dir=$OUTPUT_DIR \
    --mixed_precision=fp16 \
    --resolution=256 \
    --num_frames=16 \
    --learning_rate=2e-6 \
    --loss_type="huber" \
    --adam_weight_decay=0.0 \
    --dataloader_num_workers=4 \
    --validation_steps=100 \
    --checkpointing_steps=100 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=16 \
    --seed=453645634 \
    --enable_xformers_memory_efficient_attention \
    --report_to tensorboard \
    --tracker_project_name="motion-consistency-model" \
    --tracker_run_name=$RUN_NAME \
    --dataset_path $VIDEO_DATA_PATH \
    --num_train_epochs 10 \
    --use_8bit_adam \
    --use_lora \
    --max_grad_norm 5 \
    --lr_scheduler cosine \
    --w_min 5 \
    --w_max 15 \
    --frame_interval 8 \
    --disc_loss_type wgan \
    --disc_loss_weight 0.5 \
    --disc_learning_rate 1e-6 \
    --disc_lambda_r1 1e-4 \
    --disc_start_step 400 \
    --disc_gt_data $IMAGE_DATA_NAME \
    --disc_gt_data_path $IMAGE_DATA_PATH \
    --disc_tsn_num_frames 2 \
    --cd_target learn \
    --timestep_scaling_factor 4 \
    --cd_pred_x0_portion 0.5 \
    --num_ddim_timesteps 50 \
    --resume_from_checkpoint latest \
    --ema_decay 0.98 \
    --lr_warmup_steps 300

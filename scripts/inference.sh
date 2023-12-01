#!/bin/bash
CUDA_VISIBLE_DEVICES=3
CKPT_PATH="/work/hpc/pgl/lung-diffusion/logs/train_diffusion/runs/2023-11-12_19-04-33/checkpoints/last.ckpt"
EXPERIMENT="ddpm_vqmodel_f16_celeba.yaml"
python src/inference_diffusion.py experiment=$EXPERIMENT ckpt_path=$CKPT_PATH task_name="inference_diffusion"

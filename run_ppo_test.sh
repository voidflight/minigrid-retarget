#!/bin/bash

python -m src.run_ppo --exp_name "Blue-Purple-Find-Seed" \
    --seed 84 \
    --cuda \
    --track \
    --wandb_project_name "Retargeting-Search" \
    --env_id "FetchObstacles-6x6-v0" \
    --view_size 5 \
    --total_timesteps 6000000 \
    --learning_rate 0.00075 \
    --hidden_size 128 \
    --num_envs 32 \
    --num_steps 512 \
    --num_minibatches 4 \
    --update_epochs 8 \
    --clip_coef 0.2 \
    --ent_coef 0.01 \
    --vf_coef 0.0005 \
    --max_steps 1000 \
    --one_hot_obs \
    --num_checkpoints 10 \

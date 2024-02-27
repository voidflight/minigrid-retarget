#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 16:34:37 2024

@author: catgoddess
"""

from src.config import (
    EnvironmentConfig,
    OnlineTrainConfig,
    RunConfig,
    TransformerModelConfig,
)
from src.ppo.utils import parse_args
from src.environments.registration import register_envs
import torch 

torch.set_default_dtype(torch.float32)

if __name__ == "__main__":
    args = parse_args()
    register_envs()

    device = "cuda" if args.cuda else "cpu"
    
    run_config = RunConfig(
        exp_name=args.exp_name,
        seed=args.seed,
        device=device,
        track=args.track,
        wandb_project_name=args.wandb_project_name,
        wandb_entity=args.wandb_entity,
    )

    environment_config = EnvironmentConfig(
        env_id=args.env_id,
        one_hot_obs=args.one_hot_obs,
        fully_observed=args.fully_observed,
        max_steps=args.max_steps,
        capture_video=args.capture_video,
        view_size=args.view_size,
        device=run_config.device,
    )

    online_config = OnlineTrainConfig(
        hidden_size=args.hidden_size,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        trajectory_path=args.trajectory_path,
        fully_observed=args.fully_observed,
        device=run_config.device,
    )
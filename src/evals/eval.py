#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 16:41:12 2024

@author: catgoddess
"""

import time
import warnings
from typing import Optional, Union

import gymnasium as gym
import torch as t

import wandb
from src.config import (
    EnvironmentConfig,
    LSTMModelConfig,
    OnlineTrainConfig,
    RunConfig,
    TransformerModelConfig,
)
from src.environments.environments import make_env
from src.environments.registration import register_envs
from src.ppo.train import train_ppo
from src.ppo.utils import set_global_seeds
from src.utils.trajectory_writer import TrajectoryWriter

warnings.filterwarnings("ignore", category=DeprecationWarning)

def test_runner(
    run_config: RunConfig,
    environment_config: EnvironmentConfig,
    online_config: OnlineTrainConfig,
    model_config: Optional[Union[TransformerModelConfig, LSTMModelConfig]],):
    
    target_types = ["key"] * 2
    target_colors = ["blue", "purple"]
    
    # Verify environment is registered
    register_envs()
    all_envs = [env_spec for env_spec in gym.envs.registry]
    assert (
        environment_config.env_id in all_envs
    ), f"Environment {environment_config.env_id} not registered."
    
    # wandb initialisation,
    run_name = f"{environment_config.env_id}__{run_config.exp_name}__{run_config.seed}__{int(time.time())}"
    if run_config.track:
        run = wandb.init(
            project=run_config.wandb_project_name,
            entity=run_config.wandb_entity,
            config=combine_args(
                run_config, environment_config, online_config, model_config
            ),  # vars is equivalent to args.__dict__
            name=run_name,
            save_code=True,
        )
    
    # add run_name to args
    run_config.run_name = run_name
    
    set_global_seeds(run_config.seed)
    
    target_type = target_types[0]
    target_color = target_colors[0]
    
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                config=environment_config,
                seed=environment_config.seed + i,
                idx=i,
                run_name=run_name,
                target_color = target_color,
            )
            for i in range(online_config.num_envs)
        ]
    )
    
    for env in envs.envs:
        env.switch_target(target_type, target_color)
    
    # evaluation rollouts here
    
    if run_config.track:
        run.finish()


def combine_args(
    run_config,
    environment_config,
    online_config,
    transformer_model_config=None,
):
    args = {}
    args.update(run_config.__dict__)
    args.update(environment_config.__dict__)
    args.update(online_config.__dict__)
    if transformer_model_config is not None:
        args.update(transformer_model_config.__dict__)
    return args

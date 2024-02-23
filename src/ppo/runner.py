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


def ppo_runner(
    run_config: RunConfig,
    environment_config: EnvironmentConfig,
    online_config: OnlineTrainConfig,
    model_config: Optional[Union[TransformerModelConfig, LSTMModelConfig]],
):
    """
    Executes Proximal Policy Optimization (PPO) training on a specified environment with provided hyperparameters.

    Args:
    - run_config (RunConfig): An object containing general run configuration details.
    - environment_config (EnvironmentConfig): An object containing environment-specific configuration details.
    - online_config (OnlineTrainConfig): An object containing online training configuration details.
    - model_config (Optional[Union[TransformerModelConfig, LSTMModelConfig]]): An optional object containing either Transformer or LSTM model configuration details.

    Returns: None.
    """
    target_types = ["key"] * 4
    target_colors = ["grey", "blue", "purple", "yellow"]
    
    if online_config.trajectory_path:
        trajectory_writer = TrajectoryWriter(
            online_config.trajectory_path,
            run_config=run_config,
            environment_config=environment_config,
            online_config=online_config,
            model_config=model_config,
        )
    else:
        trajectory_writer = None

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

    # make envs
    set_global_seeds(run_config.seed)
    
    envs_list = []
    for i, color in enumerate(target_colors):
        
        envs = gym.vector.SyncVectorEnv(
            [
                make_env(
                    config=environment_config,
                    seed=environment_config.seed + i,
                    idx=i,
                    run_name=run_name,
                    target_color = color,
                )
                for i in range(online_config.num_envs)
            ]
        )
        for env in envs.envs:
            env.switch_target(target_types[i], color)
            
        envs_list.append(envs)
        
        wandb.define_metric(f"step-{color}")
        wandb.define_metric(f"{target_types[i]}_{color}_avg_value", step_metric = f"step-{color}")

    agent = train_ppo(
        run_config=run_config,
        online_config=online_config,
        environment_config=environment_config,
        model_config=model_config,
        envs_list=envs_list,
        trajectory_writer=trajectory_writer,
    )
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

#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING, Type, Union
import os
from habitat.core.env import Env, RLEnv
from habitat.datasets import make_dataset
from pathlib import Path
if TYPE_CHECKING:
    from omegaconf import DictConfig


def make_env_fn(
    config: "DictConfig",
    env_class: Union[Type[Env], Type[RLEnv]],
    dataset=None,
) -> Union[Env, RLEnv]:
    r"""Creates an env of type env_class with specified config and rank.
    This is to be passed in as an argument when creating VectorEnv.

    Args:
        config: root exp config that has core env config node as well as
            env-specific config node.
        env_class: class type of the env to be created.
        dataset: If specified, load the environment using this dataset.

    Returns:
        env object created according to specification.
    """
    def get_resume_episode(result_path):
        last_episode = 0
        if os.path.exists(result_path):
            with open(result_path, 'r') as f:
                data = f.readlines()
            last_episode = len(data)
        else:
            last_episode = 0
        return last_episode
    
    def get_env_variable(name, default=None, required=False):
        value = os.environ.get(name, default)
        if required and value is None:
            raise EnvironmentError(f"lack environment parameters: {name}")
        return value

    if "habitat" in config:
        config = config.habitat
    if dataset is None:
        dataset = make_dataset(config.dataset.type, config=config.dataset)

    start_split = get_env_variable("start_split")
    end_split = get_env_variable("end_split")
    result_path  = get_env_variable("result_path")
    project_root = Path.cwd()
    outputs_dir = project_root / 'outputs'
    result_path = outputs_dir /result_path
    print("start_split: ",start_split)
    print("end_split: ",end_split)
    print("result_path: ",result_path)
    last_episodes = get_resume_episode(f"{result_path}/final_result.txt")
    last_episodes=int(last_episodes/3)
    print("last_episodes: ",last_episodes)
    dataset.episodes = dataset.episodes[int(start_split) + int(last_episodes):int(end_split)] 
    
    
    env = env_class(config=config, dataset=dataset)
    env.seed(config.seed)
    return env

import math
from typing import Callable
from stable_baselines3 import PPO, SAC, TD3
from sb3_contrib import TRPO
import torch
from pathlib import Path
import yaml
from utils.normal_utils import ObjDict
from net.net import CNNTransformerActorCriticPolicy
from net.featuresextractor import MLPTransformerExtractor, MLPTransformerExtractor_single_channel

policy_kwargs_on_policy = dict(activation_fn=torch.nn.ReLU,
                     net_arch=dict(pi=[256, 256], vf=[256, 256]))

policy_kwargs_off_policy = dict(net_arch=dict(pi=[256, 256], qf=[256, 256]))

policy_kwargs_with_feature_ext = dict(
    features_extractor_class=MLPTransformerExtractor,
    features_extractor_kwargs=dict(
        features_dim=48,
        k=21,
        hidden_size=64,
        output_size_mlp=16,
        output_size_transformer=32,
        final_output_size=48,
        num_heads=2,
        num_layers=2
        ),
    activation_fn=torch.nn.ReLU,
    net_arch=[48, 128, 128],
    
)

class CosineScheduler:
    def __init__(self, max_update, base_lr=0.01, final_lr=0,
               warmup_steps=0, warmup_begin_lr=0):
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps

    def get_warmup_lr(self, epoch):
        increase = (self.base_lr_orig - self.warmup_begin_lr) \
                       * float(epoch) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase

    def __call__(self, epoch):
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch)
        if epoch <= self.max_update:
            self.base_lr = self.final_lr + (
                self.base_lr_orig - self.final_lr) * (1 + math.cos(
                math.pi * (epoch - self.warmup_steps) / self.max_steps)) / 2
        return self.base_lr




def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func



def load_train_model(algo, env, lr, save_dir, transformer = False):
    custom = False
    if algo == 'ppo':
        if transformer:
            model = PPO(
                "MultiInputPolicy",
                env=env,
                learning_rate=lr,
                gamma=0.99,
                clip_range=0.1,
                policy_kwargs= policy_kwargs_with_feature_ext,
                tensorboard_log=save_dir + "tensorboard",
                verbose=1,
            )
        else:
            model = PPO(
                "MlpPolicy",
                env=env,
                learning_rate=lr,
                gamma=0.99,
                clip_range=0.1,
                policy_kwargs= policy_kwargs_on_policy,
                tensorboard_log=save_dir + "tensorboard",
                verbose=1,
            )
    elif algo == 'sac':
        model = SAC(
            "MlpPolicy",
            env=env,
            learning_rate=lr,
            gamma=0.99,
            policy_kwargs= policy_kwargs_off_policy,
            tensorboard_log=save_dir + "tensorboard",
            verbose=1,
        )
    elif algo == 'td3':
        model = TD3(
            "MlpPolicy",
            env=env,
            learning_rate=lr,
            gamma=0.99,
            policy_kwargs= policy_kwargs_off_policy,
            tensorboard_log=save_dir + "tensorboard",
            verbose=1,
        )
    elif algo == 'trpo':
        model = TRPO(
            "MlpPolicy",
            env=env,
            learning_rate=lr,
            gamma=0.99,
            tensorboard_log=save_dir + "tensorboard",
            verbose=1,
        )

    else:
        raise NotImplementedError("This algorithm is not configured.")
    
    return model


def load_test_model(algo, SEED, env):
    if algo == 'ppo':
        model = PPO.load(f"./results/{algo}/{SEED}_tmp/best_model.zip", env=env)
    elif algo == 'sac':
        model = SAC.load(f"./results/{algo}/{SEED}_tmp/best_model.zip", env=env)
    elif algo == 'td3':
        model = TD3.load(f"./results/{algo}/{SEED}_tmp/best_model.zip", env=env)
    elif algo == 'trpo':
        model = TRPO.load(f"./results/{algo}/{SEED}_tmp/best_model.zip", env=env)
    else:
        raise NotImplementedError("This algorithm is not configured.")
    
    return model


def load_bc_model(algo, env, lr, save_dir):
    model = load_train_model(algo, env, lr, save_dir)
    
    return model

def load_bc_policy(algo, env, lr, save_dir):
    model = load_train_model(algo, env, lr, save_dir)
    policy = model.policy

    return policy

def load_dummy_model(algo, trainer, env):
    if algo == 'ppo':
        model = PPO.load(f"./results/{algo}_dummy/{trainer}/best_model.zip", env=env)
    elif algo == 'sac':
        model = SAC.load(f"./results/{algo}_dummy/{trainer}/best_model.zip", env=env)
    elif algo == 'td3':
        model = TD3.load(f"./results/{algo}_dummy/{trainer}/best_model.zip", env=env)
    elif algo == 'trpo':
        model = TRPO.load(f"./results/{algo}_dummy/{trainer}/best_model.zip", env=env)
    else:
        raise NotImplementedError("This algorithm is not configured.")
    
    return model

def load_config(cfg_name):
    """Load config file."""
    config_file = yaml.safe_load(Path('train.yaml').read_text())

    if cfg_name == "train":
        return ObjDict(config_file["train"])
    elif cfg_name == "env":
        return ObjDict(config_file["env"])
    elif cfg_name == "vehicle":
        return ObjDict(config_file["vehicle"])
    else:
        raise AttributeError("No such attribute: " + cfg_name)

def load_config_real(cfg_name):
    """Load config file."""
    config_file = yaml.safe_load(Path('train_real.yaml').read_text())

    if cfg_name == "train":
        return ObjDict(config_file["train"])
    elif cfg_name == "env":
        return ObjDict(config_file["env"])
    elif cfg_name == "vehicle":
        return ObjDict(config_file["vehicle"])
    else:
        raise AttributeError("No such attribute: " + cfg_name)
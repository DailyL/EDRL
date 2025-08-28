import sys 
import os
import numpy as np
import inspect
import random
import tempfile

from gymnasium import Wrapper,Env
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common import policies
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy

from imitation.algorithms import bc
from imitation.algorithms.adversarial.gail import GAIL
from imitation.algorithms.adversarial.airl import AIRL
from imitation.algorithms.dagger import SimpleDAggerTrainer
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.data import rollout, serialize


from imitation.util import logger as imit_logger
#from imitation.data.wrappers import RolloutInfoWrapper
from wrapper import RolloutInfoWrapper
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.train_utils import load_dummy_model
from utils.train_utils import CosineScheduler,linear_schedule, load_train_model,load_config,load_bc_policy

from env.make_env import make_vec_test_env_il,make_vec_test_env
from net.net import CNNTransformerActorCriticPolicy
from generate_demos import generate_and_save, load_traj




def load_trainer_model(model_name, env, model, transitions, SEED):
    rng = np.random.default_rng(0)
    log_dir = f'./results/bc/{model_name}/{SEED}_tmp/tensorboard/'
    custom_logger = imit_logger.configure(
        folder=log_dir,
        format_strs=["tensorboard", "stdout"],
    )

    # reward net used for GAIL, AIRL
    reward_net = BasicRewardNet(
            observation_space=env.observation_space,
            action_space=env.action_space,
            normalize_input_layer=RunningNorm,
        )

    if model_name =='bc':

        trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        custom_logger=custom_logger,
        rng=rng,
        policy=model.policy,
    )

    elif model_name == 'gail':
        trainer = GAIL(
            demonstrations=transitions,
            demo_batch_size=100,
            gen_replay_buffer_capacity=512,
            n_disc_updates_per_round=8,
            venv=env,
            gen_algo=model,
            reward_net=reward_net,
        )
    
    elif model_name == 'airl':
        trainer = AIRL(
            demonstrations=transitions,
            demo_batch_size=100,
            gen_replay_buffer_capacity=512,
            n_disc_updates_per_round=16,
            venv=env,
            gen_algo=model,
            reward_net=reward_net,
        )

    else:
        raise NotImplementedError("This algorithm is not configured.")
    
    return trainer

    
    
    


    

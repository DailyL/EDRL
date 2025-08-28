import sys 
import os
import numpy as np
import inspect
import random
import argparse
import gymnasium as gym
import matplotlib.pyplot as plt
import time
import torch
from gymnasium import Wrapper,Env
from gymnasium.spaces import Box
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common import policies
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy

from imitation.algorithms import bc
from imitation.data import rollout, serialize

from imitation.util import logger as imit_logger
#from imitation.data.wrappers import RolloutInfoWrapper
from wrapper import RolloutInfoWrapper
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.train_utils import load_dummy_model
from utils.train_utils import CosineScheduler,linear_schedule, load_train_model,load_config,load_bc_policy,load_bc_model

from env.make_env import make_vec_test_env_il,make_vec_test_env,make_vec_train_bc_env, make_vec_train_il_env, make_vec_test_il_env, make_bc_env_purestate, make_vec_test_env_purestate
from net.net import CNNTransformerActorCriticPolicy
from generate_demos import generate_and_save, load_traj, load_traj_env, load_traj_multi, load_traj_multi_npy
from train_utils import load_trainer_model

from imitation.util.util import make_vec_env
from metadrive.examples.top_down_metadrive import draw_multi_channels_top_down_observation
from env.util.obs_setter import PureStateObservation, StateBEVObservation, flatten_obs, flatten_space, unflatten_obs

def sample_expert_transitions(expert, env, rng):
    print("Sampling expert transitions.")
    rollouts = rollout.rollout(
        expert,
        env,
        rollout.make_sample_until(min_timesteps=None, min_episodes=1),
        rng=rng,
    )
    return rollout.flatten_trajectories(rollouts)


def train(model_name, scenario_name, obs_type, num_scenarios):

    SEED = random.randint(0,20000)
    train_config = load_config("train")
    save_dir = f"{train_config.logdir}/{SEED}_tmp/"
    npy_traj = False


    env = load_traj_env(obs_type=obs_type, trainer=model_name)


    custom_model = load_bc_model('ppo', env, lr=0.0001, save_dir=save_dir)
    
    if npy_traj:
        transitions = load_traj_multi_npy(trainer_name=model_name, obs_type=obs_type)
    else:
        transitions = load_traj_multi(trainer_name=model_name, obs_type=obs_type)

    trainer = load_trainer_model(
        model_name=model_name,
        env=env,
        model=custom_model,
        transitions=transitions,
        SEED=SEED,
        )

    print("Training a policy using {}, saved in {} _tmp".format(model_name, SEED))

    train_fn(trainer=trainer, algo=model_name)

    trainer.policy.save(f'./results/bc/{model_name}/{SEED}_tmp/best_model.zip')




def train_fn(trainer, algo):
    if algo =='bc':
        trainer.train(n_epochs=1)

    elif algo == 'gail' or algo == 'airl':
        trainer.train(3000)  # Train for 2_000_000 steps to match expert
    
    else:
        raise NotImplementedError("This algorithm is not configured.")
    
def test(model_name, SEED):

    #env = load_traj_env(obs_type=obs_type,trainer=model_name)
    env = make_vec_test_env_purestate(idm=False)
    algo = 'ppo'
    lr = 0.00003

    train_config = load_config("train")
    save_dir = f"{train_config.logdir}/{SEED}_tmp/"
    os.makedirs(save_dir, exist_ok=True)

    # Start point of RL training
    model = load_train_model(
        algo=algo,
        env=env,
        lr=lr,
        save_dir=save_dir,
        )

    model.policy = model.policy.load(f"./results/bc/{model_name}/{SEED}_tmp/best_model.zip")

    n_eval_episodes = 10
    max_steps = 2000

    test_fn(env=env, 
            model=model, 
            n_eval_episodes=n_eval_episodes, 
            algo=model_name, 
            max_steps=max_steps,
            obs_type=obs_type,
            )

    

def test_fn(env, model, n_eval_episodes, algo, max_steps, obs_type):

    if obs_type == "bevs":
        test_fn_bev(env=env, 
                    model=model, 
                    n_eval_episodes=n_eval_episodes, 
                    algo=algo, 
                    max_steps=max_steps,
                    )
        
    elif obs_type == "purestate":
        obs = env.reset()
        for k in range (n_eval_episodes):
            total_reward = 0
            done = False
            obs = env.reset()
            for i in range(max_steps):
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                total_reward += reward
                
                # Render the environment if render method is available
                if hasattr(env, 'render'):
                    ret = env.render(mode="topdown")
                
                if done:
                    print(f"Episode {k+1} reward:", total_reward)
                    break  # Exit the loop if the episode is done
    else:
        raise NotImplementedError("Observation Type Unknown.") 
    


def test_fn_bev(env, model, n_eval_episodes, algo, max_steps):
    if algo == 'bc':
        obs, _ = env.reset()
        for k in range (n_eval_episodes):
            total_reward = 0
            done = False
            obs, _ = env.reset()
            for i in range(max_steps):
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, _, info = env.step(action)
                
                total_reward += reward
                
                # Render the environment if render method is available
                if hasattr(env, 'render'):
                    ret = env.render(mode="topdown", 
                                    screen_record=True,
                                    window=True,
                                    screen_size=(600, 600), 
                                    )
                
                if done:
                    print(f"Episode {k+1} reward:", total_reward)
                    break  # Exit the loop if the episode is done
    elif algo == 'airl' or algo == 'gail':
        obs = env.reset()
        for k in range (n_eval_episodes):
            total_reward = 0
            done = False
            obs = env.reset()
            for i in range(max_steps):
                action, _states = model.predict(obs, deterministic=True)
                action = action*0.5
                obs, reward, done, info = env.step(action)
                total_reward += reward
                if i % 50 == 0:
                    self_render(obs)
                if done:
                    print(f"Episode {k+1} reward:", total_reward)
                    break  # Exit the loop if the episode is done
    else:
        raise NotImplementedError("This algorithm is not configured.")



def self_render(obs):
    bev_shape = Box(-0.0, 1.0, (84, 84, 5), np.float32).shape
    state_shape= Box(-0.0, 1.0, (21,), np.float32).shape
    obs = unflatten_obs(torch.from_numpy(obs), bev_shape, state_shape)

    obs['bev'] = obs['bev'].squeeze(0)

    draw_multi_channels_top_down_observation(obs=obs['bev'])






def reshape_for_render(obs, show_time = 1):

    obs = obs.flatten()
    obs = obs[:-21]
    height, width = 84, 84
    num_images = 5

    # Reshape the array into 5 images of size 84x84
    images = obs.reshape((height, width, num_images))

    # Display the images using Matplotlib
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))
    count = 0

    def close_event():
        plt.close()  # timer calls this function after 3 seconds and closes the window

    timer = fig.canvas.new_timer(
        interval=show_time * 1000
    )  # creating a timer object and setting an interval of 3000 milliseconds
    timer.add_callback(close_event)

    for i, ax in enumerate(axes):
        count += 1
        ax.imshow(images[i], cmap='gray')
        ax.set_title(f'Image {i+1}')
        ax.axis('off')
    timer.start()
    plt.show()







if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algorithm", help="Which algorithm use to train", default='bc')
    parser.add_argument("-s", "--scenarios", help="Scenarios for training", default='SUMO')
    parser.add_argument("-o", "--obs_type", help="Which kind of Observations for training", default='purestate')
    args = parser.parse_args()

    algo = args.algorithm
    scenarios = args.scenarios
    obs_type = args.obs_type
    train_imitation = False
    num_scenarios = 1

    if train_imitation:
        train(algo, scenarios, obs_type, num_scenarios)
    else:
        test(algo, SEED=19079)
    

    
    

    

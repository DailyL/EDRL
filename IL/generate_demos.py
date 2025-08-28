from imitation.data import rollout, serialize
import numpy as np
from pathlib import Path 
from joblib import Parallel, delayed
import os, sys,  glob
import random
import datasets
import pickle
from imitation.data.types import Trajectory
from metadrive.policy.idm_policy import IDMPolicy
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.train_utils import load_dummy_model
from utils.train_utils import CosineScheduler,linear_schedule, load_train_model,load_config,load_bc_policy

from env.make_env import make_vec_test_env_il,make_vec_test_env, make_vec_train_bc_env, make_vec_train_il_env, make_bc_env_purestate
from imitation.data.huggingface_utils import TrajectoryDatasetSequence, trajectories_to_dict
from IL_utils import collect_traj

def sample_expert_transitions(expert, env, rng, min_episodes = 1):
    print("Sampling expert transitions.")
    rollouts = rollout.rollout(
        expert,
        env,
        rollout.make_sample_until(min_timesteps=None, min_episodes=min_episodes),
        rng=rng,
        unwrap=False,
    )
    return (rollouts)

def save(save_path, trajs):
    with open(f'{save_path}', 'wb') as f:
        pickle.dump(trajs, f)

def load(save_path):
    with open(f'{save_path}', 'rb') as f:
        trajs = pickle.load(f)
    return trajs
def load_single_file(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def save_npy(save_path, trajs):
    np.save(save_path, trajs, allow_pickle=True)


def load_npy(save_path):
    try:
        data = np.load(save_path, allow_pickle=True).tolist()
        return data
    except EOFError:
        print(f"Error: File {save_path} is corrupted or incomplete.")
        return None

def load_traj_multi_npy(trainer_name, obs_type="purestate",):

    rollouts_path = Path(f"../IL_rollouts/{obs_type}/{trainer_name}")
    pattern = "*.npy"

    rollouts_names =  glob.glob(os.path.join(rollouts_path, pattern))

    transitions = load_multi_npy(rollouts_names)


    print(rollout.rollout_stats(trajectories=transitions))

    return transitions


def load_multi_npy(save_paths, num_files=3):
    parallel = False
    combined_data = []
    print(save_paths)
    if parallel: 
        # Randomly choose 'num_files' files from the list
    
        # Load data in parallel
        data_list = Parallel(n_jobs=-1)(delayed(load_single_file)(fp) for fp in save_paths)
        print("Load data done!")
        
        for data in data_list:
            if isinstance(data, list):
                combined_data.extend(data)
            else:
                combined_data.append(data)
    else:
        for filename in save_paths:
            data = load_npy(filename)
            if isinstance(data, list):
                combined_data.extend(data)
            else:
                combined_data.append(data)  
                    
    return combined_data

def load_multi(save_paths, num_files=2):

    parallel = False
    combined_data = []
    selected_paths = random.sample(save_paths, min(num_files, len(save_paths)))


    if parallel: 
        # Randomly choose 'num_files' files from the list
    
        # Load data in parallel
        data_list = Parallel(n_jobs=-1)(delayed(load_single_file)(fp) for fp in selected_paths)
        print("Load data done!")
        
        for data in data_list:
            if isinstance(data, list):
                combined_data.extend(data)
            else:
                combined_data.append(data)
    else:
        for filename in selected_paths:
            if os.path.getsize(filename) > 0:
                    try:
                        with open(filename, 'rb') as file:
                            data = pickle.load(file)
                            combined_data.extend(data)
                    except EOFError:
                        print(f"Error: The file {filename} is incomplete or corrupted.")
            else:
                print(f"The file '{filename}' is empty.")

    return combined_data

def generate_and_save(expert_name, scenario_name, expert, trainer_name, min_episodes=100):
    """

    Args:
        expert_name : default IDM (teacher policy)

        scenario_name : SUMO or the name of datasets(real)

        expert : Can be any of the following:
            1) A stable_baselines3 policy or algorithm trained on the gym environment.
            2) A Callable that takes an ndarray of observations and returns an ndarray
            of corresponding actions.
            3) None, in which case actions will be sampled randomly.

        trainer_name : name of the traner, to solve the dict obs problem in imitation
                       for BC, Dict obs is ok
                       for GAIL and AIRL, flatten obs into array first

        min_episodes : minimal episodes
    """

    rollouts_name = f"rollouts_{expert_name}_{scenario_name}.pkl"
    rollouts_path = Path(f"../IL_rollouts/{trainer_name}/{min_episodes}_{rollouts_name}")

    rng = np.random.default_rng(0)

    transitions = sample_expert_transitions(expert, expert.get_env(), rng, min_episodes=min_episodes)

    save(rollouts_path,trajs=transitions)

    return rollout.rollout_stats(trajectories=transitions)


def load_traj(expert_name, scenario_name, trainer_name, obs_type, num_scenarios):

    rollouts_name = f"rollouts_{expert_name}_{scenario_name}.pkl"
    rollouts_path = Path(f"../IL_rollouts/{obs_type}/{trainer_name}/{num_scenarios}_{rollouts_name}")

    transitions = load(rollouts_path)

    return transitions

def load_traj_multi(trainer_name, obs_type="purestate"):

    rollouts_path = Path(f"../IL_rollouts/{obs_type}/{trainer_name}")
    pattern = "*.pkl"

    rollouts_names =  glob.glob(os.path.join(rollouts_path, pattern))
    print(rollouts_names)
    transitions = load_multi(rollouts_names)


    print(rollout.rollout_stats(trajectories=transitions))

    return transitions

def load_traj_env(obs_type, trainer, collect=False):

    if obs_type == "bevs":
        if trainer =='bc':
            return make_vec_train_bc_env(idm=collect)
        elif trainer == 'gail' or trainer == 'airl':
            return make_vec_train_il_env(idm=collect)
        else:
            raise NotImplementedError("This algorithm is not configured.")
    elif obs_type == "purestate":
        return make_bc_env_purestate(idm=collect)
    else:
        raise NotImplementedError("Observation Type Unknown.") 


def generate_and_save_idm(min_episode = 1):

    env = make_vec_test_env(idm=True)
    
    env.reset()
    for _ in range(min_episode):
        obs, _ = env.reset()
        try:
            while True:

                obs, reward, termination, truncate, info = env.step([0, 3])
                env.render(mode="topdown", 
                        window=True,
                        screen_record=True,
                        screen_size=(700, 870),
                        )
                if termination:
                    break
        finally:
            env.close()

def generate_and_save_no_experts(env, trainer_name, scenario_name, obs_type, min_episodes = 1):

    SEED = random.randint(0,20000)
    save_traj_npy = False
    rng = np.random.default_rng(0)

    transitions = collect_traj(
        venv=env, 
        sample_until=rollout.make_sample_until(min_timesteps=None, min_episodes=min_episodes),
        rng=rng,
        )

    if save_traj_npy:
        rollouts_name = f"rollouts_{expert_name}_{scenario_name}_{SEED}.npy"
        rollouts_path = Path(f"../IL_rollouts/{obs_type}/{trainer_name}/{min_episodes}_{rollouts_name}")
        save_npy(rollouts_path,trajs=transitions)
    else: 
        rollouts_name = f"rollouts_{expert_name}_{scenario_name}_{SEED}.pkl"
        rollouts_path = Path(f"../IL_rollouts/{obs_type}/{trainer_name}/{min_episodes}_{rollouts_name}")
        save(rollouts_path,trajs=transitions)

    print(rollout.rollout_stats(trajectories=transitions))
    
    

if __name__ == "__main__":

    save_demo = True
    expert_name = "IDM"
    demo_name = "SUMO"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algorithm", help="Which algorithm use to train", default='bc')
    parser.add_argument("-n", "--num_episode", help="How may episodes", default=1)
    parser.add_argument("-o", "--obs_type", help="Which kind of Observations for training", default='purestate')

    args = parser.parse_args()
    trainer = args.algorithm
    num_episode = args.num_episode
    obs_type = args.obs_type

    
    env = load_traj_env(obs_type, trainer, collect=True)


    if save_demo:
        if expert_name == "IDM":
            generate_and_save_no_experts(
                env=env,
                trainer_name=trainer,
                scenario_name=demo_name,
                obs_type=obs_type,
                min_episodes=num_episode,
                )
        else:
            expert = load_dummy_model(algo='ppo', trainer=trainer, env=env)
            generate_and_save(
                expert_name=expert_name, 
                scenario_name=demo_name,
                expert=expert, 
                trainer_name=trainer,
                obs_type = obs_type,
                min_episodes=num_episode,
                )        
    else:
        transitions = load_traj(
            expert_name=expert_name, 
            scenario_name=demo_name,
            trainer_name=trainer,
            obs_type=obs_type,
            num_scenarios=num_episode,
            )



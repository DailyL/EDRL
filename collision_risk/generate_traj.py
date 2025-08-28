import numpy as np
from pathlib import Path 
from joblib import Parallel, delayed
import os, sys,  glob
import random
import pickle
from metadrive.policy.idm_policy import IDMPolicy
import argparse
import cv2
import csv
import json
import math
import torch
import pathlib

from metadrive.engine.asset_loader import AssetLoader
from metadrive.scenario.utils import read_dataset_summary,read_scenario_data


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.make_env import make_wale_env, make_wale_real_env
from metadrive.envs.top_down_env import TopDownSingleFrameMetaDriveEnv
from cr_utils.geometry import transform_trajectories, point_in_rectangle
from cr_utils.data_utils import CRDataset




def save_timestep_to_csv(csv_file_path, timestep: int, float_value: float, list1: list, list2: list):
    # Open the CSV file in append mode
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestep, float_value, json.dumps(list1), json.dumps(list2)])


def save_timestep(data, timestep: int, float_value: float, list1: list, list2: list):
    data.append({
        'timestep': timestep,
        'ego_heading': float_value,
        'ego_pose': list1,
        'nbrs': list2
    })

def load_scenarios(parent_path, dataset_lists):

    len_datasets = len(dataset_lists)
    seed = random.randint(0, len_datasets-1)
    dataset = dataset_lists[seed]

    path = pathlib.Path(AssetLoader.file_path(AssetLoader.asset_path, f'{parent_path}/scenario-nuplan-{dataset}', unix_style=False))
    summary, scenario_ids, mapping = read_dataset_summary(path)
    len_scenarios = len(scenario_ids)

    return len_scenarios, path, mapping


def collect_traj(env, min_episodes, max_steps):

    SEED = random.randint(0,20000)

    trajs_folder = f"collision_risk/data/raw_data/{SEED}"
    os.makedirs(trajs_folder, exist_ok=True) 


    for k in range (min_episodes):
        done = False
        obs = env.reset()
        traj = []
        trajs_path = f"{trajs_folder}/episode_{k}/"
        
        os.makedirs(trajs_path, exist_ok=False)
        for i in range(max_steps):
            obs, reward, done, _,  info = env.step([0,0])
            cv2.imshow('obs',obs)
            env.render(mode="top_down", 
                       window=True,
                       screen_record=False, 
                       screen_size=(500, 400))
            
            cv2.imwrite(f"{trajs_path}{SEED}_episode_{k}_{i}.png", obs)
            #ego_pose, ego_heading, nbrs = (env.wrap_info())

            #save_timestep(traj, i, ego_heading, ego_pose, nbrs)
            
            if done:
                with open(f"{trajs_path}raw_traj.pkl", 'wb') as f:
                    pickle.dump(traj, f)
                break

import matplotlib.pyplot as plt
def draw_multi_channels_top_down_observation(obs, show_time=4):
    num_channels = obs.shape[-1]
    assert num_channels == 5
    channel_names = [
        "Road and navigation", "Ego now and previous pos", "Neighbor at step t", "Neighbor at step t-1",
        "Neighbor at step t-2"
    ]
    fig, axs = plt.subplots(1, num_channels, figsize=(15, 4), dpi=80)
    count = 0

    def close_event():
        plt.close()  # timer calls this function after 3 seconds and closes the window

    timer = fig.canvas.new_timer(
        interval=show_time * 1000
    )  # creating a timer object and setting an interval of 3000 milliseconds
    timer.add_callback(close_event)

    for i, name in enumerate(channel_names):
        count += 1
        ax = axs[i]
        ax.imshow(obs[..., i], cmap="bone")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(name)
        # print("Drawing {}-th semantic map!".format(count))
    fig.suptitle("Multi-channels Top-down Observation")
    timer.start()
    plt.show()

def collect_traj_nuplan(env, min_episodes, max_steps):


    SEED = random.randint(0,20000)

    trajs_folder = f"collision_risk/data/raw_data/{SEED}"
    os.makedirs(trajs_folder, exist_ok=True)
    print("Collecting Trajectories and saving to: ", trajs_folder)
    env_counter = 0

    test = True

    if test:
        dataset_lists = ["boston", "pittsburgh"]
        dataset_parent_path = '/home/dianzhaoli/mdsn/dataset'
    else:
        dataset_lists = ["boston", "pittsburgh", "singapore", "vegas_1", "vegas_2", "vegas_3", "vegas_4"]
        dataset_parent_path = '/data/horse/ws/dili862d-scenarionet/mdsn/' 

    len_scenarios, data_path, mapping = load_scenarios(parent_path = dataset_parent_path, 
                                            dataset_lists= dataset_lists,
                                            )
    
    # Initilize the environment with the first scenario
    for file_path, file_name in mapping.items():

        full_path = pathlib.Path(f'{data_path}/{file_name}/{file_path}')
        assert full_path.exists(), f"{full_path} does not exist"
        scenario_description = read_scenario_data(full_path)
        print("Running scenario: ", scenario_description["id"])
        env.set_scenario(scenario_description)
        break



    for k in range (min_episodes):

        if env_counter >= 500 or env_counter > len_scenarios:
            env_counter = 0
            len_scenarios, data_path, mapping = load_scenarios(parent_path = dataset_parent_path, 
                                            dataset_lists= dataset_lists)
        

        done = False
        obs = env.reset()
        traj = []
        trajs_path = f"{trajs_folder}/episode_{k}/"
        
        os.makedirs(trajs_path, exist_ok=False)
        for i in range(max_steps):
            obs, reward, done, _,  info = env.step([0,0])

            env.render(mode="top_down")
            cv2.imwrite(f"{trajs_path}{SEED}_episode_{k}_{i}.png", obs)
            ego_pose, ego_heading, nbrs = (env.wrap_info())

            save_timestep(traj, i, ego_heading, ego_pose, nbrs)
            
            if done:
                with open(f"{trajs_path}raw_traj.pkl", 'wb') as f:
                    pickle.dump(traj, f)

                # Reset Scenario
                full_path = pathlib.Path(f'{data_path}/{list(mapping.items())[env_counter][1]}/{list(mapping.items())[env_counter][0]}')
                scenario_description = read_scenario_data(full_path)
                env.set_scenario(scenario_description)
                env_counter += 1
                break
 
from env.make_env import env_config, vehicle_cfg, SCENARIO_ENV_CONFIG, make_wale_real_env
from env.state_env import Wale_RealEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.obs.top_down_obs import TopDownObservation

def reset_env(data_source, idm=True):

    test = True

    if test:
        dataset_lists = ["boston", "pittsburgh"]
        dataset_parent_path = '/home/dianzhaoli/mdsn/dataset'
    else:
        if data_source == 'nuplan':
            dataset_lists = ["vegas_5", "vegas_6"]
            dataset_parent_path = '/data/horse/ws/dili862d-scenarionet/mdsn/' 
        elif data_source == 'waymo':
            dataset_lists = ["training", "training_20s"]
            dataset_parent_path = '/data/horse/ws/dili862d-scenarionet/mdsn/' 

    len_datasets = len(dataset_lists)
    seed = random.randint(0, len_datasets-1)
    dataset = dataset_lists[seed]
    
    data_directory = AssetLoader.file_path(AssetLoader.asset_path, f'{dataset_parent_path}/scenario-{data_source}-{dataset}', unix_style=False)

    SCENARIO_ENV_CONFIG['data_directory'] = data_directory
    SCENARIO_ENV_CONFIG['sequential_seed'] = False


    cfg = env_config

    cfg.update(
        SCENARIO_ENV_CONFIG
        )
    
    cfg.update(
        vehicle_cfg
    )
    if idm:
        cfg.update(
            dict(
                agent_policy=ReplayEgoCarPolicy
            )
        )
    cfg.update(
        {
            "agent_observation": TopDownObservation,
        }
    )

    return Wale_RealEnv(cfg)



def collect_traj_real(min_episodes, max_steps, data_source):

    SEED = random.randint(0,20000)

    trajs_folder = f"collision_risk/data/raw_data/{SEED}"
    os.makedirs(trajs_folder, exist_ok=True)
    print("Collecting Trajectories and saving to: ", trajs_folder)

    env_counter = 0
    max_env_counter = 10 #When to reset the whole env, tochange to other dataset

    env = reset_env(data_source=data_source)


    for k in range (min_episodes):

        if env_counter >= max_env_counter:
            env.close()
            env = reset_env(data_source=data_source)
            env_counter = 0
        done = False
        obs = env.reset()
        traj = []
        trajs_path = f"{trajs_folder}/episode_{k}/"
        
        os.makedirs(trajs_path, exist_ok=False)
        for i in range(max_steps):
            obs, reward, done, _,  info = env.step([0,0])

            env.render(mode="top_down")
            cv2.imwrite(f"{trajs_path}{SEED}_episode_{k}_{i}.png", obs)
            ego_pose, ego_heading, nbrs = (env.wrap_info())

            save_timestep(traj, i, ego_heading, ego_pose, nbrs)
            
            if done:
                with open(f"{trajs_path}raw_traj.pkl", 'wb') as f:
                    pickle.dump(traj, f)
                env_counter += 1
                break

    preprocess_traj(
            args=data_args,
            trajs_path=trajs_folder
        )
            
            



def preprocess_traj(args, trajs_path):

    folder = trajs_path

    pp = args["past_points"]  # past points
    fp = args["future_points"]  # future points

    read_trajs_data(trajs_path, args)


def read_trajs_data(folder, args):

    subfolders = [ f.path for f in os.scandir(folder) if f.is_dir()]
    subfolders = sorted(subfolders)
    trajs_path = []

    for subfolder in subfolders:
        trajs_path = [os.path.join(subfolder, f) for f in os.listdir(subfolder) if f.endswith("raw_traj.pkl")]  # .pkl path
        paths = Path(trajs_path[0])
        parent_directories = list(paths.parents)
        process_raw_traj(trajs_path[0], args, episode_num = str(parent_directories[1].name) + "_" +str(parent_directories[0].name))




def process_raw_traj(path, args, episode_num):
    folder = os.path.dirname(path)
    pp = args["past_points"]  # past points
    fp = args["future_points"]  # future points
    hist_list = []
    fut_list = []
    nbrs_list = []
    id_list = []
    sliwi_size = args["sliwi_size"]
    shrink_percentage = args["shrink_percentage"]

    keys = ['id', 'hist', 'fut', 'nbrs']

    with open(path, 'rb') as file:
        raw_trajs = pickle.load(file)

        ego_pos = []
        ego_heading = []
        nbrs = []

        for i in range(len(raw_trajs)):
            ego_pos.append(raw_trajs[i]["ego_pose"])
            ego_heading.append(raw_trajs[i]["ego_heading"])
            nbrs.append(raw_trajs[i]["nbrs"])

        if len(ego_pos) <= (fp + 1):
            return
        if len(nbrs) >= (pp):
            nbrs = nbrs[pp:]   # remove the nbrs of first 30 steps, because they are not useful

        # Here for each data point (his, nbrs, futs)
        for time_step in range(
            0, len(ego_pos) - fp, sliwi_size
        ):
            # Generate history, future, and nbrs
            hist = []
            agg_nbrs = []

            for i in reversed(range(pp)):
                if time_step - i >= 0:
                    hist.append(np.array(ego_pos[time_step - i]))
                    if nbrs[time_step - i]:  # Check if there are neighbors at this step
                        agg_nbrs.append(np.array(nbrs[time_step - i]))  # Append all neighbors at this step
                    else:
                        agg_nbrs.append([np.nan, np.nan])  # Handle missing neighbors with NaN
                    
                else:
                    hist.append([np.nan, np.nan])
                    agg_nbrs.append([np.nan, np.nan])
                    

            translation = hist[-1]
            rotation = ego_heading[time_step]

            # Adapt rotation if heading is based on the y-axis
            #rotation -= math.pi / 2

            agg_nbrs = transform_trajectories(agg_nbrs, translation, rotation)
            hist = transform_trajectories([hist], translation, rotation)   # Transform the history poses according to the last pose (for relative position)

            fut = ego_pos[time_step + 1 : time_step + fp + 1]

            fut = transform_trajectories([fut], translation, rotation)
            
            
            # Generate neighbor array
            nbrs_array, pir_list, r1, r2 = generate_nbrs_array(agg_nbrs)

            # All NaN to zeros
            hist = np.array(hist)
            fut = np.array(fut)

            hist[np.isnan(hist)] = 0
            fut[np.isnan(fut)] = 0
            nbrs_array[np.isnan(nbrs_array)] = 0

            id = f"{episode_num}_{time_step}"



            hist_list.append(hist)
            nbrs_list.append(nbrs_array)
            fut_list.append(fut)
            id_list.append(id)
    
    trajs = dict(zip(keys, [id_list, hist_list, fut_list, nbrs_list]))

    

    with open(f"{folder}/processed_traj.pkl", 'wb') as f:
        pickle.dump(trajs, f)

    
def generate_nbrs_array(nbrs_list, pp=30, window_size=[18, 78]):
    

    # Define window to identify neihbors
    r1 = [int(-i / 2) for i in window_size]  # [-9, -39]
    r2 = [int(i / 2) for i in window_size]
    nbrs = np.zeros((3, 13, pp, 2))
    pir_list = []

    for nbr in nbrs_list:
        if not np.isnan(nbr.all()):
            pir = point_in_rectangle(r1, r2, nbr)
            if pir:
                nbrs[pir] = nbr
                pir_list.append(pir)
        else:
            pass    
    return nbrs, pir_list, r1, r2
    




if __name__ == "__main__":

    #env = make_wale_env(idm=True)
    #env = make_wale_real_env(idm=True)

    data_args = {
    "past_points": 30,  # number of past points as network input
    "future_points": 40,  # number of future points as ground truth for prediction
    "dpi": 300,  # dpi of rendered image
    "resolution": 256,  # resolution of the rendered image
    "watch_radius": 64,  # radius in m covered by scene image
    "exclude_handcrafted": True,  # exlcude handcrafted scenes
    "sliwi_size": 1,  # size of sliding window in dataset generation
    "shrink_percentage": 0.5,  # percentage of artificially added shrinked (with past points < "past_points") samples
    }

    collect = True
    collect_real_scenario = True
    
    data_source = 'nuplan'
    
    
    if collect_real_scenario:
        collect_traj_real(
            min_episodes=10,
            max_steps=5000,
            data_source=data_source,
        )
    else:
        collect_traj(
            env=env,
            min_episodes=10,
            max_steps=2000,
        )
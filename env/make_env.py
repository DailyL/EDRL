import random
import logging
import os
import sys
from metadrive import TopDownMetaDrive
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.envs import BaseEnv
from metadrive.obs.observation_base import DummyObservation
from metadrive.obs.state_obs import StateObservation, LidarStateObservation
from metadrive.obs.top_down_obs import TopDownObservation
from metadrive.obs.top_down_obs_multi_channel import TopDownMultiChannel
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.policy.expert_policy import ExpertPolicy
from metadrive.utils import print_source, CONFIG

from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.component.sensors.semantic_camera import SemanticCamera
from metadrive.component.sensors.depth_camera import DepthCamera
from env.base_env import StateLidarEnv
from env.state_env import PureStateEnv, StateBEVEnv, StateBEVEnv_wrapped, Obs_Flatten, WaleEnv, Wale_RealEnv, PureStateEnv_TUDRL, PureStateEnv_SRL, Traj_Nav_Env_TUDRL, Traj_Nav_Env_SRL
from env.util.obs_setter import PureStateObservation,StateBEVObservation, WaleObservation, PureStateObservation_woHist,PureStateObservation_woHist,State_with_pede_woHist
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from IL.wrapper import RolloutInfoWrapper
from imitation.data.wrappers import BufferingWrapper


from utils.train_utils import load_config, load_config_real


env_config = load_config("env")
real_env_config = load_config_real("env")
vehicle_cfg = load_config("vehicle")
make_dummy_model = False   # Use for making dummy model for the training of imitation library (GAIL, AIRL) 


def make_statebev_env():
    cfg = env_config
    env = StateBEVEnv(cfg)
    return env

def make_wale_env(idm):

    cfg = env_config
    
    if idm:
        cfg.update(
            dict(
                agent_policy=IDMPolicy
            )
        )
    
    cfg.update(
        vehicle_cfg
    )
    cfg.update(
        {
            "agent_observation": TopDownObservation,
        }
    )
    env = WaleEnv(cfg)
    
    return env

def make_vec_env_purestate(log_dir):
    cfg = env_config
    cfg.update(
        vehicle_cfg
    )
    cfg.update(
        {
            "agent_observation": PureStateObservation
        }
    )
    env = PureStateEnv(cfg) 
    check_env(env,warn=True)
    env = Monitor(env, log_dir)
    env = DummyVecEnv([lambda: env])
    return env


def make_tudrl_purestate():
    cfg = env_config
    cfg.update(
        vehicle_cfg
    )
    cfg.update(
        {
            "agent_observation": PureStateObservation_woHist
        }
    )
    env = Traj_Nav_Env_TUDRL(cfg) 
    return env


def make_srl_purestate(cost_fn_mode):
    cfg = env_config
    cfg.update(
        vehicle_cfg
    )
    cfg.update(
        {
            "agent_observation": PureStateObservation_woHist
        }
    )
    #env = PureStateEnv_TUDRL(cfg) 
    env = Traj_Nav_Env_SRL(cfg) 
    env.set_cost_fn_mode(cost_fn_mode)
    return env


def make_srl_env():
    cfg = env_config
    cfg.update(
        vehicle_cfg
    )
    cfg.update(
        {
            "agent_observation": PureStateObservation
        }
    )
    env = PureStateEnv_SRL(cfg) 
    return env


def make_bc_env_purestate(idm):

    cfg = env_config
    
    if idm:
        cfg.update(
            dict(
                agent_policy=IDMPolicy
            )
        )
    
    cfg.update(
        vehicle_cfg
    )
    cfg.update(
        {
            "agent_observation": PureStateObservation,
        }
    )
    env = PureStateEnv(cfg)
    check_env(env,warn=True)
    Monitor(env)
    env = DummyVecEnv([lambda: (env)])
    env = BufferingWrapper(env)

    return (env)


def make_vec_test_env_purestate(idm):

    cfg = env_config
    
    if idm:
        cfg.update(
            dict(
                agent_policy=IDMPolicy
            )
        )
    
    cfg.update(
        vehicle_cfg
    )
    cfg.update(
        {
            "agent_observation": PureStateObservation,
        }
    )
    env = PureStateEnv(cfg)
    env = DummyVecEnv([lambda: (env)])
    
    return env



def make_vec_env(log_dir):
    cfg = env_config
    cfg.update(
        vehicle_cfg
    )
    cfg.update(
        {
            "agent_observation": StateBEVObservation
        }
    )
    env = StateBEVEnv(cfg)
    if make_dummy_model:
        env = Obs_Flatten(env)
    
    check_env(env,warn=True)
    env = Monitor(env, log_dir)
    env = DummyVecEnv([lambda: env])
    
    return env


def make_vec_test_env(idm):

    cfg = env_config
    
    if idm:
        cfg.update(
            dict(
                agent_policy=IDMPolicy
            )
        )
    
    cfg.update(
        vehicle_cfg
    )
    cfg.update(
        {
            "agent_observation": StateBEVObservation,
        }
    )
    env = StateBEVEnv(cfg)
    
    return env


def make_vec_test_env_il(idm):

    cfg = env_config
    
    if idm:
        cfg.update(
            dict(
                agent_policy=IDMPolicy
            )
        )
    
    cfg.update(
        vehicle_cfg
    )
    cfg.update(
        {
            "agent_observation": StateBEVObservation,
        }
    )
    env = StateBEVEnv(cfg)
    check_env(env,warn=True)
    Monitor(env)
    env = DummyVecEnv([lambda: (env)])
    env = BufferingWrapper(env)

    return (env)


def make_vec_train_bc_env(idm):

    cfg = env_config
    
    if idm:
        cfg.update(
            dict(
                agent_policy=IDMPolicy
            )
        )
    
    cfg.update(
        vehicle_cfg
    )
    cfg.update(
        {
            "agent_observation": StateBEVObservation,
        }
    )
    env = StateBEVEnv(cfg)
    check_env(env,warn=True)
    Monitor(env)
    env = DummyVecEnv([lambda: (env)])
    env = BufferingWrapper(env)

    return (env)


def make_vec_train_il_env(idm):

    cfg = env_config
    
    if idm:
        cfg.update(
            dict(
                agent_policy=IDMPolicy
            )
        )

    cfg.update(
        vehicle_cfg
    )
    cfg.update(
        {
            "agent_observation": StateBEVObservation,
        }
    )
    env = StateBEVEnv(cfg)
    env = Obs_Flatten(env)
    check_env(env,warn=True)
    Monitor(env)
    env = DummyVecEnv([lambda: (env)])
    env = BufferingWrapper(env)

    return env





def make_vec_test_il_env(idm):

    cfg = env_config
    
    if idm:
        cfg.update(
            dict(
                agent_policy=IDMPolicy
            )
        )

    cfg.update(
        vehicle_cfg
    )
    cfg.update(
        {
            "agent_observation": StateBEVObservation,
        }
    )
    env = StateBEVEnv(cfg)
    env = Obs_Flatten(env)
    env = DummyVecEnv([lambda: (env)])

    return env


from metadrive.engine.asset_loader import AssetLoader 
from env.state_env import RealTrafficEnv, RealTrafficNoCostEnv
from metadrive.component.navigation_module.node_network_navigation import NodeNetworkNavigation

data_directory = AssetLoader.file_path("/home/dianzhaoli/EDRL/filtered_dataset/evaluation_2/", unix_style=False)

SCENARIO_ENV_CONFIG = dict(
    # ===== Scenario Config =====
    data_directory=data_directory,
    start_scenario_index=0,
    num_scenarios=6,
    sequential_seed=True,  # Whether to set seed (the index of map) sequentially across episodes
    worker_index=0,  # Allowing multi-worker sampling with Rllib
    num_workers=1,  # Allowing multi-worker sampling with Rllib

    # ===== Curriculum Config =====
    curriculum_level=1,  # i.e. set to 5 to split the data into 5 difficulty level
    episodes_to_evaluate_curriculum=None,
    target_success_rate=0.8,

    # ===== Map Config =====
    store_map=True,
    store_data=False,
    need_lane_localization=True,
    no_map=False,
    map_region_size=1024,
    cull_lanes_outside_map=True,

    # ===== Scenario =====
    no_traffic=False,  # nothing will be generated including objects/pedestrian/vehicles
    no_static_vehicles=False,  # static vehicle will be removed
    no_light=False,  # no traffic light
    reactive_traffic=True,  # turn on to enable idm traffic
    filter_overlapping_car=True,  # If in one frame a traffic vehicle collides with ego car, it won't be created.
    default_vehicle_in_traffic=False,
    skip_missing_light=True,
    static_traffic_object=True,
    show_sidewalk=True,
)


def make_real_env(cost_fn_mode):

    cfg = real_env_config

    cfg.update(
        vehicle_cfg
    )
    cfg.update(
        SCENARIO_ENV_CONFIG
        )
    
    cfg.update(
        {
            "agent_observation": State_with_pede_woHist
        }
    )

    env = RealTrafficEnv(cfg)
    env.set_cost_fn_mode(cost_fn_mode)

    return env

def make_real_no_cost_env(cost_fn_mode):

    cfg = real_env_config

    cfg.update(
        vehicle_cfg
    )
    cfg.update(
        SCENARIO_ENV_CONFIG
        )
    
    cfg.update(
        {
            "agent_observation": State_with_pede_woHist
        }
    )

    env = RealTrafficNoCostEnv(cfg)
    env.set_cost_fn_mode(cost_fn_mode)

    return env


from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.envs.scenario_env import ScenarioOnlineEnv

def make_wale_real_env(idm):

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
    env = Wale_RealEnv(cfg)
    
    return env









def make_env():
    env = TopDownMetaDrive(
        dict(
            # We also support using two renderer (Panda3D renderer and Pygame renderer) simultaneously. You can
            # try this by uncommenting next line.
            # use_render=True,

            # You can also try to uncomment next line with "use_render=True", so that you can control the ego vehicle
            # with keyboard in the main window.
            # manual_control=True,
            map="SSSS",
            traffic_density=0.1,
            num_scenarios=100,
            start_seed=random.randint(0, 1000),
            agent_observation=StateObservation,
        )
    )

    return env



    


from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.component.sensors.semantic_camera import SemanticCamera
from metadrive.component.sensors.depth_camera import DepthCamera
import cv2
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.obs.observation_base import BaseObservation
from metadrive.obs.image_obs import ImageObservation
from metadrive.obs.state_obs import StateObservation, LidarStateObservation
from metadrive.obs.top_down_obs import TopDownObservation
from metadrive.obs.top_down_obs_multi_channel import TopDownMultiChannel
import os



class PureStateObservation(BaseObservation):
    """
    Pure State Observation
    Ego state + Navi info + Other vehicles info
      9 or 11   +   10     +     4*nums  + 4*nums(past)
    """
    def __init__(self, config):
        super(PureStateObservation, self).__init__(config)
        self.state = LidarStateObservation(config)
        self.enable_lidar = False
        self.num_others = self.config["vehicle_config"]["lidar"]["num_others"]
        self.past_nebrs_obs = np.zeros(self.num_others*4, dtype=np.float32)
    @property
    def observation_space(self):
        shape = list(self.state.observation_space.shape)
        if self.enable_lidar:
            shape[0] -= self.num_others*4
        else:
            if self.config["vehicle_config"]["lidar"]["num_lasers"] > 0:
                lidar_dim = self.config["vehicle_config"]["lidar"]["num_lasers"]
                shape[0] -= lidar_dim
                shape[0] += self.num_others*4
        return gym.spaces.Box(-0.0, 1.0, shape=tuple(shape), dtype=np.float32)

    def observe(self, vehicle):
        if self.enable_lidar:
            obs = self.state.observe(vehicle)
            obs = obs[self.num_others*4:]
            return obs
        
        else:
            if self.config["vehicle_config"]["lidar"]["num_lasers"] > 0:
                N = vehicle.config["lidar"]["num_lasers"]
                obs = self.state.observe(vehicle)
                obs = obs[:-N]
                past_obs = obs[-self.num_others*4:]
                obs = np.concatenate([obs,self.past_nebrs_obs])
                self.past_nebrs_obs = past_obs
                return obs

            else:
                return self.state.observe(vehicle)

class PureStateObservation_woHist(BaseObservation):
    """
    Pure State Observation
    Ego state + Navi info + Other vehicles info
      9 or 11   +   10     +     4*nums 
    """
    def __init__(self, config):
        super(PureStateObservation_woHist, self).__init__(config)
        self.state = LidarStateObservation(config)
        self.enable_lidar = False
        self.num_others = self.config["vehicle_config"]["lidar"]["num_others"]
    @property
    def observation_space(self):
        shape = list(self.state.observation_space.shape)
        if self.enable_lidar:
            shape[0] -= self.num_others*4
        else:
            if self.config["vehicle_config"]["lidar"]["num_lasers"] > 0:
                lidar_dim = self.config["vehicle_config"]["lidar"]["num_lasers"]
                shape[0] -= lidar_dim
        return gym.spaces.Box(-0.0, 1.0, shape=tuple(shape), dtype=np.float32)

    def observe(self, vehicle):
        if self.enable_lidar:
            obs = self.state.observe(vehicle)
            obs = obs[self.num_others*4:]
            return obs
        
        else:
            if self.config["vehicle_config"]["lidar"]["num_lasers"] > 0:
                N = vehicle.config["lidar"]["num_lasers"]
                obs = self.state.observe(vehicle)
                obs = obs[:-N]
                return obs

            else:
                return self.state.observe(vehicle)


from metadrive.utils.math import norm, clip
from metadrive.component.traffic_participants.pedestrian import Pedestrian 
from typing import Set


class State_with_pede_woHist(BaseObservation):
    """
    Pure State Observation
    Ego state + Navi info + Other vehicles info
      9 or 11   +   10     +     4*nums  + 2 * nums(pedestrians)
    """
    def __init__(self, config):
        super(State_with_pede_woHist, self).__init__(config)
        self.state = LidarStateObservation(config)
        self.enable_lidar = False
        self.detect_ped = True
        self.num_others = self.config["vehicle_config"]["lidar"]["num_others"]
        self.num_ped = self.config["vehicle_config"]["lidar"]["num_pedestrians"]

    @property
    def observation_space(self):
        shape = list(self.state.observation_space.shape)
        if self.enable_lidar:
            shape[0] -= self.num_others*4
        else:
            if self.config["vehicle_config"]["lidar"]["num_lasers"] > 0:
                lidar_dim = self.config["vehicle_config"]["lidar"]["num_lasers"]
                shape[0] -= lidar_dim
            if self.detect_ped:
                shape[0] += self.num_ped*2
        return gym.spaces.Box(-0.0, 1.0, shape=tuple(shape), dtype=np.float32)

    def observe(self, vehicle):
        if self.enable_lidar:
            obs = self.state.observe(vehicle)
            obs = obs[self.num_others*4:]
            return obs
        
        else:
            if self.config["vehicle_config"]["lidar"]["num_lasers"] > 0:
                N = vehicle.config["lidar"]["num_lasers"]
                obs = self.state.observe(vehicle)
                obs = obs[:-N]
            else:
                obs = self.state.observe(vehicle)
            if self.detect_ped:
                ped_obs = self.obs_surrounding_objects(vehicle)
                obs = np.concatenate([obs, ped_obs])
            return obs


    def obs_surrounding_objects(self, vehicle):

        _, detected_objects = self.engine.get_sensor("lidar").perceive(
                vehicle,
                physics_world=self.engine.physics_world.dynamic_world,
                num_lasers=vehicle.config["lidar"]["num_lasers"],
                distance=vehicle.config["lidar"]["distance"],
                show=vehicle.config["show_lidar"],
            )

        surrounding_pedestrians_info = self.get_walkers_info(
            detected_objects, 
            vehicle, 
            perceive_distance=vehicle.config["lidar"]["distance"], 
            num_others= self.num_ped,
            )


        return surrounding_pedestrians_info
    
    def get_surrounding_walkers(self, detected_objects) -> Set:

        walkers = set()
        objs = detected_objects
        for ret in objs:
            if isinstance(ret, Pedestrian):
                walkers.add(ret)
        return walkers
    
    def get_walkers_info(self, detected_objects, ego_vehicle, perceive_distance, num_others= 4):

        walkers = list(self.get_surrounding_walkers(detected_objects))

        walkers.sort(
            key=lambda v: norm(ego_vehicle.position[0] - v.position[0], ego_vehicle.position[1] - v.position[1])
        )

        if walkers:
            walkers.sort(
                key=lambda v: norm(ego_vehicle.position[0] - v.position[0], ego_vehicle.position[1] - v.position[1])
            )

        walkers += [None] * num_others
        res = []

        ego_position = ego_vehicle.position

        for walker in walkers[:num_others]:
            if walker is not None:
                relative_position = ego_vehicle.convert_to_local_coordinates(walker.position, ego_position)
                res.append(clip((relative_position[0] / perceive_distance + 1) / 2, 0.0, 1.0))
                res.append(clip((relative_position[1] / perceive_distance + 1) / 2, 0.0, 1.0))
            else:
                res += [0.0] * 2
        
        return res

class StateBEVObservation(BaseObservation):

    def __init__(self, config):
        super(StateBEVObservation, self).__init__(config)
        self.state = PureStateObservation(config)
        self.bev = TopDownMultiChannel(
            self.config["vehicle_config"],
            onscreen=self.config["use_render"],
            clip_rgb=self.config["norm_pixel"],
            frame_stack=self.config["frame_stack"],
            post_stack=self.config["post_stack"],
            frame_skip=self.config["frame_skip"],
            resolution=(self.config["resolution_size"], self.config["resolution_size"]),
            max_distance=self.config["distance"])

    @property
    def observation_space(self):
        """
        Combine BEV with state obs
        """

        bev_space = self.bev.observation_space
        state_space = self.state.observation_space

        # Combine them using gym.spaces.Dict
        combined_space = gym.spaces.Dict({
            "bev": bev_space,
            "state": state_space
        })
        return combined_space


    def observe(self, vehicle):

        bev_obs = self.bev.observe(vehicle)
        state_obs = self.state.observe(vehicle)

        return (bev_obs, state_obs)
    
    def render(self):
        self.bev.render()

    def reset(self, env, vehicle):
        self.bev.reset(env,vehicle=vehicle)



class WaleObservation(BaseObservation):

    def __init__(self, config):
        super(WaleObservation, self).__init__(config)
        self.state = PureStateObservation(config)
        self.bev = TopDownObservation(
            self.config["vehicle_config"],
            onscreen=self.config["use_render"],
            clip_rgb=self.config["norm_pixel"],
            frame_stack=self.config["frame_stack"],
            post_stack=self.config["post_stack"],
            frame_skip=self.config["frame_skip"],
            resolution=(self.config["resolution_size"], self.config["resolution_size"]),
            max_distance=self.config["distance"])
    @property
    def observation_space(self):
        """
        Combine BEV with state obs
        """

        bev_space = self.bev.observation_space
        state_space = self.state.observation_space

        # Combine them using gym.spaces.Dict
        combined_space = gym.spaces.Dict({
            "bev": bev_space,
            "state": state_space
        })
        return combined_space


    def observe(self, vehicle):

        bev_obs = self.bev.observe(vehicle)
        state_obs = self.state.observe(vehicle)

        return (bev_obs, state_obs)
    
    def render(self):
        self.bev.render()

    def reset(self, env, vehicle):
        self.bev.reset(env,vehicle=vehicle)










# Functions for unflatten and flatten observations

def unflatten_obs(flat_obs, bev_shape, state_shape):
    """
    Convert a flattened observation back into the original dictionary format.
    """
    bev_size = np.prod(bev_shape)
    state_size = np.prod(state_shape)
    bev_flat = flat_obs[:, :bev_size]
    state_flat = flat_obs[:, bev_size:bev_size + state_size]
    bev = bev_flat.view(-1, *bev_shape)
    state = state_flat.view(-1, *state_shape)
    return {'bev': bev, 'state': state}



def flatten_obs(obs):
    """
    Flatten the observation dictionary into a single NumPy array.
    """
    bev_flat = obs['bev'].flatten()
    state_flat = obs['state'].flatten()
    return np.concatenate([bev_flat, state_flat])
    
def flatten_space(space):
    """
    Flatten the observation space into a single Box space.
    """
    bev_space = space['bev']
    state_space = space['state']
    
    flat_dim = bev_space.shape[0] * bev_space.shape[1] * bev_space.shape[2] + state_space.shape[0]
    return spaces.Box(low=-0.0, high=1.0, shape=(flat_dim,), dtype=np.float32)



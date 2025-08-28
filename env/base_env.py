from metadrive.envs import BaseEnv
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.obs.observation_base import DummyObservation
from metadrive.obs.state_obs import StateObservation, LidarStateObservation
from metadrive.utils import Config
import logging
import numpy as np


class StateLidarEnv(MetaDriveEnv):
    def default_config(self) -> Config:
        config = super(StateLidarEnv, self).default_config()
        return config

    def __init__(self, config = None):
        super(StateLidarEnv, self).__init__(config)

    def reward_function(self, vehicle_id: str):
        vehicle = self.agents[vehicle_id]
        step_info = dict()

        # Reward for moving forward in current lane
        if vehicle.lane in vehicle.navigation.current_ref_lanes:
            current_lane = vehicle.lane
            positive_road = 1
        else:
            current_lane = vehicle.navigation.current_ref_lanes[0]
            current_road = vehicle.navigation.current_road
            positive_road = 1 if not current_road.is_negative_road() else -1
        long_last, _ = current_lane.local_coordinates(vehicle.last_position)
        long_now, lateral_now = current_lane.local_coordinates(vehicle.position)

        # reward for lane keeping, without it vehicle can learn to overtake but fail to keep in lane
        if self.config["use_lateral_reward"]:
            lateral_factor = np.clip(1 - 2 * abs(lateral_now) / vehicle.navigation.get_current_lane_width(), 0.0, 1.0)
        else:
            lateral_factor = 1.0

        reward = 0.0
        reward += self.config["driving_reward"] * (long_now - long_last) * lateral_factor * positive_road
        reward += self.config["speed_reward"] * (vehicle.speed_km_h / vehicle.max_speed_km_h) * positive_road

        step_info["step_reward"] = reward

        if self._is_arrive_destination(vehicle):
            reward = +self.config["success_reward"]
        elif self._is_out_of_road(vehicle):
            reward = -self.config["out_of_road_penalty"]
        elif vehicle.crash_vehicle:
            reward = -self.config["crash_vehicle_penalty"]
        elif vehicle.crash_object:
            reward = -self.config["crash_object_penalty"]

        step_info["route_completion"] = vehicle.navigation.route_completion

        return reward, step_info

    def cost_function(self, vehicle_id: str):
        return 0, {}
    
    def get_single_observation(self):
        o = LidarStateObservation(self.config)
        return o
    



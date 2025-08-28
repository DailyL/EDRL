import logging
import numpy as np
import gymnasium as gym
import math
import random
import time as tm
from gymnasium import spaces
from typing import Union, Dict, AnyStr, Optional, Tuple, Callable, Any, ClassVar, Set
import pygame
from metadrive import TopDownMetaDrive
from metadrive.envs import BaseEnv
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.obs.observation_base import DummyObservation
from metadrive.obs.top_down_obs import TopDownObservation
from metadrive.obs.top_down_obs_multi_channel import TopDownMultiChannel
from metadrive.obs.state_obs import StateObservation, LidarStateObservation
from metadrive.utils import Config
from metadrive.utils.math import wrap_to_pi
from env.util.obs_setter import (PureStateObservation, StateBEVObservation, flatten_obs, flatten_space, 
                                 WaleObservation, PureStateObservation_woHist, State_with_pede_woHist)
from env.util.env_utils import (add_noise_to_trajectory, VectorFiledGuidance, get_surrounding_vehicles, 
                                convert_action_to_target_points, convert_action_to_target_point, 
                                calc_cross_track_error, VehiclePIDController, find_target_lane, 
                                compute_2d_ttc, compute_ttc_components)
from env.util.path_planning import (quintic_polynomials_planner, calc_frenet_trajectories, 
                                    Frenet_State, calc_frenet_trajectory, traj_planner, 
                                    pid_traj_controller)
from imitation.data import rollout, types
from metadrive.examples.top_down_metadrive import draw_multi_channels_top_down_observation
from metadrive.engine.top_down_renderer import draw_top_down_map_native, draw_top_down_trajectory


from collision_risk.collision_probability import check_collisions, get_inv_mahalanobis_dist, get_collision_probability
from collision_risk.predictor import Prediction
from collision_risk.helpers.prediction_helper import (get_orientation_velocity_and_shape_of_prediction,
                                                      get_orientation_velocity_and_shape_of_ego, 
                                                      constant_v_propagation, map_gt_to_prediction)
import torch.autograd.profiler as profiler
from collision_risk.risk_cost import cal_risk
from collision_risk.calc_traj_costs import calc_traj_costs


class WaleEnv(MetaDriveEnv):
    def default_config(self) -> Config:
        config = super(WaleEnv, self).default_config()
        return config

    def __init__(self, config = None):
        super(WaleEnv, self).__init__(config)
    
    def get_single_observation(self):
        o = TopDownObservation(
            self.config,
            onscreen=self.config["use_render"],
            clip_rgb=False,
            )
        return o
    
    def wrap_info(self):

        ego_vehicle = self.vehicle
        engine = self.engine
        ego_vehicle.config["lidar"]["distance"] = 50   # max detection range 39m or 50m
        num_others = ego_vehicle.config["lidar"]["num_others"]

        surrounding_vehicles = get_surrounding_vehicles(ego_vehicle, engine)
        ego_pose = ego_vehicle.position
        ego_heading = ego_vehicle.heading_theta
        

        nbrs = []
        for vehicle in surrounding_vehicles:
            if vehicle is not None:
                nbrs.append(vehicle.position)
        
        return [ego_pose[0], ego_pose[1]], ego_heading, nbrs



class Wale_RealEnv(ScenarioEnv):
    def default_config(self) -> Config:
        config = super(Wale_RealEnv, self).default_config()
        
        return config

    def __init__(self, config = None):
        super(Wale_RealEnv, self).__init__(config)
    
    def get_single_observation(self):
        o = TopDownObservation(
            self.config,
            onscreen=self.config["use_render"],
            clip_rgb=False,
            )
        return o
    
    def wrap_info(self):

        ego_vehicle = self.vehicle
        engine = self.engine
        ego_vehicle.config["lidar"]["distance"] = 50   # max detection range 39m or 50m
        num_others = ego_vehicle.config["lidar"]["num_others"]

        surrounding_vehicles = get_surrounding_vehicles(ego_vehicle, engine)
        ego_pose = ego_vehicle.position
        ego_heading = ego_vehicle.heading_theta
        

        nbrs = []
        for vehicle in surrounding_vehicles:
            if vehicle is not None:
                nbrs.append(vehicle.position)
        
        return [ego_pose[0], ego_pose[1]], ego_heading, nbrs
    
    



import matplotlib.pyplot as plt
class PureStateEnv_TUDRL(MetaDriveEnv):
    def default_config(self) -> Config:
        config = super(PureStateEnv_TUDRL, self).default_config()
        return config

    def __init__(self, config = None):
        super(PureStateEnv_TUDRL, self).__init__(config)

        self.predictor = Prediction(SEED=4064)

        

    def reward_function(self, vehicle_id: str):
        vehicle = self.agents[vehicle_id]
        step_info = dict()
        heading_penalty = 0.1
        cross_track_reward = 0.1
        

        # Reward for moving forward in current lane
        if vehicle.lane in vehicle.navigation.current_ref_lanes:
            current_lane = vehicle.lane
            positive_road = 1
        else:
            current_lane = vehicle.navigation.current_ref_lanes[-1]
            current_road = vehicle.navigation.current_road
            positive_road = 1 if not current_road.is_negative_road() else -1
        long_last, _ = current_lane.local_coordinates(vehicle.last_position)
        long_now, lateral_now = current_lane.local_coordinates(vehicle.position)


        # reward for lane keeping, without it vehicle can learn to overtake but fail to keep in lane
        if self.config["use_lateral_reward"]:
            lateral_factor = np.clip(1 - 2 * abs(lateral_now) / vehicle.navigation.get_current_lane_width(), 0.0, 1.0)
        else:
            lateral_factor = 1.0
        
        # Heading diff
        ref_line_heading = current_lane.heading_theta_at(long_now + 1)
        heading_diff = wrap_to_pi(abs(vehicle.heading_theta - ref_line_heading)) / np.pi

        # Crosscrack error 
        x_d, e_y = VectorFiledGuidance(vehicle)

        #r_speed = ( 1 - (abs(vehicle.speed_km_h -  vehicle.max_speed_km_h) / vehicle.max_speed_km_h)) *  positive_road
        r_speed = (vehicle.speed_km_h/vehicle.max_speed_km_h) *  positive_road
        r_heading = -heading_diff
        r_progress = (long_now - long_last) * lateral_factor * positive_road
        r_crosscrack = math.pow(0.001, (abs(e_y) * 0.6))

        #print(r_speed, r_progress, r_heading,  r_crosscrack)
        
        reward = self.config["speed_reward"] * r_speed + \
                 self.config["driving_reward"] * r_progress + \
                 r_heading * heading_penalty + \
                 r_crosscrack * cross_track_reward
        #print(self.config["speed_reward"] * r_speed, self.config["driving_reward"] * r_progress , r_heading * heading_penalty, r_crosscrack * cross_track_reward)
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
        
        
        
        surface = self.render(
            mode="top_down",
            num_stack=10, 
            text={
                "Speed Reward" : round(self.config["speed_reward"] * r_speed, 2),
                "Driving Reward" : round(self.config["driving_reward"] * r_progress, 2),
                "Heading Penalty" : round(r_heading * heading_penalty, 2),
                "Cross Track Reward" : round(r_crosscrack * cross_track_reward, 2)
                })
        
        #time.sleep(0.5)

        return reward, step_info

    def cost_function(self, vehicle_id: str):


        vehicle = self.agents[vehicle_id]
        v_id = vehicle.id
        prediction_result = self.predictor.step(engine=self.engine)
        ego_pred = vehicle.position #init

        const_ego = True

        
        FAST = False

        if FAST:
            for result in prediction_result:
                if result['id'] == v_id:
                    ego_pred = result
            collision_prediction = get_inv_mahalanobis_dist(
                traj = ego_pred,
                obj_predictions = prediction_result,
                                
            )

        else:
            prediction_result = get_orientation_velocity_and_shape_of_prediction(
                obj_predictions = prediction_result,
                engine = self.engine,
                dt=0.1
            )
            for result in prediction_result:
                if result['id'] == v_id:
                    ego_pred = result
                    prediction_result.remove(result)
                if any(key not in result for key in ['v_list', 'orientation_list', 'pos_list', 'shape']):
                    print(f"Object {result['id']} is missing required keys. Why HERE")
                    continue 
            if const_ego:
                ego_pred = constant_v_propagation(vehicle)

            total_risk_cost = calc_traj_costs(
                ego_traj = ego_pred,
                obj_predictions = prediction_result,
                engine = self.engine
            )

        return total_risk_cost, {}
    
    def get_single_observation(self):
        o = PureStateObservation_woHist(self.config)
        return o
    
    def step(self, actions: Union[Union[np.ndarray, list], Dict[AnyStr, Union[list, np.ndarray]], int]):
        

        obs, reward, termination, truncate, info = super().step(actions)
        print(self.agent.speed)

        return obs, reward, termination, info
    
    def reset(self):
        obs, info = super().reset()
        self.predictor._hard_init_buffer(engine=self.engine)
        

        return obs


from env.planner.planner import Frenet_planner, Quintic_planner
from metadrive.utils import concat_step_infos
from metadrive.constants import TerminationState, TerrainProperty



class PureStateEnv(MetaDriveEnv):
    def default_config(self) -> Config:
        config = super(PureStateEnv, self).default_config()
        return config

    def __init__(self, config = None):
        super(PureStateEnv, self).__init__(config)

    def reward_function(self, vehicle_id: str):
        vehicle = self.agents[vehicle_id]
        step_info = dict()
        heading_penalty = 0.1
        cross_track_reward = 0.1
        

        # Reward for moving forward in current lane
        if vehicle.lane in vehicle.navigation.current_ref_lanes:
            current_lane = vehicle.lane
            positive_road = 1
        else:
            current_lane = vehicle.navigation.current_ref_lanes[-1]
            current_road = vehicle.navigation.current_road
            positive_road = 1 if not current_road.is_negative_road() else -1
        long_last, _ = current_lane.local_coordinates(vehicle.last_position)
        long_now, lateral_now = current_lane.local_coordinates(vehicle.position)


        # reward for lane keeping, without it vehicle can learn to overtake but fail to keep in lane
        if self.config["use_lateral_reward"]:
            lateral_factor = np.clip(1 - 2 * abs(lateral_now) / vehicle.navigation.get_current_lane_width(), 0.0, 1.0)
        else:
            lateral_factor = 1.0
        
        # Heading diff
        ref_line_heading = current_lane.heading_theta_at(long_now + 1)
        heading_diff = wrap_to_pi(abs(vehicle.heading_theta - ref_line_heading)) / np.pi

        # Crosscrack error 
        x_d, e_y = VectorFiledGuidance(vehicle)

        #r_speed = ( 1 - (abs(vehicle.speed_km_h -  vehicle.max_speed_km_h) / vehicle.max_speed_km_h)) *  positive_road
        r_speed = (vehicle.speed_km_h/vehicle.max_speed_km_h) *  positive_road
        r_heading = -heading_diff
        r_progress = (long_now - long_last) * lateral_factor * positive_road
        r_crosscrack = math.pow(0.001, (abs(e_y) * 0.6))

        #print(r_speed, r_progress, r_heading,  r_crosscrack)
        
        reward = self.config["speed_reward"] * r_speed + \
                 self.config["driving_reward"] * r_progress + \
                 r_heading * heading_penalty + \
                 r_crosscrack * cross_track_reward
        #print(self.config["speed_reward"] * r_speed, self.config["driving_reward"] * r_progress , r_heading * heading_penalty, r_crosscrack * cross_track_reward)
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
        
        surface = self.render(
            mode="top_down",
            num_stack=10, 
            text={
                "Speed Reward" : round(self.config["speed_reward"] * r_speed, 2),
                "Driving Reward" : round(self.config["driving_reward"] * r_progress, 2),
                "Heading Penalty" : round(r_heading * heading_penalty, 2),
                "Cross Track Reward" : round(r_crosscrack * cross_track_reward, 2)
                })
        #time.sleep(0.5)
        
        

        return reward, step_info

    def cost_function(self, vehicle_id: str):


        return 0, {}
    
    def get_single_observation(self):
        o = PureStateObservation(self.config)
        return o
    
    def step(self, actions: Union[Union[np.ndarray, list], Dict[AnyStr, Union[list, np.ndarray]], int]):
        obs, reward, termination, truncate, info = super().step(actions)

        return obs, reward, termination, truncate, info



class Traj_Nav_Env_TUDRL(MetaDriveEnv):
    def default_config(self) -> Config:
        config = super(Traj_Nav_Env_TUDRL, self).default_config()
        return config

    def __init__(self, config = None):
        super(Traj_Nav_Env_TUDRL, self).__init__(config)
        self.predict_trj = False
        self.c_x_acc = 0 
        self.c_y_acc = 0
        self.c_x_v_past, self.c_y_v_past = 0, 0
        self.dt = 0.1
        self.frenet = True
        self.desire_target_time = 1.0
        self.target_time = self.desire_target_time

        if self.frenet:
            self.planner = Frenet_planner(pid_control=True)
        else:
            self.planner = Quintic_planner(pid_control=True)

    def reward_function(self, vehicle_id: str):
        vehicle = self.agents[vehicle_id]
        step_info = dict()
        heading_penalty = 0.1
        cross_track_reward = 0.1
        time_reward = 0.1
        desired_speed = 45 #km/h

        # Reward for planner with target time (Gaussian)

        r_time = (1/math.sqrt(2*math.pi))*math.exp(-0.5*(self.target_time - self.desire_target_time)**2)
        #r_time = ( 1 - (abs(self.target_time -  self.desire_target_time) / self.desire_target_time))

        # Reward for moving forward in current lane
        if vehicle.lane in vehicle.navigation.current_ref_lanes:
            current_lane = vehicle.lane
            positive_road = 1
        else:
            current_lane = vehicle.navigation.current_ref_lanes[-1]
            current_road = vehicle.navigation.current_road
            positive_road = 1 if not current_road.is_negative_road() else -1
        long_last, _ = current_lane.local_coordinates(vehicle.last_position)
        long_now, lateral_now = current_lane.local_coordinates(vehicle.position)

        # reward for lane keeping, without it vehicle can learn to overtake but fail to keep in lane
        if self.config["use_lateral_reward"]:
            lateral_factor = np.clip(1 - 2 * abs(lateral_now) / vehicle.navigation.get_current_lane_width(), 0.0, 1.0)
        else:
            lateral_factor = 1.0
        
        # Heading diff
        ref_line_heading = current_lane.heading_theta_at(long_now + 1)
        heading_diff = wrap_to_pi(abs(vehicle.heading_theta - ref_line_heading)) / np.pi

        # Crosscrack error 
        x_d, e_y = VectorFiledGuidance(vehicle)

        r_speed = ( 1 - (abs(vehicle.speed_km_h -  desired_speed) / desired_speed)) *  positive_road
        #r_speed = (vehicle.speed_km_h/vehicle.max_speed_km_h) *  positive_road
        r_heading = -heading_diff
        r_progress = (long_now - long_last) * lateral_factor * positive_road
        r_crosscrack = np.exp(-abs(e_y) * 0.05)

        #print(r_speed, r_progress, r_heading,  r_crosscrack)
        
        reward = self.config["speed_reward"] * r_speed + \
                 self.config["driving_reward"] * r_progress + \
                 r_heading * heading_penalty + \
                 r_crosscrack * cross_track_reward + \
                 r_time * time_reward
        
        step_info["step_reward"] = reward

        if self._is_arrive_destination(vehicle):
            reward += self.config["success_reward"]
        elif self._is_out_of_road(vehicle):
            reward -= self.config["out_of_road_penalty"]
        elif vehicle.crash_vehicle:
            reward -= self.config["crash_vehicle_penalty"]
        elif vehicle.crash_object:
            reward -= self.config["crash_object_penalty"]

        step_info["route_completion"] = vehicle.navigation.route_completion
        
        
        """
        self.render(
            mode="bev",
            num_stack=10, 
            text={
                "Speed Reward" : round(self.config["speed_reward"] * r_speed, 2),
                "Driving Reward" : round(self.config["driving_reward"] * r_progress, 2),
                "Heading Penalty" : round(r_heading * heading_penalty, 2),
                "Cross Track Reward" : round(r_crosscrack * cross_track_reward, 2)
                })
        """
        #time.sleep(0.5)

        return reward, step_info

    def cost_function(self, vehicle_id: str):
        vehicle = self.agents[vehicle_id]
        v_id = vehicle.id
        ego_pred = vehicle.position #init
        const_ego = True
        total_risk_cost = 0

        if self.predict_trj:
            prediction_result = self.predictor.step(engine=self.engine) 
        
            FAST = False

            if FAST:
                for result in prediction_result:
                    if result['id'] == v_id:
                        ego_pred = result
                collision_prediction = get_inv_mahalanobis_dist(
                    traj = ego_pred,
                    obj_predictions = prediction_result,
                                    
                )

            else:
                prediction_result = get_orientation_velocity_and_shape_of_prediction(
                    obj_predictions = prediction_result,
                    engine = self.engine,
                    dt=self.dt
                )
                for result in prediction_result:
                    if result['id'] == v_id:
                        ego_pred = result
                        prediction_result.remove(result)
                    if any(key not in result for key in ['v_list', 'orientation_list', 'pos_list', 'shape']):
                        print(f"Object {result['id']} is missing required keys. Why HERE")
                        continue 
                if const_ego:
                    ego_pred = constant_v_propagation(vehicle)

                total_risk_cost = calc_traj_costs(
                    ego_traj = ego_pred,
                    obj_predictions = prediction_result,
                    engine = self.engine
                )

        return total_risk_cost, {}
    
    def get_single_observation(self):
        o = PureStateObservation_woHist(self.config)
        return o
    
    def step(self, actions: Union[Union[np.ndarray, list], Dict[AnyStr, Union[list, np.ndarray]], int]):

        # Get current vehicle info

        ego_vehicle =self.agent
        if ego_vehicle.lane in ego_vehicle.navigation.current_ref_lanes:
            current_lane = ego_vehicle.lane
        else:
            current_lane = ego_vehicle.navigation.current_ref_lanes[0]
        
        long_now, lateral_now = current_lane.local_coordinates(ego_vehicle.position)
        long_last, lateral_last = current_lane.local_coordinates(ego_vehicle.last_position)

        c_x_v = (long_now -long_last)/ self.dt
        c_y_v = (lateral_now - lateral_last)/ self.dt

        # ACC computation
        c_x_acc = (c_x_v - self.c_x_v_past) / self.dt
        c_y_acc = (c_y_v - self.c_y_v_past) / self.dt

        c_a = math.sqrt(c_x_acc**2 + c_y_acc**2)


        self.target_time, target_lateral, target_speed_long = convert_action_to_target_point(
            actions=actions, vehicle=ego_vehicle,current_lane=current_lane, velocity=[c_x_v,c_y_v]
        )

        #target_time = 1
        #target_time, target_long, target_lateral, target_speed_long = 1, 10, 0, 5
        #print(self.target_time, target_lateral, target_speed_long)
        # Plan and execute trajectory
        target_long = None #not used

        if self.frenet:
            current_frenet_state = Frenet_State(
                s=long_now,
                s_d=c_x_v,
                s_dd=self.c_x_acc,
                d=lateral_now,
                d_d=c_y_v,
                d_dd=self.c_y_acc
            )
            fp_list,fp_cost = self.planner.plan(
                frenet_state=current_frenet_state,
                target_lateral=target_lateral + current_frenet_state.d,
                target_long=target_long,
                target_speed_long=target_speed_long,
                target_time=self.target_time,
                current_lane=current_lane
            )
        else:
            fp_list = self.planner.plan(
                sx=ego_vehicle.position[0], sy=ego_vehicle.position[1],
                syaw=ego_vehicle.heading_theta, sv=ego_vehicle.speed,
                sa=math.sqrt(self.c_x_acc**2 + self.c_y_acc**2),
                gx=target_long, gy=target_lateral,
                gyaw=ego_vehicle.heading_theta, gv=target_speed_long,
                ga=None, max_accel=5, max_jerk=3, gT=self.target_time
            )
        pid_control = True
        cumulated_reward = 0
        #self.planner.plot_traj(fp_list)
        #self.planner.plot_traj(fp_list) Unable for training
        #print(current_frenet_state)
        #print(fp_list.s, fp_list.d, fp_list.yaw, fp_list.v)
        for i in range(len(fp_list.t)):
            
            if pid_control:
                action = self.planner.act(
                    target_x=fp_list.x[i], target_y=fp_list.y[i],
                    target_yaw=fp_list.yaw[i], target_speed=fp_list.v[i],
                    ego_vehicle=ego_vehicle
                )

                obs, reward, termination, truncate, info = super().step(action)

                self.render(mode="bev")
            else:
                ego_vehicle.set_position([fp_list.x[i], fp_list.y[i]])
                ego_vehicle.set_heading_theta(fp_list.yaw[i])
                obs, reward, termination, truncate, info = super().step([0,0])
                #self.render(mode="bev")

            cumulated_reward += reward
            if termination or truncate:
                break
        cumulated_reward = cumulated_reward - fp_cost[1] - fp_cost[2]*0.05

        # Update past velocities
        self.c_x_v_past, self.c_y_v_past = c_x_v, c_y_v
        return obs, cumulated_reward, termination, info
    
    def reset(self):
        obs, info = super().reset()
        self.padd = None
        return obs
    




class PureStateEnv_SRL(MetaDriveEnv):

    def default_config(self) -> Config:
        config = super(PureStateEnv_SRL, self).default_config()
        return config

    def __init__(self, config = None):
        super(PureStateEnv_SRL, self).__init__(config)
        self.predictor = Prediction(SEED=4064)


    def reward_function(self, vehicle_id: str):
        vehicle = self.agents[vehicle_id]
        step_info = dict()
        heading_penalty = 0.1
        cross_track_reward = 0.1
        

        # Reward for moving forward in current lane
        if vehicle.lane in vehicle.navigation.current_ref_lanes:
            current_lane = vehicle.lane
            positive_road = 1
        else:
            current_lane = vehicle.navigation.current_ref_lanes[-1]
            current_road = vehicle.navigation.current_road
            positive_road = 1 if not current_road.is_negative_road() else -1
        long_last, _ = current_lane.local_coordinates(vehicle.last_position)
        long_now, lateral_now = current_lane.local_coordinates(vehicle.position)


        # reward for lane keeping, without it vehicle can learn to overtake but fail to keep in lane
        if self.config["use_lateral_reward"]:
            lateral_factor = np.clip(1 - 2 * abs(lateral_now) / vehicle.navigation.get_current_lane_width(), 0.0, 1.0)
        else:
            lateral_factor = 1.0
        
        # Heading diff
        ref_line_heading = current_lane.heading_theta_at(long_now + 1)
        heading_diff = wrap_to_pi(abs(vehicle.heading_theta - ref_line_heading)) / np.pi

        # Crosscrack error 
        x_d, e_y = VectorFiledGuidance(vehicle)

        #r_speed = ( 1 - (abs(vehicle.speed_km_h -  vehicle.max_speed_km_h) / vehicle.max_speed_km_h)) *  positive_road
        r_speed = (vehicle.speed_km_h/vehicle.max_speed_km_h) *  positive_road
        r_heading = -heading_diff
        r_progress = (long_now - long_last) * lateral_factor * positive_road
        r_crosscrack = math.pow(0.001, (abs(e_y) * 0.6))

        #print(r_speed, r_progress, r_heading,  r_crosscrack)
        
        reward = self.config["speed_reward"] * r_speed + \
                 self.config["driving_reward"] * r_progress + \
                 r_heading * heading_penalty + \
                 r_crosscrack * cross_track_reward
        #print(self.config["speed_reward"] * r_speed, self.config["driving_reward"] * r_progress , r_heading * heading_penalty, r_crosscrack * cross_track_reward)
        step_info["step_reward"] = reward

        if self._is_arrive_destination(vehicle):
            reward  =+ self.config["success_reward"]
        elif self._is_out_of_road(vehicle):
            reward  =- self.config["out_of_road_penalty"]
        elif vehicle.crash_vehicle:
            reward =- self.config["crash_vehicle_penalty"]
        elif vehicle.crash_object:
            reward =- self.config["crash_object_penalty"]

        step_info["route_completion"] = vehicle.navigation.route_completion
        
        

        """
        
        surface = self.render(
            mode="top_down",
            num_stack=10, 
            text={
                "Speed Reward" : round(self.config["speed_reward"] * r_speed, 2),
                "Driving Reward" : round(self.config["driving_reward"] * r_progress, 2),
                "Heading Penalty" : round(r_heading * heading_penalty, 2),
                "Cross Track Reward" : round(r_crosscrack * cross_track_reward, 2)
                })
        """
        #time.sleep(0.5)     

        return reward, step_info

    def cost_function(self, vehicle_id: str):
        step_info = dict()
        vehicle = self.agents[vehicle_id]
        v_id = vehicle.id
        predict = False
        total_risk_cost = 0
        if predict:
            prediction_result = self.predictor.step(engine=self.engine)
            ego_pred = vehicle.position #init

            const_ego = False
            total_risk_cost = 1e-6
            
            prediction_result = get_orientation_velocity_and_shape_of_prediction(
                obj_predictions = prediction_result,
                engine = self.engine,
                dt=0.1
            )

            for result in prediction_result:
                if result['id'] == v_id:
                    ego_pred = result
                    prediction_result.remove(result)
                if any(key not in result for key in ['v_list', 'orientation_list', 'pos_list', 'shape']):
                    print(f"Object {result['id']} is missing required keys. Why HERE")
                    continue 
            if const_ego:
                ego_pred = constant_v_propagation(vehicle)

            total_risk_cost = calc_traj_costs(
                ego_traj = ego_pred,
                obj_predictions = prediction_result,
                engine = self.engine
            )
        step_info["cost"] = total_risk_cost
        


        return total_risk_cost, step_info

    
    def get_single_observation(self):
        o = PureStateObservation(self.config)
        return o
    

    def reset(self):
        obs, info = super().reset()

        self.predictor._hard_init_buffer(engine=self.engine)
        
        return obs, info
    



class Traj_Nav_Env_SRL(MetaDriveEnv):

    def default_config(self) -> Config:
        config = super(Traj_Nav_Env_SRL, self).default_config()
        return config

    def __init__(self, config = None):
        super(Traj_Nav_Env_SRL, self).__init__(config)
        self.predictor = Prediction(SEED=11263)
        self.predict_trj = True
        self.c_x_acc = 0 
        self.c_y_acc = 0
        self.c_x_v_past, self.c_y_v_past = 0, 0
        self.dt = 0.1
        self.frenet = True
        self.desire_target_time = 2.0
        self.target_time = self.desire_target_time
        self.ego_planned_traj = []
        self.cost_fn_mode = "standard"
        self.test_mode = True

        if self.test_mode:
            date = "2024-05-17"
            cost_limit = 1.5
            mode = "ethical"
            self.cost_fn_mode = mode

            self.risk_logger = risk_logger(
                file_path=f"risk_results/{date}_{cost_limit}/{mode}",
                fieldnames=[
                            "ego_risk_max",
                            "obj_risk_max",
                            "ego_harm_max",
                            "obj_harm_max",
                            "bayes_cost",
                            "equality_cost",
                            "maximin_cost",
                            "ego_cost",
                            "obj_id_type",
                            ])



        

        if self.frenet:
            self.planner = Frenet_planner(pid_control=True)
        else:
            self.planner = Quintic_planner(pid_control=True)

    @property
    def action_space(self) -> gym.Space:

        return spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)


    def step(self, actions: Union[Union[np.ndarray, list], Dict[AnyStr, Union[list, np.ndarray]], int]):
        # Get current vehicle info

        ego_vehicle =self.agent
        if ego_vehicle.lane in ego_vehicle.navigation.current_ref_lanes:
            current_lane = ego_vehicle.lane
        else:
            current_lane = ego_vehicle.navigation.current_ref_lanes[0]
        
        long_now, lateral_now = current_lane.local_coordinates(ego_vehicle.position)
        long_last, lateral_last = current_lane.local_coordinates(ego_vehicle.last_position)

        c_x_v = (long_now -long_last)/ self.dt
        c_y_v = (lateral_now - lateral_last)/ self.dt

        # ACC computation
        c_x_acc = (c_x_v - self.c_x_v_past) / self.dt
        c_y_acc = (c_y_v - self.c_y_v_past) / self.dt

        c_a = math.sqrt(c_x_acc**2 + c_y_acc**2)


        self.target_time, target_lateral, target_speed_long = convert_action_to_target_point(
            actions=actions, vehicle=ego_vehicle,current_lane=current_lane, velocity=[c_x_v,c_y_v]
        )

        # Plan and execute trajectory
        target_long = None #not used

        if self.frenet:
            current_frenet_state = Frenet_State(
                s=long_now,
                s_d=c_x_v,
                s_dd=self.c_x_acc,
                d=lateral_now,
                d_d=c_y_v,
                d_dd=self.c_y_acc
            )
            fp_list,fp_cost = self.planner.plan(
                frenet_state=current_frenet_state,
                target_lateral=target_lateral + current_frenet_state.d,
                target_long=target_long,
                target_speed_long=target_speed_long,
                target_time=self.target_time,
                current_lane=current_lane
            )
        else:
            fp_list = self.planner.plan(
                sx=ego_vehicle.position[0], sy=ego_vehicle.position[1],
                syaw=ego_vehicle.heading_theta, sv=ego_vehicle.speed,
                sa=math.sqrt(self.c_x_acc**2 + self.c_y_acc**2),
                gx=target_long, gy=target_lateral,
                gyaw=ego_vehicle.heading_theta, gv=target_speed_long,
                ga=None, max_accel=5, max_jerk=3, gT=self.target_time
            )
        pid_control = True
        cumulated_reward = 0
        self.ego_planned_traj = fp_list

        traj_cost = self.traj_cost_function('default_agent')

        for i in range(len(fp_list.t)):
            
            if pid_control:
                action = self.planner.act(
                    target_x=fp_list.x[i], target_y=fp_list.y[i],
                    target_yaw=fp_list.yaw[i], target_speed=fp_list.v[i],
                    ego_vehicle=ego_vehicle
                )

                obs, reward, termination, truncate, info = super().step(action)

                #self.render(mode="bev")
            else:
                ego_vehicle.set_position([fp_list.x[i], fp_list.y[i]])
                ego_vehicle.set_heading_theta(fp_list.yaw[i])
                obs, reward, termination, truncate, info = super().step([0,0])
                #self.render(mode="bev")

            cumulated_reward += reward
            if termination or truncate:
                break
        #cumulated_reward = cumulated_reward - fp_cost[1] - fp_cost[2]*0.05
        cumulated_reward = cumulated_reward - fp_cost[2]*0.05

        # Update past velocities
        self.c_x_v_past, self.c_y_v_past = c_x_v, c_y_v


        info["cost"] = traj_cost

        return obs, cumulated_reward, termination, info


    def reward_function(self, vehicle_id: str):
        vehicle = self.agents[vehicle_id]
        step_info = dict()
        heading_penalty = 0.1
        cross_track_reward = 0.1
        time_reward = 0.1
        desired_speed = 45 #km/h
        use_lateral_reward = False
        speed_reward = 0.7
        driving_reward = 0.1

        # Reward for planner with target time (Gaussian)

        #r_time = (1/math.sqrt(2*math.pi))*math.exp(-0.5*(self.target_time - self.desire_target_time)**2)
        #r_time = ( 1 - (abs(self.target_time -  self.desire_target_time) / self.desire_target_time))

        # Reward for moving forward in current lane
        if vehicle.lane in vehicle.navigation.current_ref_lanes:
            current_lane = vehicle.lane
            positive_road = 1
        else:
            current_lane = vehicle.navigation.current_ref_lanes[-1]
            current_road = vehicle.navigation.current_road
            positive_road = 1 if not current_road.is_negative_road() else -1
        long_last, _ = current_lane.local_coordinates(vehicle.last_position)
        long_now, lateral_now = current_lane.local_coordinates(vehicle.position)


        # reward for lane keeping, without it vehicle can learn to overtake but fail to keep in lane
        if use_lateral_reward:
            lateral_factor = np.clip(1 - 2 * abs(lateral_now) / vehicle.navigation.get_current_lane_width(), 0.0, 1.0)
        else:
            lateral_factor = 1.0
        
        # Heading diff
        #ref_line_heading = current_lane.heading_theta_at(long_now + 1)
        #heading_diff = wrap_to_pi(abs(vehicle.heading_theta - ref_line_heading)) / np.pi

        # Crosscrack error 
        #x_d, e_y = VectorFiledGuidance(vehicle)

        r_speed = ( 1 - (abs(vehicle.speed_km_h -  desired_speed) / desired_speed)) *  positive_road
        #r_speed = (vehicle.speed_km_h/vehicle.max_speed_km_h) *  positive_road
        #r_heading = -heading_diff
        r_progress = (long_now - long_last) * lateral_factor * positive_road
        #r_crosscrack = np.exp(-abs(e_y) * 0.05)

        
 
        
        reward = speed_reward * r_speed + \
                 driving_reward * r_progress
        
        
        
        step_info["step_reward"] = reward

        if self._is_arrive_destination(vehicle):
            reward += self.config["success_reward"]
        elif self._is_out_of_road(vehicle):
            reward -= self.config["out_of_road_penalty"]
        elif vehicle.crash_vehicle:
            reward -= self.config["crash_vehicle_penalty"]
        elif vehicle.crash_object:
            reward -= self.config["crash_object_penalty"]
        elif vehicle.crash_human:
            reward -= self.config["crash_human_penalty"]

        step_info["route_completion"] = vehicle.navigation.route_completion
        
        

        return reward, step_info

    def traj_cost_function(self, vehicle_id: str):

        vehicle = self.agents[vehicle_id]
        predict = self.predict_trj
        total_risk_cost = 1e-6
        n_points = len(np.arange(0.0, self.target_time, 0.1))


        if predict:
            neib_trajs_list = self.predictor.step(engine=self.engine, ego_vehicle=vehicle)

            neib_trajs_list = self.truncate_data(neib_trajs_list, n_points)

            # Extend ego planned traj for risk prediction        
            ego_traj = get_orientation_velocity_and_shape_of_ego(
                ego_traj=self.ego_planned_traj,
                ego=vehicle,
                n_points=n_points,
            )

            # Extend predicted trajs of neibours for risk prediction    
            neib_trajs_list = get_orientation_velocity_and_shape_of_prediction(
                obj_predictions = neib_trajs_list,
                engine = self.engine,
                dt=0.1
            )

            total_risk_cost = calc_traj_costs(
            ego_traj = ego_traj,
            obj_predictions = neib_trajs_list,
            engine = self.engine,
            n_points=n_points,
            cost_fn_mode=self.cost_fn_mode,
            only_vehicle = True,
            risk_harm_logger=self.risk_logger,
            test_mode=self.test_mode,
        )
            
        return total_risk_cost


    def cost_function(self, vehicle_id: str):
        
        """ step_info = dict()
        vehicle = self.agents[vehicle_id]
        v_id = vehicle.id
        predict = self.predict_trj
        total_risk_cost = 1e-6
        n_points = len(np.arange(0.0, self.target_time, 0.1))


        if predict:
            neib_trajs_list = self.predictor.step(engine=self.engine, ego_vehicle=vehicle)

            neib_trajs_list = self.truncate_data(neib_trajs_list, n_points)

            # Extend ego planned traj for risk prediction        
            ego_traj = get_orientation_velocity_and_shape_of_ego(
                ego_traj=self.ego_planned_traj,
                ego=vehicle,
                n_points=n_points,
            )

            # Extend predicted trajs of neibours for risk prediction    
            neib_trajs_list = get_orientation_velocity_and_shape_of_prediction(
                obj_predictions = neib_trajs_list,
                engine = self.engine,
                dt=0.1
            )

            total_risk_cost = calc_traj_costs(
            ego_traj = ego_traj,
            obj_predictions = neib_trajs_list,
            engine = self.engine,
            n_points=n_points,
            cost_fn_mode=self.cost_fn_mode,
            only_vehicle = True,
        )
            
        step_info["cost"] = total_risk_cost """
        
        # Dummy cost function
        step_info = dict()
        total_risk_cost = 1e-6
        step_info["cost"] = total_risk_cost

        return total_risk_cost, step_info

    
    def get_single_observation(self):
        o = PureStateObservation_woHist(self.config)
        return o
    

    def reset(self):
        if self.test_mode:
            self.risk_logger.close()
            self.risk_logger.reset()

        obs, info = super().reset()

        self.predictor._hard_init_buffer(engine=self.engine)
        
        return obs, info
    
    def set_cost_fn_mode(self, mode):
        self.cost_fn_mode = mode
        print('Cost function mode set to:', mode)


    def truncate_data(self, data, n):
        truncated = []
        for entry in data:
            truncated_entry = {
                'id': entry['id'],
                'pos_list': entry['pos_list'][:n],  # Keep first n positions
                'cov_list': entry['cov_list'][:n]    # Keep first n covariance matrices
            }
            truncated.append(truncated_entry)
        return truncated

class StateBEVEnv(MetaDriveEnv):
    def default_config(self) -> Config:
        config = super(StateBEVEnv, self).default_config()
        config.update(
            {
                "frame_skip": 5,
                "frame_stack": 3,
                "post_stack": 5,
                "norm_pixel": True,
                "resolution_size": 84,
                "distance": 30
            }
        )
        return config

    def __init__(self, config = None):
        super(StateBEVEnv, self).__init__(config)

    def get_single_observation(self, _=None):

        return StateBEVObservation(self.config)

    
    def reward_function(self, vehicle_id: str):
        vehicle = self.agents[vehicle_id]
        step_info = dict()
        heading_penalty = 0.1
        max_speed = vehicle.max_speed_km_h *0.75

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
        # heading diff
        ref_line_heading = current_lane.heading_theta_at(long_now)
        heading_diff = wrap_to_pi(abs(vehicle.heading_theta - ref_line_heading)) / np.pi

        r_speed = ( 1 - (abs(vehicle.speed_km_h -  max_speed) / max_speed)) *  positive_road
        r_heading = -heading_diff
        r_progress = (long_now - long_last) * lateral_factor * positive_road


        reward = self.config["speed_reward"] * r_speed +  self.config["driving_reward"] * r_progress + r_heading * heading_penalty
        
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
    
    def reset(self, seed: Union[None, int] = None):

        obs, _ = super().reset(seed)
        if not isinstance(obs, dict) or not set(obs.keys()) == {"bev", "state"}:
            # Assuming obs is a list or a numpy array, convert it to a dictionary
            obs = self.convert_obs_to_dict(obs)
        assert set(obs.keys()) == {"bev", "state"}, "The observation keys returned by `reset()` must match the observation space keys."
        return obs, _

    
    def step(self, actions: Union[Union[np.ndarray, list], Dict[AnyStr, Union[list, np.ndarray]], int]):
        obs, reward, termination, truncate, info = super().step(actions)
        if not isinstance(obs, dict) or not set(obs.keys()) == {"bev", "state"}:
            # Assuming obs is a list or a numpy array, convert it to a dictionary
            obs = self.convert_obs_to_dict(obs)
        assert set(obs.keys()) == {"bev", "state"}, "The observation keys returned by `step()` must match the observation space keys."


        return obs, reward, termination, truncate, info
    


    def convert_obs_to_dict(self, obs: Tuple[np.ndarray, np.ndarray]) -> Dict[str, np.ndarray]:
        if isinstance(obs, tuple) and len(obs) == 2:
            bev, state = obs
            obs_dict = {
                "bev": bev,
                "state": state
            }
        else:
            raise ValueError("Unexpected observation format")

        return obs_dict





class StateBEVEnv_wrapped(StateBEVEnv):
    def __init__(self, config = None):
        super(StateBEVEnv_wrapped, self).__init__(config)
        self._obs = None
        self._rews = None
    
    def reset(self, seed: Union[None, int] = None):

        obs, _ = super().reset(seed)
        if not isinstance(obs, dict) or not set(obs.keys()) == {"bev", "state"}:
            # Assuming obs is a list or a numpy array, convert it to a dictionary
            obs = self.convert_obs_to_dict(obs)
        assert set(obs.keys()) == {"bev", "state"}, "The observation keys returned by `reset()` must match the observation space keys."
        self._obs = [types.maybe_wrap_in_dictobs(obs)]
        self._rews = []
        return obs, _
    
    def step(self, actions: Union[Union[np.ndarray, list], Dict[AnyStr, Union[list, np.ndarray]], int]):

        obs, rew, terminated, truncated, info = super().step(actions)
        done = terminated or truncated
        self._obs.append(types.maybe_wrap_in_dictobs(obs))
        self._rews.append(rew)

        if done:
            assert "rollout" not in info
            info["rollout"] = {
                "obs": types.stack_maybe_dictobs(self._obs),
                "rews": np.stack(self._rews),
            }
        return obs, rew, terminated, truncated, info


from metadrive.component.traffic_participants.pedestrian import Pedestrian 
from metadrive.component.traffic_participants.cyclist import Cyclist
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.utils.math import norm, clip
from metadrive.manager.scenario_data_manager import ScenarioDataManager, ScenarioOnlineDataManager
import os
from env.util.logger import risk_logger, frame_logger
import cv2
class RealTrafficEnv(ScenarioEnv):
    def default_config(self) -> Config:
        config = super(RealTrafficEnv, self).default_config()
        config.update(
        {
            "vehicle_config": 
            {
              "lidar": {
                  "num_pedestrians": 4,
              }
            }
        }
        )
        return config
    

    def __init__(self, config = None):
        super(RealTrafficEnv, self).__init__(config)
        self.predictor = Prediction(SEED=11263)
        self.predict_trj = False
        self.c_x_acc = 0 
        self.c_y_acc = 0
        self.c_x_v_past, self.c_y_v_past = 0, 0
        self.dt = 0.1
        self.frenet = True
        self.desire_target_time = 2.0
        self.target_time = self.desire_target_time
        self.ego_planned_traj = []
        #self.lazy_init()
        self.test_mode = True
        self.date = "2024-06-07"
        self.cost_limit = 1
        self.cost_fn_mode = "ethical"


        if self.test_mode:
            date = self.date
            cost_limit =self.cost_limit
            self.risk_logger = risk_logger(
                file_path=f"risk_results/{date}_{cost_limit}/{self.cost_fn_mode}",
                fieldnames=[
                            "ego_risk_max",
                            "obj_risk_max",
                            "ego_harm_max",
                            "obj_harm_max",
                            "bayes_cost",
                            "equality_cost",
                            "maximin_cost",
                            "ego_cost",
                            "obj_id_type",
                            "ego_speed",
                            "ego_long",
                            "ego_lat",
                            "info",
                            "obs",
                            "ttc_walker",
                            "ttc_vehicle",
                            "relevante_objects",
                            ])
            self.frame_logger = frame_logger(
                base_path=f"risk_results/{date}_{cost_limit}/{self.cost_fn_mode}"
            )

        else:
            self.risk_logger = None
            self.frame_logger = None

        if self.frenet:
            self.planner = Frenet_planner(pid_control=True)
        else:
            self.planner = Quintic_planner(pid_control=True)

    @property
    def action_space(self) -> gym.Space:

        return spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

    
    def get_surrounding_walkers(self, detected_objects) -> Set:

        walkers = set()
        objs = detected_objects
        for ret in objs:
            if isinstance(ret, Pedestrian) or isinstance(ret, Cyclist):
                walkers.add(ret)
        return walkers
    
    def get_relevante_objects(self):

        _, objs = self.obs_surrounding_objects()

        true_id = ['6817', '3266', '5132', '1582', '1373', '2093', '196','1529']
        id_ls = self.engine.managers['traffic_manager'].obj_id_to_scenario_id
        if id_ls is None:
            print("Warning: obj_id_to_scenario_id() returned None")
            matching_ids = []
        else:
            matching_ids = [k for k, v in id_ls.items() if v in true_id]
        relevante_objs = []
        for obj in objs:
            if hasattr(obj, "name") and obj.id in matching_ids:
                relevante_objs.append(obj.id)
            else:
                pass
        return relevante_objs
        

    def get_surrounding_vehicles(self, detected_objects) -> Set:

        vehicles = set()
        objs = detected_objects
        for ret in objs:
            if isinstance(ret, BaseVehicle):
                vehicles.add(ret)
        return vehicles

    def get_walkers_info(self, detected_objects, ego_vehicle, perceive_distance=50, num_others= 4):

        walkers = list(self.get_surrounding_walkers(detected_objects))

        walkers.sort(
            key=lambda v: norm(ego_vehicle.position[0] - v.position[0], ego_vehicle.position[1] - v.position[1])
        )

        if walkers:
            walkers.sort(
                key=lambda v: norm(self.agent.position[0] - v.position[0], self.agent.position[1] - v.position[1])
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




    def obs_surrounding_objects(self):

        cloud_points, detected_objects = self.engine.get_sensor("lidar").perceive(
                self.agent,
                physics_world=self.engine.physics_world.dynamic_world,
                num_lasers=self.agent.config["lidar"]["num_lasers"],
                distance=self.agent.config["lidar"]["distance"],
                show=self.agent.config["show_lidar"],
            )


        return cloud_points, detected_objects
    
    def groundtruth_trajs(self, n_points):

        traffic_manager = self.engine._managers['traffic_manager']
        traffic_data = traffic_manager.current_traffic_data

        ego_vehicle = traffic_manager.ego_vehicle


        current_surrounding_vehicles = get_surrounding_vehicles(ego_vehicle,self.engine)


        # Find the surrounding vehicles ids
        surround_v_ids = []
        if current_surrounding_vehicles:
            for v in current_surrounding_vehicles:
                if v is not None:
                    surround_v_ids.append(v.id)
        

        # All obj ids to scenario ids
        obj_id_to_scenario_id_lst = traffic_manager.obj_id_to_scenario_id

        # Filter the traffic data ind list with target id list
        filtered_obj_id_lst = {key: obj_id_to_scenario_id_lst[key] for key in surround_v_ids if key in obj_id_to_scenario_id_lst}

        
        #Match the surrounding vehicles ids with the groundtruth data

        traffic_data_dict = {item: data for item, data in traffic_data.items()}

        # raw traffic data with correct vehicle ids in scenario
        filtered_traffic_data = [(key, traffic_data_dict[item]) for key, item in filtered_obj_id_lst.items() if item in traffic_data_dict]

        # Extract only relevant data
        ground_truth_trajs = []
        for key, data in filtered_traffic_data:
            ground_truth_trajs.append(
                {
                    "id": key,
                    "position": data['state']['position'][:,:2][:n_points],
                    "heading": data['state']['heading'][:n_points],
                    "velocity": data['state']['velocity'][:n_points],
                }
            )
        ground_truth_trajs = map_gt_to_prediction(ground_truth_trajs, engine=self.engine, n_points=n_points)


        return ground_truth_trajs


    def step(self, actions: Union[Union[np.ndarray, list], Dict[AnyStr, Union[list, np.ndarray]], int]):
        

        

        

        # Get current vehicle info

        ego_vehicle =self.agent
        if ego_vehicle.lane in ego_vehicle.navigation.current_ref_lanes:
            current_lane = ego_vehicle.lane
        else:
            current_lane = ego_vehicle.navigation.current_ref_lanes[0]


        
        #cloud_points, detected_objects = self.obs_surrounding_objects()

        
        
        long_now, lateral_now = current_lane.local_coordinates(ego_vehicle.position)
        long_last, lateral_last = current_lane.local_coordinates(ego_vehicle.last_position)

        c_x_v = (long_now -long_last)/ self.dt
        c_y_v = (lateral_now - lateral_last)/ self.dt

        # ACC computation
        c_x_acc = (c_x_v - self.c_x_v_past) / self.dt
        c_y_acc = (c_y_v - self.c_y_v_past) / self.dt

        c_a = math.sqrt(c_x_acc**2 + c_y_acc**2)


        self.target_time, target_lateral, target_speed_long = convert_action_to_target_point(
            actions=actions, vehicle=ego_vehicle,current_lane=current_lane, velocity=[c_x_v,c_y_v]
        )

        #target_time = 1
        #target_time, target_long, target_lateral, target_speed_long = 1, 10, 0, 5
        #print(self.target_time, target_lateral, target_speed_long)
        # Plan and execute trajectory
        target_long = None #not used

        if self.frenet:
            current_frenet_state = Frenet_State(
                s=long_now,
                s_d=c_x_v,
                s_dd=self.c_x_acc,
                d=lateral_now,
                d_d=c_y_v,
                d_dd=self.c_y_acc
            )
            fp_list,fp_cost = self.planner.plan(
                frenet_state=current_frenet_state,
                target_lateral=target_lateral + current_frenet_state.d,
                target_long=target_long,
                target_speed_long=target_speed_long,
                target_time=self.target_time,
                current_lane=current_lane
            )
        
        else:
            fp_list = self.planner.plan(
                sx=ego_vehicle.position[0], sy=ego_vehicle.position[1],
                syaw=ego_vehicle.heading_theta, sv=ego_vehicle.speed,
                sa=math.sqrt(self.c_x_acc**2 + self.c_y_acc**2),
                gx=target_long, gy=target_lateral,
                gyaw=ego_vehicle.heading_theta, gv=target_speed_long,
                ga=None, max_accel=5, max_jerk=3, gT=self.target_time
            )
        pid_control = True
        cumulated_reward = 0
        self.ego_planned_traj = fp_list

        traj_cost = self.traj_cost_function('default_agent')

        ego_long_ls = []
        ego_lat_ls = []
        ego_speed_ls = []
        ttc_walker_ls = []
        ttc_vehicle_ls = []
        info_ls = []
        relevante_object_ls = []

        for i in range(len(fp_list.t)):
            
            if pid_control:
                action = self.planner.act(
                    target_x=fp_list.x[i], target_y=fp_list.y[i],
                    target_yaw=fp_list.yaw[i], target_speed=fp_list.v[i],
                    ego_vehicle=ego_vehicle
                )

                obs, reward, termination, truncate, info = super().step(action)

                if self.test_mode:
                    ret = self.render(mode="bev",
                    screen_size=(300, 300),)
                    if ret is not None:
                        self.frame_logger.save_frame(ret)


            else:
                ego_vehicle.set_position([fp_list.x[i], fp_list.y[i]])
                ego_vehicle.set_heading_theta(fp_list.yaw[i])
                obs, reward, termination, truncate, info = super().step([0,0])
                #self.render(mode="bev")
            if self.test_mode:
                ego_lat_ls.append(ego_vehicle.navigation.last_current_lat[-1])
                ego_long_ls.append(ego_vehicle.navigation.last_current_long[-1])
                ego_speed_ls.append(ego_vehicle.speed)
                info_ls.append(info)
                ttc_walkers, ttc_vehicles = self.get_ttc_metrics(ego_vehicle)
                relevante_objects = self.get_relevante_objects()
                relevante_object_ls.append(relevante_objects)
                ttc_walker_ls.append(ttc_walkers)
                ttc_vehicle_ls.append(ttc_vehicles)


            cumulated_reward += reward
            if termination or truncate:
                break
        #cumulated_reward = cumulated_reward - fp_cost[1] - fp_cost[2]*0.05
        cumulated_reward = cumulated_reward - fp_cost[2]*0.05

        # Update past velocities
        self.c_x_v_past, self.c_y_v_past = c_x_v, c_y_v

        info["cost"] = traj_cost


        if self.test_mode:
            #Additional log for evaluation
            log_data = {
                        "ego_speed": ego_speed_ls,
                        "ego_long": ego_long_ls,
                        "ego_lat": ego_lat_ls,
                        "info": info_ls,
                        "obs": obs,
                        "ttc_walker": ttc_walker_ls,
                        "ttc_vehicle": ttc_vehicle_ls,
                        "relevante_objects": relevante_object_ls,
                    }
            
            self.risk_logger.add_log(log_data)

        
        return obs, cumulated_reward, termination, info

    def done_function(self, vehicle_id: str):

        done, done_info = super().done_function(vehicle_id)

        if done_info['crash_vehicle']:
            done = True
        elif done_info['crash_human']:
            done = True

        return done, done_info

    def traj_cost_function(self, vehicle_id: str):
        vehicle = self.agents[vehicle_id]
        predict = self.predict_trj
        total_risk_cost = 1e-6
        n_points = len(np.arange(0.0, self.target_time, 0.1))

        if predict:
            neib_trajs_list = self.predictor.step(engine=self.engine, ego_vehicle=vehicle)
            neib_trajs_list = self.truncate_data(neib_trajs_list, n_points)
        else:
            neib_trajs_list = self.groundtruth_trajs(n_points=n_points)

        
        # Extend ego planned traj for risk prediction        
        ego_traj = get_orientation_velocity_and_shape_of_ego(
            ego_traj=self.ego_planned_traj,
            ego=vehicle,
            n_points=n_points
        )

        # Extend predicted trajs of neibours for risk prediction    
        neib_trajs_list = get_orientation_velocity_and_shape_of_prediction(
            obj_predictions = neib_trajs_list,
            engine = self.engine,
            dt=0.1
        )
        total_risk_cost = calc_traj_costs(
            ego_traj = ego_traj,
            obj_predictions = neib_trajs_list,
            engine = self.engine,
            n_points=n_points,
            cost_fn_mode=self.cost_fn_mode,
            only_vehicle = False,
            risk_harm_logger=self.risk_logger,
            test_mode=self.test_mode,

        )
        return total_risk_cost
        
    def reward_function(self, vehicle_id: str):
        vehicle = self.agents[vehicle_id]
        step_info = dict()
        heading_penalty = 0.1
        cross_track_reward = 0.1
        time_reward = 0.1
        desired_speed = 45 #km/h
        use_lateral_reward = False
        speed_reward = 0.7
        driving_reward = 0.1

        # Reward for planner with target time (Gaussian)

        #r_time = (1/math.sqrt(2*math.pi))*math.exp(-0.5*(self.target_time - self.desire_target_time)**2)

        # Reward for moving forward in current lane
        if vehicle.lane in vehicle.navigation.current_ref_lanes:
            current_lane = vehicle.lane
            positive_road = 1
        else:
            current_lane = vehicle.navigation.current_ref_lanes[-1]
            current_road = vehicle.navigation.current_road
            positive_road = 1 if not current_road.is_negative_road() else -1
        long_last, _ = current_lane.local_coordinates(vehicle.last_position)
        long_now, lateral_now = current_lane.local_coordinates(vehicle.position)


        # reward for lane keeping, without it vehicle can learn to overtake but fail to keep in lane
        if use_lateral_reward:
            lateral_factor = np.clip(1 - 2 * abs(lateral_now) / vehicle.navigation.get_current_lane_width(), 0.0, 1.0)
        else:
            lateral_factor = 1.0
        
        # Heading diff
        #ref_line_heading = current_lane.heading_theta_at(long_now + 1)
        #heading_diff = wrap_to_pi(abs(vehicle.heading_theta - ref_line_heading)) / np.pi

        # Crosscrack error 
        #x_d, e_y = VectorFiledGuidance(vehicle)

        r_speed = ( 1 - (abs(vehicle.speed_km_h -  desired_speed) / desired_speed)) *  positive_road
        #r_speed = (vehicle.speed_km_h/vehicle.max_speed_km_h) *  positive_road
        #r_heading = -heading_diff
        r_progress = (long_now - long_last) * lateral_factor * positive_road
        #r_crosscrack = math.pow(0.001, (abs(e_y) * 0.6))

        
        reward = speed_reward * r_speed + \
                 driving_reward * r_progress
        

        if self._is_arrive_destination(vehicle):
            reward += self.config["success_reward"]
        elif self._is_out_of_road(vehicle):
            reward -= self.config["out_of_road_penalty"]
        elif vehicle.crash_vehicle:
            reward -= self.config["crash_vehicle_penalty"]
        elif vehicle.crash_object:
            reward -= self.config["crash_object_penalty"]
        elif vehicle.crash_human:
            reward -= self.config["crash_human_penalty"]

        step_info["route_completion"] = vehicle.navigation.route_completion

        step_info["step_reward"] = reward

        # Add additional information to step_info for evaluations
        step_info["lateral_dist"] = lateral_now


        return reward, step_info

   

    
    def get_single_observation(self):
        o = State_with_pede_woHist(self.config)
        return o
    

    def reset(self):
        if self.test_mode:
            self.risk_logger.close()
            self.risk_logger.reset()

            self.frame_logger.reset()



        obs, info = super().reset()

        #self.predictor._hard_init_buffer(engine=self.engine)
        
        return obs, info

    def set_cost_fn_mode(self, mode):
        self.cost_fn_mode = mode
        print('Cost function mode set to:', mode)
    
    def truncate_data(self, data, n):
        truncated = []
        for entry in data:
            truncated_entry = {
                'id': entry['id'],
                'pos_list': entry['pos_list'][:n],  # Keep first n positions
                'cov_list': entry['cov_list'][:n]    # Keep first n covariance matrices
            }
            truncated.append(truncated_entry)
        return truncated

    def get_ttc_metrics(self, vehicle):

        NUM_others = 8

        _, detected_objects = self.engine.get_sensor("lidar").perceive(
                vehicle,
                physics_world=self.engine.physics_world.dynamic_world,
                num_lasers=vehicle.config["lidar"]["num_lasers"],
                distance=vehicle.config["lidar"]["distance"],
                show=vehicle.config["show_lidar"],
            )

        surrounding_walkers = self.get_surrounding_walkers(detected_objects)
        surrounding_vehicles = get_surrounding_vehicles(vehicle=vehicle, engine=self.engine)
        surrounding_walkers = list(surrounding_walkers)[:NUM_others]
        
        two_d_ttc_walkers = []
        two_d_ttc_vehicles = []

        for walker in surrounding_walkers:
            if walker is not None:
                ttc = compute_ttc_components(
                    ego=vehicle,
                    other=walker,
                    )
                two_d_ttc_walkers.append(ttc)
            else:
                two_d_ttc_walkers.append(np.inf)

        
        for sur_vehicle in surrounding_vehicles:
            if sur_vehicle is not None:
                ttc = compute_ttc_components(
                    ego=vehicle,
                    other=sur_vehicle,
                    )
                two_d_ttc_vehicles.append(ttc)
            else:
                two_d_ttc_vehicles.append(np.inf)

        return two_d_ttc_walkers, two_d_ttc_vehicles  






class RealTrafficNoCostEnv(ScenarioEnv):
    def default_config(self) -> Config:
        config = super(RealTrafficNoCostEnv, self).default_config()
        config.update(
        {
            "vehicle_config": 
            {
              "lidar": {
                  "num_pedestrians": 4,
              }
            }
        }
        )
        return config
    

    def __init__(self, config = None):
        super(RealTrafficNoCostEnv, self).__init__(config)
        self.predictor = Prediction(SEED=11263)
        self.predict_trj = False
        self.c_x_acc = 0 
        self.c_y_acc = 0
        self.c_x_v_past, self.c_y_v_past = 0, 0
        self.dt = 0.1
        self.frenet = True
        self.desire_target_time = 2.0
        self.target_time = self.desire_target_time
        self.ego_planned_traj = []
        #self.lazy_init()
        self.test_mode = True

        self.date = "2024-06-25"
        self.cost_limit = 0
        self.cost_fn_mode = "standard"

        if self.test_mode:
            date = self.date
            cost_limit = self.cost_limit


            self.risk_logger = risk_logger(
                file_path=f"risk_results/{date}_{cost_limit}/{self.cost_fn_mode}",
                fieldnames=[
                            "ego_risk_max",
                            "obj_risk_max",
                            "ego_harm_max",
                            "obj_harm_max",
                            "bayes_cost",
                            "equality_cost",
                            "maximin_cost",
                            "ego_cost",
                            "obj_id_type",
                            "ego_speed",
                            "ego_long",
                            "ego_lat",
                            "info",
                            "obs",
                            "ttc_walker",
                            "ttc_vehicle",
                            "relevante_objects",
                            ])
            self.frame_logger = frame_logger(
                base_path=f"risk_results/{date}_{cost_limit}/{self.cost_fn_mode}"
            )

        else:
            self.risk_logger = None
            self.frame_logger = None

        if self.frenet:
            self.planner = Frenet_planner(pid_control=True)
        else:
            self.planner = Quintic_planner(pid_control=True)

    @property
    def action_space(self) -> gym.Space:

        return spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

    
    def get_surrounding_walkers(self, detected_objects) -> Set:

        walkers = set()
        objs = detected_objects
        for ret in objs:
            if isinstance(ret, Pedestrian):
                walkers.add(ret)
        return walkers
    
    def get_relevante_objects(self):

        _, objs = self.obs_surrounding_objects()

        true_id = ['6817', '3266', '5132', '1582', '1373', '2093', '196','1529']
        id_ls = self.engine.managers['traffic_manager'].obj_id_to_scenario_id
        if id_ls is None:
            print("Warning: obj_id_to_scenario_id() returned None")
            matching_ids = []
        else:
            matching_ids = [k for k, v in id_ls.items() if v in true_id]
        relevante_objs = []
        for obj in objs:
            if hasattr(obj, "name") and obj.id in matching_ids:
                relevante_objs.append(obj.id)
            else:
                pass
        return relevante_objs
    
    def get_surrounding_vehicles(self, detected_objects) -> Set:

        vehicles = set()
        objs = detected_objects
        for ret in objs:
            if isinstance(ret, BaseVehicle):
                vehicles.add(ret)
        return vehicles
    

    def get_walkers_info(self, detected_objects, ego_vehicle, perceive_distance=50, num_others= 4):

        walkers = list(self.get_surrounding_walkers(detected_objects))

        walkers.sort(
            key=lambda v: norm(ego_vehicle.position[0] - v.position[0], ego_vehicle.position[1] - v.position[1])
        )

        if walkers:
            walkers.sort(
                key=lambda v: norm(self.agent.position[0] - v.position[0], self.agent.position[1] - v.position[1])
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




    def obs_surrounding_objects(self):

        cloud_points, detected_objects = self.engine.get_sensor("lidar").perceive(
                self.agent,
                physics_world=self.engine.physics_world.dynamic_world,
                num_lasers=self.agent.config["lidar"]["num_lasers"],
                distance=self.agent.config["lidar"]["distance"],
                show=self.agent.config["show_lidar"],
            )


        return cloud_points, detected_objects
    
    def groundtruth_trajs(self, n_points):

        traffic_manager = self.engine._managers['traffic_manager']
        traffic_data = traffic_manager.current_traffic_data

        ego_vehicle = traffic_manager.ego_vehicle

        
        current_surrounding_vehicles = get_surrounding_vehicles(ego_vehicle,self.engine)

        # Find the surrounding vehicles ids
        surround_v_ids = []
        if current_surrounding_vehicles:
            for v in current_surrounding_vehicles:
                if v is not None:
                    surround_v_ids.append(v.id)
        

        # All obj ids to scenario ids
        obj_id_to_scenario_id_lst = traffic_manager.obj_id_to_scenario_id

        # Filter the traffic data ind list with target id list
        filtered_obj_id_lst = {key: obj_id_to_scenario_id_lst[key] for key in surround_v_ids if key in obj_id_to_scenario_id_lst}

        
        #Match the surrounding vehicles ids with the groundtruth data

        traffic_data_dict = {item: data for item, data in traffic_data.items()}

        # raw traffic data with correct vehicle ids in scenario
        filtered_traffic_data = [(key, traffic_data_dict[item]) for key, item in filtered_obj_id_lst.items() if item in traffic_data_dict]

        # Extract only relevant data
        ground_truth_trajs = []
        for key, data in filtered_traffic_data:
            ground_truth_trajs.append(
                {
                    "id": key,
                    "position": data['state']['position'][:,:2][:n_points],
                    "heading": data['state']['heading'][:n_points],
                    "velocity": data['state']['velocity'][:n_points],
                }
            )
        ground_truth_trajs = map_gt_to_prediction(ground_truth_trajs, engine=self.engine, n_points=n_points)


        return ground_truth_trajs


    def step(self, actions: Union[Union[np.ndarray, list], Dict[AnyStr, Union[list, np.ndarray]], int]):
        


        # Get current vehicle info

        ego_vehicle =self.agent
        if ego_vehicle.lane in ego_vehicle.navigation.current_ref_lanes:
            current_lane = ego_vehicle.lane
        else:
            current_lane = ego_vehicle.navigation.current_ref_lanes[0]
        
        
        long_now, lateral_now = current_lane.local_coordinates(ego_vehicle.position)
        long_last, lateral_last = current_lane.local_coordinates(ego_vehicle.last_position)

        c_x_v = (long_now -long_last)/ self.dt
        c_y_v = (lateral_now - lateral_last)/ self.dt

        # ACC computation
        c_x_acc = (c_x_v - self.c_x_v_past) / self.dt
        c_y_acc = (c_y_v - self.c_y_v_past) / self.dt

        c_a = math.sqrt(c_x_acc**2 + c_y_acc**2)


        self.target_time, target_lateral, target_speed_long = convert_action_to_target_point(
            actions=actions, vehicle=ego_vehicle,current_lane=current_lane, velocity=[c_x_v,c_y_v]
        )

        
        target_long = None #not used

        if self.frenet:
            current_frenet_state = Frenet_State(
                s=long_now,
                s_d=c_x_v,
                s_dd=self.c_x_acc,
                d=lateral_now,
                d_d=c_y_v,
                d_dd=self.c_y_acc
            )
            fp_list,fp_cost = self.planner.plan(
                frenet_state=current_frenet_state,
                target_lateral=target_lateral + current_frenet_state.d,
                target_long=target_long,
                target_speed_long=target_speed_long,
                target_time=self.target_time,
                current_lane=current_lane
            )
        
        else:
            fp_list = self.planner.plan(
                sx=ego_vehicle.position[0], sy=ego_vehicle.position[1],
                syaw=ego_vehicle.heading_theta, sv=ego_vehicle.speed,
                sa=math.sqrt(self.c_x_acc**2 + self.c_y_acc**2),
                gx=target_long, gy=target_lateral,
                gyaw=ego_vehicle.heading_theta, gv=target_speed_long,
                ga=None, max_accel=5, max_jerk=3, gT=self.target_time
            )
        pid_control = True
        cumulated_reward = 0
        self.ego_planned_traj = fp_list

        traj_cost = self.traj_cost_function('default_agent')

        ego_long_ls = []
        ego_lat_ls = []
        ego_speed_ls = []
        ttc_walker_ls = []
        ttc_vehicle_ls = []
        info_ls = []
        relevante_object_ls = []

        for i in range(len(fp_list.t)):
            
            if pid_control:
                action = self.planner.act(
                    target_x=fp_list.x[i], target_y=fp_list.y[i],
                    target_yaw=fp_list.yaw[i], target_speed=fp_list.v[i],
                    ego_vehicle=ego_vehicle
                )

                obs, reward, termination, truncate, info = super().step(action)

                if self.test_mode:
                    ret = self.render(mode="bev",
                    screen_size=(300, 300),)
                    if ret is not None:
                        self.frame_logger.save_frame(ret)
            else:
                ego_vehicle.set_position([fp_list.x[i], fp_list.y[i]])
                ego_vehicle.set_heading_theta(fp_list.yaw[i])
                obs, reward, termination, truncate, info = super().step([0,0])
                #self.render(mode="bev")
            
            if self.test_mode:
                ego_lat_ls.append(ego_vehicle.navigation.last_current_lat[-1])
                ego_long_ls.append(ego_vehicle.navigation.last_current_long[-1])
                ego_speed_ls.append(ego_vehicle.speed)
                info_ls.append(info)
                ttc_walkers, ttc_vehicles = self.get_ttc_metrics(ego_vehicle)
                relevante_objects = self.get_relevante_objects()
                relevante_object_ls.append(relevante_objects)
                ttc_walker_ls.append(ttc_walkers)
                ttc_vehicle_ls.append(ttc_vehicles)

            cumulated_reward += reward
            if termination or truncate:
                break
        #cumulated_reward = cumulated_reward - fp_cost[1] - fp_cost[2]*0.05
        cumulated_reward = cumulated_reward - fp_cost[2]*0.05  - traj_cost

        # Update past velocities
        self.c_x_v_past, self.c_y_v_past = c_x_v, c_y_v

        info["cost"] = traj_cost
        if self.test_mode:
            #Additional log for evaluation
            log_data = {
                        "ego_speed": ego_speed_ls,
                        "ego_long": ego_long_ls,
                        "ego_lat": ego_lat_ls,
                        "info": info_ls,
                        "obs": obs,
                        "ttc_walker": ttc_walker_ls,
                        "ttc_vehicle": ttc_vehicle_ls,
                        "relevante_objects": relevante_object_ls,
                    }
            
            self.risk_logger.add_log(log_data)
        
        return obs, cumulated_reward, termination, info

    def done_function(self, vehicle_id: str):

        done, done_info = super().done_function(vehicle_id)

        if done_info['crash_vehicle']:
            done = True
        elif done_info['crash_human']:
            done = True

        return done, done_info

    def traj_cost_function(self, vehicle_id: str):
        vehicle = self.agents[vehicle_id]
        predict = self.predict_trj
        total_risk_cost = 1e-6
        n_points = len(np.arange(0.0, self.target_time, 0.1))

        if predict:
            neib_trajs_list = self.predictor.step(engine=self.engine, ego_vehicle=vehicle)
        else:
            neib_trajs_list = self.groundtruth_trajs(n_points=n_points)

        # Extend ego planned traj for risk prediction        
        ego_traj = get_orientation_velocity_and_shape_of_ego(
            ego_traj=self.ego_planned_traj,
            ego=vehicle,
            n_points=n_points
        )

        # Extend predicted trajs of neibours for risk prediction    
        neib_trajs_list = get_orientation_velocity_and_shape_of_prediction(
            obj_predictions = neib_trajs_list,
            engine = self.engine,
            dt=0.1
        )
        total_risk_cost = calc_traj_costs(
            ego_traj = ego_traj,
            obj_predictions = neib_trajs_list,
            engine = self.engine,
            n_points=n_points,
            cost_fn_mode=self.cost_fn_mode,
            only_vehicle = False,
            risk_harm_logger=self.risk_logger,
            test_mode=self.test_mode,

        )
        return total_risk_cost
        
    def reward_function(self, vehicle_id: str):
        vehicle = self.agents[vehicle_id]
        step_info = dict()
        heading_penalty = 0.1
        cross_track_reward = 0.1
        time_reward = 0.1
        desired_speed = 45 #km/h
        use_lateral_reward = False
        speed_reward = 0.7
        driving_reward = 0.1

    

        # Reward for moving forward in current lane
        if vehicle.lane in vehicle.navigation.current_ref_lanes:
            current_lane = vehicle.lane
            positive_road = 1
        else:
            current_lane = vehicle.navigation.current_ref_lanes[-1]
            current_road = vehicle.navigation.current_road
            positive_road = 1 if not current_road.is_negative_road() else -1
        long_last, _ = current_lane.local_coordinates(vehicle.last_position)
        long_now, lateral_now = current_lane.local_coordinates(vehicle.position)


        # reward for lane keeping, without it vehicle can learn to overtake but fail to keep in lane
        if use_lateral_reward:
            lateral_factor = np.clip(1 - 2 * abs(lateral_now) / vehicle.navigation.get_current_lane_width(), 0.0, 1.0)
        else:
            lateral_factor = 1.0
        
       

        r_speed = ( 1 - (abs(vehicle.speed_km_h -  desired_speed) / desired_speed)) *  positive_road
        #r_speed = (vehicle.speed_km_h/vehicle.max_speed_km_h) *  positive_road
        #r_heading = -heading_diff
        r_progress = (long_now - long_last) * lateral_factor * positive_road
        #r_crosscrack = math.pow(0.001, (abs(e_y) * 0.6))

        
        reward = speed_reward * r_speed + \
                 driving_reward * r_progress
        

        if self._is_arrive_destination(vehicle):
            reward += self.config["success_reward"]
        elif self._is_out_of_road(vehicle):
            reward -= self.config["out_of_road_penalty"]
        elif vehicle.crash_vehicle:
            reward -= self.config["crash_vehicle_penalty"]
        elif vehicle.crash_object:
            reward -= self.config["crash_object_penalty"]
        elif vehicle.crash_human:
            reward -= self.config["crash_human_penalty"]

        step_info["route_completion"] = vehicle.navigation.route_completion

        step_info["step_reward"] = reward

        return reward, step_info

    def cost_function(self, vehicle_id: str):

        # Dummy cost function
        step_info = dict()
        total_risk_cost = 1e-6
        step_info["cost"] = total_risk_cost

        return total_risk_cost, step_info

    
    def get_single_observation(self):
        o = State_with_pede_woHist(self.config)
        return o
    

    def reset(self):
        if self.test_mode:
            self.risk_logger.close()
            self.risk_logger.reset()

            self.frame_logger.reset()

        obs, info = super().reset()

        #self.predictor._hard_init_buffer(engine=self.engine)
        
        return obs, info

    def set_cost_fn_mode(self, mode):
        self.cost_fn_mode = mode
        print('Cost function mode set to:', mode)

    def get_ttc_metrics(self, vehicle):

        NUM_others = 4

        _, detected_objects = self.engine.get_sensor("lidar").perceive(
                vehicle,
                physics_world=self.engine.physics_world.dynamic_world,
                num_lasers=vehicle.config["lidar"]["num_lasers"],
                distance=vehicle.config["lidar"]["distance"],
                show=vehicle.config["show_lidar"],
            )

        surrounding_walkers = self.get_surrounding_walkers(detected_objects)
        surrounding_vehicles = get_surrounding_vehicles(vehicle=vehicle, engine=self.engine)

        surrounding_walkers = list(surrounding_walkers)[:NUM_others]
        
        two_d_ttc_walkers = []
        two_d_ttc_vehicles = []

        for walker in surrounding_walkers:
            if walker is not None:
                ttc = compute_ttc_components(
                    ego=vehicle,
                    other=walker,
                    )
                two_d_ttc_walkers.append(ttc)
            else:
                two_d_ttc_walkers.append(np.inf)

        for sur_vehicle in surrounding_vehicles:
            if sur_vehicle is not None:
                ttc = compute_ttc_components(
                    ego=vehicle,
                    other=sur_vehicle,
                    )
                two_d_ttc_vehicles.append(ttc)
            else:
                two_d_ttc_vehicles.append(np.inf)

        return two_d_ttc_walkers, two_d_ttc_vehicles


class Obs_Flatten(gym.Env):
    def __init__(self, env):
        self.env = env
        self.observation_space = flatten_space(env.observation_space)
        self.action_space = env.action_space

    def reset(self, seed: Union[None, int] = None):
        obs, _ = self.env.reset(seed=seed)


        if isinstance(obs, dict) and set(obs.keys()) == {"bev", "state"}:
            # Flatten the observations
            obs = flatten_obs(obs)
        else:
            raise ValueError("The observation format returned by `reset()` is not as expected.")
        
        return obs, _

    
    def step(self, actions: Union[Union[np.ndarray, list], Dict[AnyStr, Union[list, np.ndarray]], int]):
        obs, reward, termination, truncate, info = self.env.step(actions)

        if isinstance(obs, dict) and set(obs.keys()) == {"bev", "state"}:
            # Flatten the observations
            obs = flatten_obs(obs)
        else:
            raise ValueError("The observation format returned by `step()` is not as expected.")
        
        return obs, reward, termination, truncate, info
    
    



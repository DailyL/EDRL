import numpy as np
from metadrive.component.vehicle.vehicle_type import DefaultVehicle, XLVehicle, LVehicle, MVehicle, SVehicle, VaryingDynamicsVehicle, vehicle_type
from metadrive.component.traffic_participants.cyclist import Cyclist
from metadrive.component.traffic_participants.pedestrian import Pedestrian
from metadrive.component.static_object.traffic_object import TrafficCone, TrafficWarning, TrafficBarrier
from metadrive.component.traffic_light.base_traffic_light import BaseTrafficLight
from ..collision_probability import separating_axis_theorem, get_four_corners

# Dictionary for existence of protective crash structure.
obstacle_types = {
    DefaultVehicle,
    XLVehicle,
    LVehicle,
    MVehicle,
    SVehicle,
    VaryingDynamicsVehicle,
    Cyclist,
    Pedestrian,
    TrafficCone,
    TrafficWarning,
    TrafficBarrier,
}

def get_obstacle_mass(obstacle_type):

    if obstacle_type in obstacle_types:
        return obstacle_type.MASS
    elif obstacle_type == BaseTrafficLight:
        return 20
    else:
        return 0
    


def calc_crash_angle(ego_traj, obj):
    """
    Calculate the collision angle if there is any collision possible
    """
    if isinstance(ego_traj, dict):
        # EGO vehicle trajs
        ego_traj_pose = ego_traj['pos_list']
        pred_length = len(ego_traj_pose)
        traj_x = ego_traj_pose[:, 0][0:pred_length]
        traj_y = ego_traj_pose[:, 1][0:pred_length]
        ego_length = ego_traj['shape']['length']
        ego_width = ego_traj['shape']['width']
    
    else:
        return None

    obj_id = obj['id']
    obj_mean_list = obj['pos_list']
    obj_cov_list = obj['cov_list']

    # Handle missing 'orientation_list' key
    if 'orientation_list' not in obj:
        print(f"Warning: 'orientation_list' missing for object {obj_id}. Skipping this object.")
        print(obj)
        return None

    obj_yaw_list = obj['orientation_list']
    obj_length = obj['shape']['length']
    obj_width = obj['shape']['width']
    obj_crush_angle = []
    obj_inv_cov_list = [np.linalg.pinv(cov) for cov in obj_cov_list]

    for i in range(1, len(traj_x)):
        if i >= len(obj_mean_list):
            break

        # get the current position of the ego vehicle
        ego_pos = [traj_x[i], traj_y[i]]
        ego_yaw = ego_traj['orientation_list'][i]
        
        # get the mean and the covariance of the prediction
        mean = obj_mean_list[i - 1]
        obj_yaw = obj_yaw_list[i - 1]
        
        # Check for overlap using SAT (Separating Axis Theorem)
        ego_corners = get_four_corners(
            position = ego_pos,
            length = ego_length,
            width = ego_width,
            angle = ego_yaw
        )
        obj_corners = get_four_corners(
            position = mean,
            length = obj_length,
            width = obj_width,
            angle = obj_yaw
        )
        
        if separating_axis_theorem(ego_corners, obj_corners):
            crush_angle = calculate_relative_angle(
                heading_ego = ego_yaw,
                heading_other = obj_yaw,
            )

            obj_crush_angle.append((crush_angle, ego_yaw, obj_yaw))
        else:
            obj_crush_angle.append(None) 
                
                                
    return obj_crush_angle if obj_crush_angle else None



def calculate_relative_angle(heading_ego, heading_other):
    """ Calculate the relative angle between two vehicle headings in radians, then convert to degrees. """
    # Compute the relative angle
    relative_angle = heading_ego - heading_other
    
    # Normalize the angle to the range [-pi, pi]
    relative_angle = np.arctan2(np.sin(relative_angle), np.cos(relative_angle))
    
    # Convert to degrees
    return relative_angle


def angle_range(angle):
    """
    Return an angle in [rad] in the interval ]-pi; pi].

    Args:
        angle (float): Angle in rad.
    Returns:
        float: angle in rad in the interval ]-pi; pi]

    """
    while angle <= -np.pi or angle > np.pi:
        if angle <= -np.pi:
            angle += 2 * np.pi
        elif angle > np.pi:
            angle -= 2 * np.pi

    return angle



def calc_delta_v(vehicle_1, vehicle_2, pdof):
    """
    Calculate the difference between pre-crash and post-crash speed.

    Args:
        vehicle_1 (HarmParameters): dictionary with crash relevant parameters
            for the first vehicle
        vehicle_2 (HarmParameters): dictionary with crash relevant parameters
            for the second vehicle
        pdof (float): crash angle [rad].

    Returns:
        float: Delta v for the first vehicle
        float: Delta v for the second vehicle
    """
    delta_v = np.sqrt(
        np.power(vehicle_1.velocity, 2)
        + np.power(vehicle_2.velocity, 2)
        + 2 * vehicle_1.velocity * vehicle_2.velocity * np.cos(pdof)
    )

    veh_1_delta_v = vehicle_2.mass / (vehicle_1.mass + vehicle_2.mass) * delta_v
    veh_2_delta_v = vehicle_1.mass / (vehicle_1.mass + vehicle_2.mass) * delta_v

    return veh_1_delta_v, veh_2_delta_v
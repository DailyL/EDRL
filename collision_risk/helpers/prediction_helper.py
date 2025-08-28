import numpy as np 
from metadrive.component.vehicle.vehicle_type import BaseVehicle, DefaultVehicle, XLVehicle, LVehicle, MVehicle, SVehicle, VaryingDynamicsVehicle, vehicle_type
from metadrive.component.traffic_participants.cyclist import Cyclist
from metadrive.component.traffic_participants.pedestrian import Pedestrian
from metadrive.component.static_object.traffic_object import TrafficCone, TrafficWarning, TrafficBarrier
from metadrive.component.traffic_light.base_traffic_light import BaseTrafficLight
from metadrive.type import MetaDriveType
from metadrive.utils.math import norm, clip
from metadrive.component.traffic_participants.pedestrian import Pedestrian
from metadrive.component.traffic_participants.cyclist import Cyclist 
from typing import Set


def get_orientation_velocity_and_shape_of_prediction(
    obj_predictions, engine, safety_margin_length=1.0, safety_margin_width=0.5, dt = 0.1
):
    """
    Extend the prediction by adding information about the orientation, velocity and the shape of the predicted obstacle.

    Args:
        predictions (dict): Prediction dictionary that should be extended.
        scenario (Scenario): Considered scenario.

    Returns:
        dict: Extended prediction dictionary.
    """

    updated_predictions = []

    for obj_pred in obj_predictions:

        try: 
            obj = engine.get_objects()[obj_pred['id']]
        except KeyError:
            # Log missing object and skip it
            #print(f"Warning: Object with ID {obj_pred['id']} not found in engine.")
            continue

        pred_traj = obj_pred['pos_list']
        pred_length = len(pred_traj)

        

        if pred_length == 0:
            continue

        # for predictions with only one timestep, the gradient can not be derived --> use current orientation
        if pred_length == 1:

            pred_orientation = [obj.heading_theta]
            pred_v = [obj.speed_km_h]/3.6

        else:

            # Generate time vector
            t = np.arange(0.0, pred_length * dt, dt)

            # Extract x and y positions from the trajectory

            x, y = pred_traj[:, 0], pred_traj[:, 1]


            # Calculate the gradients (dx/dt, dy/dt)
            dx = np.gradient(x, t)
            dy = np.gradient(y, t)

            # if the vehicle does barely move, use the initial orientation
            # otherwise small uncertainties in the position can lead to great orientation uncertainties
            if np.all(np.abs(dx) < 1e-4) and np.all(np.abs(dy) < 1e-4):
                pred_orientation = np.full(pred_length, obj.heading_theta)
            # if the vehicle moves, calculate the orientation
            else:
                pred_orientation = np.arctan2(dy, dx)
            
            # get the velocity from the derivation of the position
            #pred_v = np.sqrt((np.power(dx, 2) + np.power(dy, 2)))
            pred_v = np.hypot(dx, dy)

        
        # add the new information to the prediction dictionary
        obj_pred['orientation_list'] = pred_orientation
        obj_pred['v_list'] = pred_v
        obj_pred['shape'] = {
            'length': obj.LENGTH + safety_margin_length,
            'width': obj.WIDTH + safety_margin_width,
        }
        obj_pred['type'] = get_type_from_class(type(obj))

        updated_predictions.append(obj_pred)

    return updated_predictions

def get_orientation_velocity_and_shape_of_ego(
    ego_traj, ego, n_points = 40, safety_margin_length=1.0, safety_margin_width=0.5,
):
    """
    Extend the prediction by adding information about the orientation, velocity and the shape of the predicted obstacle.

    Args:
        predictions (dict): Prediction dictionary that should be extended.
        scenario (Scenario): Considered scenario.

    Returns:
        dict: Extended prediction dictionary.
    """
    vehicle_id = ego.id
    length, width = ego.LENGTH, ego.WIDTH

    if ego_traj:
        if len(ego_traj.x) > 1:
            pos_list = np.column_stack((ego_traj.x, ego_traj.y))
            v_list = ego_traj.v
            yaw_list = ego_traj.yaw

            if len(pos_list) < n_points:
                #handel not enough future points
                last_pos = pos_list[-1]  # Get last position (x, y)
                last_speed = v_list[-1]  # Get last speed
                last_heading = yaw_list[-1]

                extra_positions = []
                extra_speeds = []
                extra_headings = []

                for _ in range(n_points - len(pos_list)):
                    last_pos = [
                        last_pos[0] + last_speed * np.cos(last_heading),  # Update x
                        last_pos[1] + last_speed * np.sin(last_heading)   # Update y
                    ]
                    extra_positions.append(last_pos)
                    extra_speeds.append(last_speed)  # Keep speed constant
                    extra_headings.append(last_heading)  # Keep heading constant

                pos_list = np.vstack((pos_list, np.array(extra_positions)))
                v_list = np.hstack((v_list, np.array(extra_speeds)))
                yaw_list = np.hstack((yaw_list, np.array(extra_headings)))
            
                # ADD gaussian noise to trajs
            pos_list, cov_list = add_noise_random_covariance(
                original_traj=pos_list,
                add_noise=False,
            )


            # Pack into dictionary
            ego_traj_after = {
                'id': vehicle_id,
                'pos_list': pos_list,
                'cov_list': cov_list,
                'orientation_list': yaw_list,
                'v_list': v_list,
                'shape': {'length': length + safety_margin_length, 'width': width + safety_margin_width,},
                'type':  get_type_from_class(type(ego)) 
            }

        else:
            print('No valid ego traj provided, use current position as traj')
            ego_traj_after = constant_v_propagation(ego)
    else:
        ego_traj_after = constant_v_propagation(ego)


    return ego_traj_after

def get_type_from_class(obj_class):
    if obj_class is BaseVehicle:
        return MetaDriveType.VEHICLE
    elif obj_class is DefaultVehicle:
        return DefaultVehicle
    elif obj_class is XLVehicle:
        return XLVehicle
    elif obj_class is LVehicle:
        return LVehicle
    elif obj_class is MVehicle:
        return MVehicle
    elif obj_class is SVehicle:
        return SVehicle
    elif obj_class is VaryingDynamicsVehicle:
        return SVehicle
    elif obj_class is Pedestrian:
        return Pedestrian
    elif obj_class is Cyclist:
        return Cyclist
    elif obj_class is BaseTrafficLight:
        return MetaDriveType.TRAFFIC_LIGHT
    elif obj_class is TrafficBarrier:
        return MetaDriveType.TRAFFIC_BARRIER
    elif obj_class is TrafficCone:
        return MetaDriveType.TRAFFIC_CONE
    else:
        return MetaDriveType.OTHER



def constant_v_propagation(ego_vehicle, pred_hor = 40, dt = 0.1):

    """
    Generate future poses with constant speed and current heading
    """
    vehicle_id = ego_vehicle.id
    x, y = ego_vehicle.position
    current_orientation = ego_vehicle.heading_theta
    current_v = ego_vehicle.speed_km_h/3.6

    length, width = ego_vehicle.LENGTH, ego_vehicle.WIDTH



    # Initialize lists to store results
    pos_list = []
    cov_list = []
    orientation_list = []
    v_list = []

    for i in range(pred_hor):
        # Calculate new position based on constant speed and heading
        x += current_v * dt * np.cos(current_orientation)
        y += current_v * dt * np.sin(current_orientation)
        pos_list.append([x, y])
        
        # Generate a simple orientation model (slightly decreasing for example)
        orientation = current_orientation - i * 0.001  # adjust to model your system's orientation changes
        orientation_list.append(orientation)
        
        # Update velocity slightly to simulate a non-constant speed (e.g., Gaussian noise)
        current_v += np.random.normal(0, 0.1)
        v_list.append(current_v)
        
        # Covariance matrix (uncertainty grows slightly with each step)
        cov_matrix = np.array([[0.1 + i * 0.01, 0.01 * np.random.randn()], [0.01 * np.random.randn(), 0.05 + i * 0.01]])
        cov_list.append(cov_matrix)
    
    # Convert lists to numpy arrays
    pos_list = np.array(pos_list, dtype=float)
    cov_list = np.array(cov_list, dtype=float)
    orientation_list = np.array(orientation_list, dtype=float)
    v_list = np.array(v_list, dtype=float)
    
    # Pack into dictionary
    prediction_data = {
        'id': vehicle_id,
        'pos_list': pos_list,
        'cov_list': cov_list,
        'orientation_list': orientation_list,
        'v_list': v_list,
        'shape': {'length': length, 'width': width},
        'type':  get_type_from_class(type(ego_vehicle)) 
    }
    
    return prediction_data



def map_gt_to_prediction(ground_truth_trajs, engine, n_points):
    """
    Convert the traj data from engine
    To current traj, according to current position of vehicle 

    Args:
        ground_truth_trajs (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    new_ground_truth_trajs = []
    for v_traj in ground_truth_trajs:
        current_v = engine.get_objects()[(v_traj["id"])]
        v_current_pos = current_v.position

        traj_list = start_from_current_point_traj(
            trajs=v_traj["position"],
            start_point=v_current_pos,
            length=n_points,
            )
        
        # ADD gaussian noise to trajs
        """ traj_list, cov_list = add_noise_to_trajectory(
            trajectory=traj_list,
            add_noise=True,
        ) """

        traj_list, cov_list = add_noise_random_covariance(
            original_traj=traj_list,
            add_noise=True,
        )
        
        new_ground_truth_trajs.append(
            {
                "id": v_traj["id"],
                "pos_list": traj_list,
                "cov_list": cov_list,
            }
        )
    return new_ground_truth_trajs




def start_from_current_point_traj(trajs, start_point, length=40):

    # Compute distances and find the closest index
    dists = np.linalg.norm(trajs - start_point, axis=1)
    closest_idx = np.argmin(dists)  # Index of the closest point

    # Create the new trajectory starting from closest_idx
    new_traj = trajs[closest_idx:]

    # Extend if not enough points
    if len(new_traj) < length:
        last_point = new_traj[-1]
        padding = np.tile(last_point, (length - len(new_traj), 1))  # Repeat last point
        new_traj = np.vstack((new_traj, padding))  # Stack vertically
    
    return new_traj[:length]  # Ensure final shape is (length, 2)

    
def add_noise_to_trajectory(
    trajectory,  # Single trajectory (list/array of [x, y] points)
    noise_mean=(0, 0),
    noise_cov=[[1, 0], [0, 1]],
    add_noise=True,
    dtype=np.float32
):
    """
    Add Gaussian noise to a single trajectory (optional) and return 
    the noisy trajectory + covariance matrices as NumPy arrays.
    
    Args:
        trajectory (list): A single trajectory (list of (x, y) points).
        noise_mean (tuple): Mean of the Gaussian noise.
        noise_cov (list): 2x2 covariance matrix for the noise.
        add_noise (bool): If True, add noise; otherwise return original.
        dtype: Data type for arrays (default: float32).
        
    Returns:
        noisy_trajectory (np.ndarray): Shape [N, 2].
        cov_list (np.ndarray): Shape [N, 2, 2].
    """


    noise_mean = np.array(noise_mean, dtype=dtype)
    noise_cov = np.array(noise_cov, dtype=dtype)
    zero_cov = np.zeros_like(noise_cov)
    
    # Convert trajectory to NumPy array
    trajectory_np = np.array(trajectory, dtype=dtype)
    n_points = trajectory_np.shape[0]
    
    if add_noise:
        noise = np.random.multivariate_normal(
            noise_mean, noise_cov, n_points
        ).astype(dtype)
        noisy_trajectory = trajectory_np + noise
        cov_list = np.tile(noise_cov, (n_points, 1, 1))
    else:
        noisy_trajectory = trajectory_np.copy()
        cov_list = np.tile(zero_cov, (n_points, 1, 1))
    
    return noisy_trajectory, cov_list


def add_noise_random_covariance(original_traj, var_range=(0.1, 1), rho_range=(-0.2, 0.2), add_noise=True,):
    """
    Adds zero-mean Gaussian noise with random variances and covariance to a trajectory.
    
    Parameters:
    original_traj (np.ndarray): N x 2 array of (x, y) points.
    var_range (tuple): Range (min, max) for sampling variances. Default is (0.1, 1.0).
    rho_range (tuple): Range (min, max) for sampling correlation coefficient. Default is (-1, 1).
    
    Returns:
    noisy_traj (np.ndarray): Noisy trajectory with shape N x 2.
    covariances (list): List of 2x2 covariance matrices (one per point).
    """
    original_traj = np.array(original_traj)

    n = original_traj.shape[0]
    noisy_traj = np.zeros_like(original_traj)
    covariances = []
    
    if add_noise:
        for i in range(n):
            # Scale variance by timestep (e.g., linearly)
            t_factor = (i + 1) / n  # Normalized to [1/n, 1]
            var_x = np.random.uniform(var_range[0] * t_factor, var_range[1] * t_factor)
            var_y = np.random.uniform(var_range[0] * t_factor, var_range[1] * t_factor)

            """ # Randomly sample variances for x and y
            var_x = np.random.uniform(var_range[0], var_range[1])
            var_y = np.random.uniform(var_range[0], var_range[1]) """
            
            # Randomly sample correlation coefficient (ρ)
            rho = np.random.uniform(rho_range[0], rho_range[1])
            
            # Compute covariance term (cov_xy = ρ * σ_x * σ_y)
            cov_xy = rho * np.sqrt(var_x * var_y)
            
            # Build the covariance matrix
            cov_matrix = np.array([[var_x, cov_xy], [cov_xy, var_y]])
            
            # Generate zero-mean Gaussian noise for this point
            noise = np.random.multivariate_normal(mean=[0, 0], cov=cov_matrix)
            noisy_traj[i] = original_traj[i] + noise
            covariances.append(cov_matrix)
    else:
        noisy_traj = original_traj.copy()
        covariances = [np.zeros((2, 2)) for _ in range(n)]
    
    return noisy_traj, covariances


def unprot_partic_trajs(engine, n_points):


    traffic_manager = engine._managers['traffic_manager']
    traffic_data = traffic_manager.current_traffic_data
    ego_vehicle = traffic_manager.ego_vehicle

    pedetrains, cyclist = obs_surrounding_objects(vehicle=ego_vehicle,engine=engine)
    unprotected_road_users = pedetrains + cyclist


    # Find the surrounding vehicles ids
    road_user_ids = []
    if unprotected_road_users:
        for ru in unprotected_road_users:
            if ru is not None:
                road_user_ids.append(ru.id)
    
    # All obj ids to scenario ids
    obj_id_to_scenario_id_lst = traffic_manager.obj_id_to_scenario_id

    # Filter the traffic data ind list with target id list
    filtered_obj_id_lst = {key: obj_id_to_scenario_id_lst[key] for key in road_user_ids if key in obj_id_to_scenario_id_lst}

    #Match the surrounding vehicles ids with the groundtruth data

    traffic_data_dict = {item: data for item, data in traffic_data.items()}

    # raw traffic data with correct vehicle ids in scenario
    filtered_traffic_data = [(key, traffic_data_dict[item]) for key, item in filtered_obj_id_lst.items() if item in traffic_data_dict]

    # Extract only relevant data
    unprotected_road_users_trajs = []
    for key, data in filtered_traffic_data:
        unprotected_road_users_trajs.append(
            {
                "id": key,
                "position": data['state']['position'][:,:2][:n_points],
                "heading": data['state']['heading'][:n_points],
                "velocity": data['state']['velocity'][:n_points],
            }
        )

    unprotected_road_users_trajs = map_gt_to_prediction(unprotected_road_users_trajs, engine, n_points=n_points)

    unprotected_road_users_trajs = get_orientation_velocity_and_shape_of_prediction(
            obj_predictions = unprotected_road_users_trajs,
            engine = engine,
            dt=0.1
        )
    
    
    return unprotected_road_users_trajs







def obs_surrounding_objects(vehicle, engine):
    num_ped = 4
    _, detected_objects = engine.get_sensor("lidar").perceive(
                vehicle,
                physics_world=engine.physics_world.dynamic_world,
                num_lasers=vehicle.config["lidar"]["num_lasers"],
                distance=vehicle.config["lidar"]["distance"],
                show=vehicle.config["show_lidar"],
            )
    
    return get_pedetrain_cyclist_info(detected_objects, num_ped, vehicle)


    


def get_surrounding_walkers(detected_objects):

        walkers = set()
        objs = detected_objects
        for ret in objs:
            if isinstance(ret, Pedestrian):
                walkers.add(ret)
        return walkers


def get_surrounding_cyclist(detected_objects):

        cyclist = set()
        objs = detected_objects
        for ret in objs:
            if isinstance(ret, Cyclist):
                cyclist.add(ret)
        return cyclist




def get_pedetrain_cyclist_info(detected_objects, num_others, vehicle):

    pedetrains = list(get_surrounding_walkers(detected_objects))
    cyclist = list(get_surrounding_cyclist(detected_objects))

    
    pedetrains.sort(
        key=lambda v: norm(vehicle.position[0] - v.position[0], vehicle.position[1] - v.position[1])
    )
    pedetrains = pedetrains[:num_others]

    cyclist.sort(
        key=lambda v: norm(vehicle.position[0] - v.position[0], vehicle.position[1] - v.position[1])
    )
    cyclist = cyclist[:num_others]
    
    
    return pedetrains, cyclist
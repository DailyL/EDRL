import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2





def get_collision_probability(traj, obj_prediction):

    """
    Main function for collision probabilities calculation

    """
    collision_prob_dict = {}
    if isinstance(traj, dict):
        # EGO vehicle trajs
        ego_traj_pose = traj['pos_list']
        pred_length = len(ego_traj_pose)
        traj_x = ego_traj_pose[:, 0][0:pred_length]
        traj_y = ego_traj_pose[:, 1][0:pred_length]
        ego_length = traj['shape']['length']
        ego_width = traj['shape']['width']
    
    else:
        return collision_prob_dict

    valid_obj_predictions = [
        obj for obj in obj_prediction
        if 'orientation_list' in obj and 'cov_list' in obj
    ]

    
    for obj in valid_obj_predictions:
        obj_id = obj['id']
        obj_mean_list = obj['pos_list']
        obj_cov_list = obj['cov_list']
        obj_yaw_list = obj['orientation_list']
        obj_length = obj['shape']['length']
        obj_width = obj['shape']['width']
        obj_probs = []
        try:
            obj_inv_cov_list = [np.linalg.pinv(cov) for cov in obj_cov_list]
        except np.linalg.LinAlgError:
            print(f"Warning: Covariance matrix inversion failed for object {obj_id}. Skipping.")
            continue


        for i in range(len(traj_x)):
            if i >= len(obj_mean_list):
                break

            # get the current position of the ego vehicle
            ego_pos = [traj_x[i], traj_y[i]]
            ego_yaw = traj['orientation_list'][i]
            
            # get the mean and the covariance of the prediction
            mean = obj_mean_list[i]
            obj_yaw = obj_yaw_list[i]
            iv = obj_inv_cov_list[i]
            
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
                # If there's a potential overlap, calculate the collision probability based on uncertainty

                prob = prob_with_mahalanobis_chi_squared(
                    ego_pose = ego_pos,
                    obj_pose = mean,
                    cov_mat = iv
                )
            else:
                prob = 0  # No overlap, no collision probability
            
            obj_probs.append(prob)

        collision_prob_dict[obj_id] = obj_probs

                                
    return collision_prob_dict



def prob_with_mahalanobis_chi_squared(ego_pose, obj_pose, cov_mat, collision_threshold = 0.0):
    """
    Calculate the collision probability using the Mahalanobis distance
    and the chi-squared distribution.

    Parameters:
    r (np.array): The relative position vector (x, y) between two vehicles.
    cov (np.array): The combined covariance matrix (2x2).
    collision_threshold (float): The threshold for collision detection (e.g., size of bounding boxes).

    Returns:
    float: Collision probability.
    """
    # Compute the Mahalanobis distance
    d_m = mahalanobis(ego_pose, obj_pose, cov_mat)

    
    prob = chi2.cdf(d_m**2, df=2)

    # Compute the collision probability using a chi-squared distribution
    return prob
    


def separating_axis_theorem(corners1, corners2, tolerance=0.1, early_exit = True):
    """
    Apply the Separating Axis Theorem (SAT) to check if two rectangles collide.
    
    Parameters:
    corners1 (np.array): The corners of the first rectangle (4x2 array).
    corners2 (np.array): The corners of the second rectangle (4x2 array).
    tolerance (float): The tolerance value for near-collision detection.
    early_exit : IF we do early exit, first simple calculate the distances
    
    Returns:
    bool: True if the rectangles overlap (collide), False otherwise.
    """
    if early_exit:
        # Calculate bounding box centers
        center1 = np.mean(corners1, axis=0)
        center2 = np.mean(corners2, axis=0) 

        # Calculate bounding box half-diagonals
        half_diagonal1 = np.sqrt(np.sum((corners1[0] - corners1[2])**2)) / 2
        half_diagonal2 = np.sqrt(np.sum((corners2[0] - corners2[2])**2)) / 2
        
        # Calculate distance between centers
        center_distance = np.linalg.norm(center1 - center2)
        
        # Early exit: if center distance is greater than the sum of the half-diagonals
        if center_distance > (half_diagonal1 + half_diagonal2 + tolerance):
            return False  # No collision possible, exit early
        

    axes = np.array([
        corners1[1] - corners1[0],  # Edge 1 of rect1
        corners1[3] - corners1[0],  # Edge 2 of rect1
        corners2[1] - corners2[0],  # Edge 1 of rect2
        corners2[3] - corners2[0]   # Edge 2 of rect2
    ])
    
    axes = np.array([axis / np.linalg.norm(axis) for axis in axes])
    
    for axis in axes:
        projections1 = np.dot(corners1, axis)
        projections2 = np.dot(corners2, axis)
        if max(projections1) < min(projections2) - tolerance or max(projections2) < min(projections1) - tolerance:
            return False  # Separating axis found, no collision
    return True  # No separating axis found, collision occurs




def get_four_corners(position, length, width, angle):
    """
    Get the corners of a rectangle given its position, size, and orientation.
    
    Parameters:
    position (np.array): The center position of the rectangle (x, y).
    length (float): The length of the rectangle.
    width (float): The width of the rectangle.
    angle (float): The orientation angle of the rectangle (in radians).
    
    Returns:
    np.array: The four corners of the rectangle.
    """
    half_length = length / 2.0
    half_width = width / 2.0
    corners = np.array([[-half_length, -half_width],
                        [ half_length, -half_width],
                        [ half_length,  half_width],
                        [-half_length,  half_width]])
    rotated_corners = np.array([rotate_point(corner, angle) for corner in corners])

    return rotated_corners + np.array(position)


def rotate_point(point, angle):
    """Rotate a 2D point by an angle (in radians)."""
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle),  np.cos(angle)]])
    return np.dot(rotation_matrix, point)


def get_unit_vector(angle: float):
    """
    Get the unit vector for a given angle.

    Args:
        angle (float): Considered angle.

    Returns:
        float: Unit vector of the considered angle
    """
    return np.array([np.cos(angle), np.sin(angle)])




def get_inv_mahalanobis_dist(traj, obj_predictions, safety_margin=1.0):

    """
    Calculate the collision probabilities of a trajectory and predictions (mahalanobis).

    Args:
        traj: Predicted trajectory.
        predictions (dict): Predictions of the visible obstacles.

    Returns:
        dict: Collision probability of the trajectory per time step with the
            prediction for every visible obstacle.
    """
    collision_prob_dict = {}

    # EGO vehicle trajs
    traj = traj['pos_list']
    pred_length = len(traj)
    traj_x = traj[:, 0][0:pred_length]
    traj_y = traj[:, 1][0:pred_length]


    for obj in obj_predictions:

        obj_id = obj['id']
        obj_mean_list = obj['pos_list']
        obj_cov_list = obj['cov_list']
        obj_inv_cov_list = [np.linalg.inv(cov) for cov in obj_cov_list]
        inv_dist = []

        for i in range(len(traj_x)):
            if i < len(obj_mean_list):
                u = [traj_x[i], traj_y[i]]
                v = obj_mean_list[i]
                iv = obj_inv_cov_list[i]

                # 1e-4 is regression param to be similar to collision probability     ?????? Should we estimate another one?
                inv_dist.append(round((1e-4 / mahalanobis(u, v, iv)), 4))
            else:
                inv_dist.append(1e-6)
        collision_prob_dict[obj_id] = inv_dist

    return collision_prob_dict




############################################################################################# 
# Simple model, with the prediction of the traj of surrounding_vehicles with current velocity 



def distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def check_collisions(ego_vehicle, surrounding_vehicles, time_horizon=5, step=0.1, threshold_distance=2.0):
    p1 = ego_vehicle.position
    v1 = ego_vehicle.speed
    collision_data = []

    for t in np.arange(0, time_horizon + step, step):
        for vehicle in surrounding_vehicles:
            if vehicle is not None:
                p2 = vehicle.position
                v2 = vehicle.speed
                # Predict positions at time t
                p1_t = predict_position(p1, v1, t)
                p2_t = predict_position(p2, v2, t)
                
                # Compute distance between vehicles
                dist = distance(p1_t, p2_t)
                
                if dist <= threshold_distance:
                    # If there's a collision, compute the collision angle
                    collision_angle = compute_collision_angle(p1, v1, p2, v2, t)
                    collision_data.append((t, dist, collision_angle))
            else:
                pass
    return collision_data


def predict_position(p, v, t):
    return p + v * t

def relative_velocity(v1, v2):
    return v2 - v1

def relative_position(p1, p2):
    return p2 - p1

def compute_collision_angle(p1, v1, p2, v2, t):
    # Predict future positions
    p1_t = predict_position(p1, v1, t)
    p2_t = predict_position(p2, v2, t)
    
    # Relative position and velocity
    p_rel = relative_position(p1_t, p2_t)
    v_rel = relative_velocity(v1, v2)
    
    # Dot product and magnitudes
    dot_product = np.dot(p_rel, v_rel)
    p_rel_mag = np.linalg.norm(p_rel)
    v_rel_mag = np.linalg.norm(v_rel)
    
    # Compute collision angle
    cos_phi = dot_product / (p_rel_mag * v_rel_mag)
    phi = np.arccos(np.clip(cos_phi, -1.0, 1.0))  # Clip to avoid numerical errors
    
    return np.degrees(phi)  # Convert to degrees
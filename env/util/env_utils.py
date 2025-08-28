import math
import numpy as np
from scipy.spatial import distance
from typing import Set
from metadrive.utils.math import norm, clip
from metadrive.component.vehicle.PID_controller import PIDController
from metadrive.utils.math import wrap_to_pi
import random

class VehiclePIDController(object):

    def __init__(self, max_throttle=1, max_steering=1):

        self.max_throt = max_throttle
        self.max_steer = max_steering
        self.past_steering = 0

        self._lon_controller = PIDController(0.1, 0.001, 0.3)
        self._lat_controller = PIDController(1, .002, 0.05)

        self.heading_pid = PIDController(1.7, 0.01, 3.5)
        self.lateral_pid = PIDController(0.3, .002, 0.05)

        self.long_pid = PIDController(0.1, 0.001, 0.3)
    
    def run_long_step(self, current_speed, target_speed):
        acc = self.long_pid.get_result(current_speed - target_speed)
        return acc
    
    def run_lat_step(self,ego_vehicle, target_position, current_direction, target_direction):

        cross_track_error = calc_cross_track_error(ego_vehicle, target_position[0], target_position[1])

        steering = self.heading_pid.get_result(-wrap_to_pi(target_direction - current_direction))

        #steering += self.lateral_pid.get_result(-cross_track_error)

        return steering

    
    def run_step(self, current_speed, target_speed, ego_vehicle, target_position, current_direction, target_direction, stanly_control = True):

        acc = self.run_long_step(current_speed, target_speed)

        if not stanly_control:
            steering = self.run_lat_step(ego_vehicle, target_position, current_direction, target_direction)
        else:
            steering = self.run_stanly_controller(ego_vehicle, target_position, current_direction, target_direction)


         # Steering regulation: changes cannot happen abruptly, can't steer too much.
        smooth = False
        limit = False
        
        if smooth:
            if steering > self.past_steering + 0.5:
                steering = self.past_steering + 0.5
            elif steering < self.past_steering - 0.5:
                steering = self.past_steering - 0.5
        if limit:
            if steering >= 0:
                steering = min(self.max_steer, steering)
            else:
                steering = max(-self.max_steer, steering)

        self.past_steering = steering

        return [steering, acc]
    def initial_steering(self):
        self.past_steering = 0

    def run_stanly_controller(self, ego_vehicle, target_position, current_direction, target_direction):

        """

        Args:
            ego_vehicle (_type_): ego vehicle 
            target_position (_type_): target pose [x,y]
            current_direction (_type_): current heading
            target_direction (_type_): target heading

        Returns:
            sigma: Stanly control steering 
        """

        yaw_path = target_direction

        # stanley parameters
        ke = 0.3
        kv = 10

        x, y = ego_vehicle.position[0], ego_vehicle.position[1]
        v = ego_vehicle.speed

        psi = target_direction - current_direction # Heading error

        psi = (psi + np.pi) % (2 * np.pi) - np.pi

        crosstrack_error = calc_cross_track_error(ego_vehicle, target_position[0], target_position[1])

        yaw_cross_track = np.arctan2(y - target_position[1], x - target_position[0])

        yaw_modify = yaw_path -yaw_cross_track
        if yaw_modify > np.pi:
            yaw_modify -= 2 *np.pi
        if yaw_modify < -np.pi :
            yaw_modify += 2* np.pi
        
        crosstrack_error = abs(crosstrack_error) if yaw_modify > 0 else -abs(crosstrack_error)

        psi_crosstrack = np.arctan(ke * crosstrack_error / (kv + v))

        sigma = psi + psi_crosstrack

        return sigma




    


def find_target_lane(target_point, lanes):
    dist_list = []
    valid_lanes = []

    # Filter lanes that contain the target_point
    for lane in lanes:
        if lane.point_on_lane(target_point):  # Check if the point is within the lane polygon
            valid_lanes.append(lane)
            dist = lane.distance(target_point)
            dist_list.append(dist)
    
    if not valid_lanes:
        return None
    
    # Find the closest valid lane
    min_dist_index = dist_list.index(min(dist_list))
    return valid_lanes[min_dist_index]






def VectorFiledGuidance(vehicle):
    true_vfg = False
    x_infi = 1
    k_ey = 0.5

    if vehicle.lane in vehicle.navigation.current_ref_lanes:
        current_lane = vehicle.lane
    else:
        current_lane = vehicle.navigation.current_ref_lanes[-1]

    if true_vfg:
        # If want to calculate true VFG error, need to know how to call the reference points on target lane of current position
        r_d = distance.euclidean(current_lane.position, vehicle.position)
        e_y = math.sin(
            x_path - math.atan(
                (current_lane.position[1]-vehicle.position[1])/
                (current_lane.position[0]-vehicle.position[0])
                )
            )*r_d
    else:
         _, e_y = current_lane.local_coordinates(vehicle.position)

    # Target heading

    long_now, _ = current_lane.local_coordinates(vehicle.position)
    x_path = current_lane.heading_theta_at(long_now +1 )

    x_d = x_infi * math.atan(k_ey*e_y) + x_path

    return x_d, e_y

def get_vehicles(detected_objects) -> Set:
        from metadrive.component.vehicle.base_vehicle import BaseVehicle
        vehicles = set()
        objs = detected_objects
        for ret in objs:
            if isinstance(ret, BaseVehicle):
                vehicles.add(ret)
        return vehicles



def get_surrounding_vehicles(vehicle, engine):
    num_others = vehicle.config["lidar"]["num_others"]

    _, detected_objects = vehicle.lidar.perceive(
                                vehicle,
                                physics_world=engine.physics_world.dynamic_world,
                                num_lasers=vehicle.config["lidar"]["num_lasers"],
                                distance=vehicle.config["lidar"]["distance"],
                                )
    
    surrounding_vehicles = list(get_vehicles(detected_objects))
    surrounding_vehicles.sort(
        key=lambda v: norm(vehicle.position[0] - v.position[0], vehicle.position[1] - v.position[1])
    )
    surrounding_vehicles = surrounding_vehicles[:num_others]
    
    return surrounding_vehicles



def scale(value, min_out, max_out):
    # General scaling from [-1, 1] to [min_out, max_out] without rounding
    return ((value + 1) / 2) * (max_out - min_out) + min_out


def continuous_to_discrete(value, min_value, max_value, interval = 0.1):

    num_intervals = np.floor((max_value - min_value) / interval)
    # Scale the value to a 0 to (num_intervals - 1) range
    scaled_value = (value - min_value) / (max_value - min_value) * (num_intervals - 1)
    discrete_value = int(np.round(scaled_value))  # or use np.floor to round down
    return discrete_value


def scale_and_round(value, min_out=0.1, max_out=5, step=0.1):

    # Step 1: Scale the value from [-1, 1] to [0.1, 5]
    scaled_value = ((value + 1) / 2) * (max_out - min_out) + min_out
    
    # Step 2: Round to the nearest 0.1
    rounded_value = round(scaled_value / step) * step
    return rounded_value



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





def convert_action_to_target_point(actions, vehicle, current_lane, velocity):

    """
    Args:
    actions: action by RL agent, witin [-1,1], 4 D 
             1. Target Time interval
             2. Target longitudinal position
             3. Target lateral position
             4. Target longitudinal velocity
    vehicle: Vehicle object in Metadrive
    
    Return:
    Target points

    """

    [c_x_v, c_y_v] = velocity  # Now use x y speed, can change to lateral and longitudinal?

    MAX_SPEED = vehicle.max_speed_m_s


    MAX_ACCEL_lat = 2.0 
    MAX_ACCEL_long = 5.0 
    MAX_DECEL_long = 7.0


    MAX_ROAD_WIDTH = current_lane.width * 2

    # 1. Target Time Interval (e.g., scale to range [0.1, 10] seconds and round to 0.1s)
    target_time = scale_and_round(actions[0], min_out=0.2, max_out=2, step=0.1)

    #target_time = 1

    # 2. Target Longitudinal Position (max possible distance based on max speed and target time)
    #MAX_possible_dist = target_time * MAX_SPEED
    #target_long = scale(actions[1], min_out=0, max_out=MAX_possible_dist)


    # 3. Target Lateral Position (limit within road width and lateral acceleration constraints)
    max_lateral_position = min((MAX_ROAD_WIDTH / 2), c_y_v + MAX_ACCEL_lat * target_time)
    target_lateral = scale(actions[1], min_out=-max_lateral_position, max_out=max_lateral_position)


    # 4. Target Longitudinal Speed (limited by max speed and longitudinal acceleration/deceleration)
    max_target_speed = min(MAX_SPEED, c_x_v + MAX_ACCEL_long * target_time)
    min_target_speed = max(0, c_x_v - MAX_DECEL_long * target_time)
    
    target_speed_long = scale(actions[2], min_out=min_target_speed, max_out=max_target_speed)

    #target_acc = scale(actions[4], min_out=-MAX_DECEL_long, max_out=MAX_ACCEL_long)
    
    return target_time, target_lateral, target_speed_long


def calc_cross_track_error(vehicle, target_x, target_y):
    dx = target_x - vehicle.position[0]
    dy = target_y - vehicle.position[1]
    cross_track_error = dy * math.cos(vehicle.heading_theta) - dx * math.sin(vehicle.heading_theta)
    return cross_track_error    


def convert_action_to_target_points(actions, vehicle, current_lane, velocity):

    """
    Args:
    actions: action by RL agent, witin [-1,1], 4 D 
             1. Target Time interval
             2. Target longitudinal position
             3. Target lateral position
             4. Target longitudinal velocity
    vehicle: Vehicle object in Metadrive
    
    Return:
    Target points

    """

    [c_x_v, c_y_v] = velocity  # Now use x y speed, can change to lateral and longitudinal?

    MAX_SPEED = vehicle.max_speed_m_s


    MAX_ACCEL_lat = 2.0 
    MAX_ACCEL_long = 4.0 
    MAX_DECEL_long = 5.0

    current_road = vehicle.navigation.current_road

    MAX_ROAD_WIDTH = current_lane.width * 2

    # 1. Target Time Interval (e.g., scale to range [0.1, 10] seconds and round to 0.1s)
    target_time = scale_and_round(actions[0], min_out=0.2, max_out=3, step=0.1)

    # 2. Target Longitudinal Position (max possible distance based on max speed and target time)
    MAX_possible_dist = target_time * MAX_SPEED
    target_long = scale(actions[1], min_out=0, max_out=MAX_possible_dist)


    # 3. Target Lateral Position (limit within road width and lateral acceleration constraints)
    max_lateral_position = min((MAX_ROAD_WIDTH / 2), c_y_v + MAX_ACCEL_lat * target_time)
    target_lateral = scale(actions[2], min_out=-max_lateral_position, max_out=max_lateral_position)


    # 4. Target Longitudinal Speed (limited by max speed and longitudinal acceleration/deceleration)
    max_target_speed = min(MAX_SPEED, c_x_v + MAX_ACCEL_long * target_time)
    min_target_speed = max(0, c_x_v - MAX_DECEL_long * target_time)
    
    target_speed_long = scale(actions[3], min_out=min_target_speed, max_out=max_target_speed)

    target_acc = scale(actions[4], min_out=-MAX_DECEL_long, max_out=MAX_ACCEL_long)
    
    return target_time, target_long, target_lateral, target_speed_long, target_acc



import bisect
class CubicSpline1D:
    """
    1D Cubic Spline class

    Parameters
    ----------
    x : list
        x coordinates for data points. This x coordinates must be
        sorted
        in ascending order.
    y : list
        y coordinates for data points

    """

    def __init__(self, x, y):

        h = np.diff(x)
        if np.any(h < 0):
            raise ValueError("x coordinates must be sorted in ascending order")

        self.a, self.b, self.c, self.d = [], [], [], []
        self.x = x
        self.y = y
        self.nx = len(x)  # dimension of x

        # calc coefficient a
        self.a = [iy for iy in y]

        # calc coefficient c
        A = self.__calc_A(h)
        B = self.__calc_B(h, self.a)
        self.c = np.linalg.solve(A, B)

        # calc spline coefficient b and d
        for i in range(self.nx - 1):
            d = (self.c[i + 1] - self.c[i]) / (3.0 * h[i])
            b = 1.0 / h[i] * (self.a[i + 1] - self.a[i]) \
                - h[i] / 3.0 * (2.0 * self.c[i] + self.c[i + 1])
            self.d.append(d)
            self.b.append(b)

    def calc_position(self, x):
        """
        Calc `y` position for given `x`.

        if `x` is outside the data point's `x` range, return None.

        Returns
        -------
        y : float
            y position for given x.
        """
        if x < self.x[0]:
            return None
        elif x > self.x[-1]:
            return None

        i = self.__search_index(x)
        dx = x - self.x[i]
        position = self.a[i] + self.b[i] * dx + \
            self.c[i] * dx ** 2.0 + self.d[i] * dx ** 3.0

        return position

    def calc_first_derivative(self, x):
        """
        Calc first derivative at given x.

        if x is outside the input x, return None

        Returns
        -------
        dy : float
            first derivative for given x.
        """

        if x < self.x[0]:
            return None
        elif x > self.x[-1]:
            return None

        i = self.__search_index(x)
        dx = x - self.x[i]
        dy = self.b[i] + 2.0 * self.c[i] * dx + 3.0 * self.d[i] * dx ** 2.0
        return dy

    def calc_second_derivative(self, x):
        """
        Calc second derivative at given x.

        if x is outside the input x, return None

        Returns
        -------
        ddy : float
            second derivative for given x.
        """

        if x < self.x[0]:
            return None
        elif x > self.x[-1]:
            return None

        i = self.__search_index(x)
        dx = x - self.x[i]
        ddy = 2.0 * self.c[i] + 6.0 * self.d[i] * dx
        return ddy

    def __search_index(self, x):
        """
        search data segment index
        """
        return bisect.bisect(self.x, x) - 1

    def __calc_A(self, h):
        """
        calc matrix A for spline coefficient c
        """
        A = np.zeros((self.nx, self.nx))
        A[0, 0] = 1.0
        for i in range(self.nx - 1):
            if i != (self.nx - 2):
                A[i + 1, i + 1] = 2.0 * (h[i] + h[i + 1])
            A[i + 1, i] = h[i]
            A[i, i + 1] = h[i]

        A[0, 1] = 0.0
        A[self.nx - 1, self.nx - 2] = 0.0
        A[self.nx - 1, self.nx - 1] = 1.0
        return A

    def __calc_B(self, h, a):
        """
        calc matrix B for spline coefficient c
        """
        B = np.zeros(self.nx)
        for i in range(self.nx - 2):
            B[i + 1] = 3.0 * (a[i + 2] - a[i + 1]) / h[i + 1]\
                - 3.0 * (a[i + 1] - a[i]) / h[i]
        return B


class CubicSpline2D:
    """
    Cubic CubicSpline2D class

    Parameters
    ----------
    x : list
        x coordinates for data points.
    y : list
        y coordinates for data points.

    """

    def __init__(self, x, y):
        self.s = self.__calc_s(x, y)
        self.sx = CubicSpline1D(self.s, x)
        self.sy = CubicSpline1D(self.s, y)

    def __calc_s(self, x, y):
        dx = np.diff(x)
        dy = np.diff(y)
        self.ds = np.hypot(dx, dy)
        s = [0]
        s.extend(np.cumsum(self.ds))
        return s

    def calc_position(self, s):
        """
        calc position

        Parameters
        ----------
        s : float
            distance from the start point. if `s` is outside the data point's
            range, return None.

        Returns
        -------
        x : float
            x position for given s.
        y : float
            y position for given s.
        """
        x = self.sx.calc_position(s)
        y = self.sy.calc_position(s)

        return x, y

    def calc_curvature(self, s):
        """
        calc curvature

        Parameters
        ----------
        s : float
            distance from the start point. if `s` is outside the data point's
            range, return None.

        Returns
        -------
        k : float
            curvature for given s.
        """
        dx = self.sx.calc_first_derivative(s)
        ddx = self.sx.calc_second_derivative(s)
        dy = self.sy.calc_first_derivative(s)
        ddy = self.sy.calc_second_derivative(s)
        k = (ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2)**(3 / 2))
        return k

    def calc_yaw(self, s):
        """
        calc yaw

        Parameters
        ----------
        s : float
            distance from the start point. if `s` is outside the data point's
            range, return None.

        Returns
        -------
        yaw : float
            yaw angle (tangent vector) for given s.
        """
        dx = self.sx.calc_first_derivative(s)
        dy = self.sy.calc_first_derivative(s)
        yaw = math.atan2(dy, dx)
        return yaw




def compute_2d_ttc(ego, other, min_t=0.0):
    """
    ego, other: vehicles from env.vehicles.values(), each has .position (x,y) and .velocity (vx,vy)
    returns TTC >0 if collision predicted, else np.inf
    """

    # Get vehicle dimensions (use diagonal as a conservative estimate)
    ego_radius = np.sqrt((ego.LENGTH/2)**2 + (ego.WIDTH/2)**2)
    other_radius = np.sqrt((other.LENGTH/2)**2 + (other.WIDTH/2)**2)
    
    # Sum of radii is the distance at which we consider a collision
    collision_distance = ego_radius + other_radius

    # relative position and velocity
    r = np.array(other.position) - np.array(ego.position)
    v = np.array(other.velocity) - np.array(ego.velocity)
    v_norm2 = np.dot(v, v)
    if v_norm2 < 1e-6:  # Relative velocity near zero
        # Check if already within collision distance
        if np.linalg.norm(r) <= collision_distance:
            return 0.0
        return np.inf
    
    rv = np.dot(r, v)
    if rv >= 0:  # Not approaching
        return np.inf
    
    rr = np.dot(r, r)
    
    # If already colliding
    if rr <= collision_distance**2:
        return 0.0
    
    # Solve for time when distance = collision_distance
    c = rr - collision_distance**2
    disc = rv**2 - v_norm2 * c
    
    if disc < 0:  # No collision predicted
        return np.inf
    
    ttc = (-rv - np.sqrt(disc)) / v_norm2
    
    if ttc < min_t:  # Too soon or in the past
        return np.inf
    
    return ttc


def compute_ttc_components(ego, other, min_t=0.0, return_2d_ttc=True):
    """
    Compute longitudinal and lateral time-to-collision (TTC) using a point mass model.
    Vehicles are treated as point masses without physical dimensions.
    
    Args:
        ego: Ego vehicle object
        other: Other vehicle object
        min_t: Minimum time threshold for TTC calculation
        return_2d_ttc: Whether to calculate and return the 2D TTC
        
    Returns:
        dict: Contains 'vehicle_id', 'ttc', 'long_ttc', 'lat_ttc'
    """
    if return_2d_ttc:
        # Calculate overall 2D TTC for point masses
        # Use zero collision distance (point masses)
        r = np.array(other.position) - np.array(ego.position)
        v = np.array(other.velocity) - np.array(ego.velocity)
        v_norm2 = np.dot(v, v)
        
        if v_norm2 < 1e-6:  # Relative velocity near zero
            # Check if already at same position (unlikely)
            if np.linalg.norm(r) <= 1e-6:
                ttc_2d = 0.0
            else:
                ttc_2d = np.inf
        else:
            rv = np.dot(r, v)
            if rv >= 0:  # Not approaching
                ttc_2d = np.inf
            else:
                # Time to collision is when distance = 0
                ttc_2d = -rv / v_norm2
                if ttc_2d < min_t:
                    ttc_2d = np.inf
    
    # Calculate relative position and velocity in ego's reference frame
    r_world = np.array(other.position) - np.array(ego.position)
    v_world = np.array(other.velocity) - np.array(ego.velocity)
    
    # Convert to ego's local coordinates (longitudinal = x, lateral = y)
    heading = ego.heading_theta
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    
    # Rotate to ego's frame
    r_long = r_world[0] * cos_h + r_world[1] * sin_h
    r_lat = -r_world[0] * sin_h + r_world[1] * cos_h
    v_long = v_world[0] * cos_h + v_world[1] * sin_h
    v_lat = -v_world[0] * sin_h + v_world[1] * cos_h
    
    # Initialize TTC values
    long_ttc = np.inf
    lat_ttc = np.inf
    
    # Calculate longitudinal TTC (point mass)
    if abs(v_long) > 1e-6:  # Check if there's relative velocity in longitudinal direction
        t_long = r_long / -v_long if v_long < 0 else np.inf
        if t_long >= min_t:
            long_ttc = t_long
    elif abs(r_long) < 1e-6:  # Already at same longitudinal position
        long_ttc = 0.0
    
    # Calculate lateral TTC (point mass)
    if abs(v_lat) > 1e-6:  # Check if there's relative velocity in lateral direction
        t_lat = r_lat / -v_lat if v_lat < 0 else np.inf
        if t_lat >= min_t:
            lat_ttc = t_lat
    elif abs(r_lat) < 1e-6:  # Already at same lateral position
        lat_ttc = 0.0
    
    if return_2d_ttc:
        # Return all TTCs and vehicle ID
        return {
            'vehicle_id': other.id,
            'ttc': ttc_2d,
            'long_ttc': long_ttc,
            'lat_ttc': lat_ttc
        }
    else:
        # Return only longitudinal and lateral TTCs
        return {
            'vehicle_id': other.id,
            'long_ttc': long_ttc,
            'lat_ttc': lat_ttc
        }
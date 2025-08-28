import numpy as np 
from metadrive.component.vehicle.vehicle_type import DefaultVehicle, XLVehicle, LVehicle, MVehicle, SVehicle, VaryingDynamicsVehicle, vehicle_type
from metadrive.component.traffic_participants.cyclist import Cyclist
from metadrive.component.traffic_participants.pedestrian import Pedestrian
from metadrive.component.static_object.traffic_object import TrafficCone, TrafficWarning, TrafficBarrier
from metadrive.component.traffic_light.base_traffic_light import BaseTrafficLight
from collision_risk.configs.load_json import load_harm_parameter_json, load_risk_json
from collision_risk.helpers.properties import get_obstacle_mass, calc_crash_angle

from collision_risk.cr_utils.logistic_regression import (
    get_protected_log_reg_harm,
    get_unprotected_log_reg_harm,
)
from collision_risk.cr_utils.reference_speed import (
    get_protected_ref_speed_harm,
    get_unprotected_ref_speed_harm,
)
from collision_risk.cr_utils.reference_speed_symmetrical import (
    get_protected_inj_prob_ref_speed_complete_sym,
    get_protected_inj_prob_ref_speed_ignore_angle,
    get_protected_inj_prob_ref_speed_reduced_sym,
)
from collision_risk.cr_utils.reference_speed_asymmetrical import (
    get_protected_inj_prob_ref_speed_complete,
    get_protected_inj_prob_ref_speed_reduced,
)
from collision_risk.cr_utils.gidas import (
    get_protected_gidas_harm,
    get_unprotected_gidas_harm,
)
from collision_risk.cr_utils.logistic_regression_symmetrical import (
    get_protected_inj_prob_log_reg_complete_sym,
    get_protected_inj_prob_log_reg_ignore_angle,
    get_protected_inj_prob_log_reg_reduced_sym,
)
from collision_risk.cr_utils.logistic_regression_asymmetrical import (
    get_protected_inj_prob_log_reg_complete,
    get_protected_inj_prob_log_reg_reduced,
)


# Dictionary for existence of protective crash structure.
obstacle_protection = {
    DefaultVehicle: True,
    XLVehicle: True,
    LVehicle: True,
    MVehicle: False,
    SVehicle: False,
    VaryingDynamicsVehicle: True,
    Cyclist: False,
    Pedestrian: False,
    TrafficCone: None,
    TrafficWarning: None,
    TrafficBarrier: None,
    BaseTrafficLight: None,
}

params_mode = load_risk_json()
params_harm = load_harm_parameter_json()

"""Class with relevant parameters harm estimation."""


class HarmParameters:
    """Harm parameters class."""

    def __init__(self):
        """
        Initialize the harm parameter class.

        Parameters:
            :param type (ObstacleType): type of object according to CommonRoad
                obstacle types
            :param protection (Boolean): displays if object has a protective
                crash structure
            :param mass (Float): mass of object in kg, estimated by type if
                exact value not existent
            :param velocity (Float): current velocity of object in m/s
            :param yaw (Float): current yaw of object in rad
            :param size (Float): size of object in square meters
                (length * width)
            :param harm (Float): estimated harm value
            :param prob (Float): estimated collision probability
            :param risk (Float): estimated collision risk (prob * harm)
        """
        # obstacle parameters
        self.type = None
        self.protection = None
        self.mass = None
        self.velocity = None
        self.yaw = None
        self.size = None
        self.harm = None
        self.prob = None
        self.risk = None


def harm_model(
    ego_type,
    obj_type,
    ego_velocity: float,
    ego_yaw: float,
    obstacle_velocity: float,
    obstacle_yaw: float,
    pdof: float,
    ego_angle: float,
    obs_angle: float,
    modes,
    coeffs,
):
    """
    Get the harm for two possible collision partners.

    Args:
        scenario (Scenario): Considered scenario.
        ego_vehicle_id (Int): ID of ego vehicle.
        vehicle_params (Dict): Parameters of ego vehicle (1, 2 or 3).
        ego_velocity (Float): Velocity of ego vehicle [m/s].
        ego_yaw (Float): Yaw of ego vehicle [rad].
        obstacle_id (Int): ID of considered obstacle.
        obstacle_size (Float): Size of obstacle in [m²] (length * width)
        obstacle_velocity (Float): Velocity of obstacle [m/s].
        obstacle_yaw (Float): Yaw of obstacle [rad].
        pdof (float): Crash angle between ego vehicle and considered
            obstacle [rad].
        ego_angle (float): Angle of impact area for the ego vehicle.
        obs_angle (float): Angle of impact area for the obstacle.
        modes (Dict): Risk modes. Read from risk.json.
        coeffs (Dict): Risk parameters. Read from risk_parameters.json

    Returns:
        float: Harm for the ego vehicle.
        float: Harm for the other collision partner.
        HarmParameters: Class with independent variables for the ego
            vehicle
        HarmParameters: Class with independent variables for the obstacle
            vehicle
    """
    # create dictionaries with crash relevant parameters
    ego_vehicle = HarmParameters()
    obstacle = HarmParameters()

    # assign parameters to dictionary
    ego_vehicle.type = ego_type
    obstacle.type = obj_type
    ego_vehicle.protection = obstacle_protection[ego_vehicle.type]
    obstacle.protection = obstacle_protection[obstacle.type]

    if ego_vehicle.protection is not None:
        ego_vehicle.mass = get_obstacle_mass(
            obstacle_type=ego_vehicle.type
        )
        ego_vehicle.velocity = ego_velocity
        ego_vehicle.yaw = ego_yaw
        ego_vehicle.size = ego_type.WIDTH * ego_type.LENGTH
    else:
        ego_vehicle.mass = None
        ego_vehicle.velocity = None
        ego_vehicle.yaw = None
        ego_vehicle.size = None

    if obstacle.protection is not None:
        obstacle.velocity = obstacle_velocity
        obstacle.yaw = obstacle_yaw
        obstacle.size = obj_type.WIDTH * obj_type.LENGTH
        obstacle.mass = get_obstacle_mass(
            obstacle_type=obstacle.type
        )
    else:
        obstacle.mass = None
        obstacle.velocity = None
        obstacle.yaw = None
        obstacle.size = None

    # get model based on selection
    if modes["harm_mode"] == "log_reg":
        # select case based on protection structure
        if obstacle.protection is True:
            ego_vehicle.harm, obstacle.harm = get_protected_log_reg_harm(
                ego_vehicle=ego_vehicle,
                obstacle=obstacle,
                pdof=pdof,
                ego_angle=ego_angle,
                obs_angle=obs_angle,
                modes=modes,
                coeffs=coeffs,
            )
        elif obstacle.protection is False:
            ego_vehicle.harm, obstacle.harm = get_unprotected_log_reg_harm(
                ego_vehicle=ego_vehicle, obstacle=obstacle, pdof=pdof, coeff=coeffs
            )
        else:
            ego_vehicle.harm = 1
            obstacle.harm = 1

    elif modes["harm_mode"] == "ref_speed":
        # select case based on protection structure
        if obstacle.protection is True:
            ego_vehicle.harm, obstacle.harm = get_protected_ref_speed_harm(
                ego_vehicle=ego_vehicle,
                obstacle=obstacle,
                pdof=pdof,
                ego_angle=ego_angle,
                obs_angle=obs_angle,
                modes=modes,
                coeffs=coeffs,
            )
        elif obstacle.protection is False:
            ego_vehicle.harm, obstacle.harm = get_unprotected_ref_speed_harm(
                ego_vehicle=ego_vehicle, obstacle=obstacle, pdof=pdof, coeff=coeffs
            )
        else:
            ego_vehicle.harm = 1
            obstacle.harm = 1

    elif modes["harm_mode"] == "gidas":
        # select case based on protection structure
        if obstacle.protection is True:
            ego_vehicle.harm, obstacle.harm = get_protected_gidas_harm(
                ego_vehicle=ego_vehicle, obstacle=obstacle, pdof=pdof, coeff=coeffs
            )
        elif obstacle.protection is False:
            ego_vehicle.harm, obstacle.harm = get_unprotected_gidas_harm(
                ego_vehicle=ego_vehicle, obstacle=obstacle, pdof=pdof, coeff=coeffs
            )
        else:
            ego_vehicle.harm = 1
            obstacle.harm = 1

    else:
        raise ValueError(
            "Please select a valid mode for harm estimation "
            "(log_reg, ref_speed, gidas)"
        )

    return ego_vehicle.harm, obstacle.harm, ego_vehicle, obstacle


def cal_harm(ego_traj, obj_prediction, coeffs = params_harm, modes = params_mode):

    """
    Main function to compute harm
    """

    if isinstance(ego_traj, tuple):
        print(f"Invalid input type for ego_traj: {type(ego_traj)}")
        return None, None
    else:
        ego_mass = get_obstacle_mass(obstacle_type = DefaultVehicle)
        ego_pos = np.array(ego_traj['pos_list'], dtype= float)
        ego_v = np.array(ego_traj['v_list'], dtype= float)
        ego_yaw = np.array(ego_traj['orientation_list'], dtype= float)
    
        ego_harm_traj = {}
        obj_harm_traj = {}

        for obj in obj_prediction:
            # Skip if any required keys are missing
            if any(key not in obj for key in ['v_list', 'orientation_list', 'pos_list', 'shape']):
                print(f"Object {obj['id']} is missing required keys. Skipping...")
                continue

            try:
                obj_type = obj['type']
                pred_path = np.array(obj['pos_list'], dtype=float)
                pred_length = min(len(ego_pos), len(pred_path))
                
                if pred_length == 0:
                    continue

                # get the size, the velocity and the orientation of the predicted vehicle

                pred_v = np.array(obj['v_list'], dtype=float)[:pred_length]
                pred_yaw = np.array(obj['orientation_list'], dtype=float)[:pred_length]
                pred_size = obj['shape']['length'] * obj['shape']['width']
                obstacle_mass = get_obstacle_mass(obstacle_type = obj_type)

                ego_harm_fun, obstacle_harm_fun = get_harm_model(
                    modes = modes,
                    obj_type = obj_type,
                )

               # Initialize harm lists
                ego_harm_obst = np.zeros(pred_length)
                obst_harm_obst = np.zeros(pred_length)


                if modes["crash_angle_simplified"] is False:
                    obj_crush_angle = calc_crash_angle(
                        ego_traj = ego_traj,
                        obj = obj, 
                    )
                    for i in range(pred_length):
                        if obj_crush_angle[i] is not None: 
                            # get the harm ego harm and the harm of the collision opponent
                            ego_harm, obst_harm, ego_harm_data, obst_harm_data = harm_model(
                                ego_type = DefaultVehicle, 
                                obj_type = obj_type,
                                ego_velocity = ego_traj['v_list'][i],
                                ego_yaw = ego_traj['orientation_list'][i],
                                obstacle_velocity = obj['v_list'][i],
                                obstacle_yaw = obj['orientation_list'][i],
                                pdof = obj_crush_angle[0][i],
                                ego_angle = obj_crush_angle[1][i],
                                obs_angle = obj_crush_angle[2][i],
                                modes = modes,
                                coeffs = coeffs,
                            )
                            # store information to calculate harm and harm value in list
                            ego_harm_obst[i] = ego_harm
                            obst_harm_obst[i] = obst_harm
                        else:
                            ego_harm_obst[i] = 1e-6
                            obst_harm_obst[i] = 1e-6

                else:
                    
                    # crash angle between ego vehicle and considered obstacle [rad]
                    pdof_array = pred_yaw - ego_yaw[:pred_length]
                    rel_angle_array = np.arctan2(pred_path[:, 1] - ego_pos[:pred_length, 1],
                                             pred_path[:, 0] - ego_pos[:pred_length, 0])
                    ego_angle_array = rel_angle_array - ego_yaw[:pred_length]
                    obs_angle_array = rel_angle_array - pred_yaw

                    # calculate the difference between pre-crash and post-crash speed
                    # Delta V calculation
                    delta_v_array = np.sqrt(
                        ego_v[:pred_length] ** 2 +
                        pred_v ** 2 +
                        2 * ego_v[:pred_length] * pred_v * np.cos(pdof_array)
                    )
                    ego_delta_v = (obstacle_mass / (ego_mass + obstacle_mass)) * delta_v_array
                    obstacle_delta_v = (ego_mass / (ego_mass + obstacle_mass)) * delta_v_array

                    # Calculate harm for ego and obstacle
                    ego_harm_obst = ego_harm_fun(
                        velocity = ego_delta_v, 
                        angle = ego_angle_array, 
                        coeff = coeffs,
                        )
                    
                    obst_harm_obst = obstacle_harm_fun(
                        velocity = obstacle_delta_v,
                        angle = obs_angle_array, 
                        coeff = coeffs,
                        )
                
                ego_harm_traj[obj['id']] = ego_harm_obst
                obj_harm_traj[obj['id']] = obst_harm_obst
            except KeyError as err:
                print(f"Object {obj['id']} encountered an key error for {err}. Removing from list.")
                print(obj)
                continue  # if the objective is disappeared in the env
        
        return ego_harm_traj, obj_harm_traj


    






def get_harm_model(modes, obj_type):
    """
    Get harm model
    """

    # obstacle protection type
    obs_protection = obstacle_protection[obj_type]

    if modes["harm_mode"] == "log_reg":
        # select case based on protection structure
        if obs_protection is True:
            # calculate harm based on angle mode
            if modes["ignore_angle"] is False:
                if modes["sym_angle"] is False:
                    if modes["reduced_angle_areas"] is False:
                        # use log reg complete
                        # calculate harm for the ego vehicle
                        ego_harm = get_protected_inj_prob_log_reg_complete

                        # calculate harm for the obstacle vehicle
                        obstacle_harm = get_protected_inj_prob_log_reg_complete

                    else:
                        # use log reg reduced
                        # calculate harm for the ego vehicle
                        ego_harm = get_protected_inj_prob_log_reg_reduced

                        # calculate harm for the obstacle vehicle
                        obstacle_harm = get_protected_inj_prob_log_reg_reduced
                else:
                    if modes["reduced_angle_areas"] is False:
                        # use log reg sym complete
                        # calculate harm for the ego vehicle
                        ego_harm = get_protected_inj_prob_log_reg_complete_sym

                        # calculate harm for the obstacle vehicle
                        obstacle_harm = get_protected_inj_prob_log_reg_complete_sym
                    else:
                        # use log reg sym reduced
                        # calculate harm for the ego vehicle
                        ego_harm = get_protected_inj_prob_log_reg_reduced_sym

                        # calculate harm for the obstacle vehicle
                        obstacle_harm = get_protected_inj_prob_log_reg_reduced_sym
            else:
                # use log reg delta v
                # calculate harm for the ego vehicle
                ego_harm = get_protected_inj_prob_log_reg_ignore_angle

                # calculate harm for the obstacle vehicle
                obstacle_harm = get_protected_inj_prob_log_reg_ignore_angle

        elif obs_protection is False:
            # calc ego harm
            ego_harm = get_protected_inj_prob_log_reg_ignore_angle

            # calculate obstacle harm
            # logistic regression model
            obstacle_harm = lambda velocity,angle,coeff : 1 / (  # noqa E731
                1
                + np.exp(
                    coeff["pedestrian"]["const"]
                    - coeff["pedestrian"]["speed"] * velocity
                )
            )
        else:
            ego_harm = lambda velocity,angle,coeff : 1  # noqa E731
            obstacle_harm = lambda velocity,angle,coeff : 1  # noqa E731

    elif modes["harm_mode"] == "ref_speed":
        # select case based on protection structure
        if obs_protection is True:
            # calculate harm based on angle mode
            if modes["ignore_angle"] is False:
                if modes["sym_angle"] is False:
                    if modes["reduced_angle_areas"] is False:
                        # use log reg complete
                        # calculate harm for the ego vehicle
                        ego_harm = get_protected_inj_prob_ref_speed_complete

                        # calculate harm for the obstacle vehicle
                        obstacle_harm = get_protected_inj_prob_ref_speed_complete

                    else:
                        # use log reg reduced
                        # calculate harm for the ego vehicle
                        ego_harm = get_protected_inj_prob_ref_speed_reduced

                        # calculate harm for the obstacle vehicle
                        obstacle_harm = get_protected_inj_prob_ref_speed_reduced
                else:
                    if modes["reduced_angle_areas"] is False:
                        # use log reg sym complete
                        # calculate harm for the ego vehicle
                        ego_harm = get_protected_inj_prob_ref_speed_complete_sym

                        # calculate harm for the obstacle vehicle
                        obstacle_harm = get_protected_inj_prob_ref_speed_complete_sym
                    else:
                        # use log reg sym reduced
                        # calculate harm for the ego vehicle
                        ego_harm = get_protected_inj_prob_ref_speed_reduced_sym

                        # calculate harm for the obstacle vehicle
                        obstacle_harm = get_protected_inj_prob_ref_speed_reduced_sym
            else:
                # use log reg delta v
                # calculate harm for the ego vehicle
                ego_harm = get_protected_inj_prob_ref_speed_ignore_angle

                # calculate harm for the obstacle vehicle
                obstacle_harm = get_protected_inj_prob_ref_speed_ignore_angle

        elif obs_protection is False:
            # calc ego harm
            ego_harm = get_protected_inj_prob_ref_speed_ignore_angle

            # calculate obstacle harm
            # logistic regression model
            obstacle_harm = lambda velocity,angle,coeff : 1 / (  # noqa E731
                1
                + np.exp(
                    coeff["pedestrian"]["const"]
                    - coeff["pedestrian"]["speed"] * velocity
                )
            )
        else:
            ego_harm = lambda velocity,angle,coeff : 1  # noqa E731
            obstacle_harm = lambda velocity,angle,coeff : 1  # noqa E731

    elif modes["harm_mode"] == "gidas":
        # select case based on protection structure
        if obs_protection is True:
            ego_harm = lambda velocity,angle,coeff : 1 / (  # noqa E731
                1 + np.exp(-coeff["gidas"]["const"] - coeff["gidas"]["speed"] * velocity)
            )

            obs_harm = lambda velocity,angle,coeff : 1 / (  # noqa E731
                1
                + np.exp(-coeff["gidas"]["const"] - coeff["gidas"]["speed"] * velocity)
            )
        elif obs_protection is False:
            # calc ego harm
            ego_harm = lambda velocity,angle,coeff : 1 / (  # noqa E731
                1 + np.exp(-coeff["gidas"]["const"] - coeff["gidas"]["speed"] * velocity)
            )

            # calculate obstacle harm
            # logistic regression model
            obstacle_harm = lambda velocity, angle, coeff: 1 / (  # noqa E731
                1
                + np.exp(
                    coeff["pedestrian_MAIS2+"]["const"]
                    - coeff["pedestrian_MAIS2+"]["speed"] * velocity
                )
            )
        else:
            ego_harm = lambda velocity, angle, coeff: 1  # noqa E731
            obstacle_harm = lambda velocity, angle, coeff: 1  # noqa E731

    else:
        raise ValueError(
            "Please select a valid mode for harm estimation "
            "(log_reg, ref_speed, gidas)"
        )

    return ego_harm, obstacle_harm














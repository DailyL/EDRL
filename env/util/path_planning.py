import numpy as np
import math
import copy
from dataclasses import dataclass

from env.util.planner import QuinticPolynomial, QuarticPolynomial
from env.util.env_utils import CubicSpline2D, VehiclePIDController
import matplotlib.pyplot as plt


def traj_planner(planner_name, ego_vehicle, current_lane, target_lateral, target_long, target_speed_long, target_time, frenet_state=None, padd=None, c_a=0, target_acc=0):
    if planner_name =='Frenet' or 'frenet':

        fp_list = calc_frenet_trajectory(
                start_state=frenet_state,
                t_d=target_lateral, t_s=target_long, t_T=target_time, t_v=target_speed_long ,
                current_lane=current_lane, padd=padd,
            )
    
    elif planner_name =='Quintic' or 'quintic':

        fp_list = quintic_polynomials_planner(
                sx = ego_vehicle.position[0], sy= ego_vehicle.position[1], syaw = ego_vehicle.heading_theta, sv=ego_vehicle.speed, sa = c_a,
                gx = target_long, gy = target_lateral, gyaw = ego_vehicle.heading_theta, gv = target_speed_long,
                ga = target_acc, max_accel=5, max_jerk=3, gT=target_time
            )
    else:
        raise NotImplementedError
    

def pid_traj_controller(target_x, target_y, target_yaw, target_speed, ego_vehicle, pid_controller):


    
    action = pid_controller.run_step(
                        current_speed = ego_vehicle.speed, target_speed = target_speed, ego_vehicle=ego_vehicle,
                        target_position=[target_x, target_y], current_direction=ego_vehicle.heading_theta,
                        target_direction = target_yaw,
                    )
    
    return action

    










show_animation = True

def transfer_local_to_global(x_loc, y_loc, yaw_loc, x_rel, y_rel, vx_rel, vy_rel):

    x_global = x_loc + x_rel * np.cos(yaw_loc) - y_rel * np.sin(yaw_loc)
    y_global = y_loc + x_rel * np.sin(yaw_loc) + y_rel * np.cos(yaw_loc)

    vx_goal = vx_rel * np.cos(yaw_loc) - vy_rel * np.sin(yaw_loc)
    vy_goal = vx_rel * np.sin(yaw_loc) + vy_rel * np.cos(yaw_loc)
    return x_global, y_global, vx_goal, vy_goal


class Quintic_planner(object):

    def __init__(self, pid_control):

        self.pid_control = pid_control
        self.pid_controller = VehiclePIDController()
    
    def plan(self, sx, sy, syaw, sv, sa, gx, gy, gyaw, gv, ga, max_accel, max_jerk, gT, dt = 0.1):

        fp = quintic_polynomials_planner(sx, sy, syaw, sv, sa, gx, gy, gyaw, gv, ga, max_accel, max_jerk, gT, dt = dt)

        return fp
    

    def act(self, target_x, target_y, target_yaw, target_speed, ego_vehicle):

        if self.pid_control:
            action = pid_traj_controller(
                target_x, target_y, target_yaw, target_speed, ego_vehicle, self.pid_controller
            )
            return action
        else:
            pass

    def plot_traj(self, fp_list):

        plt.ion()  # Enable interactive mode
        area = 20
        for i in range(len(fp_list.t)):
            plt.cla()  # Clear axis for each frame

            position = [fp_list.x[i], fp_list.y[i]]

            plt.plot(position[0], position[1], 'go')

            # Plot headings as arrows
            headings = fp_list.yaw[i:]  # Extract heading values
            dx = np.cos(headings)           # X-components of arrows
            dy = np.sin(headings)           # Y-components of arrows
            
            plt.quiver(fp_list.x[i:], fp_list.y[i:], dx, dy, angles='xy', scale_units='xy', scale=1, color='blue', label="Heading")
            plt.xlim(fp_list.x[i] - area, fp_list.x[i] + area)
            plt.ylim(fp_list.y[i] - area, fp_list.y[i] + area)
            plt.grid(True)
            
            plt.draw()     # Force a redraw of the current figure
            plt.pause(0.01)  # Short pause to update plot in real-time

        plt.ioff()  # Disable interactive mode when done


def traj_planner(planner_name, ego_vehicle, current_lane, target_lateral, target_long, target_speed_long, target_time, frenet_state=None, padd=None, c_a=0, target_acc=0):
    if planner_name =='Frenet' or 'frenet':

        fp_list = calc_frenet_trajectory(
                start_state=frenet_state,
                t_d=target_lateral, t_s=target_long, t_T=target_time, t_v=target_speed_long ,
                current_lane=current_lane, padd=padd,
            )
    
    elif planner_name =='Quintic' or 'quintic':

        fp_list = quintic_polynomials_planner(
                sx = ego_vehicle.position[0], sy= ego_vehicle.position[1], syaw = ego_vehicle.heading_theta, sv=ego_vehicle.speed, sa = c_a,
                gx = target_long, gy = target_lateral, gyaw = ego_vehicle.heading_theta, gv = target_speed_long,
                ga = target_acc, max_accel=5, max_jerk=3, gT=target_time
            )
    else:
        raise NotImplementedError


class Frenet_planner(object):

    def __init__(self, pid_control):

        self.pid_control = pid_control
        self.pid_controller = VehiclePIDController()
        self.padd = None
    
    def plan(self, frenet_state, target_lateral, target_long, target_time, target_speed_long, current_lane):

        fp = calc_frenet_trajectory(
                start_state=frenet_state,
                t_d=target_lateral, t_s=target_long, t_T=target_time, t_v=target_speed_long ,
                current_lane=current_lane, padd=self.padd,
            )
        self.padd = fp

        return fp
    

    def act(self, target_x, target_y, target_yaw, target_speed, ego_vehicle):

        if self.pid_control:
            action = pid_traj_controller(
                target_x, target_y, target_yaw, target_speed, ego_vehicle, self.pid_controller
            )
            return action
        else:
            pass
    def plot_traj(self, fp_list):

        plt.ion()  # Enable interactive mode
        area = 20
        for i in range(len(fp_list.t)):
            plt.cla()  # Clear axis for each frame

            position = [fp_list.x[i], fp_list.y[i]]

            plt.plot(position[0], position[1], 'go')

            # Plot headings as arrows
            headings = fp_list.yaw[i:]  # Extract heading values
            dx = np.cos(headings)           # X-components of arrows
            dy = np.sin(headings)           # Y-components of arrows
            
            plt.quiver(fp_list.x[i:], fp_list.y[i:], dx, dy, angles='xy', scale_units='xy', scale=1, color='blue', label="Heading")
            plt.xlim(fp_list.x[i] - area, fp_list.x[i] + area)
            plt.ylim(fp_list.y[i] - area, fp_list.y[i] + area)
            plt.grid(True)
            
            plt.draw()     # Force a redraw of the current figure
            plt.pause(0.01)  # Short pause to update plot in real-time

        plt.ioff()  # Disable interactive mode when done            



def quintic_polynomials_planner(sx, sy, syaw, sv, sa, gx, gy, gyaw, gv, ga, max_accel, max_jerk, gT, dt = 0.1):


    
    """
    quintic polynomial planner

    input
        s_x: start x position [m]
        s_y: start y position [m]
        s_yaw: start yaw angle [rad]
        sa: start accel [m/ss]
        gx: goal x position [m]
        gy: goal y position [m]
        gyaw: goal yaw angle [rad]
        ga: goal accel [m/ss]
        max_accel: maximum accel [m/ss]
        max_jerk: maximum jerk [m/sss]
        dt: time tick [s]

    return
        time: time result
        rx: x position result list
        ry: y position result list
        ryaw: yaw angle result list
        rv: velocity result list
        ra: accel result list

    """
    v_max = 20

    ga = max(-max_accel, ga)
    ga = min(ga, ga)

    vxg = gv * math.cos(gyaw)
    vyg = gv * math.sin(gyaw)
    vxs = sv * math.cos(syaw)
    vys = sv * math.sin(syaw)

    gx, gy, vxg, vyg = transfer_local_to_global(sx, sy, syaw, gx, gy, vxg, vyg)



    MIN_T = 0.1
    MAX_T = 5


    axs = sa * math.cos(syaw)
    ays = sa * math.sin(syaw)
    axg = ga * math.cos(gyaw)
    ayg = ga * math.sin(gyaw)

    time, rx, ry, ryaw, rv, ra, rj = [], [], [], [], [], [], []

    for T in np.arange(MIN_T, gT, dt):
        xqp = QuinticPolynomial(sx, vxs, axs, gx, vxg, axg, T)
        yqp = QuinticPolynomial(sy, vys, ays, gy, vyg, ayg, T)

        time, rx, ry, ryaw, rv, ra, rj = [], [], [], [], [], [], []

        for t in np.arange(0.0, T + dt, dt):
            time.append(t)
            rx.append(xqp.calc_point(t))
            ry.append(yqp.calc_point(t))

            vx = xqp.calc_first_derivative(t)
            vy = yqp.calc_first_derivative(t)
            v = np.hypot(vx, vy)
            yaw = math.atan2(vy, vx)
            rv.append(v)
            ryaw.append(yaw)

            ax = xqp.calc_second_derivative(t)
            ay = yqp.calc_second_derivative(t)
            a = np.hypot(ax, ay)
            if len(rv) >= 2 and rv[-1] - rv[-2] < 0.0:
                a *= -1
            ra.append(a)

            jx = xqp.calc_third_derivative(t)
            jy = yqp.calc_third_derivative(t)
            j = np.hypot(jx, jy)
            if len(ra) >= 2 and ra[-1] - ra[-2] < 0.0:
                j *= -1
            rj.append(j)
    fp = QuinticTrajectory(
        t=time,
        x=rx,
        y=ry,
        yaw=ryaw,
        v=rv,
        a=ra,
    )

    return fp




class FrenetTrajectory:
    """Trajectory in frenet Coordinates with longitudinal and lateral position and up to 3rd derivative. It also includes the global pose and curvature."""

    def __init__(
        self,
        t: float = None,
        d: float = None,
        d_d: float = None,
        d_dd: float = None,
        d_ddd: float = None,
        s: float = None,
        s_d: float = None,
        s_dd: float = None,
        s_ddd: float = None,
        x: float = None,
        y: float = None,
        yaw: float = None,
        v: float = None,
        curv: float = None,
    ):
        """
        Initialize a frenét trajectory.

        Args:
            t ([float]): List for the time. Defaults to None.
            d ([float]): List for the lateral offset. Defaults to None.
            d_d: ([float]): List for the lateral velocity. Defaults to None.
            d_dd ([float]): List for the lateral acceleration. Defaults to None.
            d_ddd ([float]): List for the lateral jerk. Defaults to None.
            s ([float]): List for the covered arc length of the spline. Defaults to None.
            s_d ([float]): List for the longitudinal velocity. Defaults to None.
            s_dd ([float]): List for the longitudinal acceleration. Defaults to None.
            s_ddd ([float]): List for the longitudinal jerk. Defaults to None.
            x ([float]): List for the x-position. Defaults to None.
            y ([float]): List for the y-position. Defaults to None.
            yaw ([float]): List for the yaw angle. Defaults to None.
            v([float]): List for the velocity. Defaults to None.
            curv ([float]): List for the curvature. Defaults to None.
        """
        # time vector
        self.t = t

        # frenet coordinates
        self.d = d
        self.d_d = d_d
        self.d_dd = d_dd
        self.d_ddd = d_ddd
        self.s = s
        self.s_d = s_d
        self.s_dd = s_dd
        self.s_ddd = s_ddd

        # Global coordinates
        self.x = x
        self.y = y
        self.yaw = yaw
        # Velocity
        self.v = v
        # Curvature
        self.curv = curv

        # Validity
        self.valid_level = 0
        self.reason_invalid = None

        # Cost
        self.cost = 0

        # Risk
        self.ego_risk_dict = []
        self.obst_risk_dict = []

@dataclass
class Frenet_State:
    s: float
    s_d: float
    s_dd: float
    d: float
    d_d: float
    d_dd: float
    """
    
    d ([float]): the lateral offset. 
    d_d: ([float]): the lateral velocity. 
    d_dd ([float]): the lateral acceleration. 
    d_ddd ([float]): the lateral jerk. 
    s ([float]): the covered arc length of the spline. 
    s_d ([float]): the longitudinal velocity. 
    s_dd ([float]): the longitudinal acceleration. 
    s_ddd ([float]): the longitudinal jerk.
    """

class QuinticTrajectory:
    """Trajectory in frenet Coordinates with longitudinal and lateral position and up to 3rd derivative. It also includes the global pose and curvature."""

    def __init__(
        self,
        t: float = None,
        x: float = None,
        y: float = None,
        yaw: float = None,
        v: float = None,
        a: float = None,
    ):
        
        # time vector
        self.t = t

        # frenet coordinates
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.a = a



def calc_frenet_trajectories(
    start_state: Frenet_State,
    t_d: float,
    t_s,
    t_T,
    t_v,
    current_lane,
    dt = 0.1,
    v_thr: float = 3.0,
):
    """
    Calculate all possible frenet trajectories from a given starting point and target lateral deviations, times and velocities.

    Args:
        c_s (float): Start longitudinal position.
        c_s_d (float): Start longitudinal velocity.
        c_s_dd (float): Start longitudinal acceleration
        c_d (float): Start lateral position.
        c_d_d (float): Start lateral velocity.
        c_d_dd (float): Start lateral acceleration.
        t_T : Target time interval
        dt (float): Time step size of the trajectories.
        csp (CubicSpline2D): Reference spline of the global path.
        v_thr (float): Threshold velocity to distinguish slow and fast trajectories.
        exec_times_dict (dict): Dictionary for execution times. Defaults to None.

    Returns:
        [FrenetTrajectory]: List with all frenét trajectories.
    """

    min_velocity = 0.0001

    start_state.s_d = max(start_state.s_d, min_velocity)

    c_d = start_state.d 
    c_d_d = start_state.d_d
    c_d_dd = start_state.d_dd
    c_s = start_state.s
    c_s_d = start_state.s_d
    c_s_dd = start_state.s_dd


    t_v = max(t_v, min_velocity)

    # Determine motion mode
    if abs(c_s_d) < v_thr or abs(t_v) < v_thr:
        lat_mode = "low_velocity"
    else:
        lat_mode = "high_velocity"


    # quartic polynomial in longitudinal direction

    
    qp_long = QuarticPolynomial(
        xs=start_state.s, vxs=start_state.s_d, axs=start_state.s_dd, vxe=t_v, axe=0.0, time = t_T
    )

    # Consider target longitual position
    """
    
    qp_long = QuinticPolynomial(
        xs=c_s, vxs=c_s_d, axs=c_s_dd, xe=t_s, vxe=t_v, axe=0.0, time=t_T,
    )
    """  
    # time vector
    t = np.arange(0.0, t_T, dt)

    # longitudinal position and derivatives
    s = qp_long.calc_point(t)
    s_d = qp_long.calc_first_derivative(t)
    s_dd = qp_long.calc_second_derivative(t)
    s_ddd = qp_long.calc_third_derivative(t)

    s0, ds = s[0], s[-1] - s[0]


    if lat_mode == "high_velocity":

        qp_lat = QuinticPolynomial(
            xs=c_d, vxs=c_d_d, axs=c_d_dd, xe=t_d, vxe=0.0, axe=0.0, time=t_T,
        )

        # lateral distance and derivatives
        d = qp_lat.calc_point(t)
        d_d = qp_lat.calc_first_derivative(t)
        d_dd = qp_lat.calc_second_derivative(t)
        d_ddd = qp_lat.calc_third_derivative(t)

        d_d_time = d_d
        d_dd_time = d_dd
        d_d = d_d / s_d
        d_dd = (d_dd - d_d * s_dd) / np.power(s_d, 2)

    # for low velocities, we have ds/dt and dd/ds
    elif lat_mode == "low_velocity":
        # singularity
        if ds == 0:
            ds = 0.00001

        # the quintic polynomial shows dd/ds, so d(c_s)/ds and dd(c_s)/dds is needed
        if c_s_d != 0.0:
            c_d_d_not_time = c_d_d / c_s_d
            c_d_dd_not_time = (c_d_dd - c_s_dd * c_d_d_not_time) / (
                c_s_d ** 2
            )
        else:
            c_d_d_not_time = 0.0
            c_d_dd_not_time = 0.0

        # Upper boundary for ds to avoid bad lat polynoms (solved by  if ds > abs(dT)?)
        # ds = max(ds, 0.1)

        qp_lat = QuinticPolynomial(
            xs=c_d, vxs=c_d_d_not_time, axs=c_d_dd_not_time, xe=t_d, vxe=0.0, axe=0.0, time=ds,
        )

        
        # use universal function feature to perform array operation
        # lateral distance and derivatives
        d = qp_lat.calc_point(s - s0)
        d_d = qp_lat.calc_first_derivative(s - s0)
        d_dd = qp_lat.calc_second_derivative(s - s0)
        d_ddd = qp_lat.calc_third_derivative(s - s0)

        # since dd/ds, a conversion to dd/dt is needed
        d_d_time = s_d * d_d
        d_dd_time = s_dd * d_d + np.power(s_d, 2) * d_dd



    # calculate the position of the reference path
    global_path_x = np.zeros(len(s),dtype=np.float64)
    global_path_y = np.zeros(len(s),dtype=np.float64)
    
    for i in range(len(t)):
        global_path_x[i], global_path_y[i] = current_lane.position(s[i], d[i])



    # Compute derivatives and curvatures
    dx, dy = np.gradient(global_path_x, s), np.gradient(global_path_y, s)
    ddy, ddx = np.gradient(dy, s), np.gradient(dx, s)
    dddy, dddx = np.gradient(ddx, s), np.gradient(ddy, s)
    global_path_yaw = np.arctan2(dy, dx)

    # calculate yaw of the global path
    global_path_yaw = np.arctan2(dy, dx)

    # calculate the curvature of the global path
    global_path_curv = (np.multiply(dx, ddy) - np.multiply(ddx, dy)) / (
        np.power(dx, 2) + np.power(dy, 2) ** (3 / 2)
    )

    # calculate the derivation of the global path's curvature
    z = np.multiply(dx, ddy) - np.multiply(ddx, dy)
    z_d = np.multiply(dx, dddy) - np.multiply(dddx, dy)
    n = (np.power(dx, 2) + np.power(dy, 2)) ** (3 / 2)
    n_d = (3 / 2) * np.multiply(
        np.power((np.power(dx, 2) + np.power(dy, 2)), 0.5),
        (2 * np.multiply(dx, ddx) + 2 * np.multiply(dy, ddy)),
    )
    global_path_curv_d = (np.multiply(z_d, n) - np.multiply(z, n_d)) / (
        np.power(n, 2)
    )
    
    yaw_diff_array = np.arctan(d_d / (1 - global_path_curv * d))
    yaw = yaw_diff_array + global_path_yaw
    x = global_path_x - d * np.sin(global_path_yaw)
    y = global_path_y + d * np.cos(global_path_yaw)
    v = (s_d * (1 - global_path_curv * d)) / np.cos(yaw_diff_array)
    curv = (
        (
            (
                d_dd
                + (global_path_curv_d * d + global_path_curv * d_d)
                * np.tan(yaw_diff_array)
            )
            * (np.power(np.cos(yaw_diff_array), 2) / (1 - global_path_curv * d))
        )
        + global_path_curv
    ) * (np.cos(yaw_diff_array) / (1 - global_path_curv * d))

            
    # create frenet trajectory
    fp = FrenetTrajectory(
        t=t,
        d=d,
        d_d=d_d_time,
        d_dd=d_dd_time,
        d_ddd=d_ddd,
        s=s,
        s_d=s_d,
        s_dd=s_dd,
        s_ddd=s_ddd,
        x=x,
        y=y,
        yaw=yaw,
        v=v,
        curv=curv,
    )

    return fp


def scale_velocity(velocity, min_velocity=0.0001):
    return max(velocity, min_velocity)

def compute_longitudinal_trajectory(start_state, t_s, t_v, t_T):
    # Determines the appropriate polynomial (quartic/quintic) for longitudinal motion
    t_s = None
    if t_s is None:
        return QuarticPolynomial(xs=start_state.s, vxs=start_state.s_d, axs=start_state.s_dd, vxe=t_v, axe=0.0, time=t_T)
    return QuinticPolynomial(xs=start_state.s, vxs=start_state.s_d, axs=start_state.s_dd, xe=t_s, vxe=t_v, axe=0.0, time=t_T)

def compute_lateral_trajectory(start_state, t_d, t_T):
    # Lateral motion using a quintic polynomial
    return QuinticPolynomial(xs=start_state.d, vxs=start_state.d_d, axs=start_state.d_dd, xe=t_d, vxe=0.0, axe=0.0, time=t_T)

def calculate_global_path(current_lane, s, d):
    global_path_x = np.zeros(len(s), dtype=np.float64)
    global_path_y = np.zeros(len(s), dtype=np.float64)
    
    for i in range(len(s)):
        global_path_x[i], global_path_y[i] = current_lane.position(s[i], d[i])
    return global_path_x, global_path_y


from scipy.ndimage import uniform_filter1d
from scipy.interpolate import CubicSpline
def calc_frenet_trajectory(start_state, t_d, t_s, t_T, t_v, current_lane, padd, dt=0.1, v_thr=1.0):

    # Basic configurations
    min_velocity = 0.0001
    start_state.s_d = scale_velocity(start_state.s_d, min_velocity)
    t_v = scale_velocity(t_v, min_velocity)

    # Longitudinal motion using polynomial
    qp_long = compute_longitudinal_trajectory(start_state, t_s, t_v, t_T)

    t = np.arange(0.0, t_T, dt)

    # Longitudinal position and derivatives
    s = qp_long.calc_point(t)
    s_d = qp_long.calc_first_derivative(t)
    s_dd = qp_long.calc_second_derivative(t)
    s_ddd = qp_long.calc_third_derivative(t)


    # Lateral motion using a quintic polynomial
    qp_lat = compute_lateral_trajectory(start_state, t_d, t_T)
    d = qp_lat.calc_point(t)
    d_d = qp_lat.calc_first_derivative(t)
    d_dd = qp_lat.calc_second_derivative(t)
    d_ddd = qp_lat.calc_third_derivative(t)
    


    # Calculate global path coordinates
    global_path_x, global_path_y = calculate_global_path(current_lane, s, d)




    # Compute derivatives and curvatures
    dx, dy = np.gradient(global_path_x, s), np.gradient(global_path_y, s)
    ddy, ddx = np.gradient(dy, s), np.gradient(dx, s)
    dddy, dddx = np.gradient(ddx, s), np.gradient(ddy, s)
    
    global_path_yaw = np.arctan2(dy, dx)

    global_path_curv = (np.multiply(dx, ddy) - np.multiply(ddx, dy)) / np.power(dx ** 2 + dy ** 2, 1.5)

    # Derivation of global path's curvature
    z = np.multiply(dx, ddy) - np.multiply(ddx, dy)
    z_d = np.multiply(dx, dddy) - np.multiply(dddx, dy)
    n = np.power(dx ** 2 + dy ** 2, 1.5)
    n_d = 1.5 * np.sqrt(dx ** 2 + dy ** 2) * (2 * np.multiply(dx, ddx) + 2 * np.multiply(dy, ddy))
    global_path_curv_d = (np.multiply(z_d, n) - np.multiply(z, n_d)) / np.power(n, 2)



    # Calculate yaw, velocity, and curvature for frenet trajectory
    yaw_diff_array = np.arctan(d_d / (1 - global_path_curv * d))
    x = global_path_x - d * np.sin(global_path_yaw)
    y = global_path_y + d * np.cos(global_path_yaw)
    v = (s_d * (1 - global_path_curv * d)) / np.cos(yaw_diff_array)
    curv = ((d_dd + (global_path_curv_d * d + global_path_curv * d_d) * np.tan(yaw_diff_array)) 
            * (np.power(np.cos(yaw_diff_array), 2) / (1 - global_path_curv * d)) 
            + global_path_curv) * (np.cos(yaw_diff_array) / (1 - global_path_curv * d))
    

    padd = None
    if padd is not None:
        past_x = padd.x
        past_y = padd.y

        x = np.concatenate((past_x, x))
        y = np.concatenate((past_y, y))
        dx = np.gradient(x)
        dy = np.gradient(y)

        yaw = np.arctan2(dy, dx)

        x = x[(len(past_x)):]
        y = y[(len(past_y)):]
        yaw = yaw[(len(past_x)):]



    else: 
        dx = np.gradient(x)
        dy = np.gradient(y)

        yaw = np.arctan2(dy, dx)
    



    

    """
    
    
    ds = np.gradient(s)
    dd = np.gradient(d)

    yaw_local = np.arctan2(dd, ds)
    for i in range(len(yaw_local)):
        yaw_local[i] = yaw_local[i] + current_lane.heading_theta_at(s[i])
    """



    # Create Frenet trajectory object
    fp = FrenetTrajectory(
        t=t,
        d=d,
        d_d=d_d,
        d_dd=d_dd,
        d_ddd=d_ddd,
        s=s,
        s_d=s_d,
        s_dd=s_dd,
        s_ddd=s_ddd,
        x=x,
        y=y,
        yaw=yaw,
        v=v,
        curv=curv,
    )

    return fp
import numpy as np
import math

from env.planner.polynomials import QuinticPolynomial, QuarticPolynomial
from env.planner.state import QuinticTrajectory,FrenetTrajectory, Frenet_State


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

    

def transfer_local_to_global(x_loc, y_loc, yaw_loc, x_rel, y_rel, vx_rel, vy_rel):

    x_global = x_loc + x_rel * np.cos(yaw_loc) - y_rel * np.sin(yaw_loc)
    y_global = y_loc + x_rel * np.sin(yaw_loc) + y_rel * np.cos(yaw_loc)

    vx_goal = vx_rel * np.cos(yaw_loc) - vy_rel * np.sin(yaw_loc)
    vy_goal = vx_rel * np.sin(yaw_loc) + vy_rel * np.cos(yaw_loc)
    return x_global, y_global, vx_goal, vy_goal



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







def calc_frenet_trajectories(
    start_state: Frenet_State,
    t_d: float,
    t_s,
    t_T,
    t_v,
    current_lane,
    dt = 0.1,
    v_thr: float = 1.0,
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
        global_path_x[i], global_path_y[i] = current_lane.position(s[i] + c_s, d[i] + c_d)



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
    if t_s is None:
        return QuarticPolynomial(xs=start_state.s, vxs=start_state.s_d, axs=start_state.s_dd, vxe=t_v, axe=0.0, time=t_T)
    return QuinticPolynomial(xs=start_state.s, vxs=start_state.s_d, axs=start_state.s_dd, xe=t_s, vxe=t_v, axe=0.0, time=t_T)

def compute_lateral_trajectory(start_state, t_d, t_T):
    # Lateral motion using a quintic polynomial
    return QuinticPolynomial(xs=start_state.d, vxs=start_state.d_d, axs=start_state.d_dd, xe=t_d, vxe=0.0, axe=0.0, time=t_T)

def calculate_global_path(current_lane, s, d):
    """
    Calculate the global path coordinates (x, y) based on the current lane and Frenet coordinates (s, d).
    Args:
        current_lane (Lane): The current lane object which provides the position method.
        s (array-like): The array of longitudinal positions along the lane.
        d (array-like): The array of lateral offsets from the lane center.
    Returns:
        tuple: Two numpy arrays containing the x and y coordinates of the global path.
    """
    global_path_x = np.zeros(len(s), dtype=np.float64)
    global_path_y = np.zeros(len(s), dtype=np.float64)
    
    for i in range(len(s)):
        global_path_x[i], global_path_y[i] = current_lane.position(s[i], d[i])
    return global_path_x, global_path_y



from scipy.ndimage import uniform_filter1d
from scipy.interpolate import CubicSpline


def calc_frenet_trajectory(start_state, t_d, t_s, t_T, t_v, current_lane, padd, dt=0.1, v_thr=3.0):

    if abs(start_state.s_d) < v_thr or abs(t_v) < v_thr:
        lat_mode = "low_velocity"
    else:
        lat_mode = "high_velocity"
    
    # Basic configurations
    min_velocity = 0.0001
    start_state.s_d = scale_velocity(start_state.s_d, min_velocity)
    t_v = scale_velocity(t_v, min_velocity)

    # Longitudinal motion using polynomial
    qp_long = compute_longitudinal_trajectory(start_state, t_s, t_v, t_T)

    t = np.arange(0.0, t_T, dt)

    # Calculate longitudinal trajectory derivatives
    s, s_d, s_dd, s_ddd = (
        qp_long.calc_point(t),
        qp_long.calc_first_derivative(t),
        qp_long.calc_second_derivative(t),
        qp_long.calc_third_derivative(t),
    )

    if lat_mode == 'high_velocity':
        # Lateral motion using a quintic polynomial
        qp_lat = compute_lateral_trajectory(start_state, t_d, t_T)

        d, d_d, d_dd, d_ddd = (
            qp_lat.calc_point(t),
            qp_lat.calc_first_derivative(t),
            qp_lat.calc_second_derivative(t),
            qp_lat.calc_third_derivative(t),
        )

        d_d_time, d_dd_time = d_d, d_dd
        d_d /= s_d
        d_dd = (d_dd - d_d * s_dd) / np.power(s_d, 2)

    elif lat_mode == 'low_velocity':
        
        ds = max(s[-1] - s[0], 0.00001)
        
        if start_state.s_d != 0.0:
            c_d_d = start_state.d_d / start_state.s_d
            c_d_dd = (start_state.d_dd - start_state.s_dd * c_d_d) / np.power(start_state.s_d, 2)
        else:
            c_d_d, c_d_dd = 0.0, 0.0
        
        qp_lat = QuinticPolynomial(xs=start_state.d, vxs=c_d_d, axs=c_d_dd, xe=t_d, vxe=0.0, axe=0.0, time=ds)

        d = qp_lat.calc_point(s - s[0])
        d_d = qp_lat.calc_first_derivative(s - s[0])
        d_dd = qp_lat.calc_second_derivative(s - s[0])
        d_ddd = qp_lat.calc_third_derivative(s - s[0])

        d_d_time, d_dd_time = s_d * d_d, s_dd * d_d + np.power(s_d, 2) * d_dd
    

    # Calculate global path coordinates
    global_path_x, global_path_y = calculate_global_path(current_lane, s, d)

    # Compute derivatives and curvatures
    dx, dy = np.gradient(global_path_x, s), np.gradient(global_path_y, s)
    ddy, ddx = np.gradient(dy, s), np.gradient(dx, s)
    dddy, dddx = np.gradient(ddx, s), np.gradient(ddy, s)
    
    global_path_yaw = np.arctan2(dy, dx)

    epsilon = 1e-6  # Small constant to avoid division by zero
    global_path_curv = np.divide(
        np.multiply(dx, ddy) - np.multiply(ddx, dy),
        np.maximum(np.power(dx**2 + dy**2, 1.5), epsilon)
    )

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

    # Calculate overall velocity in the global frame
    v_x = s_d * np.cos(global_path_yaw) - d_d_time * np.sin(global_path_yaw)
    v_y = s_d * np.sin(global_path_yaw) + d_d_time * np.cos(global_path_yaw)
    v_global = np.sqrt(v_x**2 + v_y**2)

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
    
    # Create Frenet trajectory object
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
        yaw=global_path_yaw,
        v=v_global,  # Use the overall global velocity
        curv=curv,
    )

    return fp

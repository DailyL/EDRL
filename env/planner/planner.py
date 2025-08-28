import numpy as np
import matplotlib.pyplot as plt
from env.planner.path_planning import quintic_polynomials_planner, pid_traj_controller, calc_frenet_trajectory
from env.util.env_utils import VehiclePIDController


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

    def plot_traj(self, fp_list, now_long = 0, now_lateral=0):

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

from env.planner.polynomials import QuinticPolynomial
class Frenet_planner(object):

    def __init__(self, pid_control):

        self.pid_control = pid_control
        self.pid_controller = VehiclePIDController()
        self.padd = None
        self.fp = None

        # cost weights
        self.K_J = 0.0001
        self.K_V = 0.1
        self.K_P = 0.01
        self.K_LAT = 1.0
        self.K_LON = 1.0
    
    def plan(self, frenet_state, target_lateral, target_long, target_time, target_speed_long, current_lane):

        self.fp = calc_frenet_trajectory(
                start_state=frenet_state,
                t_d=target_lateral, t_s=target_long, t_T=target_time, t_v=target_speed_long ,
                current_lane=current_lane, padd=self.padd,
            )
        self.padd = self.fp

        cost_fp = self.calc_cost(
            target_speed_long,
            frenet_state,
            target_long,
            target_lateral,
        )

        return self.fp, cost_fp
    

    def act(self, target_x, target_y, target_yaw, target_speed, ego_vehicle):

        if self.pid_control:
            action = pid_traj_controller(
                target_x, target_y, target_yaw, target_speed, ego_vehicle, self.pid_controller
            )
            return action
        else:
            pass


    def calc_cost(self, target_speed_long, frenet_state, target_long, target_lat):

        target_long = 10

        c_s = frenet_state.s
        c_d = frenet_state.d


        # Jerk cost
        Jd = sum(np.power(self.fp.d_ddd, 2))  # square of jerk (lateral)
        Js = sum(np.power(self.fp.s_ddd, 2))  # square of jerk (long)

        #Acc cost
        Ad = sum(np.power(self.fp.d_dd, 2))  # square of jerk (lateral)
        As = sum(np.power(self.fp.s_dd, 2))  # square of jerk (long)

        # Speed cost, square of diff from target speed
        Ss = (target_speed_long - self.fp.s_d[-1]) ** 2


        # Position cost
        Ps = (c_s + target_long - self.fp.s[-1])**2
        Pd = (c_d + target_lat- self.fp.d[-1])**2


        # Cost long
        Cs = self.K_J * As + self.K_V * Ss + self.K_P * Ps

        # Cost lateral
        Cd = self.K_J * Ad + self.K_P * Pd

        first_cost = self.K_LAT * Cd + self.K_LON * Cs
        
        # Second cost

        s_dd_min = -7
        s_dd_max = 5
        infesible_cost = False
        sec_cost = 0

        if any(sd < 0 for sd in self.fp.s_d):
            infesible_cost = True

        if any(sdd < s_dd_min or sdd > s_dd_max for sdd in self.fp.s_dd):
            infesible_cost = True
        
        if any(curv > 1/6 for curv in self.fp.curv):
            infesible_cost = True

        if infesible_cost:
            sec_cost = -5


        # third cost

        summme = sum(abs(x) for x in self.fp.s_ddd)

        third_cost = summme * len(self.fp.s_ddd) * 0.1


        return [first_cost, sec_cost, third_cost]



    def plot_traj(self, fp_list, now_long = 0, now_lateral=0):

        plt.ion()  # Enable interactive mode
        area = 20
        for i in range(len(fp_list.t)):
            plt.cla()  # Clear axis for each frame
            position = [fp_list.s[i], fp_list.d[i]]

            plt.plot(position[0], position[1], 'go')

            # Plot headings as arrows
            headings = fp_list.yaw[i:]  # Extract heading values
            dx = np.cos(headings)           # X-components of arrows
            dy = np.sin(headings)           # Y-components of arrows
            
            plt.quiver(fp_list.s[i:], fp_list.d[i:], dx, dy, angles='xy', scale_units='xy', scale=1, color='blue', label="Heading")
            plt.xlim(fp_list.s[i] - area, fp_list.s[i] + area)
            plt.ylim(fp_list.d[i] - area, fp_list.d[i] + area)
            plt.grid(True)
            
            plt.draw()     # Force a redraw of the current figure
            plt.pause(0.01)  # Short pause to update plot in real-time

        plt.ioff()  # Disable interactive mode when done            
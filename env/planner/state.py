from dataclasses import dataclass




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

        self.cd = 0.0
        self.cv = 0.0
        self.cf = 0.0


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

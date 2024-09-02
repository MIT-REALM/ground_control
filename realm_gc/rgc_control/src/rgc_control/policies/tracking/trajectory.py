"""Define linearly interpolated trajectories."""
from typing import List
import pickle
import numpy as np
import math
from rgc_control.policies.cubic_spline import CubicSpline2D

def pi_2_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

class SplineTrajectory2D():
    """
    The trajectory for a single robot, represented by curvilinear interpolation
    t represents the index in the spline list
    args:
        p: the array of control points for the trajectory
    """
    def __init__(self, v_ref:float, filepath: str, traj: dict = None):
        #Loads a dictionary with keys 'X' and 'Y' and converts it into spline information
        if traj is not None:
            self.traj = traj
        else:
            with open(filepath,'rb') as file:
                self.traj = pickle.load(file) 
                self.traj['Y'] = np.array(self.traj['Y']) * 5 - 4.0
                # self.traj['Y'] = np.array(self.traj['Y']) - 4.0 
                
                self.traj['X'] = np.array(self.traj['X']) / 2
                        
        self.cx,self.cy,self.cyaw,self.ck = self.calc_spline_course()
        self.v_ref = v_ref
        self.v = self.calc_speed_profile(self.v_ref)

    def calc_nearest_index(self, state):
        cx, cy, cyaw = self.cx, self.cy, self.cyaw
        
        dx = [state.x - icx for icx in cx]
        dy = [state.y - icy for icy in cy]

        d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]

        mind = min(d)

        ind = d.index(mind)

        mind = np.abs(math.sqrt(mind))

        return ind, mind

    def calc_spline_course(self, ds=0.1):
        trajectory = self.traj
        x = trajectory['X']
        y = trajectory['Y']
        sp = CubicSpline2D(x, y)
        if np.isnan(sp.s[-1]):
            print(sp.s)
        s = list(np.arange(0, sp.s[-1], ds))

        rx, ry, ryaw, rk = [], [], [], []
        for i_s in s:
            ix, iy = sp.calc_position(i_s)
            rx.append(ix)
            ry.append(iy)
            ryaw.append(sp.calc_yaw(i_s))
            rk.append(sp.calc_curvature(i_s))
        return rx, ry, ryaw, rk
    
    def calc_speed_profile(self,v_ref):
        speed_profile = [v_ref] * len(self.cyaw)
        direction = 1.0

        # Set stop point
        for i in range(len(self.cyaw) - 1):
            dyaw = abs(self.cyaw[i + 1] - self.cyaw[i])
            switch = np.pi / 4.0 <= dyaw < np.pi / 2.0

            if switch:
                direction *= -1

            if direction != 1.0:
                speed_profile[i] = - v_ref
            else:
                speed_profile[i] = v_ref

            if switch:
                speed_profile[i] = 0.0
            
            # speed down
            if i>20:
                for i in range(20):
                    speed_profile[-i] = v_ref / (50 - i)
                    if speed_profile[-i] <= 1.0 / 3.6:
                        speed_profile[-i] = 1.0 / 3.6

            return speed_profile

    def __call__(self, t: int):
        """Return the point along the trajectory at the given index"""
        return np.array([self.cx[t],self.cy[t], self.cyaw[t], self.v[t], self.ck[t]])


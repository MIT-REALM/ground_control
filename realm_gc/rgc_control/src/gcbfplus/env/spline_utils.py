import jax
import jax.numpy as np


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

    Examples
    --------
    You can interpolate 1D data points.

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> x = np.arange(5)
    >>> y = [1.7, -6, 5, 6.5, 0.0]
    >>> sp = CubicSpline1D(x, y)
    >>> xi = np.linspace(0.0, 5.0)
    >>> yi = [sp.calc_position(x) for x in xi]
    >>> plt.plot(x, y, "xb", label="Data points")
    >>> plt.plot(xi, yi , "r", label="Cubic spline interpolation")
    >>> plt.grid(True)
    >>> plt.legend()
    >>> plt.show()

    .. image:: cubic_spline_1d.png

    """

    def __init__(self, x, y):

        h = np.diff(x)
        # if np.any(h < 0):
        #     raise ValueError("x coordinates must be sorted in ascending order")

        self.a, self.b, self.c, self.d = [], [], [], []
        self.x = x
        self.y = y
        self.nx = len(x)  # dimension of x

        # calc coefficient a
        self.a = y.copy()

        # calc coefficient c
        A = self.__calc_A(h)
        B = self.__calc_B(h, self.a)
        self.c = np.linalg.solve(A, B)
        d = np.zeros(self.nx - 1)
        b = np.zeros(self.nx - 1)
        def body(i, input):
            d=  input[0]
            b = input[1]
            d = d.at[i].set((self.c[i + 1] - self.c[i]) / (3.0 * h[i]))
            b = b.at[i].set(1.0 / h[i] * (self.a[i + 1] - self.a[i]) \
                - h[i] / 3.0 * (2.0 * self.c[i] + self.c[i + 1]))
            return (d, b)
        
        d, b = jax.lax.fori_loop(0, self.nx - 1, body, (d, b))
        self.d = d
        self.b = b
        
        # # calc spline coefficient b and d
        # for i in range(self.nx - 1):
        #     d = (self.c[i + 1] - self.c[i]) / (3.0 * h[i])
        #     b = 1.0 / h[i] * (self.a[i + 1] - self.a[i]) \
        #         - h[i] / 3.0 * (2.0 * self.c[i] + self.c[i + 1])
        #     self.d.append(d)
        #     self.b.append(b)

    def calc_position(self, x):
        """
        Calc `y` position for given `x`.

        if `x` is outside the data point's `x` range, return None.

        Returns
        -------
        y : float
            y position for given x.
        """
        # if x < self.x[0]:
        #     return None
        # elif x > self.x[-1]:
        #     return None

        i = self.__search_index(x)
        dx = x - self.x[i]
        position = self.a[i] + self.b[i] * dx + \
            self.c[i] * dx ** 2.0 + self.d[i] * dx ** 3.0

        position = np.where(x < self.x[0], -10.0, position)
        position = np.where(x > self.x[-1], -10.0, position)
        
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

        # if x < self.x[0]:
        #     return None
        # elif x > self.x[-1]:
        #     return None

        i = self.__search_index(x)
        dx = x - self.x[i]
        dy = self.b[i] + 2.0 * self.c[i] * dx + 3.0 * self.d[i] * dx ** 2.0
        
        dy = np.where(x < self.x[0], -1000.0, dy)
        dy = np.where(x > self.x[-1], -1000.0, dy)
        
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

        # if x < self.x[0]:
        #     return None
        # elif x > self.x[-1]:
        #     return None

        i = self.__search_index(x)
        dx = x - self.x[i]
        ddy = 2.0 * self.c[i] + 6.0 * self.d[i] * dx
        
        ddy = np.where(x < self.x[0], -1000.0, ddy)
        ddy = np.where(x > self.x[-1], -1000.0, ddy)
        
        
        return ddy

    def __search_index(self, x):
        """
        search data segment index
        """
        # return bisect.bisect(self.x, x) - 1
        i = np.searchsorted(self.x, x, side='right') - 1
        return i
    
    def __calc_A(self, h):
        """
        calc matrix A for spline coefficient c
        """
        A = np.zeros((self.nx, self.nx))
        A = A.at[0, 0].set(1.0)
        # A[0, 0] = 1.0
        # for i in range(self.nx - 1):
        #     if i != (self.nx - 2):
        #         # A[i + 1, i + 1] = 2.0 * (h[i] + h[i + 1])
        #         A = A.at[i + 1, i + 1].set(2.0 * (h[i] + h[i + 1]))
        #     # A[i + 1, i] = h[i]
        #     A = A.at[i + 1, i].set(h[i])
        #     # A[i, i + 1] = h[i]
        #     A = A.at[i, i + 1].set(h[i])
       
        def body(i, A):
            # if i != (self.nx - 2):
                # A = A.at[i + 1, i + 1].set(2.0 * (h[i] + h[i + 1]))
            A = np.where(i < (self.nx - 2), A.at[i + 1, i + 1].set(2.0 * (h[i] + h[i + 1])), A)
            A = A.at[i + 1, i].set(h[i])
            A = A.at[i, i + 1].set(h[i])
            return A
        
        A = jax.lax.fori_loop(0, self.nx - 1, body, A)
        
        # return A
        # A[0, 1] = 0.0
        # A[self.nx - 1, self.nx - 2] = 0.0
        # A[self.nx - 1, self.nx - 1] = 1.0
        A = A.at[0, 1].set(0.0)
        A = A.at[self.nx - 1, self.nx - 2].set(0.0)
        A = A.at[self.nx - 1, self.nx - 1].set(1.0)
        return A

    def __calc_B(self, h, a):
        """
        calc matrix B for spline coefficient c
        """
        B = np.zeros(self.nx)
        # for i in range(self.nx - 2):
        #     B = B.at[i + 1].set(3.0 * (a[i + 2] - a[i + 1]) / h[i + 1]\
        #         - 3.0 * (a[i + 1] - a[i]) / h[i])
        #     # B[i + 1] = 3.0 * (a[i + 2] - a[i + 1]) / h[i + 1]\
        #     #     - 3.0 * (a[i + 1] - a[i]) / h[i]
        # return B
        def body(i, B):
            B = B.at[i + 1].set(3.0 * (a[i + 2] - a[i + 1]) / h[i + 1]\
                - 3.0 * (a[i + 1] - a[i]) / h[i])
            return B
        B = jax.lax.fori_loop(0, self.nx - 2, body, B)
        return B
        # B = B.at[1:].set(3.0 * (a[2:] - a[1:-1]) / h[1:] - 3.0 * (a[1:-1] - a[:-2]) / h[:-1])
    
class CubicSpline2D:
    """
    Cubic CubicSpline2D class

    Parameters
    ----------
    x : list
        x coordinates for data points.
    y : list
        y coordinates for data points.

    Examples
    --------
    You can interpolate a 2D data points.

    >>> import matplotlib.pyplot as plt
    >>> x = [-2.5, 0.0, 2.5, 5.0, 7.5, 3.0, -1.0]
    >>> y = [0.7, -6, 5, 6.5, 0.0, 5.0, -2.0]
    >>> ds = 0.1  # [m] distance of each interpolated points
    >>> sp = CubicSpline2D(x, y)
    >>> s = np.arange(0, sp.s[-1], ds)
    >>> rx, ry, ryaw, rk = [], [], [], []
    >>> for i_s in s:
    ...     ix, iy = sp.calc_position(i_s)
    ...     rx.append(ix)
    ...     ry.append(iy)
    ...     ryaw.append(sp.calc_yaw(i_s))
    ...     rk.append(sp.calc_curvature(i_s))
    >>> plt.subplots(1)
    >>> plt.plot(x, y, "xb", label="Data points")
    >>> plt.plot(rx, ry, "-r", label="Cubic spline path")
    >>> plt.grid(True)
    >>> plt.axis("equal")
    >>> plt.xlabel("x[m]")
    >>> plt.ylabel("y[m]")
    >>> plt.legend()
    >>> plt.show()

    .. image:: cubic_spline_2d_path.png

    >>> plt.subplots(1)
    >>> plt.plot(s, [np.rad2deg(iyaw) for iyaw in ryaw], "-r", label="yaw")
    >>> plt.grid(True)
    >>> plt.legend()
    >>> plt.xlabel("line length[m]")
    >>> plt.ylabel("yaw angle[deg]")

    .. image:: cubic_spline_2d_yaw.png

    >>> plt.subplots(1)
    >>> plt.plot(s, rk, "-r", label="curvature")
    >>> plt.grid(True)
    >>> plt.legend()
    >>> plt.xlabel("line length[m]")
    >>> plt.ylabel("curvature [1/m]")

    .. image:: cubic_spline_2d_curvature.png
    """

    def __init__(self, x, y):
        self.s = self.__calc_s(x, y)
        self.sx = CubicSpline1D(self.s, x)
        self.sy = CubicSpline1D(self.s, y)

    def __calc_s(self, x, y):
        dx = np.diff(x)
        dy = np.diff(y)
        self.ds = np.hypot(dx, dy)
        # s = np.zeros_like(x)
        s = np.cumsum(self.ds)
        s = np.concatenate((np.array([0]), s))
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
        dx = np.where(dy > -100, dx, -1000.0)
        yaw = np.where(dx > -100, np.arctan2(dy, dx), 0.0)
        return yaw



class SplineTrajectory2D():
    """
    The trajectory for a single robot, represented by curvilinear interpolation
    t represents the index in the spline list
    args:
        p: the array of control points for the trajectory
    """
    def __init__(self, v_ref:float, x, y):
        #Loads a dictionary with keys 'X' and 'Y' and converts it into spline information
        self.x = x
        self.y = y
        self.cx,self.cy,self.cyaw = self.calc_spline_course()
        self.v_ref = v_ref
        self.v = self.calc_speed_profile(self.v_ref)

    def calc_spline_course(self, ds=0.1):
        x = self.x
        y = self.y
        sp = CubicSpline2D(x, y)
        # if np.isnan(sp.s[-1]):
        #     print(sp.s)
        # s = list(np.arange(0, sp.s[-1], ds))
        # s = np.arange(0, sp.s[-1], ds)
        c = np.arange(0, 1, 1 / ds)
        s = sp.s[-1] * c
        rx = np.zeros(len(s))
        ry = np.zeros(len(s))
        ryaw = np.zeros(len(s))
        # rk = np.zeros(len(s))
        
        def body(i, input):
            rx = input[0]
            ry = input[1]
            ryaw = input[2]
            # rk = input[3]
            ix, iy = sp.calc_position(i)
            rx = np.where(ix> -2, rx.at[i].set(ix), rx)
            ry = np.where(iy> -2, ry.at[i].set(iy), ry)
            rx = np.where((ix < -2) & (i< 2), rx.at[i].set(x[0]), rx)
            ry = np.where((iy < -2) & (i< 2), ry.at[i].set(y[0]), ry)
            rx = np.where((ix < -2) & (i> len(x) - 2), rx.at[i].set(x[-1]), rx)
            ry = np.where((iy < -2) & (i> len(y) - 2), ry.at[i].set(y[-1]), ry)
            # rx = rx.at[i].set(ix)
            # ry = ry.at[i].set(iy)
            ryaw = ryaw.at[i].set(sp.calc_yaw(i))
            # rk = rk.at[i].set(sp.calc_curvature(i))
            return (rx, ry, ryaw)
        
        rx, ry, ryaw = jax.lax.fori_loop(0, len(s), body, (rx, ry, ryaw))
        # for i_s in s:
            # ix, iy = sp.calc_position(i_s)
            # rx.append(ix)
            # ry.append(iy)
            # ryaw.append(sp.calc_yaw(i_s))
            # rk.append(sp.calc_curvature(i_s))
        return rx, ry, ryaw
    
    def calc_speed_profile(self,v_ref):
        speed_profile = v_ref * np.ones(len(self.cx))
        direction = 1.0

        # Set stop point
        # for i in range(len(self.cyaw) - 1):
        #     dyaw = abs(self.cyaw[i + 1] - self.cyaw[i])
        #     switch = np.pi / 4.0 <= dyaw < np.pi / 2.0

        #     if switch:
        #         direction *= -1

        #     if direction != 1.0:
        #         speed_profile[i] = - v_ref
        #     else:
        #         speed_profile[i] = v_ref

        #     if switch:
        #         speed_profile[i] = 0.0
        #     return speed_profile
        
        def body(i, speed_profile):
            dyaw = abs(self.cyaw[i + 1] - self.cyaw[i])
            # switch = np.pi / 4.0 <= dyaw < np.pi / 2.0
            switch = (dyaw >= np.pi / 4.0) & (dyaw < np.pi / 2.0)
            # direction = 1.0 if not switch else -1
            direction = np.where(switch, -1.0, 1.0)
            # speed_profile = speed_profile.at[i].set(-v_ref if direction != 1.0 else v_ref)
            speed_profile = np.where(direction != 1.0, speed_profile.at[i].set(-v_ref), speed_profile)
            # if switch:
            #     speed_profile = speed_profile.at[i].set(0.0)
            speed_profile = np.where(switch, speed_profile.at[i].set(0.0), speed_profile)
            return speed_profile
        
        speed_profile = jax.lax.fori_loop(0, len(self.cyaw) - 1, body, speed_profile)
        return speed_profile
        # speed down
        """
        if i>20:
            for i in range(20):
                speed_profile[-i] = v_ref / (50 - i)
                if speed_profile[-i] <= 1.0 / 3.6:
                    speed_profile[-i] = 1.0 / 3.6
            return speed_profile
        """
    def __call__(self, t: int):
        """Return the point along the trajectory at the given index"""
        return np.array([self.cx[t],self.cy[t], self.cyaw[t], self.v[t], self.ck[t]])

def get_spline(state, goal):
    c = np.linspace(0, 1, 20)
    x = state[0] * (1 - c) + goal[0] * c
    y = state[1] * (1 - c) + goal[1] * c
    
    sp = SplineTrajectory2D(1.0, x, y)
    x_ref = sp.cx
    y_ref = sp.cy
    yaw_ref = sp.cyaw
    # v_ref = np.ones(len(x_ref))
    v_ref = sp.v
    
    # find where x_ref is x[-1]
    # x_ind = np.where(x_ref == x[-1])[0][0]
    # y_ind = np.where(y_ref == y[-1])[0][0]
    # min_ind = max(x_ind, y_ind) + 1
    # x_ref = x_ref[:min_ind]
    # y_ref = y_ref[:min_ind]
    # yaw_ref = yaw_ref[:min_ind]
    # v_ref = v_ref[:min_ind]
    
    return np.array([x_ref[1], y_ref[1], np.cos(yaw_ref[1]),np.sin(yaw_ref[1]), v_ref[1]])
    # return x_ref, y_ref, yaw_ref, v_ref    

# def main():
#     x = np.array([-2.5, 0.0, 2.5, 5.0, 7.5, 3.0, -1.0])
#     y = np.array([0.7, 2.0, 5.0, 6.5, 0.0, 5.0, -2.0])
#     ds = 0.1  # [m] distance of each interpolated points
#     # traj = {'X':x, 'Y':y}
#     # sp = SplineTrajectory2D(1.0, traj)
#     # get_spline(traj)
#     x_ref, y_ref, yaw_ref, v_ref = get_spline(x, y)
#     print('x_ref:', x_ref)
#     print('y_ref:', y_ref)
#     print('yaw_ref:', yaw_ref)
#     print('v_ref:', v_ref)

# if __name__ == "__main__":
#     with ipdb.launch_ipdb_on_exception():
#         main()


import numpy as np
import pickle

import matplotlib.pyplot as plt

np.random.seed(2)
n = 20

x = np.random.uniform(0.5,4.5,n).reshape(-1,1)
y = np.random.uniform(0.5,6.5,n).reshape(-1,1)

samples = np.hstack((x,y))
for i in range(n):
    print(i, samples[i])

m=15
plt.scatter(x[:m],y[:m])
plt.show()

def lane_change_trajectory(lane_width, v=1.5, t_total=3):

    t = np.linspace(0,t_total,num=400)
    curvature_factor=2.5

    x = v*t
    y = (lane_width/2)*(1-np.cos(curvature_factor * np.pi * t / t_total))

    return x, y, t

for i in range(n):
    x_ref, y_ref, _ = lane_change_trajectory(lane_width=samples[i,0])
    data = {'X':y_ref, 'Y':x_ref, "V_ref": samples[i,1],'lw':samples[i,0]}
    filename = f"realm_gc/rgc_control/saved_policies/test_experiments/test_initial_{i}.pkl"
    with open(filename,"wb") as f:
        pickle.dump(data,f)
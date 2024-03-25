import pickle
import numpy as np
import matplotlib.pyplot as plt
"""
y_ = list(np.linspace(-5,3,100))
x_ = list(np.zeros((100,)))
data={'X':x_,'Y':y_}
file = open('ego_traj_nominal_111.pkl','wb')
pickle.dump(data,file)

"""
file = open('/home/realm/ground_control_anjali/ground_control/realm_gc/rgc_control/saved_policies/base/ego_traj_failure_1_redact.pkl','rb')
data = pickle.load(file)
file.close()
x = data['X']
y = data['Y']
plt.plot(x[0:-25],y[0:-25])
plt.show()
"""
x_ = x[0:-25]
y_ = y[0:-25] 
data={'X':x_,'Y':y_}
file_ = open('ego_traj_failure_1.pkl','wb')
pickle.dump(data,file_)
file.close()
"""
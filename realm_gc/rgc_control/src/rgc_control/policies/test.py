import pickle
import numpy as np
y_ = list(np.linspace(-5,3,100))
x_ = list(np.zeros((100,)))
data={'X':x_,'Y':y_}
file = open('ego_traj_nominal_111.pkl','wb')
pickle.dump(data,file)


import pickle
import matplotlib.pyplot as plt

file=open("/home/realm/ground_control_anjali/ground_control/realm_gc/rgc_control/saved_policies/data/data_initial_0_2.43.pkl" ,'rb')
data = pickle.load(file)
file.close()
plt.scatter(data['states']['x'],data['states']['y'])
plt.show()
print(len(data['X']))
#with open("/home/realm/ground_control_anjali/ground_control/realm_gc/data_ego_traj_nominal_3_2.0.pkl",'rb') as f:
#    data = pickle.load(f)
#    for k, v in data.items():
#        print(k, v)
#        print('\n')

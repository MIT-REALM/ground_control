import pickle
import matplotlib.pyplot as plt

file=open("ground_control/realm_gc/rgc_control/saved_policies/base/ego_traj_nominal_1.pkl" ,'rb')
data = pickle.load(file)
file.close()
plt.scatter(data['X'],data['Y'])
plt.show()
print(len(data['X']))
#with open("/home/realm/ground_control_anjali/ground_control/realm_gc/data_ego_traj_nominal_3_2.0.pkl",'rb') as f:
#    data = pickle.load(f)
#    for k, v in data.items():
#        print(k, v)
#        print('\n')

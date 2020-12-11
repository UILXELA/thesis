import pickle
import numpy as np

filename=input("type in the pkl name\n")
with open(filename , 'rb') as f:
    data = pickle.load(f)

for i in range(len(data)):
    original=data[i]['point_cloud']['velodyne_path']
    modified=[]
    for j in range(len(data[i]['annos']['index'])):
        label_str=str(int(original[-10:-4]))+" "+str(data[i]['annos']['index'][j])
        modified.append(label_str)
    data[i]['annos']['dcount']=modified
    #print(data[i]['annos']['dcount'])

with open("mod_"+filename, 'wb') as f:
    pickle.dump(data,f)

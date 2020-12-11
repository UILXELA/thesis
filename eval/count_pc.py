import pickle
import numpy as np
import os

#Count # of pts in every detected object
fname_detected="./detected.txt"
fname_undetected="./undetected.txt"
#a,b=os.path.splitext(filename)
with open(fname_detected , 'rb') as f:
    detected = f.readlines()

with open(fname_undetected , 'rb') as f:
    undetected = f.readlines()

det_point_count=[]
un_point_count=[]
for i in range(len(detected)):
    (f_ind,c_ind)=detected[i].split()
    pc_fname=f_ind.decode("utf-8")+"_"+"Car_"+c_ind.decode("utf-8")+".bin"
    point_cloud_data = np.fromfile(pc_fname, '<f4')  # little-endian float32
    point_cloud_data = np.reshape(point_cloud_data, (-1, 4))    # x, y, z, r
    point_count=point_cloud_data.shape[0]
    det_point_count.append(str(point_count)+'\n')

for i in range(len(undetected)):
    (f_ind,c_ind)=undetected[i].split()
    pc_fname=f_ind.decode("utf-8")+"_"+"Car_"+c_ind.decode("utf-8")+".bin"
    point_cloud_data = np.fromfile(pc_fname, '<f4')  # little-endian float32
    point_cloud_data = np.reshape(point_cloud_data, (-1, 4))    # x, y, z, r
    point_count=point_cloud_data.shape[0]
    un_point_count.append(str(point_count)+'\n')

with open("detected_count.txt" , 'w') as f:
    f.writelines(det_point_count)

with open("undetected_count.txt" , 'w') as f:
    f.writelines(un_point_count)


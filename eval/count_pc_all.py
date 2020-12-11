import pickle
import numpy as np
import os
import glob
#Get point densities of all pcd

fnames=glob.glob("*.bin")
all_point_count=[]
for fname in fnames:
    pc_fname=fname
    point_cloud_data = np.fromfile(pc_fname, '<f4')  # little-endian float32
    point_cloud_data = np.reshape(point_cloud_data, (-1, 4))    # x, y, z, r
    point_count=point_cloud_data.shape[0]
    all_point_count.append(point_count)

mean=np.mean(all_point_count)
median=np.median(all_point_count)
print("mean:" + str(mean))
print(median)



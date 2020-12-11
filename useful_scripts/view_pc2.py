import numpy as np
import pptk


path_to_point_cloud ='000000.bin'

point_cloud_data = np.fromfile(path_to_point_cloud, '<f4')
point_cloud_data = np.reshape(point_cloud_data, (-1, 4))    
pptk.viewer(point_cloud_data[:, :3])

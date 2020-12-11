import os
import glob
import numpy as np
import math
"""
Conversion between kitti coordinates and camera coordinates is:

Cam: x,y,z ==> Kitti: -y,-z,x

"""
BASE_DIR="./"
LABEL_DIR="training/label_2"
LOC_DIR="training/locational"
REF="_out1"

def read_client_dir():
	folders = next(os.walk(BASE_DIR))[1]
	return folders

def distance_check(loc1, loc2):
	distance=np.sqrt((loc1[0]-loc2[0])**2+(loc1[1]-loc2[1])**2+(loc1[2]-loc2[2])**2)
	if distance>1:
		return True

def get_path(label_folder,loc_folder,ref_folder,fname):
	label_path=os.path.join(label_folder,fname)
	loc_path=os.path.join(loc_folder,fname)
	ref_path=os.path.join(ref_folder,fname)
	return label_path, loc_path, ref_path

def tx_label(count,save_folder):
	car_count=0
	#clients=read_client_dir()
	clients=['_out1', '_out2', '_out3', '_out4']	
	print(clients)
	for ind in range(count):
		fname = ("%06d.txt"%ind)
		txmed_labels=[]
		txmed_locs=[]
		ref_locs=[]

		for cur_client in clients:
			
			label_folder=os.path.join(cur_client,LABEL_DIR)
			loc_folder=os.path.join(cur_client,LOC_DIR)
			ref_folder=os.path.join(REF,LOC_DIR)	

			label_path, loc_path, ref_path = get_path(label_folder,loc_folder,ref_folder,fname)

		#for label_path, loc_path, ref_path in zip(label_paths, loc_paths, ref_paths):
			loc=open(loc_path,'r').read()
			ref=open(ref_path,'r').read()
			
			ref = np.array([float(i) for i in ref.split()])
			loc = np.array([float(i) for i in loc.split()])

			label_f=open(label_path,'r')
			labels = [line.rstrip('\n') for line in label_f]
			
			#if len(labels)==0:
			#	np.savetxt(os.path.join(save_folder,os.path.basename(label_path)),txmed_labels,delimiter=' ')
			#	continue
			for label in labels:
				label = np.array([float(i) for i in label.split() if i!= 'Car'])
				if cur_client==REF:
					ref_locs.append(label[[10,11,12]])
					continue
				#print(label.shape)
				yaw=np.radians(-ref[4])
				yaw2=np.radians(-loc[4])	
	
				rot = np.matrix(np.identity(2))
				rot[0,0]=np.cos(yaw)
				rot[0,1]=-np.sin(yaw)
				rot[1,0]=np.sin(yaw)
				rot[1,1]=np.cos(yaw)
				rel_loc=loc-ref
				#print(rel_loc)
				rel_loc[0],rel_loc[1] = rot*np.array([[rel_loc[0]],[rel_loc[1]]])
				#print(rel_loc)
				txm_mat = get_matrix(rel_loc)
				label_location=label[10:13]
				label[3:7]=txm_bbox(label[10:13],label[7:10])
				label_location[[1,2,0]]=label_location[[0,1,2]]
				#print(label_location)
				label_rot_y=label[13]
				label_location=np.concatenate((label_location,[1]))[np.newaxis]
				#print(label_location.shape)
				txmed_location=txm_mat*label_location.T
				label[10:13]=txmed_location[0:3].flatten()
				label[[10,11,12]]=label[[11,12,10]]
				label[13]=label_rot_y+yaw2-yaw
				label[3:7]=txm_bbox(label[10:13],label[7:10])
				label_str=[]
				for ind,elem in enumerate(label):
					if ind<7:
						label_str.append(str(int(elem)))
					else:
						label_str.append(str(elem))
	
	
				save=True
				for loc2 in ref_locs:
					if distance_check(label[[10,11,12]], loc2):
						continue
					else:
						save=False
						break
					
				for loc2 in txmed_locs:
					if distance_check(label[[10,11,12]], loc2) and save:
						continue
					else:
						save=False
						break
	
				if save:
					car_count+=1
					txmed_locs.append(label[[10,11,12]]) 
					txmed_labels.append('Car '+' '.join([elem for elem in label_str]))
			
		filename=os.path.join(save_folder,os.path.basename(label_path))
		with open(filename, 'w') as f:
			labels_str="\n".join([str(label) for label in txmed_labels])
			f.write(labels_str)
			f.close()	
		#np.savetxt(os.path.join(save_folder,os.path.basename(label_path)),txmed_labels,delimiter=' ',fmt='%c')
	print("\n\n\n{}\n\n\n".format(car_count))
	return 

def get_matrix(rel_loc,sc_x=1.0, sc_y=1.0,sc_z=1.0):
    """
    Creates matrix from carla transform.
    """
    x,y,z,pitch,yaw,roll = rel_loc
    #yaw=yaw+180 
    c_y = np.cos(np.radians(yaw))
    s_y = np.sin(np.radians(yaw))
    c_r = np.cos(np.radians(roll))
    s_r = np.sin(np.radians(roll))
    c_p = np.cos(np.radians(pitch))
    s_p = np.sin(np.radians(pitch))
    matrix = np.matrix(np.identity(4))
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z
    matrix[0, 0] = sc_x*c_p * c_y
    matrix[0, 1] = sc_y*(c_y * s_p * s_r - s_y * c_r)
    matrix[0, 2] = -sc_z*(c_y * s_p * c_r + s_y * s_r)
    matrix[1, 0] = sc_x*s_y * c_p
    matrix[1, 1] = sc_y*(s_y * s_p * s_r + c_y * c_r)
    matrix[1, 2] = sc_z*(-s_y * s_p * c_r + c_y * s_r)
    matrix[2, 0] = sc_x*s_p
    matrix[2, 1] = -sc_y*(c_p * s_r)
    matrix[2, 2] = sc_z*(c_p * c_r)
    return matrix

def get_datapoint_count(folders):
	label_count = -1
	loc_count = -1
	count = -1
	for folder in folders:
		label_dir = os.path.join(BASE_DIR,folder,LABEL_DIR)
		loc_dir = os.path.join(BASE_DIR,folder,LOC_DIR)
		label_count= len([name for name in os.listdir(label_dir) if name.endswith('.txt')])
		loc_count= len([name for name in os.listdir(loc_dir) if name.endswith('.txt')])
		if label_count == loc_count:
			if count == label_count or count == -1:
				count = label_count
			else:
				print("Folder %s doesn't match in count" %(folder))
				return None
		else:
			print("Folder %s has unmatched loc and label counts" %folder)
			return None
	return count

def txm_bbox(loc,dim):
	WINDOW_WIDTH = 1248
	WINDOW_HEIGHT = 384
	MINI_WINDOW_WIDTH = 320
	MINI_WINDOW_HEIGHT = 180
 
	WINDOW_WIDTH_HALF = WINDOW_WIDTH / 2
	WINDOW_HEIGHT_HALF = WINDOW_HEIGHT / 2
	k = np.identity(3)
	k[0, 2] = WINDOW_WIDTH_HALF
	k[1, 2] = WINDOW_HEIGHT_HALF
	f = WINDOW_WIDTH / (2.0 * math.tan(90.0 * math.pi / 360.0))
	k[0, 0] = k[1, 1] = f
	bbox=np.empty(4)
	a=np.dot(k,[loc[0]-dim[2]/2,loc[1]-dim[0],loc[2]])
	b=np.dot(k,[loc[0]+dim[2]/2,loc[1],loc[2]])
	a=a/a[2]
	b=b/b[2]
	bbox[0:2]=a[0:2]
	bbox[2:4]=b[0:2]
	#print(bbox)
	return bbox


def main():
	count=get_datapoint_count(read_client_dir())
	save_folder=os.path.join(REF,LABEL_DIR+"_other")
	if not os.path.exists(save_folder):
		os.makedirs(save_folder)
	tx_label(count,save_folder)	
	



main()







	



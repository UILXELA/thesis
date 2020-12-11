# Created by Zheng Liu
#Given data in the format of the KITTI dataset and locations of sensors, fuse point clouds

import os
import numpy as np

BASE_DIR="./"
PC_DIR="training/velodyne_voxel0.5"
LOC_DIR="training/locational"


def read_client_dir():
	folders = next(os.walk(BASE_DIR))[1]
	return folders

def get_datapoint_count(folders):
	pc_count = -1
	loc_count = -1
	count = -1
	for folder in folders:
		pc_dir = os.path.join(BASE_DIR,folder,PC_DIR)
		loc_dir = os.path.join(BASE_DIR,folder,LOC_DIR)
		pc_count= len([name for name in os.listdir(pc_dir) if name.endswith('.bin')])
		loc_count= len([name for name in os.listdir(loc_dir) if name.endswith('.txt')])
		if pc_count == loc_count:
			if count == pc_count or count == -1:
				count = pc_count
			else:
				print("Folder %s doesn't match in count" %(folder))
				return None
		else:
			print("Folder %s has unmatched loc and pc counts" %folder)
			return None
	return count

def format_saved_pc(pc):
	return (np.reshape(pc, (-1, 4)))

def pc_transform(ref,loc,pc):
	ref = np.array([float(i) for i in ref.split()])
	loc = np.array([float(i) for i in loc.split()])
	pc=format_saved_pc(pc)
	#to_unreal_transform2 = get_matrix([0,0,0,0,-ref[4],0], 1.0,1.0,1.0)
	yaw=np.radians(-ref[4])

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
	#txm_mat = np.dot(txm_mat,to_unreal_transform2)
	pc[:,1]=-pc[:,1]
	txmed_pc = txm_mat * pc.transpose()
	return txmed_pc[0:3].transpose()



def save_fused(ind,client_dir,fused_pc):
	fused_dir = os.path.join(BASE_DIR,client_dir,"fused_pc")
	if not os.path.exists(fused_dir):
		os.makedirs(fused_dir)
	fused_pc = np.append(fused_pc, np.ones((fused_pc.shape[0],1)), axis=1)
	fused_pc=np.array(fused_pc).astype(np.float32)
	#print(fused_pc)
	#print(np.amax(fused_pc))
	fused_pc.tofile(os.path.join(fused_dir,("%06d.bin"%ind)))

def fuse(ref_folder, folders, dp_count):
	for i in range(dp_count):
		print("Working on %06d.bin" %i)
		ref_pc_path = os.path.join(BASE_DIR, ref_folder, PC_DIR, ("%06d.bin"%i))
		ref_loc_path = os.path.join(BASE_DIR, ref_folder, LOC_DIR,("%06d.txt"%i))
		fused = format_saved_pc(np.fromfile(ref_pc_path, '<f4'))[:,0:3]
		ref = open(ref_loc_path, 'r').read()
		for folder in folders:
			if folder==ref_folder:
				continue
			pc_path = os.path.join(BASE_DIR, folder, PC_DIR, ("%06d.bin"%i))
			loc_path = os.path.join(BASE_DIR, folder, LOC_DIR,("%06d.txt"%i))
			pc = np.fromfile(pc_path, '<f4')
			loc = open(loc_path, 'r').read()
			#print(fused)
			txmed_pc = pc_transform(ref,loc,pc)
			txmed_pc[:,1]=-txmed_pc[:,1]
			fused=np.concatenate((fused, txmed_pc),axis=0)
			#print(np.amax(fused))
		save_fused(i,ref_folder,fused)




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

def main():
	folders=read_client_dir()
	dp_count=get_datapoint_count(folders)
	for folder in folders:
		print("Working on %s"%folder)
		fuse(folder, folders, dp_count)

main()

import os
import shutil
#for i in range(8050,8051):
#    old=str(i) + '.bin'
#    new="../new/"+'%06d.bin' % i
#    shutil.move(old,new)


file1 = open('a.txt', 'r') 
Lines = file1.readlines() 
file2 = open('b.txt', 'r') 
Lines2 = file2.readlines()

calib_DIR='./calib/' 
img_DIR='./image_2/'
label_DIR='./label_2/'
pcl_DIR='./velodyne/'

# Strips the newline character 
for line in Lines2:
    line=line.rstrip()
    print(line) 
    pcl_fname=line+'.bin'
    img_fname=line+'.png'
    txt_fname=line+'.txt'
    shutil.move(calib_DIR+txt_fname, "../testing/"+calib_DIR+txt_fname)
    #shutil.move(label_DIR+txt_fname, "../testing/"+label_DIR+txt_fname)
    #shutil.move(img_DIR+img_fname, "../testing/"+img_DIR+img_fname)
    #shutil.move(pcl_DIR+pcl_fname, "../testing/"+pcl_DIR+pcl_fname)

import os
IND=0
BASE_DIR="./"
COUNT=1062
def get_subfolders(to_read):
	folders=next(os.walk(to_read))[1]
	full_path=[]
	for folder in folders:
		full_path.append(os.path.join(to_read,folder))
	return full_path

def get_ext(to_read):
	fname=next(os.walk(to_read))[2][1]
	ext=os.path.splitext(fname)[1]
	return ext
	
def renum(start_ind,length,client):
	client="_out"+client
	folders=get_subfolders(os.path.join(BASE_DIR,client,"training"))
	for folder in folders:
		cur_ext=get_ext(folder)
		for i in range(0,start_ind):
			os.remove(os.path.join(folder,("%06d"%i)+cur_ext))
		for i in range(start_ind+length,COUNT):
			os.remove(os.path.join(folder,("%06d"%i)+cur_ext))
		for i in range(start_ind,start_ind+length):
			os.rename(os.path.join(folder,("%06d"%i)+cur_ext),os.path.join(folder,("%06d"%(i+IND))+cur_ext))

def main():
	start_ind=int(input("Starting index   "))
	length=int(input("How many files   "))
	client=input("Client #")
	renum(start_ind,length,client)

main()

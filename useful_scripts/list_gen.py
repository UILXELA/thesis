import os
n=int(input("how many datapoints?\n"))

seq = ["%06d\n"%i for i in range(n)]
#for i in range(n):
#    seq[i] = "%06d"%i
f=open("trainval.txt",'w')
f.writelines(seq)

import numpy as np
import matplotlib.pyplot as mat

def remove_zero(recall,prec,thresh):
	last_ind=len(recall)
	non_zero=last_ind
	for i in range(last_ind-1,0,-1):
		if recall[i]!=0 and prec[i]!=0:
			non_zero=i
			break
	recall=recall[:non_zero+1]
	prec=prec[:non_zero+1]
	thresh=thresh[:non_zero+1]
	#del recall[non_zero+1:last_ind]
	return recall,prec,thresh

recall,prec,thresh=np.loadtxt("./single/roc_0.csv",delimiter=',')
recall,prec,thresh=remove_zero(recall,prec,thresh)

recall2,prec2,thresh2=np.loadtxt("./dense/roc_0.csv",delimiter=',')
recall2,prec2,thresh2=remove_zero(recall2,prec2,thresh2)

recall3,prec3,thresh3=np.loadtxt("./fused_90/roc_0.csv",delimiter=',')
recall3,prec3,thresh3=remove_zero(recall3,prec3,thresh3)

recall4,prec4,thresh4=np.loadtxt("./fused_facing/roc_0.csv",delimiter=',')
recall4,prec4,thresh4=remove_zero(recall4,prec4,thresh4)

recall5,prec5,thresh5=np.loadtxt("./fused_full/roc_0.csv",delimiter=',')
recall5,prec5,thresh5=remove_zero(recall5,prec5,thresh5)

#recall4,prec4,thresh4=np.loadtxt("./all0.5/roc_0.csv",delimiter=',')
#recall4,prec4,thresh4=remove_zero(recall4,prec4,thresh4)

mat.plot(recall,prec)
mat.plot(recall2,prec2)
mat.plot(recall3,prec3)
mat.plot(recall4,prec4)
mat.plot(recall5,prec5)
#mat.plot(recall4,prec4)
mat.title("Precision-Recall Curves, Crossroads")
mat.xlabel("Recall")
mat.ylabel("Precision")
mat.legend(["single (1)","dense single (1)","perp (1,2)","opposite (1,3)","4 fused (1,2,3,4)"])
mat.show()



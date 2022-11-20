from trainer.otf import *
from trainer.eigen import eig
from numpy import linalg as la

# pict_arr = get_pict_array('/home/archkoi/Desktop/alg/105_classes_pins_dataset')
# print('training')
# x = get_mean_diff_array(pict_arr, get_mean_vspace(pict_arr))
# coeff_arr = get_coeff_array(x)
# print('loaded')

coeff_arr = np.loadtxt('/home/archkoi/Desktop/alg/Algeo02-21090/src/p.txt',usecols=range(100))

a = eig(coeff_arr,50)
c, d = la.eig(coeff_arr)
f = []
for i in range(len(a)):
    f.append((c[i],d[i]))
f = sorted(f,key=lambda p: abs(p[0]))[::-1]

np.savetxt('pyeigenresult.txt',(list(zip(*f))[0]),fmt='%.4e')
np.savetxt('myeigenresult.txt',(list(zip(*a))[0]),fmt='%.4e')
#print(d[0])

np.savetxt('pyeigenvecresult.txt',(list(zip(*a))[1]),fmt='%.4e')
np.savetxt('myeigenvecresult.txt',(list(zip(*a))[1]),fmt='%.4e')
#print(d[0])

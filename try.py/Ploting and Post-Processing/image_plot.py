import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import interpolate

im=Image.open('NCB_Fig3_Displacement.tif')
im2=Image.open('Fig.1d.tif')
imarray1=np.array(im)
imarray2=np.array(im2)
print(imarray1)
print(imarray1.shape)
print(np.min(imarray1))
print(np.max(imarray1))
L=48.924
H=46.656

x_rev=np.linspace(L,0, num=472)
y_rev=np.linspace(H,0, num=472)
x=np.linspace(0,L, num=472)
y=np.linspace(0,H, num=472)
X,Y =np.meshgrid(x,y_rev)
#f=interpolate.interp2d(x,y_rev,imarray, kind='cubic')

""" z=np.empty(((N+1),(N+1)), dtype=float)
for i in range(N+1):
    for j in range(N+1): 
        z[i,j]=p(x_plot[i],y_plot[j],0.) 
plotfunction=interpolate.interp2d(X_pl, Y_pl, z)  """

#x_new=np.linspace(0,166, num=1000)
#y_new=np.linspace(0,181, num=1000)
#X_new, Y_new = np.meshgrid(x_new,y_new)

plot_fig=imarray1.transpose()
print(np.mean(imarray1))
plt.figure()
plt.pcolormesh(X,Y,imarray1)
""" plt.pcolormesh(X,Y,imarray) """
plt.colorbar()
plt.show()
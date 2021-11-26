import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import interpolate

N=100
Size=76.464

Len=472
He=472
im=Image.open('NCB_Fig3a_Stress.tif')
calc=np.load('output_deep/output_100_deep.npy')
imarray=np.array(im)
#kernel = np.ones((2,2),np.float32)/25

#blur=cv2.filter2D(calc,-1,kernel)
#cv2.blur(calc, (3,3))

x_N_rev=np.linspace(Size,0, num=N+1)
y_N_rev=np.linspace(Size,0, num=N+1)
x_N=np.linspace(0,Size, num=N+1)
y_N=np.linspace(0,Size, num=N+1)


x_rev=np.linspace(Size,0, num=Len)
x=np.linspace(0,Size, num=He)
y=np.linspace(0,Size, num=He)
X,Y =np.meshgrid(x,y)

interpol=interpolate.interp2d(x_N,y_N,calc, kind='cubic')
interpol2=interpolate.interp2d(x,y,imarray, kind='cubic')

#print(calc.shape)
z2=np.empty((N+1,N+1), dtype=float)
for i in range(N+1):
    for j in range(N+1): 
        z2[i,j]=interpol(x_N[i],y_N[j])

print(calc)
#print(interpol2(x_N,y_N))
#print(np.max(calc))
#print(np.max(interpol2(x_N,y_N)))
#print((np.fliplr(interpol2(x_N,y_N).transpose())-calc)/calc)

#print(np.sum(np.abs((np.fliplr(interpol2(x_N,y_N).transpose())-calc)))/np.sum(np.abs(calc)))

#print(interpol(x_N,y_N))
#print((calc.transpose()-z2))

 
fig, (ax0, ax1) = plt.subplots(1,2)
im1= ax0.pcolormesh(x_N,y_N,calc)
#im1= ax0.pcolormesh(x,y,interpol(x,y))
ax0.set_title('472 x 472 Stress Calculation Fenics')
fig.colorbar(im1, ax=ax0)
#im2=ax1.pcolormesh(x_N_rev,y_N,interpol2(x_N,y_N).transpose())
im2= ax1.pcolormesh(x_rev,y,imarray.transpose())
ax1.set_title('472 x 472 Stress Calculation Comsol')
fig.colorbar(im2, ax=ax1)
plt.show()

""" plt.figure()
plt.pcolormesh(X,Y,imarray.transpose())
plt.pcolormesh(x_N,y_N,scaled_calc)
plt.colorbar()
plt.show()  """
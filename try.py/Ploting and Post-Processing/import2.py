import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import interpolate
import cv2

N=100
L=48.924
H=46.656
calc=np.load('invapodia_100_40_real.npy')
#im=Image.open('Fig.1d.tif')
im=Image.open('Fig.1e.tif')
blur=cv2.blur(calc, (5,5))
#imarray=np.array(im)
stress=np.array(im)

print(stress)
print(calc)
print(calc.shape)
print(stress.shape)
x_N=np.linspace(0,L, num=N+1)
y_N=np.linspace(0,H, num=N+1)

x_rev=np.linspace(L,0, num=302)
y_rev=np.linspace(H,0, num=288)
x=np.linspace(0,L, num=302)
y=np.linspace(0,H, num=288)
#X,Y =np.meshgrid(x,y)

#interpol=interpolate.interp2d(x_N,y_N,calc)
#print(calc.shape)
""" z=np.empty((472,472), dtype=float)
for i in range(1000):
    for j in range(1000): 
        z[i,j]=interpol(x_rev[i],y[j])  """
""" print(imarray.transpose())
print(scaled_calc)  """
fig, (ax0, ax1) = plt.subplots(1,2)

#im1= ax0.pcolormesh(X,Y,interpol(x_rev,y))
im1= ax0.pcolormesh(x_N,y_N,blur)
ax0.set_title('302 x 288 Stress Calculation Fenics')
fig.colorbar(im1, ax=ax0)
im2= ax1.pcolormesh(y_rev,x,stress.transpose())
ax1.set_title('302 x 288 Stress Calculation Comsol')
fig.colorbar(im2, ax=ax1)
plt.show()

""" plt.figure()
plt.pcolormesh(X,Y,imarray.transpose())
plt.pcolormesh(x_N,y_N,scaled_calc)
plt.colorbar()
plt.show()  """
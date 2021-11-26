import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

#Info: mu=0.49; E=450 Pa; d= 0.4 µm, R=17 µm


#Values for Meshsize
x=np.arange(50,470,20)
#Values from the convergence test
y=np.array([[ 330.71878258 ,4420.45004352],
[ 247.94922988, 3081.09614546],
[ 205.95680881 ,2447.18368833],
[ 176.66761476, 2127.2809126 ],
[ 152.93646934, 1911.19150801],
[ 135.21408017 ,1779.4601126 ],
[ 136.26758909 ,1680.61977236],
[ 135.9812443 , 1611.80462404],
[ 132.76842971 ,1558.55675692],
[ 127.8012046 ,1517.28867895],
[ 127.5903317 , 1485.06252871],
[ 126.96732125 ,1457.46451381],
[ 124.85665077, 1436.16844082],
[ 121.92702785 ,1417.83075869],
[ 119.09332388 ,1402.42344289],
[ 118.26908532 ,1389.64216315],
[ 116.7380023  ,1378.23059316],
[ 114.74109432 ,1368.72922211],
[ 112.49387925 ,1360.28666948],
[ 111.09726064, 1352.91403464],
[ 109.92293198, 1346.46559431]])

#Values for total force
value=0.001*y[:,1]
# Hertz-values in µm
con=1.2628590020950471*np.ones(21)
#con = 80*np.ones(21)

#Possible log-log-plot
#plt.plot(np.log(x),np.log(value-con), label='log-log-Plot')
plt.plot(x,value, label='FEM-model Fenics')
plt.plot(x,con, label ='Corrected Hertz Model')
plt.xlabel('Meshsize N')
plt.ylabel('Force in nN')
plt.title('Total Force Calculation for different Mesh Sizes')
plt.legend()
plt.show()




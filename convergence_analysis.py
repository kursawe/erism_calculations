#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 12:16:36 2022

@author: israalnour
"""
import numpy as np
import matplotlib.pyplot as plt
import force_inference

Size=76.464
x_plot=np.linspace(0,Size, num=472)
y_plot=np.linspace(0,Size, num=472)

# Create an array to store the max stressses 
start=5
end=160
increment=10
#mesh_size = np.linspace(5,157,50,dtype=int)
mesh_size = np.arange(start,end,increment)
force = []
force_norm = []
for N in mesh_size:
    print('N=',N)
    value = force_inference.force_inference('NCB_Fig3_Displacement.tif',N)
    force.append(value)
    force_norm.append(np.linalg.norm(value))
    print(np.linalg.norm(value))
    
    #Plot image 
    fig, ax0 = plt.subplots()
    im= ax0.pcolormesh(x_plot,y_plot,value,rasterized=True)#need to check correctness 
                                                        # of pixel locations
    ax0.set_title('The Computed Stress for N = {}'.format(N))
    fig.colorbar(im,ax=ax0)
    plt.savefig('Computed_Stress for N = {}.png'.format(N))
    
print('Force norm =', force_norm)

# Plot max_force vs mesh size
plt.figure()
plt.plot(mesh_size, force_norm)
plt.xlabel('N')
plt.ylabel('Force')
plt.savefig('convergence_test.pdf')

# Calculating the error

error = np.zeros(len(force) - 1)
for i in range(0,len(force)-1):
     error[i] = np.linalg.norm(force[i+1] - force[i]) / np.linalg.norm(force[i])
     
print('Error =',error)

# Adjust the mesh_size array
mesh_size_error = np.arange(start+increment,end,increment)
#print(mesh_size_error)

#Plot error vs mesh size
plt.figure()
plt.plot(mesh_size_error, error)
plt.xlabel('N')
plt.ylabel('Error')
plt.savefig('convergence_test_error.pdf')

# Plot the log log plot
plt.figure()
plt.loglog(mesh_size_error,error)
plt.xlabel('logN')
plt.ylabel('logError')
plt.savefig('convergence_test_logerror.pdf')


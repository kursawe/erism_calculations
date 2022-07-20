#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 17:28:15 2022

@author: israalnour
"""
from fenics import *
from ufl import nabla_div
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import interpolate


# Function that takes in the displacement image and mesh size 
#and returns an image of the simulated stress 
def force_inference(displacement_image,N):
    # Open the displacement image 
    im=Image.open(displacement_image)
    # Store the displacement per pixel in an arra
    imarray=np.array(im)
    # Blur the image
    #blur=cv2.GaussianBlur(imarray,(5,5),0)
    
    #Open the force image genertated by COMSOL 
    # Needed only if you want to produce a subplot comparing the COMSOL and fenics results
    #im2=Image.open('NCB_Fig3a_Stress.tif')
    #stress=np.array(im2)

    Size=76.464
    # Create a mesh for the images 
    x=np.linspace(0,Size, num=472)  
    y=np.linspace(0,Size, num=472)
    y_rev=np.linspace(Size,0, num=472)
    X,Y = np.meshgrid(x,y)
    # Interpolate the displacement values 
    interpolator=interpolate.interp2d(x,y_rev,imarray, kind="cubic")

    #print(imarray)
    
    # Define tolerance for the BCs
    tol = 1E-14 
    
    # STEP 1: Calculate the stress by solving the BVP

    # Create mesh for the simulated stress
    mesh=BoxMesh(Point(0,0,0), Point(Size,Size,1),N,N,3)
    mesh2=RectangleMesh(Point(0,0), Point(Size,Size),N,N)
    # mesh size is smaller near x=y=0
    # mesh size is smaller near z=0 and mapped to a [-1;0] domain along z
    mesh.coordinates()[:, 2] = -8.5*(mesh.coordinates()[:, 2]**2)
    
    # Define Function Space 
    V = VectorFunctionSpace(mesh, 'P', 2)
    V2 = FunctionSpace(mesh, "P", 1 )
    
    #Define stress function that stores the vertical stress
    p = Function(V2, name="stress")

    # Define the Boundary Conditions
    class MyExpr(UserExpression):
        def eval(self, value, x):
            value[0] = 0
            value[1] = 0
            value[2] = -interpolator(x[0],x[1])
        def value_shape(self):
            return (3,)
    
    conditions = MyExpr()
    Top = CompiledSubDomain('on_boundary && near(x[2],0,tol)', tol=tol )
    boundary_bottom = CompiledSubDomain('on_boundary && near(x[2], -8.5 , tol) ', tol=tol)
    bc1 = DirichletBC(V, conditions, Top)
    bc2 = DirichletBC(V, Constant((0,0,0)), boundary_bottom)
    bcs=[bc1,bc2]

    # Define constants (obtained from experiment)
    E = Constant(350)
    nu = Constant(0.49)
    mu = E/2/(1+nu)
    lmbda = E*nu/(1+nu)/(1-2*nu)
    
    # Define the variational problem to be solved 
    def epsilon(u):
        # return 0.5*(nabla_grad(u) + nabla_grad(u).T)
        return sym(grad(u))
    def sigma(u):
        return lmbda*nabla_div(u)*Identity(3) + 2*mu*epsilon(u)

    # Define variational problem
    u = TrialFunction(V)
    d = u.geometric_dimension()  # space dimension
    v = TestFunction(V)
    f = Constant((0, 0, 0))
    T = Constant((0,0,0))
    a = inner(sigma(u), epsilon(v))*dx
    L = dot(f, v)*dx + dot(T, v)*ds

    # Compute solution
    u = Function(V, name="displacement")
    solve(a == L, u, bcs, solver_parameters={'linear_solver':'mumps'})
    #solve(a == L, u, bcs, solver_parameters={'linear_solver':'gmres'})

    #print()
    
    #Project the vertical component of stress into p 
    p.assign(-project(sigma(u)[2, 2], V2, solver_type='mumps'))
    p.set_allow_extrapolation(True)
    

    # STEP 2: Ploting the computed stress 
    
    # Create a mesh NOT NEEDED ANYMORE 
    x_plot=np.linspace(0,Size, num=472)
    y_plot=np.linspace(0,Size, num=472)
    X_pl, Y_pl = np.meshgrid(x_plot,y_plot)
 
    # z axis will be the computed stress ie function p
    z=np.empty(((472),(472)), dtype=float)
    for i in range(472):
        for j in range(472): 
            z[i,j]=p(x_plot[i],y_plot[j],0.) 

    np.save('output.npy', z)
    
    fig, ax0 = plt.subplots()
    im= ax0.pcolormesh(x_plot,y_plot,z,rasterized=True)#need to check correctness 
                                                        # of pixel locations
    ax0.set_title('The Computed Stress for N = {}'.format(N))
    fig.colorbar(im,ax=ax0)
    plt.savefig('Computed_Stress for N = {}.png'.format(N))
    
    file_results = XDMFFile("stress_calc_real.xdmf")
    file_results.parameters["flush_output"] = True
    file_results.parameters["functions_share_mesh"] = True
    file_results.write(u, 0.)
    file_results.write(p, 0.)
    
    return z

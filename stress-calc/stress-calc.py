from fenics import *
from ufl import nabla_div
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import interpolate

im=Image.open('NCB_Fig3_Displacement.tif')
imarray=np.array(im)

im2=Image.open('NCB_Fig3a_Stress.tif')
stress=np.array(im2)

Size=76.464
x=np.linspace(0,Size, num=472)
y=np.linspace(0,Size, num=472)
y_rev=np.linspace(Size,0, num=472)
X,Y = np.meshgrid(x,y)
interpolator=interpolate.interp2d(x,y_rev,imarray, kind="cubic")
""" def bound(x,y): 
    return float(f(x,y))
 """
print(imarray)
tol = 1E-14 


""" conditions = Expression((0,0,"bound(x[0],x[1])"), degree=3, tol=tol) """

#create mesh
N = 70
mesh=BoxMesh(Point(0,0,0), Point(Size,Size,1),N,N,5)
mesh2=RectangleMesh(Point(0,0), Point(Size,Size),N,N)
# mesh size is smaller near x=y=0
# mesh size is smaller near z=0 and mapped to a [-1;0] domain along z
mesh.coordinates()[:, 2] = -8.5*(mesh.coordinates()[:, 2]**2)

V = VectorFunctionSpace(mesh, 'P', 1)
V2 = FunctionSpace(mesh, "CG", 1)


p = Function(V2, name=" stress")
""" conditions=Function(V)
conditions.assign(-project((0,0,bound), V)) """

class MyExpr(UserExpression):
    def eval(self, value, x):
        value[0] = 0
        value[1] = 0
        value[2] = -interpolator(x[0],x[1])
    def value_shape(self):
        return (3,)
conditions = MyExpr()

""" conditions=project(u_bound, V) """
""" w=np.empty((472,472))
for i in range(472):
    for j in range(472): 
        w[i,j]=conditions.ufl_evaluate(x[i],y[j])[2] """

""" plt.figure()
plt.pcolormesh(X,Y,w)
plt.colorbar()
plt.savefig('conditions.png') """

Top = CompiledSubDomain('on_boundary && near(x[2],0,tol)', tol=tol )
boundary_bottom = CompiledSubDomain('on_boundary && near(x[2], -8.5 , tol) ', tol=tol)
bc1 = DirichletBC(V, conditions, Top)
bc2 = DirichletBC(V, Constant((0,0,0)), boundary_bottom)
bcs=[bc1,bc2]
# exterior facets MeshFunction
""" facets = MeshFunction("size_t", mesh, 2)
facets.set_all(0)
Top().mark(facets, 1)
ds = Measure('ds', subdomain_data=facets) """



E = Constant(0.350)
nu = Constant(0.49)
mu = E/2/(1+nu)
lmbda = E*nu/(1+nu)/(1-2*nu)

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
solve(a == L, u, bcs, solver_parameters={'linear_solver':'lu'})

print()
p.assign(-project(sigma(u)[2, 2], V2))
""" p.set_allow_extrapolation(True) """

x_plot=np.linspace(0,Size, num=N+1)
y_plot=np.linspace(0,Size, num=N+1)
X_pl, Y_pl = np.meshgrid(x_plot,y_plot)

z=np.empty(((N+1),(N+1)), dtype=float)
for i in range(N+1):
    for j in range(N+1): 
        z[i,j]=p(x_plot[i],y_plot[j],0.) 

np.save('output_real.npy', z)
"""
plt.figure()
plt.pcolormesh(x_plot,y_plot,z)
plt.colorbar()
plt.savefig('stressmap_350.png') """
print(z)

fig, (ax0, ax1) = plt.subplots(1,2)

im1= ax0.pcolormesh(x,y,imarray)
ax0.set_title('The Displacement field')
fig.colorbar(im1, ax=ax0)
im2= ax1.pcolormesh(x_plot,y_plot,z)
ax1.set_title('The computed Stress')
fig.colorbar(im2, ax=ax1)
ax0.set_aspect('equal', adjustable='box')
ax1.set_aspect('equal', adjustable='box')
plt.savefig('output_real.png')

file_results = XDMFFile("stress_calc_real.xdmf")
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True
file_results.write(u, 0.)
file_results.write(p, 0.)
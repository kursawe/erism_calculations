from fenics import *
from ufl import nabla_div
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import interpolate
import pandas as pd

var = pd.read_excel("experiment.xlsx")
print(var)
x = np.array(var['X values'])-50.
z = np.array(var['Y values'])
print(x)
print(z)

function1D=interpolate.interp1d(x,z,kind='cubic')


tol = 1E-14 


""" conditions = Expression((0,0,"bound(x[0],x[1])"), degree=3, tol=tol) """

#create mesh
Size=150
N = 100
mesh=BoxMesh(Point(0,0,0), Point(Size,Size,1),N,N,30)
mesh2=RectangleMesh(Point(0,0), Point(Size,Size),N,N)
# mesh size is smaller near x=y=0
# mesh size is smaller near z=0 and mapped to a [-1;0] domain along z
mesh.coordinates()[:, 2] = -8.5*(mesh.coordinates()[:, 2]**2)

V = VectorFunctionSpace(mesh, 'P', 1)
V2 = FunctionSpace(mesh, "CG", 1)


p = Function(V2, name=" stress")
def function2D(x,y):
    if sqrt((x-75)**2+(y-75)**2)< 73.0: 
        return function1D(sqrt((x-75)**2+(y-75)**2))
    else:
        return 0.    
#Expression('(sqrt((x[0]-75)**2+(x[1]-75)**2))<73. ? function1D(sqrt((x[0]-75)**2+(x[1]-75)**2)):0', degree=2)
class MyExpr(UserExpression):
    def eval(self, value, x):
        value[0] = 0
        value[1] = 0
        value[2] = function2D(x[0],x[1])
    def value_shape(self):
        return (3,)
conditions = MyExpr()


Top = CompiledSubDomain('on_boundary && near(x[2],0,tol)', tol=tol )
Top2 = CompiledSubDomain('on_boundary && near(x[2],0,tol) && (pow(x[0]-75,2)+pow(x[1]-75,2))<pow(18.,2)', tol=tol )
boundary_bottom = CompiledSubDomain('on_boundary && near(x[2], -8.5 , tol) ', tol=tol)
bc1 = DirichletBC(V, conditions, Top)
bc2 = DirichletBC(V, Constant((0,0,0)), boundary_bottom)
bcs=[bc1,bc2]
# exterior facets MeshFunction

facets = MeshFunction("size_t", mesh, 2)
facets.set_all(0)
Top.mark(facets, 1)
Top2.mark(facets,2)
ds = Measure('ds', subdomain_data=facets)


E = Constant(300)
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
solve(a == L, u, bcs, solver_parameters={'linear_solver':'mumps'})

print()
p.assign(-project(sigma(u)[2, 2], V=V2, solver_type="mumps"))
p.set_allow_extrapolation(True)

print(assemble(p*ds(1)))
print(assemble(p*ds(2)))

x_plot=np.linspace(0,Size, num=N+1)
y_plot=np.linspace(0,Size, num=N+1)
X_pl, Y_pl = np.meshgrid(x_plot,y_plot)

z=np.empty(((N+1),(N+1)), dtype=float)
for i in range(N+1):
    for j in range(N+1): 
        z[i,j]=p(x_plot[i],y_plot[j],0.) 

np.save('output_afm/output_afm.npy', z)
print(z)

file_results = XDMFFile("stress_calc_afm.xdmf")
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True
file_results.write(u, 0.)
file_results.write(p, 0.)
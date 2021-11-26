from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

N = 50
""" mesh = UnitCubeMesh.create(N, N, N//2, CellType.Type.hexahedron) """
mesh=BoxMesh(Point(0,0,0), Point(1,1,1),N,N,20)
# mesh size is smaller near x=y=0
mesh.coordinates()[:, :2] = 100*mesh.coordinates()[:, :2]
# mesh size is smaller near z=0 and mapped to a [-1;0] domain along z
mesh.coordinates()[:, 2] = -8*(mesh.coordinates()[:, 2]**2)

""" L=1.0; W= 1.0; H=0.1 
mesh = BoxMesh(Point(0, 0, 0), Point(L, W, H), 50, 50, 20 ) """

# The top surface is defined as a ``SubDomain`` and corresponding exterior facets are marked as 1. Functions for imposition of Dirichlet BC on other boundaries are also defined.
# In[16]:
tol=1E-14 

class Top(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[2], 0,tol) and on_boundary
def symmetry_x(x, on_boundary):
        return near(x[0], 0,tol) and on_boundary
def symmetry_y(x, on_boundary):
        return near(x[1], 0,tol) and on_boundary
def bottom(x, on_boundary):
        return near(x[2], -8,tol) and on_boundary
    
# exterior facets MeshFunction
facets = MeshFunction("size_t", mesh, 2)
facets.set_all(0)
Top().mark(facets, 1)
ds = Measure('ds', subdomain_data=facets)


# The obstacle shape $h(x,y)$ is now defined and a small indentation depth $d$ is considered. Standard function spaces are then defined for the problem formulation and output field exports. In particular, gap and pressure will be saved later. Finally, the ``DirichletBC`` are defined.

# In[17]:


R = 17
d = 0.4
obstacle = Expression("-d +(pow(x[0]-x0,2)+pow(x[1]-y0, 2))/2/R", x0=50, y0=50, d=d, R=R, degree=2)

V = VectorFunctionSpace(mesh, "CG", 1)
V2 = FunctionSpace(mesh, "CG", 1)
V0 = FunctionSpace(mesh, "DG", 0)

u = Function(V, name="Displacement")
du = TrialFunction(V)
u_ = TestFunction(V)
gap = Function(V2, name="Gap")
p = Function(V0, name="Contact pressure")

bc =[DirichletBC(V, Constant((0., 0., 0.)), bottom),
     DirichletBC(V.sub(0), Constant(0.), symmetry_x),
     DirichletBC(V.sub(1), Constant(0.), symmetry_y)]

# The elastic constants and functions for formulating the weak form are defined. A penalization stiffness ``pen`` is introduced and the penalized weak form
# 
# $$\text{Find } u \text{ such that } \int_{\Omega} \sigma(u):\varepsilon(v)d\Omega + k_{pen}\int_{\Gamma} \langle u-h\rangle_+ v dS = 0$$
# 
# is defined. The corresponding Jacobian ``J`` is also computed using automatic differentiation.

# In[18]:


E = Constant(350)
nu = Constant(0.49)
mu = E/2/(1+nu)
lmbda = E*nu/(1+nu)/(1-2*nu)
def eps(v):
    return sym(grad(v))
def sigma(v):
    return lmbda*tr(eps(v))*Identity(3) + 2.0*mu*eps(v)
def ppos(x):
    return (x+abs(x))/2.

pen = Constant(1e4)
form = inner(sigma(u), eps(u_))*dx + pen*dot(u_[2], ppos(u[2]-obstacle))*ds(1)
J = derivative(form, u, du)


# A non-linear solver is now defined for solving the previous problem. Due to the 3D nature of the problem with a significant number of elements, an iterative Conjugate-Gradient solver is chosen for the linear solver inside the Newton iterations. Note that choosing a large penalization parameter deteriorates the problem conditioning so that solving time will drastically increase and can eventually fail. 

# In[19]:


problem = NonlinearVariationalProblem(form, u, bc, J=J)
solver = NonlinearVariationalSolver(problem)
solver.parameters["newton_solver"]["linear_solver"] = "cg"
solver.parameters["newton_solver"]["preconditioner"] = "icc"

solver.solve()


var = pd.read_excel("experiment.xlsx")
print(var)
x = list(var['X values'])
z = list(var['Y values'])

v=np.arange(50,100,0.1)
y=np.array([u(x,50,0.)[2] for x in v])
""" z=np.array([u(50,y,0.)[2] for y in v]) """
""" fig = plt.figure()
ax = fig.subplots(1,2) """
""" ax[0].plot(v,y) """
plt.plot(v,y)
plt.plot(x,z)
plt.savefig('twoinone2.png')

# As post-processing, the gap is computed (here on the whole mesh) as well as the pressure. The maximal pressure and the total force resultant (note the factor 4 due to the symmetry of the problem) are compared with respect to the analytical solution. Finally, XDMF output is performed.

# In[20]:


p.assign(-project(sigma(u)[2, 2], V0))

#Hertz-Model
a = sqrt(R*d)
#totale Kraft
F = 4/3.*(float(E)/(1-float(nu)**2))*a*d
#maximale Kraft
p0 = 3*F/(2*pi*(a**2))

#corrected Hertz Model
L=8
chi=a/L
F_cor=4/3.*(float(E)/(1-float(nu)**2))*a*d*(1+1.133*chi+ 1.283*(chi**2)+0.769*(chi**3)+0.0975*(chi**4))
p0_cor=3*F_cor/(2*pi*(a**2))

#totale Kraft FEM
print(assemble(p*ds(1)))
#maximale und totale Kraft aus dem corrected Hertz Model
print(p0_cor,F_cor)

""" print("Maximum pressure FE: {0:8.5f} Hertz: {1:8.5f} ".format(max(np.abs(p.vector().get_local())), p0))
print("Applied force    FE: {0:8.5f} Hertz: {1:8.5f} ".format(assemble(p*ds(1)), F))
print("Maximum pressure corrected Hertz: {} ".format(p0_cor))
print("Applied force corrected Hertz: {} ".format(F_cor))



file_results = XDMFFile("contact_penalty_results_BigE.xdmf")
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True
file_results.write(u, 0.)
file_results.write(p, 0.)

 """

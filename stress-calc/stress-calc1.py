
from fenics import *
from ufl import nabla_div
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import interpolate


im=Image.open('NCB_Fig3_Displacement.tif')
#Scaling due to units (1000 nm = 1 µm)
displ=0.001*np.array(im)
#blur=cv2.GaussianBlur(displ,(5,5),0)

im3=Image.open('NCB_Fig3a_Stress.tif')
stress=np.array(im3)

Size=76.464
x=np.linspace(0,Size, num=472)
y=np.linspace(0,Size, num=472)
y_rev=np.linspace(Size,0, num=472)
X,Y = np.meshgrid(x,y)
#interpolate the Displacement Image to a continous function
interpolator=interpolate.interp2d(x,y_rev,displ, kind="cubic")
tol = 1E-14 


#create mesh
N = 100
mesh=BoxMesh(Point(0,0,0), Point(Size,Size,1),N,N,2)

print(mesh.coordinates()[:, 2])
number=int((mesh.coordinates()[:, 2].shape[0])/3)

print(number)
# mesh size is smaller near z=0 and mapped to a [-8.5;0] domain along z
new_coord=-1.67*np.ones(number)
mesh.coordinates()[:, 2] = -8.5*(mesh.coordinates()[:, 2])
mesh.coordinates()[:, 2][number:((number*2))]=new_coord

""" print(number)
print(mesh.coordinates()[:, 2].shape)
print(mesh.coordinates()[:, 2][number-1])
print(mesh.coordinates()[:, 2][number:((number*2))])
print(mesh.coordinates()[:, 2][(number*2)]) """

#Define Functionspaces
V = VectorFunctionSpace(mesh, 'P', 2)
V2 = FunctionSpace(mesh, "CG", 2)

p = Function(V2, name=" stress")

#Set up the Displacement as Boundary conditions
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

# exterior facets MeshFunction
""" facets = MeshFunction("size_t", mesh, 2)
facets.set_all(0)
Top().mark(facets, 1)
ds = Measure('ds', subdomain_data=facets) """



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
solve(a == L, u, bcs, solver_parameters={'linear_solver':'gmres', 'preconditioner' :'ilu'})

#alternative Code for the solving of the system with adaptive mesh (doesn't work!!)
""" M=sigma(u)[2, 2]*dx()
u = Function(V, name="displacement")
tol2=1.e-5
problem = LinearVariationalProblem(a, L, u, bcs)
solver = AdaptiveLinearVariationalSolver(problem, M)
solver.parameters["error_control"]["dual_variational_solver"]["linear_solver"] = "cg"
solver.solve(tol2) """
#second alternative code for the solving of the system 
""" A = assemble(a, keep_diagonal= True)
L = assemble(l)

# Apply boundary conditions
[bc.apply(A) for bc in bcs]
[bc.apply(L) for bc in bcs]

# solve for the solution
A.ident_zeros()
u = Function(V, name="displacement")
# solve(A, w.vector(), L)
solver_lin = LUSolver("mumps")
solver_lin.solve(A, u.vector(), L) """


#Calculation of the vertical component of stress Tensor sigma
p.assign(-project(sigma(u)[2, 2], V=V2, solver_type="mumps"))
p.set_allow_extrapolation(True)

# Evalute stress on the given Mesh  
x_plot=np.linspace(0,Size, num=N+1)
y_plot=np.linspace(0,Size, num=N+1)
X_pl, Y_pl = np.meshgrid(x_plot,y_plot)

z=np.empty(((N+1),(N+1)), dtype=float)
for i in range(N+1):
    for j in range(N+1): 
        z[i,j]=p(x_plot[i],y_plot[j],0.) 

#save the array
np.save('output_quad.npy', z)
print(z)

#Plot Comparison of Displacement Map to Stress Map
fig, (ax0, ax1) = plt.subplots(1,2)

im1= ax0.pcolormesh(x,y,displ)
ax0.set_title('The Displacement field')
fig.colorbar(im1, ax=ax0)
im2= ax1.pcolormesh(x_plot,y_plot,z)
ax1.set_title('The computed Stress')
fig.colorbar(im2, ax=ax1)
ax0.set_aspect('equal', adjustable='box')
ax1.set_aspect('equal', adjustable='box')
plt.savefig('output_quad.png')


#Possible XDMF output for Paraview
""" file_results = XDMFFile("stress_calc_real.xdmf")
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True
file_results.write(u, 0.)
file_results.write(p, 0.) """
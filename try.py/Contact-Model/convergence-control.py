from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
    
R = 17
d = 0.4
E = Constant(450)
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

def iteration(N): 
    mesh=BoxMesh(Point(0,0,0), Point(1,1,1),N,N,N//20)
# mesh size is smaller near x=y=0
    mesh.coordinates()[:, :2] = 100*mesh.coordinates()[:, :2]
# mesh size is smaller near z=0 and mapped to a [-1;0] domain along z
    mesh.coordinates()[:, 2] = -8*(mesh.coordinates()[:, 2]**2)

    # exterior facets MeshFunction
    facets = MeshFunction("size_t", mesh, 2)
    facets.set_all(0)
    Top().mark(facets, 1)
    ds = Measure('ds', subdomain_data=facets)

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

    form = inner(sigma(u), eps(u_))*dx + pen*dot(u_[2], ppos(u[2]-obstacle))*ds(1)
    J = derivative(form, u, du)




    problem = NonlinearVariationalProblem(form, u, bc, J=J)
    solver = NonlinearVariationalSolver(problem)
    solver.parameters["newton_solver"]["linear_solver"] = "cg"
    solver.parameters["newton_solver"]["preconditioner"] = "icc"

    solver.solve()

    p.assign(-project(sigma(u)[2, 2], V0))
    
    a = sqrt(R*d)
    F = 4/3.*(float(E)/(1-float(nu)**2))*a*d
    p0 = 3*F/(2*pi*(a**2))


    return np.array([max(np.abs(p.vector().get_local())), assemble(p*ds(1))])

sizes=np.arange(270,350,20)

a = sqrt(R*d)
F = 4/3.*(float(E)/(1-float(nu)**2))*a*d
p0 = 3*F/(2*pi*(a**2))

L=8
chi=a/L
F_cor=4/3.*(float(E)/(1-float(nu)**2))*a*d*(1+1.133*chi+ 1.283*(chi**2)+0.769*(chi**3)+0.0975*(chi**4))
p0_cor=3*F_cor/(2*pi*(a**2))

print(p0_cor,F_cor)

list=[]
for m in sizes: 
    value=iteration(m)
    print(value)
    list.append(value)

with open('iteration2.txt', 'w') as f:
    for item in list:
        f.write("%s\n" % item)

from pystencils.session import *
import shapefile
import pandas as pd
import fiona
import ezdxf
import numpy


def init(value=0.5, noise=0.02):
    for b in dh.iterate():
        b['c'].fill(value)
        np.add(b['c'], noise*np.random.rand(*b['c'].shape), out=b['c'])


def timeloop(steps=120000):
    c_sync = dh.synchronization_function(['c'])
    μ_sync = dh.synchronization_function(['mu'])
    for t in range(steps):
        c_sync()
        dh.run_kernel(μ_kernel)
        μ_sync()
        dh.run_kernel(c_kernel)
    return dh.gather_array('c')


dh = ps.create_data_handling(domain_size=(256, 256), periodicity=True)
μ_field = dh.add_array('mu', latex_name='μ')
c_field = dh.add_array('c')

κ, A = sp.symbols("κ A")

c = c_field.center
μ = μ_field.center


def f(c):
    return A * c**2 * (1-c)**2

bulk_free_energy_density = f(c)
grad_sq = sum(ps.fd.diff(c, i)**2 for i in range(dh.dim))
interfacial_free_energy_density = κ/2 * grad_sq

free_energy_density = bulk_free_energy_density + interfacial_free_energy_density
plt.figure(figsize=(7,4))
plt.sympy_function(bulk_free_energy_density.subs(A, 0.8), (-0.2, 1.2))
plt.xlabel("c")
plt.title("Bulk free energy");

ps.fd.functional_derivative(free_energy_density, c)

discretize = ps.fd.Discretization2ndOrder(dx=1, dt=0.01)

μ_update_eq = ps.fd.functional_derivative(free_energy_density, c)
μ_update_eq = ps.fd.expand_diff_linear(μ_update_eq, constants=[κ])  # pull constant κ in front of the derivatives
μ_update_eq_discretized = discretize(μ_update_eq)
print(μ_update_eq_discretized)

μ_kernel = ps.create_kernel([ps.Assignment(μ_field.center, μ_update_eq_discretized.subs(A, 1.5).subs(κ, 2))]).compile()
M = sp.Symbol("M")
cahn_hilliard = ps.fd.transient(c) - ps.fd.diffusion(μ, M)
print(cahn_hilliard)
c_update = discretize(cahn_hilliard)
c_kernel = ps.create_kernel([ps.Assignment(c_field.center,c_update.subs(M, 1))]).compile()

init()
if 'is_test_run' in globals():
    timeloop(10)
    result = None
else:
    field = timeloop()
ident = "A_1-5_k2_1-2MP-s"
f_min, f_max = np.min(field), np.max(field)
field = (field - f_min) / (f_max - f_min)
x = np.linspace(0,255,256)
[X,Y] = np.meshgrid(x,x)
cs = plt.contour(field,levels=[0.5])
plt.savefig(ident+'contour_plot.png')
a = cs.collections[0].get_paths()
f = open("myfile.txt", 'w')


for i,cnt in enumerate(a):
    points = cnt.vertices.tolist()
    for point in points:
        f.write(str(point)+"#")
    f.write("\n")
f.close()
# msp.add_lwpolyline(points)}}
#doc.saveas("Hatch_test.dxf")


#plt.show()

#ani = ps.plot.scalar_field_animation(timeloop, rescale=True, frames=600)

#ani.save('animation_A15_k2.avi')

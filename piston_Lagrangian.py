#!/usr/bin/env python
# encoding: utf-8
r"""
Piston problem in Lagrangian coordinates
===================================

Solve the one-dimensional Euler equations for inviscid, compressible flow
writen in Lagrangian coordinates:

.. math::
\frac{D}{D t}\left(\begin{array}{l}
V \\
v \\
p
\end{array}\right)
+
\left(\begin{array}{ccc}
0 & -1 & 0 \\
0 & 0 & 1 \\
0 & d^2 & 0
\end{array}\right) \frac{\partial}{\partial \xi}\left(\begin{array}{l}
V \\
v \\
p
\end{array}\right)
=0

The fluid is an ideal gas, with pressure given by :math:`p=\rho (\gamma-1)e - gamma p_{\inf}`
where e is internal energy and p+{\inf} represents the cohesion effects in liquid and solid states.

Here the simplification ()_t :=D/Dt is assumed, where the latter denotes a material derivative.
xi is a lagrangian coordinate as described in Duyen Phan Thi My's PhD thesis.

The objective of this script is to solve the piston problem in Lagrangian coordinates,
where the left boundary condition is prescribed by a moving piston.

"""
from __future__ import absolute_import
from __future__ import print_function
from clawpack import riemann
#from clawpack.riemann.euler_with_efix_1D_constants import *
import euler_HLL_1D

gamma = 1.4 # Ratio of specific heats

def setup(use_petsc=False,outdir='./_output',solver_type='classic',kernel_language='Fortran',
          mx=800, tfinal= 0.05, nout=10):#tfluct_solver=True):

    if use_petsc:
        import clawpack.petclaw as pyclaw
    else:
        from clawpack import pyclaw

    if kernel_language =='Python':
        rs = riemann.euler_1D_py.euler_roe_1D
    elif kernel_language =='Fortran':
        rs = euler_HLL_1D#riemann.euler_with_efix_1D

    if solver_type=='sharpclaw':
        solver = pyclaw.SharpClawSolver1D(rs)
        solver.time_integrator = 'SSP33'
        solver.cfl_max = 0.65
        solver.cfl_desired = 0.6
        
    elif solver_type=='classic':
        solver = pyclaw.ClawSolver1D(rs)
        solver.limiters = 4
    
    #Number of equations and waves
    num_eqn = 3
    num_waves = 2
    
    #Spatial domain
    xlower = 0.0
    xupper = 1.0

    solver.kernel_language = kernel_language

    solver.bc_lower[0]=pyclaw.BC.wall
    solver.bc_upper[0]=pyclaw.BC.wall

    solver.num_waves = num_waves
    solver.num_eqn = num_eqn

    #mx = 800;
    x = pyclaw.Dimension(xlower,xupper,mx,name='x')
    domain = pyclaw.Domain([x])
    state = pyclaw.State(domain,num_eqn)

    state.problem_data['gamma'] = gamma
    if kernel_language =='Python':
        state.problem_data['efix'] = False

    x = state.grid.x.centers
    

    #Set initial conditions
    init(state,x)
    claw = pyclaw.Controller()
    claw.tfinal = tfinal
    claw.solution = pyclaw.Solution(state,domain)
    claw.solver = solver
    claw.num_output_times = nout
    claw.outdir = outdir
    claw.setplot = setplot
    claw.keep_copy = True

    return claw

def init(state,x):
    density, momentum, energy = 0, 1, 2

    state.q[density ,:] = 1.
    state.q[momentum,:] = 0.
    state.q[energy  ,:] = ( (x<0.1)*1.e3 + (0.1<=x)*(x<0.9)*1.e-2 + (0.9<=x)*1.e2 ) / (gamma - 1.)

#--------------------------
def setplot(plotdata):
#--------------------------
    """ 
    Specify what is to be plotted at each frame.
    Input:  plotdata, an instance of visclaw.data.ClawPlotData.
    Output: a modified version of plotdata.
    """ 
    plotdata.clearfigures()  # clear any old figures,axes,items data

    plotfigure = plotdata.new_plotfigure(name='', figno=0)

    plotaxes = plotfigure.new_plotaxes()
    plotaxes.axescmd = 'subplot(211)'
    plotaxes.title = 'V'

    plotitem = plotaxes.new_plotitem(plot_type='1d')
    plotitem.plot_var = 0#density
    plotitem.kwargs = {'linewidth':3}
    
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.axescmd = 'subplot(212)'
    plotaxes.title = 'u'

    plotitem = plotaxes.new_plotitem(plot_type='1d')
    plotitem.plot_var = 2#energy
    plotitem.kwargs = {'linewidth':3}
    
    return plotdata

if __name__=="__main__":
    from clawpack.pyclaw.util import run_app_from_main
    output = run_app_from_main(setup,setplot)

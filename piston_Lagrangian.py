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
import numpy as np
#from clawpack.riemann.euler_with_efix_1D_constants import *
import euler_HLL_1D

gamma = 1.4 # Ratio of specific heats

def setup(use_petsc=False,outdir='./_output',solver_type='classic',kernel_language='Fortran',
          mx=1000, tfinal= 20*np.pi, nout=10):#tfluct_solver=True):

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
    xupper = 120.0

    solver.kernel_language = kernel_language

    solver.bc_lower[0] = pyclaw.BC.custom#pyclaw.BC.wall
    solver.user_bc_lower = piston_bc
    solver.bc_upper[0] = pyclaw.BC.extrap

    solver.num_waves = num_waves
    solver.num_eqn = num_eqn

    #mx = 800;
    x = pyclaw.Dimension(xlower,xupper,mx,name='x')
    domain = pyclaw.Domain([x])
    state = pyclaw.State(domain,num_eqn)

    state.problem_data['gamma'] = gamma

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

def init(state, x):
    gamma = state.problem_data['gamma']
    rho = 1.
    V = 1./rho
    p = 1./gamma
    u = 0.
    eps = V*p/(gamma-1)+0.5*u**2

    state.q[0 ,:] = V
    state.q[1,:] = u
    state.q[2,:] = eps

def piston_bc(state,dim,t,qbc,auxbc,num_ghost):
    """Initial pulse generated at left boundary by prescribed motion"""
    M=1.
    if dim.on_lower_boundary:
        qbc[0,:num_ghost] = qbc[0,num_ghost]
        qbc[1,:num_ghost] = qbc[1,num_ghost]
        qbc[2,:num_ghost] = qbc[2,num_ghost]
        t = state.t; gamma = state.problem_data['gamma']#t1=state.problem_data['t1']; 
        xi = state.grid.x.centers; 
        deltaxi = xi[1]-xi[0]
        [V1,u1,eps1] = state.q[:,0]
        p1 = (gamma-1.)*(eps1-0.5*u1**2)/V1
        p0 = p1 - M*np.cos(t*deltaxi)
        u0 = -2*M*np.sin(t) - u1
        deltap = (p1-p0)/(p1+p0)
        V0 = V1*(gamma +deltap)/(gamma-deltap)
        eps0 = V0*p0/(gamma-1)+0.5*u0**2
        #if abs(t0)<=1.: vwall = -a1*(1.+np.cos(t0*np.pi))
        #else: vwall=0.
        for ibc in range(num_ghost-1):
            #qbc[1,num_ghost-ibc-1] = -2*M*np.sin(t)-state.q[1,0]
            #qbc[1,num_ghost-ibc-1] = 2*vwall*state.aux[1,ibc] - qbc[1,num_ghost+ibc]
            qbc[0,num_ghost-ibc-1] = V0
            qbc[1,num_ghost-ibc-1] = u0
            qbc[2,num_ghost-ibc-1] = eps0
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
    plotitem.plot_var = 1#energy
    plotitem.kwargs = {'linewidth':3}
    
    return plotdata

if __name__=="__main__":
    from clawpack.pyclaw.util import run_app_from_main
    output = run_app_from_main(setup,setplot)

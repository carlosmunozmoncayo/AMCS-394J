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

The objective of this script is to solve the Euler equations in Lagrangian coordinates,
where the left boundary condition is prescribed by a moving piston.

"""
from __future__ import absolute_import
from __future__ import print_function
from clawpack import riemann
import numpy as np
#from clawpack.riemann.euler_with_efix_1D_constants import *
import euler_HLL_1D
import euler_burgers_HLL_1D
import euler_HLL_slowing_1D
import euler_HLL_slowing_damping_1D

gamma = 1.4 # Ratio of specific heats

def setup(use_petsc=False,outdir='./_output',solver_type='sharpclaw',
          kernel_language='Fortran',time_integrator='Heun',mx=10000, tfinal= 50,
          nout=10, xmax=400, M=0.5, CFL=0.5, limiting=1, order=2,
          start_slowing=1e+4, stop_slowing=1e+5, damping_rate=20, #For Smadar Karni's approach
          x0_switch_RS=1e+4, #For switching to Burger's simple wave solution
          euler_RS='euler', damping=False,
          absorbing_BC=False):#tfluct_solver=True):

    if use_petsc:
        import clawpack.petclaw as pyclaw
    else:
        from clawpack import pyclaw

    if kernel_language =='Python':
        rs = riemann.euler_1D_py.euler_roe_1D
    elif kernel_language =='Fortran':
        #rs = euler_HLL_1D
        if euler_RS == 'euler_burger':
            rs = euler_burgers_HLL_1D
        elif euler_RS == 'euler_slowing':
            rs = euler_HLL_slowing_1D
        elif euler_RS == 'euler_slowing_damping':
            rs = euler_HLL_slowing_damping_1D
        elif euler_RS == 'euler':
            rs = euler_HLL_1D

    if solver_type=='sharpclaw':
        solver = pyclaw.SharpClawSolver1D(rs)
        solver.lim_type = limiting
        #solver.time_integrator = 'SSPLMM32'# 'SSP33'
        solver.cfl_max = 0.9
        if time_integrator == 'Heun':
            a = np.array([[0.,0.],[1.,0.]])
            b = np.array([0.5,0.5])
            c = np.array([0.,1.])
            
            #a = np.array([[0.,0.],[2./3.,0.]])
            #b = np.array([1./4.,3./4.])
            #c = np.array([0.,2./3.])

            solver.time_integrator = 'RK'
            solver.a, solver.b, solver.c = a, b, c
        solver.cfl_desired = CFL
        solver.limiters = 1# minmod limiter pyclaw.limiters.tvd.minmod
        solver.max_steps = 5000000
        if damping:
            solver.dq_src = sharpclaw_source_step_damping_scalar
        
    elif solver_type=='classic':
        solver = pyclaw.ClawSolver1D(rs)
        solver.order =2
        solver.limiters = pyclaw.limiters.tvd.minmod
        solver.cfl_max = 0.9
        solver.cfl_desired =CFL
        solver.max_steps = 5000000
    
    #Number of equations and waves
    num_eqn = 3
    num_waves = 2
    
    #Spatial domain
    xlower = 0.0
    xupper = xmax

    solver.kernel_language = kernel_language
    
    #solver.before_step = b4step

    x = pyclaw.Dimension(xlower,xupper,mx,name='x')
    domain = pyclaw.Domain([x])
    state = pyclaw.State(domain,num_eqn,2)
    state.problem_data['gamma'] = gamma
    #state.problem_data['switch'] = x_switch_RS
    
    #################################################
    #Defining before step functions
    #################################################
    def b4step_burgers(solver,state):
        #Copying density and pressure from last cell of Euler equations
        x = state.aux[0]
        i=np.max([j for j in range(len(x)) if x[j]<=x_switch_RS])
        V0, u0, eps0 = state.q[0,i], state.q[1,i], state.q[2,i]
        p0 = (gamma-1)*(eps0-0.5*u0**2)/V0

        u = state.q[1,i:]

        state.q[0,i:] = V0
        state.q[2,i:] = p0*V0/(gamma-1.)+0.5*u**2

    def b4step_absorbing_BC(solver,state):
        # Absorbing (or generating) BC as in Escalante's paper
        #Not working yet (not even implemented yet lol)
        x = state.grid.x.centers
        #i=np.max([j for j in range(len(x)) if x[j]<=x_switch_RS])
        V, u, eps = state.q[0,:], state.q[1,:], state.q[2,:]
        V_star = np.ones(len(V))
        u_star = np.zeros(len(u))
        eps_star = (1./(gamma*(gamma-1)))*np.ones(len(eps))

        c = affine_filter(x=x,a=start_slowing,b=stop_slowing)
        
        state.q[0,:] = c*V+(1-c)*V_star
        state.q[1,:] = c*u+(1-c)*u_star
        state.q[2,:] = c*eps+(1-c)*eps_star


    #################################################

    if absorbing_BC:
        solver.before_step = b4step_absorbing_BC

    #if euler_burgers:
    #    solver.before_step = b4step

    ###########################################
    #Defining custom BC. 
    #It is convenient to define it here to pass M to setup
    ###########################################
    def piston_bc(state,dim,t,qbc,auxbc,num_ghost):
    #"""Initial pulse generated at left boundary by prescribed motion"""
        if dim.on_lower_boundary:
            qbc[0,:num_ghost] = qbc[0,num_ghost]
            qbc[1,:num_ghost] = qbc[1,num_ghost]
            qbc[2,:num_ghost] = qbc[2,num_ghost]
            t = state.t; gamma = state.problem_data['gamma']
            xi = state.grid.x.centers; 
            deltaxi = xi[1]-xi[0]
            [V1,u1,eps1] = state.q[:,0]
            p1 = (gamma-1.)*(eps1-0.5*u1**2)/V1
            p0 = p1 - M*np.cos(t)*deltaxi
            u0 = -2*M*np.sin(t) - u1
            deltap = (p1-p0)/(p1+p0)
            V0 = V1*(gamma +deltap)/(gamma-deltap)
            eps0 = V0*p0/(gamma-1)+0.5*u0**2
            
            for ibc in range(num_ghost-1):
                qbc[0,num_ghost-ibc-1] = V0
                qbc[1,num_ghost-ibc-1] = u0
                qbc[2,num_ghost-ibc-1] = eps0
    ###########################################
    ###########################################

    solver.bc_lower[0] = pyclaw.BC.custom
    solver.user_bc_lower = piston_bc
    solver.bc_upper[0] = pyclaw.BC.extrap
    solver.aux_bc_lower[0] = pyclaw.BC.extrap
    solver.aux_bc_upper[0] = pyclaw.BC.extrap

    solver.num_waves = num_waves
    solver.num_eqn = num_eqn

    x = state.grid.x.centers

    
    
    
    if euler_RS == 'euler_burger':
        #Switching to burger's equation after certain point
        #The cells where x_switch_RS is 1 will use the modified RS.
        x_switch_RS = np.where(x<x0_switch_RS,1,0).astype(dtype=int)
    
    elif euler_RS == 'euler_slowing':
        #Slowing down waves gradually (with an affine function)
        x_switch_RS = affine_filter(x=x,a=start_slowing,b=stop_slowing)
    else:
        x_switch_RS =np.zeros(len(x)).astype(dtype=int)

    #if damping:
    #    x_switch_RS = np.where(x<x0_switch_RS,1,0).astype(dtype=int)

   
    #Damping right going waves (passed to the RS but not used if Smadar Karni's mthod is not used)
    #Just used in the slowing_damping RS for instance
    #exp_decay = np.exp(damping_rate*np.minimum((start_slowing-x)/(stop_slowing-start_slowing),0.))
    exp_decay = damping_rate+0.*x#damping_rate*(1.-affine_filter(x=x,a=start_slowing,b=stop_slowing))


    #Set initial conditions
    init(state,x_switch_RS,exp_decay)
    claw = pyclaw.Controller()
    claw.tfinal = tfinal
    claw.solution = pyclaw.Solution(state,domain)
    claw.solver = solver
    claw.num_output_times = nout
    claw.outdir = outdir
    claw.setplot = setplot
    claw.keep_copy = True

    return claw

def init(state, x_switch_RS, exp_decay):
    gamma = state.problem_data['gamma']
    #x_switch_RS = state.problem_data['switch']
    rho = 1.
    V = 1./rho
    p = 1./gamma
    u = 0.
    eps = V*p/(gamma-1)+0.5*u**2

    state.q[0 ,:] = V
    state.q[1,:] = u
    state.q[2,:] = eps

    #state.aux[0,:] = x
    state.aux[0,:] = x_switch_RS
    state.aux[1,:] = exp_decay

def affine_filter(x,a,b):
        #Defining relaxation zone
        #Generates an affine function equal to 1 for x<=a and equal to 0 for x>b
        Lrelax = b-a
        di = x-b #distance of points from relaxation zone's origin 
        mi = np.abs(di)/Lrelax #slope of filter in relaxation zone
        return (mi*(mi<=1)+(mi>1))*(di<=0)

def sharpclaw_source_step_damping(solver,state,dt):
    #Time integration of the system W_t+ D W = 0
    #Intended to be used in an operator (Godunov) splitting context
    #where D is given by eq. (29) of Karni's (1996)
    q = state.q
    xc = state.grid.x.centers

    #Get parameters
    gamma = state.problem_data['gamma']
    damping_flag = state.aux[0]
    exp_decay = state.aux[1]

    #Get variables
    V = q[0,:]
    u = q[1,:]
    eps = q[2,:]

    #Defining some variables
    p = (gamma-1.)*(eps-0.5*u**2)/V
    pe = (gamma-1.)/V
    pv = -p/V
    C = np.sqrt(p*pe-pv)
    
    R = np.array([[1.+0*C, 1.+0*C, 1.+0*C],
             [C,  0*C, -C],
             [u*C, -pv/pe, -u*C-p]])
    R = R.T

    #R will be a list of the matrices R corresponding to each point
    Delta = np.array([[-0*C, 0*C, 0*C],
                  [0*C, 0*C, 0*C],
                  [0*C, 0*C, exp_decay*C]])
    Delta = Delta.T

    #The same behavior as R
    R_inv = np.array([[1./(2*V*pe+2.), (p*pe+C*u*pe-pv)/(2.*C*p*pe), -pe/(2*p*pe-2*pv)],
             [V*pe/(V*pe+1.),  -u*pe/(p*pe-pv), pe/(p*pe-pv)],
             [1./(2*pe*V+2.), (-p*pe+C*u*pe+pv)/(2*C*p*pe-2*C*pv), -pe/(2*p*pe-2*pv)]])

    R_inv = R_inv.T
    
    dq = np.array([-R[i]@Delta[i]@R_inv[i]@q.T[i] for i in range(len(q.T))])
    dq = dt*dq.T
    dq = np.where(damping_flag<0.5,dq,0.)#np.where(np.isclose(damping_flag,1.), 0., dq)
    dq[0,:] = 0.
    dq[2,:] = 0.

    return dq

def sharpclaw_source_step_damping_scalar(solver,state,dt):
    #Time integration of the system W_t+ D W = 0
    #Intended to be used in an operator (Godunov) splitting context
    #where D is given by eq. (29) of Karni's (1996)
    q = state.q
    xc = state.grid.x.centers

    #Get parameters
    gamma = state.problem_data['gamma']
    damping_flag = state.aux[0]
    damping_rate = state.aux[1]

    V = q[0,:]
    u = q[1,:]
    eps = q[2,:]

    dq = np.empty(q.shape)

    #This works 
    dq[0,:] = -dt*np.where(V>1, damping_rate/(np.log(V)*V), 0.)#0.#dt*damping_rate*q[0,:]*(1-damping_flag)
    dq[1,:] = -dt*damping_rate*q[1,:]*(1-damping_flag)#np.where(np.isclose(damping_flag,1), 0., dq)
    dq[2,:] = 0.#-dt*damping_rate*q[2,:]*(1-damping_flag)

    #This doesn't work
    #dq = -dt*damping_rate*q*(1-damping_flag)
    return dq


#--------------------------
def setplot(plotdata):
#--------------------------
    """ 
    Specify what is to be plotted at each frame.
    Input:  plotdata, an instance of visclaw.data.ClawPlotData.
    Output: a modified version of plotdata.
    """ 
    plotdata.clearfigures()  # clear any old figures,axes,items data
    xmin, xmax = 0., 400.
    plotfigure = plotdata.new_plotfigure(name='', figno=0)

    plotaxes = plotfigure.new_plotaxes()
    plotaxes.axescmd = 'subplot(211)'
    plotaxes.xlimits = [xmin,xmax]
    plotaxes.title = 'V'

    plotitem = plotaxes.new_plotitem(plot_type='1d')
    plotitem.plot_var = 0#density
    plotitem.kwargs = {'linewidth':1}
    
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.axescmd = 'subplot(212)'
    plotaxes.xlimits = [xmin,xmax]
    plotaxes.ylimits = [-1.5,1.5]
    plotaxes.title = 'u'

    plotitem = plotaxes.new_plotitem(plot_type='1d')
    plotitem.plot_var = 1#energy
    plotitem.kwargs = {'linewidth':1}
    
    return plotdata

if __name__=="__main__":
    from clawpack.pyclaw.util import run_app_from_main
    output = run_app_from_main(setup,setplot)

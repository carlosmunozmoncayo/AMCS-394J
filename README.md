Piston problem for the Euler equations in Lagrangian coordinates
===================================

### A repository for some code used in my final project for the course AMCS-394J at KAUST: Contemporary Topics in Numerical Analysis.

The goal of this project is to solve the one-dimensional Euler equations for inviscid, compressible flow
writen in Lagrangian coordinates:

<img width="377" alt="image" src="https://user-images.githubusercontent.com/29715468/201913249-3b70b7a9-e296-420f-a592-b131f7d5960a.png" class="center">

The fluid is an ideal gas, with pressure given by $p=\rho (\gamma-1)e - \gamma p_{\inf}$
where e is internal energy and $p+{\inf}$ represents the cohesion effects in liquid and solid states.

Here $D/Dt$ denotes a material derivative, while $\xi$ is a lagrangian coordinate as described in Duyen Phan Thi My's PhD thesis.

We consider a one-dimensional setting, where the conditions at the left boundary are determined by a moving piston and it is desired to impose appropiate
non-reflecting boundary conditions on the right boundary.

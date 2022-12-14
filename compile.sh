#!/bin/bash

python3 -m numpy.f2py -c rp1_euler_HLL.f90 -m euler_HLL_1D
python3 -m numpy.f2py -c rp1_euler_HLL_slowing.f90 -m euler_HLL_slowing_1D
python3 -m numpy.f2py -c rp1_euler_HLL_slowing_damping.f90 -m euler_HLL_slowing_damping_1D
python3 -m numpy.f2py -c rp1_euler_burgers_HLL.f90 -m euler_burgers_HLL_1D


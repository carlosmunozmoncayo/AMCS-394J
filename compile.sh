#!/bin/bash

python3 -m numpy.f2py -c rp1_euler_HLL.f90 -m euler_HLL_1D


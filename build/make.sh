#!/bin/bash
module use /soft/modulefiles
module load craype-accel-nvidia80
export MPICH_GPU_SUPPORT_ENABLED=1

make clean
make -j16 -f Makefile catalyst_build=yes

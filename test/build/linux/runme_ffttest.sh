#!/bin/sh
# $1 = N1
# $2 = N2
export CLAMP_NOTILECHECK=ON
#rm /tmp/kernel*.cpp
#rm /tmp/libFFTKernel*.so
./fft $1 $2

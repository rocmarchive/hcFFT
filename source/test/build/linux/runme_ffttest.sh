#!/bin/sh
# $1 = N1
# $2 = N2
export CLAMP_NOTILECHECK=ON
rm ../../../kernel*.cpp
rm ../../../libFFTKernel*.so
./fft $1 $2

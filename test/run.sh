#!/bin/sh
# $1 = N1
# $2 = N2
rm ../kernel*.cpp
rm ../libFFTKernel*.so
./fft $1 $2

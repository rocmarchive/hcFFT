# Script to remove test builds

#Remove local generated cmakes and makes
rm -rf CMake* Makefile cmake*

# Remove generated executable
rm fft

# Remove autogenerated kernels and shared objects
rm ../../../kernel*.cpp
rm ../../../libFFTKernel*.so
rm /tmp/libFFTKernel*.so

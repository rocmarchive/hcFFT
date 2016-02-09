******************
1.1. Introduction
******************
--------------------------------------------------------------------------------------------------------------------------------------------

The hcFFT library is an implementation of FFT (Fast Fourier Transform) targetting the AMD heterogenous hardware via HCC compiler runtime. The computational resources of underlying AMD heterogenous compute gets exposed and exploited through the HCC C++ frontend. Refer `here <https://bitbucket.org/multicoreware/hcc/wiki/Home>`_ for more details on HCC compiler.

To use the hcFFT API, the application must allocate the required input buffers in the GPU memory space, fill them with data, call the sequence of desired hcFFT functions, and then upload the results from the GPU memory space back to the host. The hcFFT API also provides helper functions for writing and retrieving data from the GPU.

The following list enumerates the current set of FFT sub-routines that are supported so far. 

* R2C  : Single Precision real to complex valued Fast Fourier Transform
* C2R  : Single Precision complex to real valued Fast Fourier Transform
* C2C  : Single Precision complex to complex valued Fast Fourier Transform
* D2Z  : Double Precision real to complex valued Fast Fourier Transform
* Z2D  : Double Precision complex to real valued Fast Fourier Transform
* Z2Z  : Double Precision complex to complex valued Fast Fourier Transform

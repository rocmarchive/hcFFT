#####
R2C
#####

| Single precision real to complex valued transform.
|
|
|       hcfftExecR2C() executes a single-precision real-to-complex, implicitly forward, hcFFT transform plan.
|  hcFFT uses as input data the GPU memory pointed to by the idata parameter. This function stores 
|  the nonredundant Fourier coefficients in the odata array. Pointers to idata and odata are both required to be aligned 
|  to hcfftComplex data type in single-precision transforms and hcfftDoubleComplex data type in double-precision transforms. 
|  It does an out-of-place transform.
|

Functions
^^^^^^^^^

Function Prototype:
---------------------

 .. note:: **Inputs and Outputs are HCC device pointers.**

`hcfftResult <HCFFT_TYPES.html>`_ **hcfftExecR2C** (hcfftHandle plan, hcfftReal *idata, hcfftComplex *odata)

Detailed Description
^^^^^^^^^^^^^^^^^^^^

Function Documentation
^^^^^^^^^^^^^^^^^^^^^^

::

             hcfftResult hcfftExecR2C(hcfftHandle plan, hcfftReal *idata, hcfftComplex *odata)

+------------+-----------------+--------------------------------------------------------------+
|  In/out    |  Parameters     | Description                                                  |
+============+=================+==============================================================+
|    [in]    |    plan         | hcfftHandle returned by hcfftCreate.                         |
+------------+-----------------+--------------------------------------------------------------+
|    [in]    |    idata        | Pointer to the single-precision real input data              |
|            |                 | (in GPU memory) to transform.                                |
+------------+-----------------+--------------------------------------------------------------+
|    [out]   |    odata        | Pointer to the single-precision complex output data          |
|            |                 | (in GPU memory).                                             |
+------------+-----------------+--------------------------------------------------------------+

|
| Returns,

==============================    ==============================================================
STATUS                            DESCRIPTION
==============================    ==============================================================
  HCFFT_SUCCESS 	           hcFFT successfully executed the FFT plan.
  HCFFT_INVALID_PLAN 	           The plan parameter is not a valid handle.
  HCFFT_INVALID_VALUE 	           At least one of the parameters idata and odata is not valid.
  HCFFT_INTERNAL_ERROR 	           An internal driver error was detected.
  HCFFT_EXEC_FAILED 	           hcFFT failed to execute the transform on the GPU.
  HCFFT_SETUP_FAILED 	           The hcFFT library failed to initialize.
==============================    ==============================================================

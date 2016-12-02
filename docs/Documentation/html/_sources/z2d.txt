#####
Z2D
#####

| Double precision complex to real valued transform.
|
|
|       hcfftExecZ2D() executes a double-precision complex-to-real, implicitly inverse, hcFFT transform plan.
| hcFFT uses as input data the GPU memory pointed to by the idata parameter. The input array holds only the
| nonredundant complex Fourier coefficients. This function stores the real output values in the odata array.
| and pointers are both required to be aligned to hcfftComplex data type in single-precision transforms and
| hcfftDoubleComplex type in double-precision transforms. It does an out-of-place data transform.
|

Functions
^^^^^^^^^

Function Prototype:
---------------------

 .. note:: **Inputs and Outputs are HCC device pointers.**

`hcfftResult <HCFFT_TYPES.html>`_ **hcfftExecZ2D** (hcfftHandle plan, hcfftDoubleComplex *idata, hcfftDoubleReal *odata)

Detailed Description
^^^^^^^^^^^^^^^^^^^^

Function Documentation
^^^^^^^^^^^^^^^^^^^^^^

::

             hcfftResult hcfftExecZ2D(hcfftHandle plan, hcfftDoubleComplex *idata, hcfftDoubleReal *odata)

+------------+-----------------+-----------------------------------------------------------------+
|  In/out    |  Parameters     | Description                                                     |
+============+=================+=================================================================+
|    [in]    |    plan         | hcfftHandle returned by hcfftCreate.                            |
+------------+-----------------+-----------------------------------------------------------------+
|    [in]    |    idata        | Pointer to the double-precision complex input data              |
|            |                 | (in GPU memory) to transform                                    |
+------------+-----------------+-----------------------------------------------------------------+
|    [out]   |    odata        | Pointer to the double-precision real output data                |
|            |                 | (in GPU memory).                                                |
+------------+-----------------+-----------------------------------------------------------------+

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

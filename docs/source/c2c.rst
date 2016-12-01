#####
C2C
#####

| Single precision complex to complex valued transform.
|
|
|       hcfftExecC2C() executes a single-precision complex-to-complex transform plan in the transform
| direction as specified by direction parameter. hcFFT uses the GPU memory pointed to by the idata
| parameter as input data. This function stores the Fourier coefficients in the odata array. 
| It does an out-of-place data transform in the forward or backward direction.
|

Functions
^^^^^^^^^

Function Prototype:
---------------------

 .. note:: **Inputs and Outputs are HCC device pointers.**

`hcfftResult <HCFFT_TYPES.html>`_ **hcfftExecC2C** (hcfftHandle plan, hcComplex *idata, hcComplex *odata, int direction)

Detailed Description
^^^^^^^^^^^^^^^^^^^^

Function Documentation
^^^^^^^^^^^^^^^^^^^^^^

::

             hcfftResult hcfftExecC2C(hcfftHandle plan, hcComplex *idata, hcComplex *odata, int direction)

+------------+-----------------+-----------------------------------------------------------------+
|  In/out    |  Parameters     | Description                                                     |
+============+=================+=================================================================+
|    [in]    |    plan         | hcfftHandle returned by hcfftCreate.                            |
+------------+-----------------+-----------------------------------------------------------------+
|    [in]    |    idata        | Pointer to the single-precision complex input data              |
|            |                 | (in GPU memory) to transform.                                   |
+------------+-----------------+-----------------------------------------------------------------+
|    [out]   |    odata        | Pointer to the single-precision complex output data             |
|            |                 | (in GPU memory).                                                |
+------------+-----------------+-----------------------------------------------------------------+
|    [in]    |    direction    | The transform direction: HCFFT_FORWARD or HCFFT_BACKWARD.       |
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

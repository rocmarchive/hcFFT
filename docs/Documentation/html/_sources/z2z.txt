#####
Z2Z
#####

| Double precision complex to complex valued transform.
|
|
|       hcfftExecC2C() executes a double-precision complex-to-complex transform plan in the transform
| direction as specified by direction parameter. hcFFT uses the GPU memory pointed to by the idata
| parameter as input data. This function stores the Fourier coefficients in the odata array. 
| It does an out-of-place data transform in the forward or backward direction.
|

Functions
^^^^^^^^^

Function Prototype:
---------------------

 .. note:: **Inputs and Outputs are HCC device pointers.**

`hcfftResult <HCFFT_TYPES.html>`_ **hcfftExecZ2Z** (hcfftHandle plan, hcfftDoubleComplex *idata, hcfftDoubleComplex *odata, int direction)

Detailed Description
^^^^^^^^^^^^^^^^^^^^

Function Documentation
^^^^^^^^^^^^^^^^^^^^^^

::

             hcfftResult hcfftExecZ2Z(hcfftHandle plan, hcfftDoubleComplex *idata, hcfftDoubleComplex *odata, int direction)

+------------+-----------------+-----------------------------------------------------------------+
|  In/out    |  Parameters     | Description                                                     |
+============+=================+=================================================================+
|    [in]    |    plan         | hcfftHandle returned by hcfftCreate.                            |
+------------+-----------------+-----------------------------------------------------------------+
|    [in]    |    idata        | Pointer to the double-precision complex input data              |
|            |                 | (in GPU memory) to transform.                                   |
+------------+-----------------+-----------------------------------------------------------------+
|    [out]   |    odata        | Pointer to the double-precision complex output data             |
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

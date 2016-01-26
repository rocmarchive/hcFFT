#######################
HCFFT Helper functions 
#######################

1. hcfftCreate()
-----------------

`hcfftResult hcfftCreate(hcfftHandle *&plan)`

| This function Creates only an opaque handle, and allocates small data structures on the host.
|

2. hcfftDestory()
------------------

`hcfftResult <HCFFT_TYPES.html>`_ **hcfftDestroy** (hcfftHandle plan)

| This function frees all GPU resources associated with a hcFFT plan and destroys the internal plan data structure. 
| This function should be called once a plan is no longer needed, to avoid wasting GPU memory.
|
| Return Values,

==============================    =============================================
STATUS                            DESCRIPTION
==============================    =============================================
 HCFFT_SUCCESS                    hcFFT successfully destroyed the FFT plan.
 HCFFT_INVALID_PLAN 	          The plan parameter is not a valid handle.
==============================    ============================================= 

3. hcfftPlan1d()
--------------------

`hcfftResult <HCFFT_TYPES.html>`_ **hcfftPlan1d** (hcfftHandle *&plan, int nx, hcfftType type)

| This function Creates a 1D FFT plan configuration for a specified signal size and data type.
| The batch input parameter tells hcFFT how many 1D transforms to configure.
|
| Return Values,

==============================    ===================================================================
STATUS                            DESCRIPTION
==============================    ===================================================================
 HCFFT_SUCCESS         	          hcFFT successfully created the FFT plan.
 HCFFT_ALLOC_FAILED 	          The allocation of GPU resources for the plan failed.
 HCFFT_INVALID_VALUE 	          One or more invalid parameters were passed to the API.
 HCFFT_INTERNAL_ERROR             An internal driver error was detected.
 HCFFT_SETUP_FAILED 	          The hcFFT library failed to initialize.
 HCFFT_INVALID_SIZE 	          Either or both of the nx or ny parameters is not a supported sizek.
==============================    ===================================================================

4. hcfftPlan2d()
--------------------

`hcfftResult <HCFFT_TYPES.html>`_ **hcfftPlan2d** (hcfftHandle *&plan, int nx, int ny, hcfftType type)

| This function Creates a 2D FFT plan configuration according to specified signal sizes and data type.
|
| Return Values,

==============================    ===================================================================
STATUS                            DESCRIPTION
==============================    ===================================================================
 HCFFT_SUCCESS         	          hcFFT successfully created the FFT plan.
 HCFFT_ALLOC_FAILED 	          The allocation of GPU resources for the plan failed.
 HCFFT_INVALID_VALUE 	          One or more invalid parameters were passed to the API.
 HCFFT_INTERNAL_ERROR             An internal driver error was detected.
 HCFFT_SETUP_FAILED 	          The hcFFT library failed to initialize.
 HCFFT_INVALID_SIZE 	          Either or both of the nx or ny parameters is not a supported sizek.
==============================    ===================================================================

5. hcfftPlan3d()
--------------------

`hcfftResult <HCFFT_TYPES.html>`_ **hcfftPlan3d** (hcfftHandle *&plan, int nx, int ny, int nz, hcfftType type)

| This function Creates a 3D FFT plan configuration according to specified signal sizes and data type. 
| This function is the same as hcfftPlan2d() except that it takes a third size parameter nz.
|
| Return Values,

==============================    ===================================================================
STATUS                            DESCRIPTION
==============================    ===================================================================
 HCFFT_SUCCESS         	          hcFFT successfully created the FFT plan.
 HCFFT_ALLOC_FAILED 	          The allocation of GPU resources for the plan failed.
 HCFFT_INVALID_VALUE 	          One or more invalid parameters were passed to the API.
 HCFFT_INTERNAL_ERROR             An internal driver error was detected.
 HCFFT_SETUP_FAILED 	          The hcFFT library failed to initialize.
 HCFFT_INVALID_SIZE 	          Either or both of the nx or ny parameters is not a supported sizek.
==============================    ===================================================================

6. hcfftXtSetGPUs()
--------------------

`hcfftResult <HCFFT_TYPES.html>`_ **hcfftXtSetGPUs** (accelerator &acc)

| This function returns GPUs to be used with the plan
|
| Return Values,

==============================    ===================================================================
STATUS                            DESCRIPTION
==============================    ===================================================================
 HCFFT_SUCCESS         	          hcFFT successfully created the FFT plan.
 HCFFT_ALLOC_FAILED 	          The allocation of GPU resources for the plan failed.
 HCFFT_INVALID_VALUE 	          One or more invalid parameters were passed to the API.
 HCFFT_INTERNAL_ERROR             An internal driver error was detected.
 HCFFT_SETUP_FAILED 	          The hcFFT library failed to initialize.
 HCFFT_INVALID_SIZE 	          Either or both of the nx or ny parameters is not a supported sizek.
==============================    ===================================================================

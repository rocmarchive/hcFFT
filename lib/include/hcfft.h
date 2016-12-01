#ifndef _HC_FFT_H_
#define _HC_FFT_H_

#ifdef __cplusplus
extern "C" {
#endif //(__cplusplus)

struct float_2 {
  float x;
  float y;
};

struct double_2 {
  double x;
  double y;
};


typedef float hcfftReal;
typedef float_2 hcfftComplex;
typedef double hcfftDoubleReal;
typedef double_2 hcfftDoubleComplex;

/* hcfft API Specification */

// Typedef changes
typedef hcfftPlanHandle hcfftHandle;

/* The hcFFT library supports complex- and real-data transforms. The hcfftType data type is an enumeration of the types of transform data supported by hcFFT. */

typedef enum hcfftType_t {
  HCFFT_R2C = 0x2a,  // Real to complex (interleaved)
  HCFFT_C2R = 0x2c,  // Complex (interleaved) to real
  HCFFT_C2C = 0x29,  // Complex to complex (interleaved)
  HCFFT_D2Z = 0x6a,  // Double to double-complex (interleaved)
  HCFFT_Z2D = 0x6c,  // Double-complex (interleaved) to double
  HCFFT_Z2Z = 0x69   // Double-complex to double-complex (interleaved)
} hcfftType;

typedef enum hcfftResult_t {
  HCFFT_SUCCESS        = 0,  //  The hcFFT operation was successful
  HCFFT_INVALID_PLAN   = 1,  //  hcFFT was passed an invalid plan handle
  HCFFT_ALLOC_FAILED   = 2,  //  hcFFT failed to allocate GPU or CPU memory
  HCFFT_INVALID_TYPE   = 3,  //  No longer used
  HCFFT_INVALID_VALUE  = 4,  //  User specified an invalid pointer or parameter
  HCFFT_INTERNAL_ERROR = 5,  //  Driver or internal hcFFT library error
  HCFFT_EXEC_FAILED    = 6,  //  Failed to execute an FFT on the GPU
  HCFFT_SETUP_FAILED   = 7,  //  The hcFFT library failed to initialize
  HCFFT_INVALID_SIZE   = 8,  //  User specified an invalid transform size
  HCFFT_UNALIGNED_DATA = 9,  //  No longer used
  HCFFT_INCOMPLETE_PARAMETER_LIST = 10, //  Missing parameters in call
  HCFFT_INVALID_DEVICE = 11, //  Execution of a plan was on different GPU than plan creation
  HCFFT_PARSE_ERROR    = 12, //  Internal plan database error
  HCFFT_NO_WORKSPACE   = 13  //  No workspace has been provided prior to plan execution
} hcfftResult;

/* Function hcfftCreate()
Creates only an opaque handle, and allocates small data structures on the host.
*/
hcfftResult hcfftCreate(hcfftHandle*&plan);

/* Function hcfftSetStream()
Associate FFT Plan with an accelerator_view
*/
hcfftResult hcfftSetStream(hcfftHandle*&plan, hc::accelerator_view &acc_view);

/*hcFFT Basic Plans*/

/******************************************************************************************************************
 * <i>  Function hcfftPlan1d()
   Description:
       Creates a 1D FFT plan configuration for a specified signal size and data type. The batch input parameter tells
   hcFFT how many 1D transforms to configure.

   Input:
   ----------------------------------------------------------------------------------------------
   #1 plan  Pointer to a hcfftHandle object
   #2 nx  The transform size (e.g. 256 for a 256-point FFT)
   #3 type  The transform data type (e.g., HCFFT_C2C for single precision complex to complex)

   Output:
   ----------------------------------------------------------------------------------------------
   #1 plan  Contains a hcFFT 1D plan handle value

   Return Values:
   ----------------------------------------------------------------------------------------------
   HCFFT_SUCCESS  hcFFT successfully created the FFT plan.
   HCFFT_ALLOC_FAILED   The allocation of GPU resources for the plan failed.
   HCFFT_INVALID_VALUE  One or more invalid parameters were passed to the API.
   HCFFT_INTERNAL_ERROR An internal driver error was detected.
   HCFFT_SETUP_FAILED   The hcFFT library failed to initialize.
   HCFFT_INVALID_SIZE   The nx or batch parameter is not a supported size.
 ***********************************************************************************************************************
 */

hcfftResult hcfftPlan1d(hcfftHandle*&plan, int nx, hcfftType type);

/*
 * <ii> Function hcfftPlan2d()
   Description:
      Creates a 2D FFT plan configuration according to specified signal sizes and data type.

   Input:
   ----------------------------------------------------------------------------------------------
   #1 plan  Pointer to a hcfftHandle object
   #2 nx  The transform size in the x dimension (number of rows)
   #3 ny  The transform size in the y dimension (number of columns)
   #4 type  The transform data type (e.g., HCFFT_C2R for single precision complex to real)

   Output:
   ----------------------------------------------------------------------------------------------
   #1 plan  Contains a hcFFT 2D plan handle value

   Return Values:
   ----------------------------------------------------------------------------------------------
   HCFFT_SUCCESS  hcFFT successfully created the FFT plan.
   HCFFT_ALLOC_FAILED   The allocation of GPU resources for the plan failed.
   HCFFT_INVALID_VALUE  One or more invalid parameters were passed to the API.
   HCFFT_INTERNAL_ERROR An internal driver error was detected.
   HCFFT_SETUP_FAILED   The hcFFT library failed to initialize.
   HCFFT_INVALID_SIZE   Either or both of the nx or ny parameters is not a supported sizek.
*/

hcfftResult hcfftPlan2d(hcfftHandle*&plan, int nx, int ny, hcfftType type);

/*
 * <iii> Function hcfftPlan3d()
   Description:
      Creates a 3D FFT plan configuration according to specified signal sizes and data type.
   This function is the same as hcfftPlan2d() except that it takes a third size parameter nz.

   Input:
   ----------------------------------------------------------------------------------------------
   #1 plan  Pointer to a hcfftHandle object
   #2 nx  The transform size in the x dimension
   #3 ny  The transform size in the y dimension
   #4 nz  The transform size in the z dimension
   #5 type  The transform data type (e.g., HCFFT_R2C for single precision real to complex)

   Output:
   ----------------------------------------------------------------------------------------------
   #1 plan  Contains a hcFFT 3D plan handle value

   Return Values:
   ----------------------------------------------------------------------------------------------
   HCFFT_SUCCESS  hcFFT successfully created the FFT plan.
   HCFFT_ALLOC_FAILED   The allocation of GPU resources for the plan failed.
   HCFFT_INVALID_VALUE  One or more invalid parameters were passed to the API.
   HCFFT_INTERNAL_ERROR   An internal driver error was detected.
   HCFFT_SETUP_FAILED   The hcFFT library failed to initialize.
   HCFFT_INVALID_SIZE   One or more of the nx, ny, or nz parameters is not a supported size.
*/

hcfftResult hcfftPlan3d(hcfftHandle*&plan, int nx, int ny, int nz, hcfftType type);


/* Function hcfftDestroy()
   Description:
      Frees all GPU resources associated with a hcFFT plan and destroys the internal plan data structure.
   This function should be called once a plan is no longer needed, to avoid wasting GPU memory.

   Input:
   -----------------------------------------------------------------------------------------------------
   plan   The hcfftHandle object of the plan to be destroyed.

   Return Values:
   -----------------------------------------------------------------------------------------------------
   HCFFT_SUCCESS  hcFFT successfully destroyed the FFT plan.
   HCFFT_INVALID_PLAN   The plan parameter is not a valid handle.
*/

hcfftResult hcfftDestroy(hcfftHandle plan);

/* hcFFT Execution

  Functions hcfftExecC2C() and hcfftExecZ2Z()

  Description:
       hcfftExecC2C() (hcfftExecZ2Z()) executes a single-precision (double-precision) complex-to-complex transform
  plan in the transform direction as specified by direction parameter. hcFFT uses the GPU memory pointed to by the
  idata parameter as input data. This function stores the Fourier coefficients in the odata array.
  If idata and odata are the same, this method does an in-place transform.

  Input:
  ----------------------------------------------------------------------------------------------------------
  plan  hcfftHandle returned by hcfftCreate
  idata   Pointer to the complex input data (in GPU memory) to transform
  odata   Pointer to the complex output data (in GPU memory)
  direction   The transform direction: HCFFT_FORWARD or HCFFT_INVERSE

  Output:
  -----------------------------------------------------------------------------------------------------------
  odata   Contains the complex Fourier coefficients

  Return Values:
  ------------------------------------------------------------------------------------------------------------
  HCFFT_SUCCESS   hcFFT successfully executed the FFT plan.
  HCFFT_INVALID_PLAN  The plan parameter is not a valid handle.
  HCFFT_INVALID_VALUE   At least one of the parameters idata, odata, and direction is not valid.
  HCFFT_INTERNAL_ERROR  An internal driver error was detected.
  HCFFT_EXEC_FAILED   hcFFT failed to execute the transform on the GPU.
  HCFFT_SETUP_FAILED  The hcFFT library failed to initialize. */


hcfftResult hcfftExecC2C(hcfftHandle plan, hcfftComplex* idata, hcfftComplex* odata, int direction);

hcfftResult hcfftExecZ2Z(hcfftHandle plan, hcfftDoubleComplex* idata, hcfftDoubleComplex* odata, int direction);

/*
  Functions hcfftExecR2C() and hcfftExecD2Z()

  Description:
       hcfftExecR2C() (hcfftExecD2Z()) executes a single-precision (double-precision) real-to-complex, implicitly forward,
  hcFFT transform plan. hcFFT uses as input data the GPU memory pointed to by the idata parameter. This function stores
  the nonredundant Fourier coefficients in the odata array. Pointers to idata and odata are both required to be aligned
  to hcfftComplex data type in single-precision transforms and hcfftDoubleComplex data type in double-precision transforms.
  If idata and odata are the same, this method does an in-place transform. Note the data layout differences between in-place
  and out-of-place transforms as described in Parameter hcfftType.

  Input:
  -----------------------------------------------------------------------------------------------------------------------
  plan  hcfftHandle returned by hcfftCreate
  idata   Pointer to the real input data (in GPU memory) to transform
  odata   Pointer to the complex output data (in GPU memory)

  Output:
  -----------------------------------------------------------------------------------------------------------------------
  odata   Contains the complex Fourier coefficients

  Return Values:
  ------------------------------------------------------------------------------------------------------------------------
  HCFFT_SUCCESS   hcFFT successfully executed the FFT plan.
  HCFFT_INVALID_PLAN  The plan parameter is not a valid handle.
  HCFFT_INVALID_VALUE   At least one of the parameters idata and odata is not valid.
  HCFFT_INTERNAL_ERROR  An internal driver error was detected.
  HCFFT_EXEC_FAILED   hcFFT failed to execute the transform on the GPU.
  HCFFT_SETUP_FAILED  The hcFFT library failed to initialize.
*/

hcfftResult hcfftExecR2C(hcfftHandle plan, hcfftReal* idata, hcfftComplex* odata);

hcfftResult hcfftExecD2Z(hcfftHandle plan, hcfftDoubleReal* idata, hcfftDoubleComplex* odata);

/* Functions hcfftExecC2R() and hcfftExecZ2D()

  Description:
     hcfftExecC2R() (hcfftExecZ2D()) executes a single-precision (double-precision) complex-to-real,
  implicitly inverse, hcFFT transform plan. hcFFT uses as input data the GPU memory pointed to by the
  idata parameter. The input array holds only the nonredundant complex Fourier coefficients. This function
  stores the real output values in the odata array. and pointers are both required to be aligned to hcfftComplex
  data type in single-precision transforms and hcfftDoubleComplex type in double-precision transforms. If idata
  and odata are the same, this method does an in-place transform.

  Input:
  ------------------------------------------------------------------------------------------------------------------
  plan  hcfftHandle returned by hcfftCreate
  idata   Pointer to the complex input data (in GPU memory) to transform
  odata   Pointer to the real output data (in GPU memory)

  Output:
  ------------------------------------------------------------------------------------------------------------------
  odata   Contains the real output data

  Return Values:
  -------------------------------------------------------------------------------------------------------------------
  HCFFT_SUCCESS   hcFFT successfully executed the FFT plan.
  HCFFT_INVALID_PLAN  The plan parameter is not a valid handle.
  HCFFT_INVALID_VALUE   At least one of the parameters idata and odata is not valid.
  HCFFT_INTERNAL_ERROR  An internal driver error was detected.
  HCFFT_EXEC_FAILED   hcFFT failed to execute the transform on the GPU.
  HCFFT_SETUP_FAILED  The hcFFT library failed to initialize.
*/

hcfftResult hcfftExecC2R(hcfftHandle plan, hcfftComplex* idata, hcfftReal* odata);

hcfftResult hcfftExecZ2D(hcfftHandle plan, hcfftDoubleComplex* idata, hcfftDoubleReal* odata);

#ifdef __cplusplus
}
#endif //(__cplusplus)

#endif

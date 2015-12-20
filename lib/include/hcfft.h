/*hcfft API Specification*/

/*Reference: http://docs.nvidia.com/cuda/hcfft/index.html#hcfft-api-reference*/

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

hcfftResult hcfftPlan1d(hcfftHandle* plan, int nx, hcfftType type);

/*
 * <ii> Function cufftPlan2d()
   Description:
      Creates a 2D FFT plan configuration according to specified signal sizes and data type.

   Input:
   ----------------------------------------------------------------------------------------------
   #1 plan 	Pointer to a cufftHandle object
   #2 nx 	The transform size in the x dimension (number of rows)
   #3 ny 	The transform size in the y dimension (number of columns)
   #4 type 	The transform data type (e.g., CUFFT_C2R for single precision complex to real)

   Output:
   ----------------------------------------------------------------------------------------------
   #1 plan 	Contains a cuFFT 2D plan handle value

   Return Values:
   ----------------------------------------------------------------------------------------------
   CUFFT_SUCCESS 	cuFFT successfully created the FFT plan.
   CUFFT_ALLOC_FAILED 	The allocation of GPU resources for the plan failed.
   CUFFT_INVALID_VALUE 	One or more invalid parameters were passed to the API.
   CUFFT_INTERNAL_ERROR	An internal driver error was detected.
   CUFFT_SETUP_FAILED 	The cuFFT library failed to initialize.
   CUFFT_INVALID_SIZE 	Either or both of the nx or ny parameters is not a supported sizek.
*/

cufftResult cufftPlan2d(cufftHandle *plan, int nx, int ny, cufftType type);

/* 
 * <iii> Function cufftPlan3d()
   Description:
      Creates a 3D FFT plan configuration according to specified signal sizes and data type. 
   This function is the same as cufftPlan2d() except that it takes a third size parameter nz.

   Input:
   ----------------------------------------------------------------------------------------------
   #1 plan 	Pointer to a cufftHandle object
   #2 nx 	The transform size in the x dimension
   #3 ny 	The transform size in the y dimension
   #4 nz 	The transform size in the z dimension
   #5 type 	The transform data type (e.g., CUFFT_R2C for single precision real to complex)

   Output:
   ----------------------------------------------------------------------------------------------
   #1 plan 	Contains a cuFFT 3D plan handle value

   Return Values:
   ----------------------------------------------------------------------------------------------
   CUFFT_SUCCESS 	cuFFT successfully created the FFT plan.
   CUFFT_ALLOC_FAILED 	The allocation of GPU resources for the plan failed.
   CUFFT_INVALID_VALUE 	One or more invalid parameters were passed to the API.
   CUFFT_INTERNAL_ERROR 	An internal driver error was detected.
   CUFFT_SETUP_FAILED 	The cuFFT library failed to initialize.
   CUFFT_INVALID_SIZE 	One or more of the nx, ny, or nz parameters is not a supported size.
*/

cufftResult cufftPlan3d(cufftHandle *plan, int nx, int ny, int nz, cufftType type);



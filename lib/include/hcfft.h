#ifndef _HC_FFT_H_
#define _HC_FFT_H_

#include "hcfftlib.h"

/* hcfft API Specification */

/* Reference: http://docs.nvidia.com/cuda/hcfft/index.html#hcfft-api-reference */

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
 * <ii> Function hcfftPlan2d()
   Description:
      Creates a 2D FFT plan configuration according to specified signal sizes and data type.

   Input:
   ----------------------------------------------------------------------------------------------
   #1 plan 	Pointer to a hcfftHandle object
   #2 nx 	The transform size in the x dimension (number of rows)
   #3 ny 	The transform size in the y dimension (number of columns)
   #4 type 	The transform data type (e.g., HCFFT_C2R for single precision complex to real)

   Output:
   ----------------------------------------------------------------------------------------------
   #1 plan 	Contains a hcFFT 2D plan handle value

   Return Values:
   ----------------------------------------------------------------------------------------------
   HCFFT_SUCCESS 	hcFFT successfully created the FFT plan.
   HCFFT_ALLOC_FAILED 	The allocation of GPU resources for the plan failed.
   HCFFT_INVALID_VALUE 	One or more invalid parameters were passed to the API.
   HCFFT_INTERNAL_ERROR	An internal driver error was detected.
   HCFFT_SETUP_FAILED 	The hcFFT library failed to initialize.
   HCFFT_INVALID_SIZE 	Either or both of the nx or ny parameters is not a supported sizek.
*/

hcfftResult hcfftPlan2d(hcfftHandle *plan, int nx, int ny, hcfftType type);

/* 
 * <iii> Function hcfftPlan3d()
   Description:
      Creates a 3D FFT plan configuration according to specified signal sizes and data type. 
   This function is the same as hcfftPlan2d() except that it takes a third size parameter nz.

   Input:
   ----------------------------------------------------------------------------------------------
   #1 plan 	Pointer to a hcfftHandle object
   #2 nx 	The transform size in the x dimension
   #3 ny 	The transform size in the y dimension
   #4 nz 	The transform size in the z dimension
   #5 type 	The transform data type (e.g., HCFFT_R2C for single precision real to complex)

   Output:
   ----------------------------------------------------------------------------------------------
   #1 plan 	Contains a hcFFT 3D plan handle value

   Return Values:
   ----------------------------------------------------------------------------------------------
   HCFFT_SUCCESS 	hcFFT successfully created the FFT plan.
   HCFFT_ALLOC_FAILED 	The allocation of GPU resources for the plan failed.
   HCFFT_INVALID_VALUE 	One or more invalid parameters were passed to the API.
   HCFFT_INTERNAL_ERROR 	An internal driver error was detected.
   HCFFT_SETUP_FAILED 	The hcFFT library failed to initialize.
   HCFFT_INVALID_SIZE 	One or more of the nx, ny, or nz parameters is not a supported size.
*/

hcfftResult hcfftPlan3d(hcfftHandle *plan, int nx, int ny, int nz, hcfftType type);


/* Function hcfftDestroy()
   Description: 
      Frees all GPU resources associated with a hcFFT plan and destroys the internal plan data structure. 
   This function should be called once a plan is no longer needed, to avoid wasting GPU memory.

   Input:
   -----------------------------------------------------------------------------------------------------
   plan 	The hcfftHandle object of the plan to be destroyed.

   Return Values:
   -----------------------------------------------------------------------------------------------------
   HCFFT_SUCCESS 	hcFFT successfully destroyed the FFT plan.
   HCFFT_INVALID_PLAN 	The plan parameter is not a valid handle.
*/

hcfftResult hcfftDestroy(hcfftHandle plan);

#endif

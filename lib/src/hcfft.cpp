#include "hcfft.h"

// Global Static plan object
FFTPlan planObject;

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
   HCFFT_INVALID_SIZE   The nx parameter is not a supported size.
 ***********************************************************************************************************************
 */

hcfftResult hcfftPlan1d(hcfftHandle* plan, int nx, hcfftType type) {
  // Set dimension as 1D
  hcfftDim dimension = HCFFT_1D; 
  
  // Check the input type and set appropriate direction and precision
  hcfftDirection direction;
  hcfftPrecision precision;
  switch (type) {
    case HCFFT_R2C:
                     precision = HCFFT_SINGLE;
                     direction = HCFFT_FORWARD;
                     break;
    case HCFFT_C2R:
                     precision = HCFFT_SINGLE;
                     direction = HCFFT_BACKWARD;
                     break;
    case HCFFT_C2C:
                     precision = HCFFT_SINGLE;
                     direction = HCFFT_BOTH;
                     break;
    case HCFFT_D2Z:
                     precision = HCFFT_DOUBLE;
                     direction = HCFFT_FORWARD;
                     break;
    case HCFFT_Z2D:
                     precision = HCFFT_DOUBLE;
                     direction = HCFFT_BACKWARD;
                     break;
    case HCFFT_Z2Z:
                     precision = HCFFT_DOUBLE;
                     direction = HCFFT_BOTH;
                     break;
    default:    
                     // Invalid type
                     return HCFFT_INVALID_VALUE;
  }
  
  // length array to bookkeep dimension info
  size_t* length = (size_t*)malloc(sizeof(size_t) * dimension);

  if ( nx < 0 ) {
     // invalid size
     return HCFFT_INVALID_SIZE;
  } else {
    length[0] = nx;
  }
  auto planhandle = *plan;
  hcfftStatus status = planObject.hcfftCreateDefaultPlan (&planhandle, dimension, length, direction);

  if ( status == HCFFT_ERROR || status == HCFFT_INVALID ) {
    return HCFFT_INVALID_VALUE;
  }
  
  // Default options
  // set certain properties of plan with default values
  // Set Precision
  status = planObject.hcfftSetPlanPrecision(planhandle, precision);
  if ( status != HCFFT_SUCCEEDS ) {
    return HCFFT_SETUP_FAILED; 
  }

  // Set Transpose type
  status = planObject.hcfftSetPlanTransposeResult(planhandle, HCFFT_NOTRANSPOSE);
  if ( status != HCFFT_SUCCEEDS ) {
    return HCFFT_SETUP_FAILED; 
  }
  
  // Set Result location data layout
  status = planObject.hcfftSetResultLocation(planhandle, HCFFT_OUTOFPLACE); 
  if ( status != HCFFT_SUCCEEDS ) {
    return HCFFT_SETUP_FAILED; 
  }

  return HCFFT_SUCCESS;
}


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

hcfftResult hcfftPlan2d(hcfftHandle *plan, int nx, int ny, hcfftType type) {
  // Set dimension as 2D
  hcfftDim dimension = HCFFT_2D; 
  
  // Check the input type and set appropriate direction and precision
  hcfftDirection direction;
  hcfftPrecision precision;
  switch (type) {
    case HCFFT_R2C:
                     precision = HCFFT_SINGLE;
                     direction = HCFFT_FORWARD;
                     break;
    case HCFFT_C2R:
                     precision = HCFFT_SINGLE;
                     direction = HCFFT_BACKWARD;
                     break;
    case HCFFT_C2C:
                     precision = HCFFT_SINGLE;
                     direction = HCFFT_BOTH;
                     break;
    case HCFFT_D2Z:
                     precision = HCFFT_DOUBLE;
                     direction = HCFFT_FORWARD;
                     break;
    case HCFFT_Z2D:
                     precision = HCFFT_DOUBLE;
                     direction = HCFFT_BACKWARD;
                     break;
    case HCFFT_Z2Z:
                     precision = HCFFT_DOUBLE;
                     direction = HCFFT_BOTH;
                     break;
    default:    
                     // Invalid type
                     return HCFFT_INVALID_VALUE;
  }
  
  // length array to bookkeep dimension info
  size_t* length = (size_t*)malloc(sizeof(size_t) * dimension);

  if (nx < 0 || ny < 0) {
     // invalid size
     return HCFFT_INVALID_SIZE;
  } else {
    length[0] = nx;
    length[1] = ny;
  }

  hcfftStatus status = planObject.hcfftCreateDefaultPlan (plan, dimension, length, direction);

  if ( status == HCFFT_ERROR || status == HCFFT_INVALID ) {
    return HCFFT_INVALID_VALUE;
  }

  // Default options
  // set certain properties of plan with default values
  // Set Precision
  status = planObject.hcfftSetPlanPrecision(*plan, precision);
  if ( status != HCFFT_SUCCEEDS ) {
    return HCFFT_SETUP_FAILED; 
  }

  // Set Transpose type
  status = planObject.hcfftSetPlanTransposeResult(*plan, HCFFT_NOTRANSPOSE);
  if ( status != HCFFT_SUCCEEDS ) {
    return HCFFT_SETUP_FAILED; 
  }
  
  // Set Result location data layout
  status = planObject.hcfftSetResultLocation(*plan, HCFFT_OUTOFPLACE); 
  if ( status != HCFFT_SUCCEEDS ) {
    return HCFFT_SETUP_FAILED; 
  }

  return HCFFT_SUCCESS;
}

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

hcfftResult hcfftPlan3d(hcfftHandle *plan, int nx, int ny, int nz, hcfftType type) {
  // Set dimension as 3D
  hcfftDim dimension = HCFFT_3D; 
  
  // Check the input type and set appropriate direction and precision
  hcfftDirection direction;
  hcfftPrecision precision;
  switch (type) {
    case HCFFT_R2C:
                     precision = HCFFT_SINGLE;
                     direction = HCFFT_FORWARD;
                     break;
    case HCFFT_C2R:
                     precision = HCFFT_SINGLE;
                     direction = HCFFT_BACKWARD;
                     break;
    case HCFFT_C2C:
                     precision = HCFFT_SINGLE;
                     direction = HCFFT_BOTH;
                     break;
    case HCFFT_D2Z:
                     precision = HCFFT_DOUBLE;
                     direction = HCFFT_FORWARD;
                     break;
    case HCFFT_Z2D:
                     precision = HCFFT_DOUBLE;
                     direction = HCFFT_BACKWARD;
                     break;
    case HCFFT_Z2Z:
                     precision = HCFFT_DOUBLE;
                     direction = HCFFT_BOTH;
                     break;
    default:    
                     // Invalid type
                     return HCFFT_INVALID_VALUE;
  }
  
  // length array to bookkeep dimension info
  size_t* length = (size_t*)malloc(sizeof(size_t) * dimension);

  if (nx < 0 || ny < 0 || nz < 0) {
     // invalid size
     return HCFFT_INVALID_SIZE;
  } else {
    length[0] = nx;
    length[1] = ny;
    length[2] = nz;
  }

  hcfftStatus status = planObject.hcfftCreateDefaultPlan (plan, dimension, length, direction);

  if ( status == HCFFT_ERROR || status == HCFFT_INVALID ) {
    return HCFFT_INVALID_VALUE;
  } 

  // Default options
  // set certain properties of plan with default values
  // Set Precision
  status = planObject.hcfftSetPlanPrecision(*plan, precision);
  if ( status != HCFFT_SUCCEEDS ) {
    return HCFFT_SETUP_FAILED; 
  }

  // Set Transpose type
  status = planObject.hcfftSetPlanTransposeResult(*plan, HCFFT_NOTRANSPOSE);
  if ( status != HCFFT_SUCCEEDS ) {
    return HCFFT_SETUP_FAILED; 
  }
  
  // Set Result location data layout
  status = planObject.hcfftSetResultLocation(*plan, HCFFT_OUTOFPLACE); 
  if ( status != HCFFT_SUCCEEDS ) {
    return HCFFT_SETUP_FAILED; 
  }

  return HCFFT_SUCCESS;
}

/* Function hcfftDestroy()
   Description: 
      Frees all GPU resources associated with a hcFFT plan and destroys the internal plan data structure. 
   This function should be called once a plan is no longer needed, to avoid wasting GPU memory.

   Input:
   -----------------------------------------------------------------------------------------------------
   plan         The hcfftHandle object of the plan to be destroyed.

   Return Values:
   -----------------------------------------------------------------------------------------------------
   HCFFT_SUCCESS        hcFFT successfully destroyed the FFT plan.
   HCFFT_INVALID_PLAN   The plan parameter is not a valid handle.
*/

hcfftResult hcfftDestroy(hcfftHandle plan) {
  auto planHandle = plan;
  hcfftStatus status = planObject.hcfftDestroyPlan(&planHandle);
  if (status != HCFFT_SUCCEEDS) {
    return HCFFT_INVALID_PLAN;
  }
  return HCFFT_SUCCESS;
}

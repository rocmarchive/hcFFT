#include "hcfft.h"

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
  // Create plan object
  FFTPlan planObject;
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

  if (nx < 0) {
     // invalid size
     return HCFFT_INVALID_SIZE;
  } else {
    length[0] = nx;
  }

  hcfftStatus status = planObject.hcfftCreateDefaultPlan (plan, dimension, length, direction);

  if ( status == HCFFT_ERROR || status == HCFFT_INVALID ) {
    return HCFFT_INVALID_VALUE;
  } else if ( status == HCFFT_SUCCEEDS ) {
    return HCFFT_SUCCESS;
  }

  return HCFFT_SUCCESS;
}


#include "ampfftlib.h"

ampfftStatus FFTPlan::createDefaultPlan (FFTPlan* fftPlan, ampfftDim dimension, ampfftIpLayout ipLayout,
                                         ampfftOpLayout opLayout, ampfftDirection direction,
                                         ampfftResLocation location, ampfftResTransposed transposeType,
                                         void* input, void* output, int *inStride, int *outStride, int *length,
                                         int batchSize, int iDist, int oDist)
{
  if(!fftPlan || !inStride || !outStride || !length)
    return AMPFFT_INVALID;

  return AMPFFT_SUCCESS;
}

ampfftStatus FFTPlan::executePlan(FFTPlan* fftPlan)
{
  if(!fftPlan)
    return AMPFFT_INVALID;

  return AMPFFT_SUCCESS;
}


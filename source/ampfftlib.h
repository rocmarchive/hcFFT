#include <iostream>
using namespace std;

typedef enum ampfftDim_
{
  AMPFFT_1D = 1,
  AMPFFT_2D,
  AMPFFT_3D
}ampfftDim;

typedef enum ampfftLayout_
{
  AMPFFT_COMPLEX = 1,
  AMPFFT_REAL,
}ampfftIpLayout,ampfftOpLayout;

typedef enum ampfftDirection_
{
  AMPFFT_FORWARD = -1,
  AMPFFT_BACKWARD = 1,
} ampfftDirection;

typedef enum ampfftResLocation_
{
  AMPFFT_INPLACE = 1,
  AMPFFT_OUTOFPLACE,
} ampfftResLocation;

typedef enum ampfftResTransposed_ {
  AMPFFT_NOTRANSPOSE = 1,
  AMPFFT_TRANSPOSED,
} ampfftResTransposed;

typedef enum ampfftStatus_ {
  AMPFFT_SUCCESS = 0,
  AMPFFT_INVALID = -1,
  AMPFFT_ERROR = -2
} ampfftStatus;

struct fftReal
{
  float data;
};

struct fftComplex
{
  float real;
  float img;
};

class FFTPlan
{
public:
  ampfftDim dimension;
  ampfftIpLayout ipLayout;
  ampfftOpLayout opLayout;
  ampfftDirection direction;
  ampfftResLocation location;
  ampfftResTransposed transposeType;
  void* input;
  void* output;
  int *inStride;
  int *outStride;
  int *length;
  int batchSize;
  int iDist;
  int oDist;

  ampfftStatus createDefaultPlan(FFTPlan* fftPlan, ampfftDim dimension, ampfftIpLayout ipLayout,
                                 ampfftOpLayout opLayout, ampfftDirection direction,
                                 ampfftResLocation location, ampfftResTransposed_ transposeType,
                                 void* input, void* output, int *inStride, int *outStride, int *length,
                                 int batchSize, int iDist, int oDist);

  ampfftStatus executePlan(FFTPlan*);
};

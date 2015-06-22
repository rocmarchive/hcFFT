#include <iostream>
#include <stdio.h>
#include <amp.h>
#include <amp_math.h>
#include <amp_short_vectors.h>
#include "lock.h"

using namespace Concurrency;
using namespace Concurrency::graphics;

typedef size_t ampfftPlanHandle;

typedef enum ampfftPrecision_
{
	AMPFFT_SINGLE	= 1,
	AMPFFT_DOUBLE,
}ampfftPrecision;

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

typedef enum ampfftGenerators_
{
	Stockham = 1,
	Transpose,
	Copy,
}ampfftGenerators;

class FFTPlan
{
public:
  ampfftDim dimension;
  ampfftIpLayout ipLayout;
  ampfftOpLayout opLayout;
  ampfftDirection direction;
  ampfftResLocation location;
  ampfftResTransposed transposeType;
  ampfftPrecision precision;
  void* input;
  void* output;
  std::vector< size_t >	length;
  std::vector< size_t >	inStride, outStride;
  size_t batchSize;
  size_t iDist;
  size_t oDist;
  double forwardScale;
  double backwardScale;

  bool baked;
  ampfftGenerators gen;

  ampfftPlanHandle planX;
  ampfftPlanHandle planY;
  ampfftPlanHandle planZ;

  ampfftPlanHandle planTX;
  ampfftPlanHandle planTY;
  ampfftPlanHandle planTZ;

  ampfftPlanHandle planRCcopy;
  //	Performance Tuning parameters
  bool bLdsComplex;
  unsigned uLdsFraction;
  bool ldsPadding;

  size_t  large1D_Xfactor;

  size_t tmpBufSize;
  Concurrency::array_view<float ,1> *intBuffer;

  size_t tmpBufSizeRC;
  Concurrency::array_view<float ,1> *intBufferRC;

  size_t  tmpBufSizeC2R;
  Concurrency::array_view<float ,1> *intBufferC2R;

  bool transflag;

  size_t  large1D;
  bool  large2D;
  size_t  cacheSize;

  Concurrency::array_view<float ,1> *const_buffer;

  // Real-Complex simple flag
  // if this is set we do real to-and-from full complex using simple algorithm
  // where imaginary of input is set to zero in forward and imaginary not written in backward
  bool RCsimple;

  FFTPlan() : dimension (AMPFFT_1D), ipLayout (AMPFFT_COMPLEX),
              opLayout (AMPFFT_COMPLEX), location (AMPFFT_INPLACE),
              transposeType (AMPFFT_NOTRANSPOSE), precision (AMPFFT_SINGLE),
              batchSize (1), iDist(1), oDist(1), forwardScale (1.0), backwardScale (1.0),
              baked (false), gen(Stockham), planX(0), planY(0), planZ(0),
              planTX(0), planTY(0), planTZ(0), planRCcopy(0), bLdsComplex(false),
              uLdsFraction(0), ldsPadding(false), large1D_Xfactor(0), tmpBufSize(0),
	      intBuffer( NULL ), tmpBufSizeRC(0), intBufferRC(NULL), tmpBufSizeC2R(0),
	      intBufferC2R(NULL), transflag(false),large1D(0), large2D(false),
              const_buffer(NULL)
  {};

  ampfftStatus ampfftCreateDefaultPlan(ampfftPlanHandle* plHandle,ampfftDim dimension, const size_t *length);

  ampfftStatus executePlan(FFTPlan*);
};


class FFTRepo
{
  // Private constructor to stop explicit instantiation
  FFTRepo( )
  {}

  // Private copy constructor to stop implicit instantiation
  FFTRepo( const FFTRepo& );

  // Private operator= to assure only 1 copy of singleton
  FFTRepo& operator=( const FFTRepo& );

  public:

  //	Used to make the FFTRepo struct thread safe; STL is not thread safe by default
  //	Optimally, we could use a lock object per STL struct, as two different STL structures
  //	can be modified at the same time, but a single lock object is easier and performance should
  //	still be good
  static lockRAII lockRepo;

  //	Everybody who wants to access the Repo calls this function to get a repo reference
  static FFTRepo& getInstance( )
  {
    static FFTRepo fftRepo;
    return fftRepo;
  };

  ~FFTRepo( )
  {
  }
};

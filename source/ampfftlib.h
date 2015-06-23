#include <iostream>
#include <stdio.h>
#include <amp.h>
#include <amp_math.h>
#include <amp_short_vectors.h>
#include <complex>
#include "lock.h"

using namespace Concurrency;
using namespace Concurrency::graphics;

#define AMPFFT_CB_NY 0
#define	AMPFFT_CB_NZ 1
#define	AMPFFT_CB_NW 2
#define	AMPFFT_CB_N5 3
#define	AMPFFT_CB_ISX 4
#define	AMPFFT_CB_ISY 5
#define	AMPFFT_CB_ISZ 6
#define	AMPFFT_CB_ISW 7
#define	AMPFFT_CB_IS5 8
#define	AMPFFT_CB_OSX 9
#define	AMPFFT_CB_OSY 10
#define	AMPFFT_CB_OSZ 11
#define	AMPFFT_CB_OSW 12
#define	AMPFFT_CB_OS5 13

#define AMPFFT_CB_SIZE 32
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

//	The "envelope" is a set of limits imposed by the hardware
//	This will depend on the GPU(s) in the OpenCL context.
//	If there are multiple devices, this should be the least
//	common denominators.
//
struct FFTEnvelope {
	cl_ulong   limit_LocalMemSize;
	           //  this is the minimum of CL_DEVICE_LOCAL_MEM_SIZE
	size_t     limit_Dimensions;
	           //  this is the minimum of CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS
	size_t     limit_Size[8];
	           //  these are the minimima of CL_DEVICE_MAX_WORK_ITEM_SIZES[0..n]
	size_t     limit_WorkGroupSize;
	           //  this is the minimum of CL_DEVICE_MAX_WORK_GROUP_SIZE

	// ??  CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE

	FFTEnvelope ()
	:	limit_LocalMemSize (0)
	,	limit_Dimensions (0)
	,	limit_WorkGroupSize (0)
	{
		::memset (& limit_Size, 0, sizeof (limit_Size));
	}
};

struct FFTKernelGenKeyParams {
	/*
	 *	This structure distills a subset of the fftPlan data,
	 *	including all information that is used to generate the OpenCL kernel.
	 *	This structure can be used as a key to reusing kernels that have already
	 *	been compiled.
	 */
	size_t                   fft_DataDim;       // Dimensionality of the data
	size_t                   fft_N[5];          // [0] is FFT size, e.g. 1024
	                                            // This must be <= size of LDS!
	size_t                   fft_inStride [5];  // input strides
	size_t                   fft_outStride[5];  // output strides

	ampfftResLocation   fft_placeness;
	ampfftIpLayout           fft_inputLayout;
	ampfftOpLayout           fft_outputLayout;
	ampfftPrecision        fft_precision;
	double                   fft_fwdScale;
	double                   fft_backScale;

	size_t                   fft_SIMD;          // Assume this SIMD/workgroup size
	size_t                   fft_LDSsize;       // Limit the use of LDS to this many bytes.
	size_t                   fft_R;             // # of complex values to keep in working registers
	                                            // SIMD size * R must be <= size of LDS!
	size_t                   fft_MaxRadix;      // Limit the radix to this value.
	size_t			fft_MaxWorkGroupSize; // Limit for work group size
	bool                     fft_LdsComplex;    // If true, store complex values in LDS memory
	                                            // If false, store scalare values in LDS.
	                                            // Generally, false will provide more efficient kernels,
	                                            // but not always.
	                                            // see FFTPlan::bLdsComplex and ARBITRARY::LDS_COMPLEX
	bool                     fft_ldsPadding;    // default padding is false
	bool                     fft_3StepTwiddle;  // This is one pass of the "3-step" algorithm;
	                                            // so extra twiddles are applied on output.
	bool                     fft_UseFMA;        // *** TODO
	bool                     fft_RCsimple;
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

  //	Hardware Limits
  FFTEnvelope envelope;

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

  ampfftStatus ampfftDestroyPlan(ampfftPlanHandle* plHandle);

  ampfftStatus executePlan(FFTPlan*);

  ampfftStatus	ampfftGetPlanPrecision(const ampfftPlanHandle plHandle, ampfftPrecision* precision );

  ampfftStatus	ampfftSetPlanPrecision(ampfftPlanHandle plHandle, ampfftPrecision precision );

  ampfftStatus	ampfftGetPlanScale(const ampfftPlanHandle plHandle, ampfftDirection dir, float* scale );

  ampfftStatus	ampfftSetPlanScale(ampfftPlanHandle plHandle, ampfftDirection dir, float scale );

  ampfftStatus	ampfftGetPlanBatchSize(const ampfftPlanHandle plHandle, size_t* batchSize );

  ampfftStatus	ampfftSetPlanBatchSize(ampfftPlanHandle plHandle, size_t batchSize );

  ampfftStatus	ampfftGetPlanDim(const ampfftPlanHandle plHandle, ampfftDim* dim, int* size );

  ampfftStatus	ampfftSetPlanDim(ampfftPlanHandle plHandle, const ampfftDim dim );

  ampfftStatus	ampfftGetPlanLength(const ampfftPlanHandle plHandle, const ampfftDim dim, size_t* clLengths );

  ampfftStatus	ampfftSetPlanLength(ampfftPlanHandle plHandle, const ampfftDim dim, const size_t* clLengths );

  ampfftStatus	ampfftGetPlanInStride(const ampfftPlanHandle plHandle, const ampfftDim dim, size_t* clStrides );

  ampfftStatus	ampfftSetPlanInStride(ampfftPlanHandle plHandle, const ampfftDim dim, size_t* clStrides );

  ampfftStatus	ampfftGetPlanOutStride(const ampfftPlanHandle plHandle, const ampfftDim dim, size_t* clStrides );

  ampfftStatus	ampfftSetPlanOutStride(ampfftPlanHandle plHandle, const ampfftDim dim, size_t* clStrides );

  ampfftStatus	ampfftGetPlanDistance(const ampfftPlanHandle plHandle, size_t* iDist, size_t* oDist );

  ampfftStatus	ampfftSetPlanDistance(ampfftPlanHandle plHandle, size_t iDist, size_t oDist );

  ampfftStatus	ampfftGetLayout(const ampfftPlanHandle plHandle, ampfftIpLayout* iLayout, ampfftOpLayout* oLayout );

  ampfftStatus	ampfftSetLayout(ampfftPlanHandle plHandle, ampfftIpLayout iLayout, ampfftOpLayout oLayout );

  ampfftStatus	ampfftGetResultLocation(const ampfftPlanHandle plHandle, ampfftResLocation* placeness );

  ampfftStatus GetEnvelope (const FFTEnvelope **) const;

  ampfftStatus SetEnvelope ();

  ampfftStatus AllocateWriteBuffers ();

  ampfftStatus ReleaseBuffers ();

  size_t ElementSize() const;
};


class FFTRepo
{
  //	All plans that the user creates over the course of using the library are stored here.
  //	Plans can be arbitrarily created and destroyed at anytime by the user, in arbitrary order, so vector
  //	does not seem appropriate, so a map was chosen because of the O(log N) search properties
  //	A lock object is created for each plan, such that any getter/setter can lock the 'plan' object before
  //	reading/writing its values.  The lock object is kept seperate from the plan object so that the lock
  //	object can be held the entire time a plan is getting destroyed in ampfftDestroyPlan.
  typedef pair< FFTPlan*, lockRAII* > repoPlansValue;
  typedef map< ampfftPlanHandle, repoPlansValue > repoPlansType;
  repoPlansType repoPlans;

  //	Structure containing all the data we need to remember for a specific invokation of a kernel
  //	generator
  struct fftRepoValue {
    std::string ProgramString;
    std::string EntryPoint_fwd;
    std::string EntryPoint_back;

    fftRepoValue ()
    {}
  };

  typedef std::pair< ampfftGenerators, ampfftPlanHandle> fftRepoKey;
  typedef std::map< fftRepoKey, fftRepoValue > fftRepoType;
  typedef fftRepoType::iterator fftRepo_iterator;

  fftRepoType	mapFFTs;

  //	Static count of how many plans we have generated; always incrementing during the life of the library
  //	This is used as a unique identifier for plans
  static size_t planCount;

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

  ampfftStatus createPlan( ampfftPlanHandle* plHandle, FFTPlan*& fftPlan );

  ampfftStatus getPlan( ampfftPlanHandle plHandle, FFTPlan*& fftPlan, lockRAII*& planLock );

  ampfftStatus deletePlan( ampfftPlanHandle* plHandle );

  ampfftStatus setProgramEntryPoints( const ampfftGenerators gen, const ampfftPlanHandle& handle, const FFTKernelGenKeyParams& fftParam, 
                                      const char * kernel_fwd, const char * kernel_back);

  ampfftStatus getProgramEntryPoint( const ampfftGenerators gen, const ampfftPlanHandle& handle, const FFTKernelGenKeyParams& fftParam, ampfftDirection dir, std::string& kernel);

  ampfftStatus setProgramCode( const ampfftGenerators gen, const ampfftPlanHandle& handle, const FFTKernelGenKeyParams&, const std::string& kernel);

  ampfftStatus getProgramCode( const ampfftGenerators gen, const ampfftPlanHandle& handle, const FFTKernelGenKeyParams&, std::string& kernel);

  ampfftStatus releaseResources( );

  ~FFTRepo( )
  {
    releaseResources();
  }
};

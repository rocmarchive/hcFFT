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

#define BUG_CHECK(_proposition)	\
	{ bool btmp = (_proposition);	assert (btmp); if (! btmp)	return AMPFFT_ERROR; }

#define ARG_CHECK(_proposition)	\
{ bool btmp = (_proposition);	assert (btmp); if (! btmp)	return AMPFFT_ERROR; }

static inline bool IsPo2 (size_t u) {
	return (u != 0) &&  (0 == (u & (u-1)));
}

inline void BSF( unsigned long* index, size_t& mask )
{
       //_BitScanForward( index, mask );
}

template<typename T>
static inline T DivRoundingUp (T a, T b) {
	return (a + (b-1)) / b;
}

static inline size_t BitScanF (size_t n) {
	assert (n != 0);
	unsigned long tmp = 0;
	BSF (& tmp, n);
	return (size_t) tmp;
}

//	Find the smallest power of 2 that is >= n; return its power of 2 factor
//	e.g., CeilPo2 (7) returns 3 : (2^3 >= 7)
inline size_t CeilPo2 (size_t n)
{
  size_t v = 1, t = 0;
  while(v < n)
  {
    v <<= 1;
    t++;
  }
  return t;
}

inline size_t FloorPo2 (size_t n)
//	return the largest power of 2 that is <= n.
//	e.g., FloorPo2 (7) returns 4.
// *** TODO use x86 BSR instruction, using compiler intrinsics.
{
  size_t tmp;
  while (0 != (tmp = n & (n-1)))
    n = tmp;
    return n;
}


namespace ARBITRARY {
	// TODO:  These arbitrary parameters should be tuned for the type of GPU
	//	being used.  These values are probably OK for Radeon 58xx and 68xx.
	enum {
		MAX_DIMS  = 3,
			//  The clEnqueuNDRangeKernel accepts a multi-dimensional domain array.
			//  The # of dimensions is arbitrary, but limited by the OpenCL implementation
			//  usually to 3 dimensions (CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS).
			//  The kernel generator also assumes a limit on the # of dimensions.

		SIMD_WIDTH = 64,
			//  Workgroup size.  This is the # of work items that share
			//  local data storage (LDS).  This # is best for Evergreen gpus,
			//  but might change in the future.

		LDS_BANK_BITS = 5,
		LDS_BANK_SIZE = (1 << LDS_BANK_BITS),
		LDS_PADDING   = false,//true,
			//  On AMD hardware, the low-order bits of the local_id enumerate
			//  the work items that access LDS in parallel.  Ideally, we will
			//  pad our LDS arrays so that these work items access different banks
			//  of the LDS.
			//  2 ** LDS_BANK_BITS is the number of LDS banks.
			//  If LDS_PADDING is non-zero, the kernel generator should pad the
			//  LDS arrays to reduce or eliminate bank conflicts.

		LDS_FRACTION_IDEAL = 6,    // i.e., 1/6th
		LDS_FRACTION_MAX   = 4,    // i.e., 1/4
			//  For best performance, each workgroup should use 1/IDEAL'th the amount of LDS
			//  revealed by clGetDeviceInfo (.. CL_DEVICE_LOCAL_MEM_SIZE, ...)
			//  However, we can use up to 1/MAX'th of LDS per workgroup when necessary to
			//  perform the FFT in a single pass instead of multiple passes.
			//  This tuning parameter is a good value for Evergreen gpus,
			//  but might change in the future.

		LDS_COMPLEX = false,
			//  This is the default value for FFTKernelGenKeyParams::fft_LdsComplex.
			//  The generated kernels require so many bytes of LDS for each single precision
			//..complex number in the vector.
			//  If LDS_COMPLEX, then we declare an LDS array of complex numbers (8 bytes each)
			//  and swap data between workitems with a single barrier.
			//  If ! LDS_COMPLEX, then we declare an LDS array or scalar numbers (4 bytes each)
			//  and swap data between workitems in two phases, with extra barriers.
			//  The former approach uses fewer instructions and barriers;
			//  The latter uses half as much LDS space, so twice as many wavefronts can be run
			//  in parallel.

		TWIDDLE_DEE = 4,
			//  4 bits per row of matrix.
        };
};

class tofstreamRAII
{
	public:
		FILE *outFile;
		std::string fileName;
		tofstreamRAII( const std::string& name ): fileName( name )
		{
			outFile = fopen(fileName.c_str( ), "w");
		}

		~tofstreamRAII( )
		{
			fclose( outFile);
		}

		std::string& getName( )
		{
			return fileName;
		}

		void setName( const std::string& name )
		{
			fileName = name;
		}

		FILE* get( )
		{
			return outFile;
		}
};

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

class FFTRepo;

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

  ampfftStatus	ampfftSetResultLocation(ampfftPlanHandle plHandle, ampfftResLocation placeness );

  ampfftStatus	ampfftGetPlanTransposeResult(const ampfftPlanHandle plHandle, ampfftResTransposed * transposed );

  ampfftStatus	ampfftSetPlanTransposeResult(ampfftPlanHandle plHandle, ampfftResTransposed transposed );

  ampfftStatus GetEnvelope (const FFTEnvelope **) const;

  ampfftStatus SetEnvelope ();

  template <ampfftGenerators G>
  ampfftStatus GetMax1DLengthPvt (size_t *longest ) const;

  template <ampfftGenerators G>
  ampfftStatus GetKernelGenKeyPvt (FFTKernelGenKeyParams & params) const;

  template <ampfftGenerators G>
  ampfftStatus GetWorkSizesPvt (std::vector<size_t> & globalws, std::vector<size_t> & localws) const;

  template <ampfftGenerators G>
  ampfftStatus GenerateKernelPvt (const ampfftPlanHandle plHandle, FFTRepo& fftRepo) const;

  ampfftStatus GetMax1DLength (size_t *longest ) const;

  ampfftStatus GetKernelGenKey (FFTKernelGenKeyParams & params) const;

  ampfftStatus GetWorkSizes (std::vector<size_t> & globalws, std::vector<size_t> & localws) const;

  ampfftStatus GenerateKernel (const ampfftPlanHandle plHandle, FFTRepo & fftRepo) const;

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

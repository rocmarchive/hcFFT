#include <iostream>
#include <stdio.h>
#include <amp.h>
#include <amp_math.h>
#include <amp_short_vectors.h>
#include <complex>
#include <unistd.h>
#include "lock.h"

using namespace Concurrency;
using namespace Concurrency::graphics;

#define HCFFT_CB_NY 0
#define	HCFFT_CB_NZ 1
#define	HCFFT_CB_NW 2
#define	HCFFT_CB_N5 3
#define	HCFFT_CB_ISX 4
#define	HCFFT_CB_ISY 5
#define	HCFFT_CB_ISZ 6
#define	HCFFT_CB_ISW 7
#define	HCFFT_CB_IS5 8
#define	HCFFT_CB_OSX 9
#define	HCFFT_CB_OSY 10
#define	HCFFT_CB_OSZ 11
#define	HCFFT_CB_OSW 12
#define	HCFFT_CB_OS5 13
#define HCFFT_CB_SIZE 32

#define BUG_CHECK(_proposition)	\
	{ bool btmp = (_proposition);	assert (btmp); if (! btmp)	return HCFFT_ERROR; }

#define ARG_CHECK(_proposition)	\
{ bool btmp = (_proposition);	assert (btmp); if (! btmp)	return HCFFT_ERROR; }

enum BlockComputeType
{
	BCT_C2R,	// Column to row
	BCT_R2C,	// Row to column
};

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

typedef size_t hcfftPlanHandle;

typedef enum hcfftPrecision_
{
	HCFFT_SINGLE	= 1,
	HCFFT_DOUBLE,
}hcfftPrecision;

typedef enum hcfftDim_
{
  HCFFT_1D = 1,
  HCFFT_2D,
  HCFFT_3D
}hcfftDim;

typedef enum hcfftLayout_
{
  HCFFT_COMPLEX = 1,
  HCFFT_REAL,
}hcfftIpLayout,hcfftOpLayout;

typedef enum hcfftDirection_
{
  HCFFT_FORWARD = -1,
  HCFFT_BACKWARD = 1,
} hcfftDirection;

typedef enum hcfftResLocation_
{
  HCFFT_INPLACE = 1,
  HCFFT_OUTOFPLACE,
} hcfftResLocation;

typedef enum hcfftResTransposed_ {
  HCFFT_NOTRANSPOSE = 1,
  HCFFT_TRANSPOSED,
} hcfftResTransposed;

typedef enum hcfftStatus_ {
  HCFFT_SUCCESS = 0,
  HCFFT_INVALID = -1,
  HCFFT_ERROR = -2
} hcfftStatus;

typedef enum hcfftGenerators_
{
	Stockham = 1,
	Transpose,
	Copy,
}hcfftGenerators;

//	The "envelope" is a set of limits imposed by the hardware
//	This will depend on the GPU(s) in the OpenCL context.
//	If there are multiple devices, this should be the least
//	common denominators.
//
struct FFTEnvelope {
	long       limit_LocalMemSize;
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

	hcfftResLocation   fft_placeness;
	hcfftIpLayout           fft_inputLayout;
	hcfftOpLayout           fft_outputLayout;
	hcfftPrecision        fft_precision;
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
  hcfftDim dimension;
  hcfftIpLayout ipLayout;
  hcfftOpLayout opLayout;
  hcfftDirection direction;
  hcfftResLocation location;
  hcfftResTransposed transposeType;
  hcfftPrecision precision;
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
  hcfftGenerators gen;

  //	Hardware Limits
  FFTEnvelope envelope;

  hcfftPlanHandle planX;
  hcfftPlanHandle planY;
  hcfftPlanHandle planZ;

  hcfftPlanHandle planTX;
  hcfftPlanHandle planTY;
  hcfftPlanHandle planTZ;

  hcfftPlanHandle planRCcopy;
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

  FFTPlan() : dimension (HCFFT_1D), ipLayout (HCFFT_COMPLEX),
              opLayout (HCFFT_COMPLEX), location (HCFFT_INPLACE),
              transposeType (HCFFT_NOTRANSPOSE), precision (HCFFT_SINGLE),
              batchSize (1), iDist(1), oDist(1), forwardScale (1.0), backwardScale (1.0),
              baked (false), gen(Stockham), planX(0), planY(0), planZ(0),
              planTX(0), planTY(0), planTZ(0), planRCcopy(0), bLdsComplex(false),
              uLdsFraction(0), ldsPadding(false), large1D_Xfactor(0), tmpBufSize(0),
	      intBuffer( NULL ), tmpBufSizeRC(0), intBufferRC(NULL), tmpBufSizeC2R(0),
	      intBufferC2R(NULL), transflag(false),large1D(0), large2D(false),
              const_buffer(NULL), RCsimple(false), direction(HCFFT_FORWARD)
  {};

  hcfftStatus hcfftCreateDefaultPlan(hcfftPlanHandle* plHandle,hcfftDim dimension, const size_t *length);

  hcfftStatus	hcfftBakePlan(hcfftPlanHandle plHandle);

  hcfftStatus hcfftDestroyPlan(hcfftPlanHandle* plHandle);

  hcfftStatus	hcfftEnqueueTransform(hcfftPlanHandle plHandle, hcfftDirection dir, Concurrency::array_view<float, 1> *inputBuffers,
                                       Concurrency::array_view<float, 1> *outputBuffers, Concurrency::array_view<float, 1> *tmpBuffer);

  hcfftStatus executePlan(FFTPlan*);

  hcfftStatus	hcfftGetPlanPrecision(const hcfftPlanHandle plHandle, hcfftPrecision* precision );

  hcfftStatus	hcfftSetPlanPrecision(hcfftPlanHandle plHandle, hcfftPrecision precision );

  hcfftStatus	hcfftGetPlanScale(const hcfftPlanHandle plHandle, hcfftDirection dir, float* scale );

  hcfftStatus	hcfftSetPlanScale(hcfftPlanHandle plHandle, hcfftDirection dir, float scale );

  hcfftStatus	hcfftGetPlanBatchSize(const hcfftPlanHandle plHandle, size_t* batchSize );

  hcfftStatus	hcfftSetPlanBatchSize(hcfftPlanHandle plHandle, size_t batchSize );

  hcfftStatus	hcfftGetPlanDim(const hcfftPlanHandle plHandle, hcfftDim* dim, int* size );

  hcfftStatus	hcfftSetPlanDim(hcfftPlanHandle plHandle, const hcfftDim dim );

  hcfftStatus	hcfftGetPlanLength(const hcfftPlanHandle plHandle, const hcfftDim dim, size_t* clLengths );

  hcfftStatus	hcfftSetPlanLength(hcfftPlanHandle plHandle, const hcfftDim dim, const size_t* clLengths );

  hcfftStatus	hcfftGetPlanInStride(const hcfftPlanHandle plHandle, const hcfftDim dim, size_t* clStrides );

  hcfftStatus	hcfftSetPlanInStride(hcfftPlanHandle plHandle, const hcfftDim dim, size_t* clStrides );

  hcfftStatus	hcfftGetPlanOutStride(const hcfftPlanHandle plHandle, const hcfftDim dim, size_t* clStrides );

  hcfftStatus	hcfftSetPlanOutStride(hcfftPlanHandle plHandle, const hcfftDim dim, size_t* clStrides );

  hcfftStatus	hcfftGetPlanDistance(const hcfftPlanHandle plHandle, size_t* iDist, size_t* oDist );

  hcfftStatus	hcfftSetPlanDistance(hcfftPlanHandle plHandle, size_t iDist, size_t oDist );

  hcfftStatus	hcfftGetLayout(const hcfftPlanHandle plHandle, hcfftIpLayout* iLayout, hcfftOpLayout* oLayout );

  hcfftStatus	hcfftSetLayout(hcfftPlanHandle plHandle, hcfftIpLayout iLayout, hcfftOpLayout oLayout );

  hcfftStatus	hcfftGetResultLocation(const hcfftPlanHandle plHandle, hcfftResLocation* placeness );

  hcfftStatus	hcfftSetResultLocation(hcfftPlanHandle plHandle, hcfftResLocation placeness );

  hcfftStatus	hcfftGetPlanTransposeResult(const hcfftPlanHandle plHandle, hcfftResTransposed * transposed );

  hcfftStatus	hcfftSetPlanTransposeResult(hcfftPlanHandle plHandle, hcfftResTransposed transposed );

  hcfftStatus GetEnvelope (const FFTEnvelope **) const;

  hcfftStatus SetEnvelope ();

  template <hcfftGenerators G>
  hcfftStatus GetMax1DLengthPvt (size_t *longest ) const;

  template <hcfftGenerators G>
  hcfftStatus GetKernelGenKeyPvt (FFTKernelGenKeyParams & params) const;

  template <hcfftGenerators G>
  hcfftStatus GetWorkSizesPvt (std::vector<size_t> & globalws, std::vector<size_t> & localws) const;

  template <hcfftGenerators G>
  hcfftStatus GenerateKernelPvt (const hcfftPlanHandle plHandle, FFTRepo& fftRepo) const;

  hcfftStatus GetMax1DLength (size_t *longest ) const;

  hcfftStatus GetKernelGenKey (FFTKernelGenKeyParams & params) const;

  hcfftStatus GetWorkSizes (std::vector<size_t> & globalws, std::vector<size_t> & localws) const;

  hcfftStatus GenerateKernel (const hcfftPlanHandle plHandle, FFTRepo & fftRepo) const;

  hcfftStatus AllocateWriteBuffers ();

  hcfftStatus ReleaseBuffers ();

  size_t ElementSize() const;
};


class FFTRepo
{
  //	All plans that the user creates over the course of using the library are stored here.
  //	Plans can be arbitrarily created and destroyed at anytime by the user, in arbitrary order, so vector
  //	does not seem appropriate, so a map was chosen because of the O(log N) search properties
  //	A lock object is created for each plan, such that any getter/setter can lock the 'plan' object before
  //	reading/writing its values.  The lock object is kept seperate from the plan object so that the lock
  //	object can be held the entire time a plan is getting destroyed in hcfftDestroyPlan.
  typedef pair< FFTPlan*, lockRAII* > repoPlansValue;
  typedef map< hcfftPlanHandle, repoPlansValue > repoPlansType;
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

  typedef std::pair< hcfftGenerators, hcfftPlanHandle> fftRepoKey;
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

  hcfftStatus createPlan( hcfftPlanHandle* plHandle, FFTPlan*& fftPlan );

  hcfftStatus getPlan( hcfftPlanHandle plHandle, FFTPlan*& fftPlan, lockRAII*& planLock );

  hcfftStatus deletePlan( hcfftPlanHandle* plHandle );

  hcfftStatus setProgramEntryPoints( const hcfftGenerators gen, const hcfftPlanHandle& handle, const FFTKernelGenKeyParams& fftParam, 
                                      const char * kernel_fwd, const char * kernel_back);

  hcfftStatus getProgramEntryPoint( const hcfftGenerators gen, const hcfftPlanHandle& handle, const FFTKernelGenKeyParams& fftParam, hcfftDirection dir, std::string& kernel);

  hcfftStatus setProgramCode( const hcfftGenerators gen, const hcfftPlanHandle& handle, const FFTKernelGenKeyParams&, const std::string& kernel);

  hcfftStatus getProgramCode( const hcfftGenerators gen, const hcfftPlanHandle& handle, const FFTKernelGenKeyParams&, std::string& kernel);

  hcfftStatus releaseResources( );

  ~FFTRepo( )
  {
    releaseResources();
  }
};

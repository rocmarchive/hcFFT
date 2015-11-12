#include <math.h>
#include <iomanip>

#include "stockham.h"

// A structure that represents a bounding box or tile, with convenient names for the row and column addresses
// local work sizes
struct tile
{
    union
    {
        size_t x;
        size_t col;
    };

    union
    {
        size_t y;
        size_t row;
    };
};

template<>
hcfftStatus FFTPlan::GetKernelGenKeyPvt<Transpose> (FFTKernelGenKeyParams & params) const
{
    params.fft_precision    = this->precision;
    params.fft_placeness    = this->location;
    params.fft_inputLayout  = this->ipLayout;
    params.fft_outputLayout = this->opLayout;
    params.fft_3StepTwiddle = false;

    params.fft_realSpecial  = this->realSpecial;

     params.transOutHorizontal = this->transOutHorizontal;	// using the twiddle front flag to specify horizontal write
								// we do this so as to reuse flags in FFTKernelGenKeyParams
								// and to avoid making a new one 

    ARG_CHECK( this->inStride.size( ) == this->outStride.size( ) );

    if( HCFFT_INPLACE == params.fft_placeness )
    {
        //	If this is an in-place transform the
        //	input and output layout, dimensions and strides
        //	*MUST* be the same.
        //
        ARG_CHECK( params.fft_inputLayout == params.fft_outputLayout )

        for( size_t u = this->inStride.size(); u-- > 0; )
        {
            ARG_CHECK( this->inStride[u] == this->outStride[u] );
        }
    }

	params.fft_DataDim = this->length.size() + 1;
	int i = 0;
	for(i = 0; i < (params.fft_DataDim - 1); i++)
	{
        params.fft_N[i]         = this->length[i];
        params.fft_inStride[i]  = this->inStride[i];
        params.fft_outStride[i] = this->outStride[i];

	}
    params.fft_inStride[i]  = this->iDist;
    params.fft_outStride[i] = this->oDist;

    if (this->large1D != 0) {
        ARG_CHECK (params.fft_N[0] != 0)
        ARG_CHECK ((this->large1D % params.fft_N[0]) == 0)
        params.fft_3StepTwiddle = true;
	ARG_CHECK ( this->large1D  == (params.fft_N[1] * params.fft_N[0]) );
    }

    //	Query the devices in this context for their local memory sizes
    //	How we generate a kernel depends on the *minimum* LDS size for all devices.
    //
    const FFTEnvelope * pEnvelope = NULL;
    this->GetEnvelope( &pEnvelope );
    BUG_CHECK( NULL != pEnvelope );

    // TODO:  Since I am going with a 2D workgroup size now, I need a better check than this 1D use
    // Check:  CL_DEVICE_MAX_WORK_GROUP_SIZE/CL_KERNEL_WORK_GROUP_SIZE
    // CL_DEVICE_MAX_WORK_ITEM_SIZES
    params.fft_R = 1; // Dont think i'll use
    params.fft_SIMD = pEnvelope->limit_WorkGroupSize; // Use devices maximum workgroup size

    return HCFFT_SUCCESS;
}

// Constants that specify the bounding sizes of the block that each workgroup will transpose
const tile lwSize = { {16}, {16} };
const size_t reShapeFactor = 4;   // wgTileSize = { lwSize.x * reShapeFactor, lwSize.y / reShapeFactor }
const size_t outRowPadding = 0;

// This is global, but should consider to be part of FFTPlan
size_t loopCount = 0;
tile blockSize = {{0}, {0}};

template<>
hcfftStatus FFTPlan::GetWorkSizesPvt<Transpose> (std::vector<size_t> & globalWS, std::vector<size_t> & localWS) const
{
    FFTKernelGenKeyParams fftParams;
    this->GetKernelGenKeyPvt<Transpose>( fftParams );

    // We need to make sure that the global work size is evenly divisible by the local work size
    // Our transpose works in tiles, so divide tiles in each dimension to get count of blocks, rounding up for remainder items

    size_t numBlocksX = fftParams.transOutHorizontal ?
			DivRoundingUp(fftParams.fft_N[ 1 ], blockSize.y ) :
			DivRoundingUp(fftParams.fft_N[ 0 ], blockSize.x );
    size_t numBlocksY = fftParams.transOutHorizontal ?
			DivRoundingUp( fftParams.fft_N[ 0 ], blockSize.x ) :
			DivRoundingUp( fftParams.fft_N[ 1 ], blockSize.y );
    size_t numWIX = numBlocksX * lwSize.x;

    // Batches of matrices are lined up along the Y axis, 1 after the other
    size_t numWIY = numBlocksY * lwSize.y * this->batchSize;
    // fft_DataDim has one more dimension than the actual fft data, which is devoted to batch.
    // dim from 2 to fft_DataDim - 2 are lined up along the Y axis
    for(int i = 2; i < fftParams.fft_DataDim - 1; i++)
    {
	numWIY *= fftParams.fft_N[i];
    }

    globalWS.clear( );
    globalWS.push_back( numWIX );
    globalWS.push_back( numWIY );

    localWS.clear( );
    localWS.push_back( lwSize.x );
    localWS.push_back( lwSize.y );

    return HCFFT_SUCCESS;
}

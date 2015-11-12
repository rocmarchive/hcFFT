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

inline std::stringstream& hcKernWrite( std::stringstream& rhs, const size_t tabIndex )
{
    rhs << std::setw( tabIndex ) << "";
    return rhs;
}

static void OffsetCalc(std::stringstream& transKernel, const FFTKernelGenKeyParams& params, bool input )
{
	const size_t *stride = input ? params.fft_inStride : params.fft_outStride;
	std::string offset = input ? "iOffset" : "oOffset";


	hcKernWrite( transKernel, 3 ) << "size_t " << offset << " = 0;" << std::endl;
	hcKernWrite( transKernel, 3 ) << "currDimIndex = groupIndex.y;" << std::endl;


	for(size_t i = params.fft_DataDim - 2; i > 0 ; i--)
	{
		hcKernWrite( transKernel, 3 ) << offset << " += (currDimIndex/numGroupsY_" << i << ")*" << stride[i+1] << ";" << std::endl;
		hcKernWrite( transKernel, 3 ) << "currDimIndex = currDimIndex % numGroupsY_" << i << ";" << std::endl;
	}

	hcKernWrite( transKernel, 3 ) << "rowSizeinUnits = " << stride[1] << ";" << std::endl;

	if(params.transOutHorizontal)
	{
		if(input)
		{
			hcKernWrite( transKernel, 3 ) << offset << " += rowSizeinUnits * wgTileExtent.y * wgUnroll * groupIndex.x;" << std::endl;
			hcKernWrite( transKernel, 3 ) << offset << " += currDimIndex * wgTileExtent.x;" << std::endl;  
		}
		else
		{
			hcKernWrite( transKernel, 3 ) << offset << " += rowSizeinUnits * wgTileExtent.x * currDimIndex;" << std::endl;
			hcKernWrite( transKernel, 3 ) << offset << " += groupIndex.x * wgTileExtent.y * wgUnroll;" << std::endl;
		}
	}
	else
	{
		if(input)
		{
			hcKernWrite( transKernel, 3 ) << offset << " += rowSizeinUnits * wgTileExtent.y * wgUnroll * currDimIndex;" << std::endl;
			hcKernWrite( transKernel, 3 ) << offset << " += groupIndex.x * wgTileExtent.x;" << std::endl;
		}
		else
		{
			hcKernWrite( transKernel, 3 ) << offset << " += rowSizeinUnits * wgTileExtent.x * groupIndex.x;" << std::endl;
			hcKernWrite( transKernel, 3 ) << offset << " += currDimIndex * wgTileExtent.y * wgUnroll;" << std::endl;  
		}
	}

	hcKernWrite( transKernel, 3 ) << std::endl;
}




// Small snippet of code that multiplies the twiddle factors into the butterfiles.  It is only emitted if the plan tells
// the generator that it wants the twiddle factors generated inside of the transpose
static hcfftStatus genTwiddleMath( const FFTKernelGenKeyParams& params, std::stringstream& transKernel, const std::string& dtComplex, bool fwd )
{
    hcKernWrite( transKernel, 9 ) << dtComplex << " W = TW3step( (groupIndex.x * wgTileExtent.x + xInd) * (currDimIndex * wgTileExtent.y * wgUnroll + yInd) );" << std::endl;
    hcKernWrite( transKernel, 9 ) << dtComplex << " T;" << std::endl;

	if(fwd)
	{
		hcKernWrite( transKernel, 9 ) << "T.x = ( W.x * tmp.x ) - ( W.y * tmp.y );" << std::endl;
		hcKernWrite( transKernel, 9 ) << "T.y = ( W.y * tmp.x ) + ( W.x * tmp.y );" << std::endl;
	}
	else
	{
		hcKernWrite( transKernel, 9 ) << "T.x =  ( W.x * tmp.x ) + ( W.y * tmp.y );" << std::endl;
		hcKernWrite( transKernel, 9 ) << "T.y = -( W.y * tmp.x ) + ( W.x * tmp.y );" << std::endl;
	}

    hcKernWrite( transKernel, 9 ) << "tmp.x = T.x;" << std::endl;
    hcKernWrite( transKernel, 9 ) << "tmp.y = T.y;" << std::endl;

    return HCFFT_SUCCESS;
}

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

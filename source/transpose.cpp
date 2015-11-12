#include <math.h>
#include <iomanip>

#include "stockham.h"

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

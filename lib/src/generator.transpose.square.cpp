#include "generator.transpose.h"

/*sqaure action*/
template<>
hcfftStatus FFTPlan::GetKernelGenKeyPvt<Transpose_SQUARE> (FFTKernelGenKeyParams & params) const {
	params.fft_precision = this->precision;
	params.fft_placeness = this->location;
	params.fft_inputLayout = this->ipLayout;
	params.fft_outputLayout = this->opLayout;
	params.fft_3StepTwiddle = false;

	params.fft_realSpecial = this->realSpecial;

	params.transOutHorizontal = this->transOutHorizontal;	// using the twiddle front flag to specify horizontal write
	// we do this so as to reuse flags in FFTKernelGenKeyParams
	// and to avoid making a new one 

	ARG_CHECK(this->inStride.size() == this->outStride.size());

	if (HCFFT_INPLACE == params.fft_placeness)
	{
		//	If this is an in-place transform the
		//	input and output layout, dimensions and strides
		//	*MUST* be the same.
		//
		ARG_CHECK(params.fft_inputLayout == params.fft_outputLayout)

			for (size_t u = this->inStride.size(); u-- > 0; )
			{
				ARG_CHECK(this->inStride[u] == this->outStride[u]);
			}
	}

	params.fft_DataDim = this->length.size() + 1;
	int i = 0;
	for (i = 0; i < (params.fft_DataDim - 1); i++)
	{
		params.fft_N[i] = this->length[i];
		params.fft_inStride[i] = this->inStride[i];
		params.fft_outStride[i] = this->outStride[i];

	}
	params.fft_inStride[i] = this->iDist;
	params.fft_outStride[i] = this->oDist;

	if (this->large1D != 0) {
		ARG_CHECK(params.fft_N[0] != 0)
			ARG_CHECK((this->large1D % params.fft_N[0]) == 0)
			params.fft_3StepTwiddle = true;
		ARG_CHECK(this->large1D == (params.fft_N[1] * params.fft_N[0]));
	}

	//	Query the devices in this context for their local memory sizes
	//	How we generate a kernel depends on the *minimum* LDS size for all devices.
	//
	const FFTEnvelope * pEnvelope = NULL;
	this->GetEnvelope(&pEnvelope);
	BUG_CHECK(NULL != pEnvelope);

	// TODO:  Since I am going with a 2D workgroup size now, I need a better check than this 1D use
	// Check:  CL_DEVICE_MAX_WORK_GROUP_SIZE/CL_KERNEL_WORK_GROUP_SIZE
	// CL_DEVICE_MAX_WORK_ITEM_SIZES
	params.fft_R = 1; // Dont think i'll use
	params.fft_SIMD = pEnvelope->limit_WorkGroupSize; // Use devices maximum workgroup size

	params.limit_LocalMemSize = this->envelope.limit_LocalMemSize;

	params.transposeMiniBatchSize = this->transposeMiniBatchSize;
	params.transposeBatchSize = this->batchSize;

	return HCFFT_SUCCEEDS;
}

static const size_t lwSize = 256;
static const size_t reShapeFactor = 2;

template<>
hcfftStatus FFTPlan::GetWorkSizesPvt<Transpose_SQUARE> (std::vector<size_t> & globalWS, std::vector<size_t> & localWS) const {
  FFTKernelGenKeyParams params;
  this->GetKernelGenKeyPvt<Transpose_SQUARE> (params);
	size_t wg_slice;
	if (params.fft_N[0] % (16 * reShapeFactor) == 0)
		wg_slice = params.fft_N[0] / 16 / reShapeFactor;
	else
		wg_slice = (params.fft_N[0] / (16 * reShapeFactor)) + 1;

	size_t global_item_size = wg_slice*(wg_slice + 1) / 2 * 16 * 16 * this->batchSize;

	for (int i = 2; i < params.fft_DataDim - 1; i++)
	{
		global_item_size *= params.fft_N[i];
	}

	globalWS.clear();
	globalWS.push_back(global_item_size);

	localWS.clear();
	localWS.push_back(lwSize);

	return HCFFT_SUCCEEDS;
}

//	OpenCL does not take unicode strings as input, so this routine returns only ASCII strings
//	Feed this generator the FFTPlan, and it returns the generated program as a string
template<>
hcfftStatus FFTPlan::GenerateKernelPvt<Transpose_SQUARE>(const hcfftPlanHandle plHandle, FFTRepo& fftRepo, size_t count, bool exist) const {
  FFTKernelGenKeyParams params;
  this->GetKernelGenKeyPvt<Transpose_SQUARE> (params);

  if(!exist)
  {
	  std::string programCode;
    std::vector< size_t > gWorkSize;
    std::vector< size_t > lWorkSize;
    this->GetWorkSizesPvt<Transpose_SQUARE> (gWorkSize, lWorkSize);

	  hcfft_transpose_generator::genTransposeKernelBatched((void**)&twiddleslarge, acc, plHandle, params, programCode, lwSize, reShapeFactor, gWorkSize, lWorkSize, count);
	  fftRepo.setProgramCode(Transpose_SQUARE, plHandle, params, programCode);

	  // Note:  See genFunctionPrototype( )
	  if (params.fft_3StepTwiddle)
	  {
		  fftRepo.setProgramEntryPoints(Transpose_SQUARE, plHandle, params, "transpose_square_tw_fwd", "transpose_square_tw_back");
	  }
	  else
	  {
		  fftRepo.setProgramEntryPoints(Transpose_SQUARE, plHandle, params, "transpose_square", "transpose_square");
	  }
  }
  else
  {
	  //  it is a better idea to do twiddle in swap kernel if we will have a swap kernel.
	  //  for pure square transpose, twiddle will be done in transpose kernel
	  bool twiddleTransposeKernel = params.fft_3StepTwiddle && (params.transposeMiniBatchSize == 1);//when transposeMiniBatchSize == 1 it is guaranteed to be a sqaure matrix transpose
	  //	If twiddle computation has been requested, generate the lookup function
	
	  if (twiddleTransposeKernel)
	  {
		  if (params.fft_precision == HCFFT_SINGLE)
      {
		    StockhamGenerator::TwiddleTableLarge<hc::short_vector::float_2, StockhamGenerator::P_SINGLE> twLarge(params.fft_N[0] * params.fft_N[1]);
        twLarge.TwiddleLargeAV((void**)&twiddleslarge, acc);
      }
		  else
      {
    		StockhamGenerator::TwiddleTableLarge<hc::short_vector::double_2, StockhamGenerator::P_DOUBLE> twLarge(params.fft_N[0] * params.fft_N[1]);
        twLarge.TwiddleLargeAV((void**)&twiddleslarge, acc);
      }
	  }
  }
	return HCFFT_SUCCEEDS;
}

#include <math.h>
#include <iomanip>
#include "generator.transpose.h"

// These strings represent the names that are used as strKernel parameters
const std::string pmRealIn("pmRealIn");
const std::string pmImagIn("pmImagIn");
const std::string pmRealOut("pmRealOut");
const std::string pmImagOut("pmImagOut");
const std::string pmComplexIn("pmComplexIn");
const std::string pmComplexOut("pmComplexOut");

template<>
hcfftStatus FFTPlan::GetKernelGenKeyPvt<Transpose_NONSQUARE> (FFTKernelGenKeyParams & params) const {
  params.fft_precision = this->precision;
  params.fft_placeness = this->location;
  params.fft_inputLayout = this->ipLayout;
  params.fft_outputLayout = this->opLayout;
  params.fft_3StepTwiddle = false;
  params.nonSquareKernelType = this->nonSquareKernelType;
  params.fft_realSpecial = this->realSpecial;
  params.transOutHorizontal = this->transOutHorizontal; // using the twiddle front flag to specify horizontal write
  // we do this so as to reuse flags in FFTKernelGenKeyParams
  // and to avoid making a new one
  ARG_CHECK(this->inStride.size() == this->outStride.size());

  if (HCFFT_INPLACE == params.fft_placeness) {
    //  If this is an in-place transform the
    //  input and output layout
    //  *MUST* be the same.
    //
    ARG_CHECK(params.fft_inputLayout == params.fft_outputLayout)
    /*        for (size_t u = this->inStride.size(); u-- > 0; )
            {
                ARG_CHECK(this->inStride[u] == this->outStride[u]);
            }*/
  }

  params.fft_DataDim = this->length.size() + 1;
  int i = 0;

  for (i = 0; i < (params.fft_DataDim - 1); i++) {
    params.fft_N[i] = this->length[i];
    params.fft_inStride[i] = this->inStride[i];
    params.fft_outStride[i] = this->outStride[i];
  }

  params.fft_inStride[i] = this->iDist;
  params.fft_outStride[i] = this->oDist;

  if (this->large1D != 0) {
    ARG_CHECK(params.fft_N[0] != 0)
    //ToDo:ENABLE ASSERT
    //     ARG_CHECK((this->large1D % params.fft_N[0]) == 0)
    params.fft_3StepTwiddle = true;
    //ToDo:ENABLE ASSERT
    // ARG_CHECK(this->large1D == (params.fft_N[1] * params.fft_N[0]));
  }

  //  Query the devices in this context for their local memory sizes
  //  How we generate a kernel depends on the *minimum* LDS size for all devices.
  //
  const FFTEnvelope* pEnvelope = NULL;
  this->GetEnvelope(&pEnvelope);
  BUG_CHECK(NULL != pEnvelope);
  // TODO:  Since I am going with a 2D workgroup size now, I need a better check than this 1D use
  // Check:  CL_DEVICE_MAX_WORK_GROUP_SIZE/CL_KERNEL_WORK_GROUP_SIZE
  // CL_DEVICE_MAX_WORK_ITEM_SIZES
  params.fft_R = 1; // Dont think i'll use
  params.fft_SIMD = pEnvelope->limit_WorkGroupSize; // Use devices maximum workgroup size
  params.limit_LocalMemSize = this->envelope.limit_LocalMemSize;
  params.transposeMiniBatchSize = this->transposeMiniBatchSize;
  params.nonSquareKernelOrder = this->nonSquareKernelOrder;
  params.transposeBatchSize = this->batchSize;
  return HCFFT_SUCCEEDS;
}


static const size_t lwSize = 256;
static const size_t reShapeFactor = 2;

template<>
hcfftStatus FFTPlan::GetWorkSizesPvt<Transpose_NONSQUARE> (std::vector<size_t> & globalWS, std::vector<size_t> & localWS) const {
  size_t wg_slice;
  FFTKernelGenKeyParams fftParams;
  //    Translate the user plan into the structure that we use to map plans to hcPrograms
  this->GetKernelGenKeyPvt<Transpose_NONSQUARE>( fftParams );
  size_t smaller_dim = (fftParams.fft_N[0] < fftParams.fft_N[1]) ? fftParams.fft_N[0] : fftParams.fft_N[1];
  size_t bigger_dim = (fftParams.fft_N[0] >= fftParams.fft_N[1]) ? fftParams.fft_N[0] : fftParams.fft_N[1];
  size_t dim_ratio = bigger_dim / smaller_dim;
  size_t global_item_size;

  if (fftParams.nonSquareKernelType == NON_SQUARE_TRANS_TRANSPOSE_BATCHED_LEADING) {
    if (smaller_dim % (16 * reShapeFactor) == 0) {
      wg_slice = smaller_dim / 16 / reShapeFactor;
    } else {
      wg_slice = (smaller_dim / (16 * reShapeFactor)) + 1;
    }

    global_item_size = wg_slice * (wg_slice + 1) / 2 * 16 * 16 * this->batchSize;

    for (int i = 2; i < fftParams.fft_DataDim - 1; i++) {
      global_item_size *= fftParams.fft_N[i];
    }

    /*Push the data required for the transpose kernels*/
    globalWS.clear();

    if(fftParams.nonSquareKernelType == NON_SQUARE_TRANS_TRANSPOSE_BATCHED_LEADING) {
      globalWS.push_back(global_item_size * dim_ratio);
    } else if (fftParams.nonSquareKernelType == NON_SQUARE_TRANS_TRANSPOSE_BATCHED) {
      globalWS.push_back(global_item_size);
    }

    localWS.clear();
    localWS.push_back(lwSize);
  } else if (fftParams.nonSquareKernelType == NON_SQUARE_TRANS_TRANSPOSE_BATCHED) {
    if (smaller_dim % (16 * reShapeFactor) == 0) {
      wg_slice = smaller_dim / 16 / reShapeFactor;
    } else {
      wg_slice = (smaller_dim / (16 * reShapeFactor)) + 1;
    }

    global_item_size = wg_slice * (wg_slice + 1) / 2 * 16 * 16 * this->batchSize;

    for (int i = 2; i < this->length.size(); i++) {
      global_item_size *= this->length[i];
    }

    /*Push the data required for the transpose kernels*/
    globalWS.clear();
    globalWS.push_back(global_item_size);
    localWS.clear();
    localWS.push_back(lwSize);
  } else {
    /*Now calculate the data for the swap kernels */
    // general swap kernel takes care of all ratio. need clean up here
    if(dim_ratio == 2 && 0) {
      //1:2 ratio
      size_t input_elm_size_in_bytes;

      switch (fftParams.fft_precision) {
        case HCFFT_SINGLE:
          input_elm_size_in_bytes = 4;
          break;

        case HCFFT_DOUBLE:
          input_elm_size_in_bytes = 8;
          break;

        default:
          return HCFFT_INVALID;
      }

      switch (fftParams.fft_outputLayout) {
        case HCFFT_COMPLEX_INTERLEAVED:
        case HCFFT_COMPLEX_PLANAR:
          input_elm_size_in_bytes *= 2;
          break;

        case HCFFT_REAL:
          break;

        default:
          return HCFFT_INVALID;
      }

      size_t max_elements_loaded = AVAIL_MEM_SIZE / input_elm_size_in_bytes;
      size_t num_elements_loaded;
      size_t local_work_size_swap, num_grps_pro_row;

      if ((max_elements_loaded >> 1) > smaller_dim) {
        local_work_size_swap = (smaller_dim < 256) ? smaller_dim : 256;
        num_elements_loaded = smaller_dim;
        num_grps_pro_row = 1;
      } else {
        num_grps_pro_row = (smaller_dim << 1) / max_elements_loaded;
        num_elements_loaded = max_elements_loaded >> 1;
        local_work_size_swap = (num_elements_loaded < 256) ? num_elements_loaded : 256;
      }

      size_t num_reduced_row;
      size_t num_reduced_col;

      if (fftParams.fft_N[1] == smaller_dim) {
        num_reduced_row = smaller_dim;
        num_reduced_col = 2;
      } else {
        num_reduced_row = 2;
        num_reduced_col = smaller_dim;
      }

      size_t* cycle_map = new size_t[num_reduced_row * num_reduced_col * 2];
      /* The memory required by cycle_map cannot exceed 2 times row*col by design*/
      hcfft_transpose_generator::get_cycles(cycle_map, num_reduced_row, num_reduced_col);
      global_item_size = local_work_size_swap * num_grps_pro_row * cycle_map[0] * this->batchSize;

      for (int i = 2; i < fftParams.fft_DataDim - 1; i++) {
        global_item_size *= fftParams.fft_N[i];
      }

      delete[] cycle_map;
      globalWS.push_back(global_item_size);
      localWS.push_back(local_work_size_swap);
    } else {
      //if (dim_ratio == 2 || dim_ratio == 3 || dim_ratio == 5 || dim_ratio == 10)
      if (dim_ratio % 2 == 0 || dim_ratio % 3 == 0 || dim_ratio % 5 == 0 || dim_ratio % 10 == 0) {
        size_t local_work_size_swap = 256;
        std::vector<std::vector<size_t> > permutationTable;
        hcfft_transpose_generator::permutation_calculation(dim_ratio, smaller_dim, permutationTable);
        size_t global_item_size;

        if(this->large1D && (dim_ratio > 1)) {
          global_item_size = (permutationTable.size() + 2) * local_work_size_swap * this->batchSize;
        } else {
          global_item_size = (permutationTable.size() + 2) * local_work_size_swap * this->batchSize;
        }

        //for (int i = 2; i < this->length.size(); i++)
        //  global_item_size *= this->length[i];
        size_t LDS_per_WG = smaller_dim;

        while (LDS_per_WG > 1024) { //avoiding using too much lds memory. the biggest LDS memory we will allocate would be 1024*sizeof(float2/double2)*2
          if (LDS_per_WG % 2 == 0) {
            LDS_per_WG /= 2;
            continue;
          }

          if (LDS_per_WG % 3 == 0) {
            LDS_per_WG /= 3;
            continue;
          }

          if (LDS_per_WG % 5 == 0) {
            LDS_per_WG /= 5;
            continue;
          }

          return HCFFT_INVALID;
        }

        size_t WG_per_line = smaller_dim / LDS_per_WG;
        global_item_size *= WG_per_line;
        globalWS.push_back(global_item_size);
        localWS.push_back(local_work_size_swap);
      } else {
        return HCFFT_INVALID;
      }
    }
  }

  return HCFFT_SUCCEEDS;
}

//  Feed this generator the FFTPlan, and it returns the generated program as a string

template<>
hcfftStatus FFTPlan::GenerateKernelPvt<Transpose_NONSQUARE>(const hcfftPlanHandle plHandle, FFTRepo& fftRepo, size_t count, bool exist) const {
  FFTKernelGenKeyParams params;
  this->GetKernelGenKeyPvt<Transpose_NONSQUARE> (params);

  if(!exist) {
    std::string programCode;
    std::string kernelFuncName;//applied to swap kernel for now
    std::vector< size_t > gWorkSize;
    std::vector< size_t > lWorkSize;
    this->GetWorkSizesPvt<Transpose_NONSQUARE> (gWorkSize, lWorkSize);

    if (params.nonSquareKernelType == NON_SQUARE_TRANS_TRANSPOSE_BATCHED_LEADING) {
      //Requested local memory size by callback must not exceed the device LDS limits after factoring the LDS size required by transpose kernel
      hcfft_transpose_generator::genTransposeKernelLeadingDimensionBatched((void**)&twiddleslarge, acc, plHandle, params, programCode, lwSize, reShapeFactor, gWorkSize, lWorkSize, count);
    } else if (params.nonSquareKernelType == NON_SQUARE_TRANS_TRANSPOSE_BATCHED) {
      hcfft_transpose_generator::genTransposeKernelBatched((void**)&twiddleslarge, acc, plHandle, params, programCode, lwSize, reShapeFactor, gWorkSize, lWorkSize, count);
    } else {
      //general swap kernel takes care of all ratio
      hcfft_transpose_generator::genSwapKernelGeneral((void**)&twiddleslarge, acc, plHandle, params, programCode, kernelFuncName, lwSize, reShapeFactor, gWorkSize, lWorkSize, count);
    }

    fftRepo.setProgramCode(Transpose_NONSQUARE, plHandle, params, programCode);

    if (params.nonSquareKernelType == NON_SQUARE_TRANS_TRANSPOSE_BATCHED_LEADING) {
      // Note:  See genFunctionPrototype( )
      if (params.fft_3StepTwiddle) {
        fftRepo.setProgramEntryPoints(Transpose_NONSQUARE, plHandle, params, "transpose_nonsquare_tw_fwd", "transpose_nonsquare_tw_back");
      } else {
        fftRepo.setProgramEntryPoints(Transpose_NONSQUARE, plHandle, params, "transpose_nonsquare", "transpose_nonsquare");
      }
    } else if(params.nonSquareKernelType == NON_SQUARE_TRANS_TRANSPOSE_BATCHED) {
      fftRepo.setProgramEntryPoints(Transpose_NONSQUARE, plHandle, params, "transpose_square", "transpose_square");
    } else {
      if (params.fft_3StepTwiddle) { //if miniBatchSize > 1 twiddling is done in swap kernel
        std::string kernelFwdFuncName = kernelFuncName + "_tw_fwd";
        std::string kernelBwdFuncName = kernelFuncName + "_tw_back";
        fftRepo.setProgramEntryPoints(Transpose_NONSQUARE, plHandle, params, kernelFwdFuncName.c_str(), kernelBwdFuncName.c_str());
      } else {
        fftRepo.setProgramEntryPoints(Transpose_NONSQUARE, plHandle, params, kernelFuncName.c_str(), kernelFuncName.c_str());
      }
    }
  } else {
    if (params.nonSquareKernelType == NON_SQUARE_TRANS_TRANSPOSE_BATCHED_LEADING) {
      //  If twiddle computation has been requested, generate the lookup function
      if (params.fft_3StepTwiddle) {
        if (params.fft_precision == HCFFT_SINGLE) {
          StockhamGenerator::TwiddleTableLarge<hc::short_vector::float_2, StockhamGenerator::P_SINGLE> twLarge(params.fft_N[0] * params.fft_N[1]);
          twLarge.TwiddleLargeAV((void**)&twiddleslarge, acc);
        } else {
          StockhamGenerator::TwiddleTableLarge<hc::short_vector::double_2, StockhamGenerator::P_DOUBLE> twLarge(params.fft_N[0] * params.fft_N[1]);
          twLarge.TwiddleLargeAV((void**)&twiddleslarge, acc);
        }
      }
    } else if (params.nonSquareKernelType == NON_SQUARE_TRANS_TRANSPOSE_BATCHED) {
      //  it is a better idea to do twiddle in swap kernel if we will have a swap kernel.
      //  for pure square transpose, twiddle will be done in transpose kernel
      bool twiddleTransposeKernel = params.fft_3StepTwiddle && (params.transposeMiniBatchSize == 1);//when transposeMiniBatchSize == 1 it is guaranteed to be a sqaure matrix transpose
      //  If twiddle computation has been requested, generate the lookup function

      if (twiddleTransposeKernel) {
        if (params.fft_precision == HCFFT_SINGLE) {
          StockhamGenerator::TwiddleTableLarge<hc::short_vector::float_2, StockhamGenerator::P_SINGLE> twLarge(params.fft_N[0] * params.fft_N[1]);
          twLarge.TwiddleLargeAV((void**)&twiddleslarge, acc);
        } else {
          StockhamGenerator::TwiddleTableLarge<hc::short_vector::double_2, StockhamGenerator::P_DOUBLE> twLarge(params.fft_N[0] * params.fft_N[1]);
          twLarge.TwiddleLargeAV((void**)&twiddleslarge, acc);
        }
      }
    } else {
      size_t smaller_dim = (params.fft_N[0] < params.fft_N[1]) ? params.fft_N[0] : params.fft_N[1];
      size_t bigger_dim = (params.fft_N[0] >= params.fft_N[1]) ? params.fft_N[0] : params.fft_N[1];
      size_t dim_ratio = bigger_dim / smaller_dim;

      if(dim_ratio % 2 != 0 && dim_ratio % 3 != 0 && dim_ratio % 5 != 0 && dim_ratio % 10 != 0) {
        return HCFFT_INVALID;
      }

      //twiddle in swap kernel (for now, swap with twiddle seems to always be the second kernel after transpose)
      bool twiddleSwapKernel = params.fft_3StepTwiddle && (dim_ratio > 1);

      //twiddle in swap kernel
      //twiddle in or out should be using the same twiddling table
      if (twiddleSwapKernel) {
        if (params.fft_precision == HCFFT_SINGLE) {
          StockhamGenerator::TwiddleTableLarge<hc::short_vector::float_2, StockhamGenerator::P_SINGLE> twLarge(smaller_dim * smaller_dim * dim_ratio);
          twLarge.TwiddleLargeAV((void**)&twiddleslarge, acc);
        } else {
          StockhamGenerator::TwiddleTableLarge<hc::short_vector::double_2, StockhamGenerator::P_DOUBLE> twLarge(smaller_dim * smaller_dim * dim_ratio);
          twLarge.TwiddleLargeAV((void**)&twiddleslarge, acc);
        }
      }
    }
  }

  return HCFFT_SUCCEEDS;
}

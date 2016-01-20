#include <math.h>
#include <list>
#include "stockham.h"

using namespace StockhamGenerator;

namespace CopyGenerator {
// Copy kernel
template <Precision PR>
class CopyKernel {
  size_t N;
  size_t Nt;
  const FFTKernelGenKeyParams params;
  bool h2c, c2h;
  bool general;

  inline std::string OffsetCalc(const std::string &off, bool input = true) {
    std::string str;
    const size_t* pStride = input ? params.fft_inStride : params.fft_outStride;
    std::string batch = "batch";

    switch(params.fft_DataDim) {
      case 5: {
          str += "\t{\n\tuint ocalc1 = ";
          str += batch;
          str += "%";
          str += SztToStr(params.fft_N[1] * params.fft_N[2] * params.fft_N[3]);
          str += ";\n";
          str += "\tuint ocalc0 = ";
          str += "ocalc1";
          str += "%";
          str += SztToStr(params.fft_N[1] * params.fft_N[2]);
          str += ";\n";
          str += "\t";
          str += off;
          str += " = ";
          str += "(";
          str += batch;
          str += "/";
          str += SztToStr(params.fft_N[1] * params.fft_N[2] * params.fft_N[3]);
          str += ")*";
          str += SztToStr(pStride[4]);
          str += " + ";
          str += "(ocalc1";
          str += "/";
          str += SztToStr(params.fft_N[1] * params.fft_N[2]);
          str += ")*";
          str += SztToStr(pStride[3]);
          str += " + ";
          str += "(ocalc0";
          str += "/";
          str += SztToStr(params.fft_N[1]);
          str += ")*";
          str += SztToStr(pStride[2]);
          str += " + ";
          str += "(ocalc0";
          str += "%";
          str += SztToStr(params.fft_N[1]);
          str += ")*";
          str += SztToStr(pStride[1]);
          str += ";\n";
          str += "\t}\n";
        }
        break;

      case 4: {
          str += "\t{\n\tuint ocalc0 = ";
          str += batch;
          str += "%";
          str += SztToStr(params.fft_N[1] * params.fft_N[2]);
          str += ";\n";
          str += "\t";
          str += off;
          str += " = ";
          str += "(";
          str += batch;
          str += "/";
          str += SztToStr(params.fft_N[1] * params.fft_N[2]);
          str += ")*";
          str += SztToStr(pStride[3]);
          str += " + ";
          str += "(ocalc0";
          str += "/";
          str += SztToStr(params.fft_N[1]);
          str += ")*";
          str += SztToStr(pStride[2]);
          str += " + ";
          str += "(ocalc0";
          str += "%";
          str += SztToStr(params.fft_N[1]);
          str += ")*";
          str += SztToStr(pStride[1]);
          str += ";\n";
          str += "\t}\n";
        }
        break;

      case 3: {
          str += "\t";
          str += off;
          str += " = ";
          str += "(";
          str += batch;
          str += "/";
          str += SztToStr(params.fft_N[1]);
          str += ")*";
          str += SztToStr(pStride[2]);
          str += " + ";
          str += "(";
          str += batch;
          str += "%";
          str += SztToStr(params.fft_N[1]);
          str += ")*";
          str += SztToStr(pStride[1]);
          str += ";\n";
        }
        break;

      case 2: {
          str += "\t";
          str += off;
          str += " = ";
          str += batch;
          str += "*";
          str += SztToStr(pStride[1]);
          str += ";\n";
        }
        break;

      default:
        assert(false);
    }

    return str;
  }

 public:
  CopyKernel( const FFTKernelGenKeyParams &paramsVal) :
    params(paramsVal)

  {
    N = params.fft_N[0];
    Nt = 1 + N / 2;
    h2c = ( (params.fft_inputLayout == HCFFT_HERMITIAN_PLANAR) ||
            (params.fft_inputLayout == HCFFT_HERMITIAN_INTERLEAVED) ) ? true : false;
    c2h = ( (params.fft_outputLayout == HCFFT_HERMITIAN_PLANAR) ||
            (params.fft_outputLayout == HCFFT_HERMITIAN_INTERLEAVED) ) ? true : false;
    general = !(h2c || c2h);
    // We only do out-of-place copies at this point
    assert(params.fft_placeness == HCFFT_OUTOFPLACE);
  }

  void GenerateKernel(const hcfftPlanHandle plHandle, std::string &str, vector< size_t > gWorkSize, vector< size_t > lWorkSize, size_t count) {
    std::string rType  = RegBaseType<PR>(1);
    std::string r2Type  = RegBaseType<PR>(2);
    bool inIlvd; // Input is interleaved format
    bool outIlvd; // Output is interleaved format
    inIlvd  = ((params.fft_inputLayout == HCFFT_COMPLEX_INTERLEAVED) ||
               (params.fft_inputLayout == HCFFT_HERMITIAN_INTERLEAVED) ) ? true : false;
    outIlvd = ((params.fft_outputLayout == HCFFT_COMPLEX_INTERLEAVED) ||
               (params.fft_outputLayout == HCFFT_HERMITIAN_INTERLEAVED) ) ? true : false;
    // Pragma
    std::string sfx = FloatSuffix<PR>();
    // Copy kernel begin
    str += "extern \"C\"\n { void ";

    // Function name
    if(general) {
      str += "copy_general";
    } else {
      if(h2c) {
        str += "copy_h2c";
      } else {
        str += "copy_c2h";
      }
    }

    str += SztToStr(count);
    str += "(std::map<int, void*> vectArr, accelerator &acc)";
    str += "{\n\t";
    int arg = 0;

    if(inIlvd) {
      str += r2Type;
      str += " *gbIn = static_cast<";
      str += r2Type;
      str += "*> (vectArr[";
      str += SztToStr(arg);
      str += "]);\n";
      arg++;
    } else {
      str += rType;
      str += " *gbInRe = static_cast<";
      str += rType;
      str += "*> (vectArr[";
      str += SztToStr(arg);
      str += "]);\n";
      arg++;
      str += rType;
      str += " *gbInIm = static_cast";
      str += rType;
      str += "*> (vectArr[";
      str += SztToStr(arg);
      str += "]);\n";
      arg++;
    }

    if(outIlvd) {
      str += r2Type;
      str += " *gbOut = static_cast<";
      str += r2Type;
      str += "*> (vectArr[";
      str += SztToStr(arg);
      str += "]);\n";
      arg++;
    } else {
      str += rType;
      str += " *gbInRe = static_cast<";
      str += rType;
      str += "*> (vectArr[";
      str += SztToStr(arg);
      str += "]);\n";
      arg++;
      str += rType;
      str += " *gbOutIm = static_cast<";
      str += rType;
      str += "*> (vectArr[";
      str += SztToStr(arg);
      str += "]);\n";
      arg++;
    }

    str += "\thc::extent<2> grdExt( ";
    str += SztToStr(gWorkSize[0]);
    str += ", 1 ); \n";
    str += "\thc::tiled_extent<2> t_ext = grdExt.tile( ";
    str += SztToStr(lWorkSize[0]);
    str += ", 1);\n";
    str += "\thc::parallel_for_each(t_ext, [=] (hc::tiled_index<2> tidx) [[hc]]\n\t {";

    // Initialize
    if(general) {
      str += "\tuint me = tidx.local[0];\n\t";
      str += "\tuint batch = tidx.tile[0];\n\t";
    } else {
      str += "\tuint me = tidx.global[0];\n\t";
    }

    // Declare memory pointers
    str += "\n\t";
    str += "uint iOffset;\n\t";
    str += "uint oOffset;\n\t";

    // input
    if(inIlvd) {
      str += "uint inOffset;\n\t";
    } else {
      str += "uint inReOffset;\n\t";
      str += "uint inImOffset;\n\t";
    }

    // output
    if(outIlvd) {
      str += "uint outOffset;\n\t";

      if(h2c) {
        str += "\t";
        str += "uint outOffset2;\n\t";
      }
    } else {
      str += "uint outReOffset;\n\t";
      str += "uint outImOffset;\n\t";

      if(h2c) {
        str += "\t";
        str += "uint outReOffset2;\n\t";
        str += "uint outImOffset2;\n\t";
      }
    }

    // Setup registers
    str += "\t";
    str += RegBaseType<PR>(2);
    str += " R;\n\n";
    size_t NtRounded64 = DivRoundingUp<size_t>(Nt, 64) * 64;

    if(!general) {
      // Setup variables
      str += "\tuint batch, meg, mel, mel2;\n\t";
      str += "batch = me/";
      str += SztToStr(NtRounded64);
      str += ";\n\t";
      str += "meg = me%";
      str += SztToStr(NtRounded64);
      str += ";\n\t";
      str += "mel = me%";
      str += SztToStr(Nt);
      str += ";\n\t";
      str += "mel2 = (";
      str += SztToStr(N);
      str += " - mel)%";
      str += SztToStr(N);
      str += ";\n\n";
    }

    // Setup memory pointers
    str += OffsetCalc("iOffset", true);
    str += OffsetCalc("oOffset", false);
    // offset strings
    std::string inF, inF2, outF, outF2;

    if(general) {
      inF = inF2 = outF = outF2 = "";
    } else {
      inF   = "(mel*";
      inF   += SztToStr(params.fft_inStride[0]);
      inF   += ")";
      inF2  = "(mel2*";
      inF2  += SztToStr(params.fft_inStride[0]);
      inF2  += ")";
      outF  = "(mel*";
      outF  += SztToStr(params.fft_outStride[0]);
      outF  += ")";
      outF2 = "(mel2*";
      outF2 += SztToStr(params.fft_outStride[0]);
      outF2 += ")";
    }

    str += "\n\t";

    // inputs
    if(inIlvd) {
      str += "inOffset = iOffset + ";
      str += inF;
      str += ";\n\t";
    } else {
      str += "inReOffset = iOffset + ";
      str += inF;
      str += ";\n\t";
      str += "inImOffset = iOffset + ";
      str += inF;
      str += ";\n\t";
    }

    // outputs
    if(outIlvd) {
      str += "outOffset = oOffset + ";
      str += outF;
      str += ";\n";

      if(h2c) {
        str += "\t";
        str += "outOffset2 = oOffset + ";
        str += outF2;
        str += ";\n";
      }
    } else {
      str += "outReOffset = oOffset + ";
      str += outF;
      str += ";\n\t";
      str += "outImOffset = oOffset + ";
      str += outF;
      str += ";\n";

      if(h2c) {
        str += "\t";
        str += "outReOffset2 = oOffset + ";
        str += outF2;
        str += ";\n\t";
        str += "outImOffset2 = oOffset + ";
        str += outF2;
        str += ";\n";
      }
    }

    str += "\n\t";

    // Do the copy
    if(general) {
      str += "for(uint t=0; t<";
      str += SztToStr(N / 64);
      str += "; t++)\n\t{\n\t\t";

      if(inIlvd) {
        str += "R = gbIn[inOffset + me + t*64];\n\t\t";
      } else {
        str += "R.x = gbInRe[inReOffset + me + t*64];\n\t\t";
        str += "R.y = gbInIm[inImOffset + me + t*64];\n\t\t";
      }

      if(outIlvd) {
        str += "gbOut[outOffset + me + t*64] = R;\n";
      } else {
        str += "gbOutRe[outReOffset + me + t*64] = R.x;\n\t\t";
        str += "gbOutIm[outImOffset + me + t*64] = R.y;\n";
      }

      str += "\t}\n\n";
    } else {
      str += "if(meg < ";
      str += SztToStr(Nt);
      str += ")\n\t{\n\t";

      if(c2h) {
        if(inIlvd) {
          str += "R = gbIn[inOffset];\n\t";
        } else {
          str += "R.x = gbInRe[inReOffset];\n\t";
          str += "R.y = gbInIm[inImOffset];\n\t";
        }

        if(outIlvd) {
          str += "gbOut[outOffset] = R;\n\n";
        } else {
          str += "gbOutRe[outReOffset] = R.x;\n\t";
          str += "gbOutIm[outImOffset] = R.y;\n\t";
        }
      } else {
        if(inIlvd) {
          str += "R = gbIn[inOffset];\n\t";
        } else {
          str += "R.x = gbInRe[inReOffset];\n\t";
          str += "R.y = gbInIm[inImOffset];\n\t";
        }

        if(outIlvd) {
          str += "gbOut[outOffset] = R;\n\t";
          str += "R.y = -R.y;\n\t";
          str += "gbOut[outOffset2] = R;\n\n";
        } else {
          str += "gbOutRe[outReOffset] = R.x;\n\t";
          str += "gbOutIm[outImOffset] = R.y;\n\t";
          str += "R.y = -R.y;\n\t";
          str += "gbOutRe[outReOffset2] = R.x;\n\t";
          str += "gbOutIm[outImOffset2] = R.y;\n\n";
        }
      }

      str += "}\n\n";
    }

    str += " }).wait();\n}}\n\n";
  }
};
};

template<>
hcfftStatus FFTPlan::GetKernelGenKeyPvt<Copy> (FFTKernelGenKeyParams & params) const {
  //    Query the devices in this context for their local memory sizes
  //    How we generate a kernel depends on the *minimum* LDS size for all devices.
  //
  const FFTEnvelope* pEnvelope = NULL;
  const_cast<FFTPlan*>(this)->GetEnvelope (& pEnvelope);
  BUG_CHECK (NULL != pEnvelope);
  ::memset( &params, 0, sizeof( params ) );
  params.fft_precision    = this->precision;
  params.fft_placeness    = this->location;
  params.fft_inputLayout  = this->ipLayout;
  params.fft_MaxWorkGroupSize = this->envelope.limit_WorkGroupSize;
  ARG_CHECK (this->inStride.size() == this->outStride.size())
  params.fft_outputLayout = this->opLayout;
  params.fft_DataDim = this->length.size() + 1;
  int i = 0;

  for(i = 0; i < (params.fft_DataDim - 1); i++) {
    params.fft_N[i]         = this->length[i];
    params.fft_inStride[i]  = this->inStride[i];
    params.fft_outStride[i] = this->outStride[i];
  }

  params.fft_inStride[i]  = this->iDist;
  params.fft_outStride[i] = this->oDist;
  params.fft_fwdScale  = this->forwardScale;
  params.fft_backScale = this->backwardScale;
  return HCFFT_SUCCESS;
}

template<>
hcfftStatus FFTPlan::GetWorkSizesPvt<Copy> (std::vector<size_t> & globalWS, std::vector<size_t> & localWS) const {
  FFTKernelGenKeyParams fftParams;
  this->GetKernelGenKeyPvt<Copy>( fftParams );
  bool h2c, c2h;
  h2c = ((fftParams.fft_inputLayout == HCFFT_HERMITIAN_PLANAR) || (fftParams.fft_inputLayout == HCFFT_HERMITIAN_INTERLEAVED) );
  c2h = ((fftParams.fft_outputLayout == HCFFT_HERMITIAN_PLANAR) || (fftParams.fft_outputLayout == HCFFT_HERMITIAN_INTERLEAVED) );
  bool general = !(h2c || c2h);
  size_t count = this->batchSize;

  switch(fftParams.fft_DataDim) {
    case 5:
      assert(false);

    case 4:
      count *= fftParams.fft_N[2];

    case 3:
      count *= fftParams.fft_N[1];

    case 2: {
        if(general) {
          count *= 64;
        } else {
          count *= (DivRoundingUp<size_t>((1 + fftParams.fft_N[0] / 2), 64) * 64);
        }
      }
      break;

    case 1:
      assert(false);
  }

  globalWS.push_back( count );
  localWS.push_back( 64 );
  return    HCFFT_SUCCESS;
}

template<>
hcfftStatus FFTPlan::GetMax1DLengthPvt<Copy> (size_t* longest) const {
  return FFTPlan::GetMax1DLengthPvt<Stockham>(longest);
}

using namespace CopyGenerator;

template<>
hcfftStatus FFTPlan::GenerateKernelPvt<Copy>(const hcfftPlanHandle plHandle, FFTRepo& fftRepo, size_t count) const {
  FFTKernelGenKeyParams params;
  this->GetKernelGenKeyPvt<Copy> (params);
  vector< size_t > gWorkSize;
  vector< size_t > lWorkSize;
  this->GetWorkSizesPvt<Copy> (gWorkSize, lWorkSize);
  bool h2c, c2h;
  h2c = ( (params.fft_inputLayout == HCFFT_HERMITIAN_PLANAR) || (params.fft_inputLayout == HCFFT_HERMITIAN_INTERLEAVED) );
  c2h = ( (params.fft_outputLayout == HCFFT_HERMITIAN_PLANAR) || (params.fft_outputLayout == HCFFT_HERMITIAN_INTERLEAVED) );
  bool general = !(h2c || c2h);
  std::string programCode;
  programCode = hcHeader();
  Precision pr = (params.fft_precision == HCFFT_SINGLE) ? P_SINGLE : P_DOUBLE;

  switch(pr) {
    case P_SINGLE: {
        CopyKernel<P_SINGLE> kernel(params);
        kernel.GenerateKernel(plHandle, programCode, gWorkSize, lWorkSize, count);
      }
      break;

    case P_DOUBLE: {
        CopyKernel<P_DOUBLE> kernel(params);
        kernel.GenerateKernel(plHandle, programCode, gWorkSize, lWorkSize, count);
      }
      break;
  }

  fftRepo.setProgramCode( Copy, plHandle, params, programCode);

  if( general) {
    fftRepo.setProgramEntryPoints( Copy, plHandle, params, "copy_general", "copy_general");
  } else {
    fftRepo.setProgramEntryPoints( Copy, plHandle, params, "copy_c2h", "copy_h2c");
  }

  return HCFFT_SUCCESS;
}

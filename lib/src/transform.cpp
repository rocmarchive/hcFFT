#include <dlfcn.h>
#include "hcfftlib.h"
#include <unistd.h>

//  Static initialization of the repo lock variable
lockRAII FFTRepo::lockRepo( _T( "FFTRepo" ) );

//  Static initialization of the plan count variable
size_t FFTPlan::count = 0;
size_t FFTRepo::planCount = 1;
static size_t beforeCompile = 99999999;
static size_t beforeTransform = 99999999;
static std::string kernellib;
static std::string filename;
static bool exist = false;
static size_t countKernel;
static std::vector<size_t> originalLength;

bool has_suffix(const string& s, const string& suffix) {
  return (s.size() >= suffix.size()) && equal(suffix.rbegin(), suffix.rend(), s.rbegin());
}

bool checkIfsoExist(hcfftDirection direction) {
  DIR*           d;
  struct dirent* dir;
  d = opendir("/tmp/");

  if (d) {
    while ((dir = readdir(d)) != NULL) {
      if(has_suffix(dir->d_name, ".so")) {
        string libFile(dir->d_name);

        if(libFile.substr(0, 9) != "libkernel") {
          continue;
        }

        int i = 0;
        size_t length = libFile.length();
        size_t firstocc = libFile.find_first_of("_");
        string type = libFile.substr(9, firstocc - 9);

        if(!((direction == HCFFT_FORWARD && type == "Fwd") || (direction == HCFFT_BACKWARD && type == "Back"))) {
          continue;
        }

        if( firstocc != std::string::npos) {
          ++firstocc;
          size_t iter = (libFile.substr(firstocc, length - firstocc)).find("_");

          while( iter != std::string::npos) {
            size_t N = (size_t)stoi(libFile.substr(firstocc, iter));

            if(N != originalLength[i]) {
              break;
            }

            i++;
            firstocc  += iter + 1;
            iter = (libFile.substr(firstocc, length - firstocc )).find("_");
          }
        }

        if( i == originalLength.size()) {
          return true;
        }
      }
    }

    if(closedir(d) < 0) {
      return false;
    }
  }

  return false;
}

/*----------------------------------------------------FFTPlan-----------------------------------------------------------------------------*/

//  Read the kernels that this plan uses from file, and store into the plan
hcfftStatus WriteKernel( const hcfftPlanHandle plHandle, const hcfftGenerators gen, const FFTKernelGenKeyParams& fftParams, string filename, bool writeFlag) {
  FFTRepo& fftRepo  = FFTRepo::getInstance( );
  std::string kernel;
  fftRepo.getProgramCode( gen, plHandle, fftParams, kernel);
  FILE* fp;

  if(writeFlag) {
    fp = fopen (filename.c_str(), "w");

    if (!fp) {
      std::cout << " File kernel.cpp open failed for writing " << std::endl;
      return HCFFT_ERROR;
    }
  } else {
    fp = fopen (filename.c_str(), "a+");

    if (!fp) {
      std::cout << " File kernel.cpp open failed for writing " << std::endl;
      return HCFFT_ERROR;
    }
  }

  size_t written = fwrite(kernel.c_str(), kernel.size(), 1, fp);

  if(!written) {
    std::cout << "Kernel Write Failed " << std::endl;
    exit(1);
  }

  fflush(fp);
  fclose(fp);
  return  HCFFT_SUCCESS;
}

//  Compile the kernels that this plan uses, and store into the plan
hcfftStatus CompileKernels(const hcfftPlanHandle plHandle, const hcfftGenerators gen, FFTPlan* fftPlan, hcfftPlanHandle plHandleOrigin, bool exist) {
  FFTRepo& fftRepo  = FFTRepo::getInstance( );
  FFTKernelGenKeyParams fftParams;
  fftPlan->GetKernelGenKey( fftParams );
  // For real transforms we comppile either forward or backward kernel
  bool r2c_transform = (fftParams.fft_inputLayout == HCFFT_REAL);
  bool c2r_transform = (fftParams.fft_outputLayout == HCFFT_REAL);
  bool real_transform = (r2c_transform || c2r_transform);
  bool h2c = ((fftParams.fft_inputLayout == HCFFT_HERMITIAN_PLANAR) || (fftParams.fft_inputLayout == HCFFT_HERMITIAN_INTERLEAVED));
  bool c2h = ((fftParams.fft_outputLayout == HCFFT_HERMITIAN_PLANAR) || (fftParams.fft_outputLayout == HCFFT_HERMITIAN_INTERLEAVED));
  bool buildFwdKernel = (gen == Stockham || gen == Transpose) ? ((!real_transform) || r2c_transform) : (r2c_transform || c2h) || (!(h2c || c2h));
  bool buildBwdKernel = (gen == Stockham || gen == Transpose) ? ((!real_transform) || c2r_transform) : (c2r_transform || h2c) || (!(h2c || c2h));
  bool writeFlag = false;
  string type;

  if(buildFwdKernel) {
    type = "Fwd_";
  }

  if (buildBwdKernel) {
    type = "Back_";
  }

  if(beforeCompile != plHandleOrigin) {
    filename = "/tmp/kernel";
    kernellib = "/tmp/libkernel";
    filename += type;
    kernellib += type;

    for(int i = 0; i < (fftParams.fft_DataDim - 1); i++) {
      filename += SztToStr(originalLength[i]);
      kernellib += SztToStr(originalLength[i]);
      filename += "_";
      kernellib += "_";
    }

    filename += ".cpp";
    kernellib += ".so";
    beforeCompile = plHandleOrigin;
    writeFlag = true;
  }

  if(!exist) {
    WriteKernel( plHandle, gen, fftParams, filename, writeFlag);

    // Check if the default compiler path exists
    std::string execCmd = NULL; 
    char fname[256] = "/opt/hcc/bin/clang++";
    if( access( fname, F_OK ) != -1 ) {
      // compiler exists
      // install_mode = true;
      string Path = "/opt/hcc/bin/";
      execCmd = Path + "/clang++ `" + Path + "/clamp-config --install --cxxflags --ldflags --shared` " + filename + " -o " + kernellib ;
    } 
    else if ( access ( getenv ("MCWHCCBUILD"), F_OK ) != -1) {
      // TODO: This path shall be removed. User shall build from default path
      // compiler doesn't exist in default path
      // check if user has specified compiler build path
      // build_mode = true;
      char* compilerPath = getenv ("MCWHCCBUILD");
      std::string Path(compilerPath);
      std::string execCmd = Path + "/compiler/bin/clang++ `" + Path + "/build/Release/bin/clamp-config --build --cxxflags --ldflags --shared` " + filename + " -o " + kernellib ;
    }
    else {
      // No compiler found
      std::cout << "HCC compiler not found" << std::endl;
      exit(1);
    }
    system(execCmd.c_str());
  }

  // get a kernel object handle for a kernel with the given name
  if(buildFwdKernel) {
    std::string entryPoint;
    fftRepo.getProgramEntryPoint( gen, plHandle, fftParams, HCFFT_FORWARD, entryPoint);
  }

  if(buildBwdKernel) {
    std::string entryPoint;
    fftRepo.getProgramEntryPoint( gen, plHandle, fftParams, HCFFT_BACKWARD, entryPoint);
  }

  return HCFFT_SUCCESS;
}

//  This routine will query the OpenCL context for it's devices
//  and their hardware limitations, which we synthesize into a
//  hardware "envelope".
//  We only query the devices the first time we're called after
//  the object's context is set.  On 2nd and subsequent calls,
//  we just return the pointer.
//
hcfftStatus FFTPlan::SetEnvelope () {
  // TODO  The caller has already acquired the lock on *this
  //  However, we shouldn't depend on it.
  envelope.limit_LocalMemSize = 32768;
  envelope.limit_WorkGroupSize = 256;
  envelope.limit_Dimensions = 3;

  for(int i = 0 ; i < envelope.limit_Dimensions; i++) {
    envelope.limit_Size[i] = 256;
  }

  return HCFFT_SUCCESS;
}

hcfftStatus FFTPlan::GetEnvelope (const FFTEnvelope** ppEnvelope) const {
  *ppEnvelope = &envelope;
  return HCFFT_SUCCESS;
}

hcfftStatus hcfftCreateDefaultPlanInternal (hcfftPlanHandle* plHandle, hcfftDim dimension, const size_t* length) {
  if( length == NULL ) {
    return HCFFT_ERROR;
  }

  size_t lenX = 1, lenY = 1, lenZ = 1;

  switch( dimension ) {
    case HCFFT_1D: {
        if( length[ 0 ] == 0 ) {
          return HCFFT_ERROR;
        }

        lenX = length[ 0 ];
      }
      break;

    case HCFFT_2D: {
        if( length[ 0 ] == 0 || length[ 1 ] == 0 ) {
          return HCFFT_ERROR;
        }

        lenX = length[ 0 ];
        lenY = length[ 1 ];
      }
      break;

    case HCFFT_3D: {
        if( length[ 0 ] == 0 || length[ 1 ] == 0 || length[ 2 ] == 0 ) {
          return HCFFT_ERROR;
        }

        lenX = length[ 0 ];
        lenY = length[ 1 ];
        lenZ = length[ 2 ];
      }
      break;

    default:
      return HCFFT_ERROR;
      break;
  }

  FFTPlan* fftPlan = NULL;
  FFTRepo& fftRepo = FFTRepo::getInstance( );
  fftRepo.createPlan( plHandle, fftPlan );
  fftPlan->baked = false;
  fftPlan->dimension = dimension;
  fftPlan->location = HCFFT_INPLACE;
  fftPlan->ipLayout = HCFFT_COMPLEX_INTERLEAVED;
  fftPlan->opLayout = HCFFT_COMPLEX_INTERLEAVED;
  fftPlan->precision = HCFFT_SINGLE;
  fftPlan->forwardScale = 1.0;
  fftPlan->backwardScale = 1.0 / static_cast< double >( lenX * lenY * lenZ );
  fftPlan->batchSize = 1;
  fftPlan->gen = Stockham; //default setting
  fftPlan->SetEnvelope();
  //  Need to devise a way to generate better names
  std::stringstream tstream;
  tstream << _T( "plan_" ) << *plHandle;
  lockRAII* planLock  = NULL;
  fftRepo.getPlan( *plHandle, fftPlan, planLock );
  planLock->setName( tstream.str( ) );

  switch( dimension ) {
    case HCFFT_1D: {
        fftPlan->length.push_back( lenX );
        fftPlan->inStride.push_back( 1 );
        fftPlan->outStride.push_back( 1 );
        fftPlan->iDist    = lenX;
        fftPlan->oDist    = lenX;
      }
      break;

    case HCFFT_2D: {
        fftPlan->length.push_back( lenX );
        fftPlan->length.push_back( lenY );
        fftPlan->inStride.push_back( 1 );
        fftPlan->inStride.push_back( lenX );
        fftPlan->outStride.push_back( 1 );
        fftPlan->outStride.push_back( lenX );
        fftPlan->iDist    = lenX * lenY;
        fftPlan->oDist    = lenX * lenY;
      }
      break;

    case HCFFT_3D: {
        fftPlan->length.push_back( lenX );
        fftPlan->length.push_back( lenY );
        fftPlan->length.push_back( lenZ );
        fftPlan->inStride.push_back( 1 );
        fftPlan->inStride.push_back( lenX );
        fftPlan->inStride.push_back( lenX * lenY );
        fftPlan->outStride.push_back( 1 );
        fftPlan->outStride.push_back( lenX );
        fftPlan->outStride.push_back( lenX * lenY );
        fftPlan->iDist    = lenX * lenY * lenZ;
        fftPlan->oDist    = lenX * lenY * lenZ;
      }
      break;
  }

  fftPlan->plHandle = *plHandle;
  return HCFFT_SUCCESS;
}

// This external entry-point should not be called from within the library. Use clfftCreateDefaultPlanInternal instead.
hcfftStatus FFTPlan::hcfftCreateDefaultPlan( hcfftPlanHandle* plHandle, const hcfftDim dim,
    const size_t* clLengths, hcfftDirection dir) {
  hcfftStatus ret = hcfftCreateDefaultPlanInternal(plHandle, dim, clLengths);
  originalLength.clear();

  for(int i = 0 ; i < dim ; i++) {
    originalLength.push_back(clLengths[i]);
  }

  if(ret == HCFFT_SUCCESS) {
    FFTRepo& fftRepo  = FFTRepo::getInstance( );
    FFTPlan* fftPlan = NULL;
    lockRAII* planLock  = NULL;
    fftRepo.getPlan( *plHandle, fftPlan, planLock );
    fftPlan->count = 0;
    fftPlan->plHandleOrigin = *plHandle;
    fftPlan->userPlan = true;
    exist = checkIfsoExist(dir);
  }

  return ret;
}

hcfftStatus FFTPlan::hcfftEnqueueTransform(hcfftPlanHandle plHandle, hcfftDirection dir, Concurrency::array_view<float, 1>* clInputBuffers,
    Concurrency::array_view<float, 1>* clOutputBuffers, Concurrency::array_view<float, 1>* clTmpBuffers) {
  hcfftStatus status = HCFFT_SUCCESS;
  std::map<int, void*> vectArr;
  FFTRepo& fftRepo  = FFTRepo::getInstance( );
  FFTPlan* fftPlan  = NULL;
  lockRAII* planLock  = NULL;
  //  At this point, the user wants to enqueue a plan to execute.  We lock the plan down now, such that
  //  after we finish baking the plan (if the user did not do that explicitely before), the plan cannot
  //  change again through the action of other thread before we enqueue this plan for execution.
  fftRepo.getPlan( plHandle, fftPlan, planLock );
  scopedLock sLock( *planLock, _T( "hcfftGetPlanBatchSize" ) );

  if( fftPlan->baked == false ) {
    hcfftBakePlan( plHandle);
  }

  if (fftPlan->ipLayout == HCFFT_REAL) {
    dir = HCFFT_FORWARD;
  } else if (fftPlan->opLayout == HCFFT_REAL) {
    dir = HCFFT_BACKWARD;
  }

  // we do not check the user provided buffer at this release
  Concurrency::array_view<float, 1>* localIntBuffer = clTmpBuffers;

  if( clTmpBuffers == NULL && fftPlan->tmpBufSize > 0 && fftPlan->intBuffer == NULL) {
    // create the intermediate buffers
    // The intermediate buffer is always interleave and packed
    // For outofplace operation, we have the choice not to create intermediate buffer
    // input ->(col+Transpose) output ->(col) output
    float* init = (float*)calloc(fftPlan->tmpBufSize / sizeof(float), sizeof(float));
    Concurrency::array<float, 1> arr = Concurrency::array<float, 1>(Concurrency::extent<1>(fftPlan->tmpBufSize / sizeof(float)), init);
    fftPlan->intBuffer = new Concurrency::array_view<float>(arr);
    free(init);
  }

  if( localIntBuffer == NULL && fftPlan->intBuffer != NULL ) {
    localIntBuffer = fftPlan->intBuffer;
  }

  if( fftPlan->intBufferRC == NULL && fftPlan->tmpBufSizeRC > 0 ) {
    float* init = (float*)calloc(fftPlan->tmpBufSizeRC / sizeof(float), sizeof(float));
    Concurrency::array<float, 1> arr = Concurrency::array<float, 1>(Concurrency::extent<1>(fftPlan->tmpBufSizeRC / sizeof(float)), init);
    fftPlan->intBufferRC = new Concurrency::array_view<float>(arr);
    free(init);
  }

  if( fftPlan->intBufferC2R == NULL && fftPlan->tmpBufSizeC2R > 0 ) {
    float* init = (float*)calloc(fftPlan->tmpBufSizeC2R / sizeof(float), sizeof(float));
    Concurrency::array<float, 1> arr = Concurrency::array<float, 1>(Concurrency::extent<1>(fftPlan->tmpBufSizeC2R / sizeof(float)), init);
    fftPlan->intBufferC2R = new Concurrency::array_view<float>(arr);
    free(init);
  }

  //  The largest vector we can transform in a single pass
  //  depends on the GPU caps -- especially the amount of LDS
  //  available
  //
  size_t Large1DThreshold = 0;
  fftPlan->GetMax1DLength (&Large1DThreshold);
  BUG_CHECK (Large1DThreshold > 1);

  if(fftPlan->gen != Copy)
    switch( fftPlan->dimension ) {
      case HCFFT_1D: {
          if (fftPlan->length[0] <= Large1DThreshold) {
            break;
          }

          if(  ( fftPlan->ipLayout == HCFFT_REAL ) && ( fftPlan->planTZ != 0) ) {
            //First transpose
            // Input->tmp
            hcfftEnqueueTransform( fftPlan->planTX, dir, clInputBuffers, localIntBuffer, NULL);
            Concurrency::array_view<float, 1>* mybuffers;

            if (fftPlan->location == HCFFT_INPLACE) {
              mybuffers = clInputBuffers;
            } else {
              mybuffers = clOutputBuffers;
            }

            //First Row
            //tmp->output
            hcfftEnqueueTransform( fftPlan->planX, dir, localIntBuffer, fftPlan->intBufferRC, NULL );
            //Second Transpose
            // output->tmp
            hcfftEnqueueTransform( fftPlan->planTY, dir, fftPlan->intBufferRC, localIntBuffer, NULL );
            //Second Row
            //tmp->tmp, inplace
            hcfftEnqueueTransform( fftPlan->planY, dir, localIntBuffer, fftPlan->intBufferRC, NULL );
            //Third Transpose
            // tmp->output
            hcfftEnqueueTransform( fftPlan->planTZ, dir, fftPlan->intBufferRC, mybuffers, NULL );
          } else if ( fftPlan->ipLayout == HCFFT_REAL ) {
            // First pass
            // column with twiddle first, OUTOFPLACE, + transpose
            hcfftEnqueueTransform( fftPlan->planX, HCFFT_FORWARD, clInputBuffers, fftPlan->intBufferRC, localIntBuffer);
            // another column FFT output, INPLACE
            hcfftEnqueueTransform( fftPlan->planY, HCFFT_FORWARD, fftPlan->intBufferRC, fftPlan->intBufferRC, localIntBuffer );
            Concurrency::array_view<float, 1>* out_local;
            out_local = (fftPlan->location == HCFFT_INPLACE) ? clInputBuffers : clOutputBuffers;
            // copy from full complex to hermitian
            hcfftEnqueueTransform( fftPlan->planRCcopy, HCFFT_FORWARD, fftPlan->intBufferRC, out_local, localIntBuffer );
          } else if( fftPlan->opLayout == HCFFT_REAL ) {
            // copy from hermitian to full complex
            hcfftEnqueueTransform( fftPlan->planRCcopy, HCFFT_BACKWARD, clInputBuffers, fftPlan->intBufferRC, localIntBuffer );
            // First pass
            // column with twiddle first, INPLACE,
            hcfftEnqueueTransform( fftPlan->planX, HCFFT_BACKWARD, fftPlan->intBufferRC, fftPlan->intBufferRC, localIntBuffer);
            Concurrency::array_view<float, 1>* out_local;
            out_local = (fftPlan->location == HCFFT_INPLACE) ? clInputBuffers : clOutputBuffers;
            // another column FFT output, OUTOFPLACE + transpose
            hcfftEnqueueTransform( fftPlan->planY, HCFFT_BACKWARD, fftPlan->intBufferRC, out_local, localIntBuffer );
            return  HCFFT_SUCCESS;
          } else {
            if (fftPlan->transflag) {
              //First transpose
              // Input->tmp
              hcfftEnqueueTransform( fftPlan->planTX, dir, clInputBuffers, localIntBuffer, NULL );
              Concurrency::array_view<float, 1>* mybuffers;

              if (fftPlan->location == HCFFT_INPLACE) {
                mybuffers = clInputBuffers;
              } else {
                mybuffers = clOutputBuffers;
              }

              //First Row
              //tmp->output
              hcfftEnqueueTransform( fftPlan->planX, dir, localIntBuffer, mybuffers, NULL );
              //Second Transpose
              // output->tmp
              hcfftEnqueueTransform( fftPlan->planTY, dir, mybuffers, localIntBuffer, NULL );
              //Second Row
              //tmp->tmp, inplace
              hcfftEnqueueTransform( fftPlan->planY, dir, localIntBuffer, NULL, NULL );
              //Third Transpose
              // tmp->output
              hcfftEnqueueTransform( fftPlan->planTZ, dir, localIntBuffer, mybuffers, NULL );
              return  HCFFT_SUCCESS;
            } else {
              if (fftPlan->large1D == 0) {
                if(fftPlan->planCopy) {
                  // Transpose OUTOFPLACE
                  hcfftEnqueueTransform( fftPlan->planTX, dir, clInputBuffers, localIntBuffer, NULL ),
                                         // FFT INPLACE
                                         hcfftEnqueueTransform( fftPlan->planX, dir, localIntBuffer, NULL, NULL);
                  // FFT INPLACE
                  hcfftEnqueueTransform( fftPlan->planY, dir, localIntBuffer, NULL, NULL );
                  Concurrency::array_view<float, 1>* mybuffers;

                  if (fftPlan->location == HCFFT_INPLACE) {
                    mybuffers = clInputBuffers;
                  } else {
                    mybuffers = clOutputBuffers;
                  }

                  // Copy kernel
                  hcfftEnqueueTransform( fftPlan->planCopy, dir, localIntBuffer, mybuffers, NULL );
                } else {
                  // First pass
                  // column with twiddle first, OUTOFPLACE, + transpose
                  hcfftEnqueueTransform( fftPlan->planX, dir, clInputBuffers, localIntBuffer, localIntBuffer);

                  if(fftPlan->planTZ) {
                    hcfftEnqueueTransform( fftPlan->planY, dir, localIntBuffer, NULL, NULL );

                    if (fftPlan->location == HCFFT_INPLACE) {
                      hcfftEnqueueTransform( fftPlan->planTZ, dir, localIntBuffer, clInputBuffers, NULL );
                    } else {
                      hcfftEnqueueTransform( fftPlan->planTZ, dir, localIntBuffer, clOutputBuffers, NULL );
                    }
                  } else {
                    //another column FFT output, OUTOFPLACE
                    if (fftPlan->location == HCFFT_INPLACE) {
                      hcfftEnqueueTransform( fftPlan->planY, dir, localIntBuffer, clInputBuffers, localIntBuffer );
                    } else {
                      hcfftEnqueueTransform( fftPlan->planY, dir, localIntBuffer, clOutputBuffers, localIntBuffer );
                    }
                  }
                }
              } else {
                // second pass for huge 1D
                // column with twiddle first, OUTOFPLACE, + transpose
                hcfftEnqueueTransform( fftPlan->planX, dir, localIntBuffer, clOutputBuffers, localIntBuffer);
                hcfftEnqueueTransform( fftPlan->planY, dir, clOutputBuffers, clOutputBuffers, localIntBuffer );
              }
            }
          }

          return  status;
          break;
        }

      case HCFFT_2D: {
          // if transpose kernel, we will fall below
          if (fftPlan->transflag && !(fftPlan->planTX)) {
            break;
          }

          //cl_event rowOutEvents = NULL;

          if (fftPlan->transflag) {
            //first time set up transpose kernel for 2D
            //First row
            hcfftEnqueueTransform( fftPlan->planX, dir, clInputBuffers, clOutputBuffers, NULL );
            Concurrency::array_view<float, 1>* mybuffers;

            if (fftPlan->location == HCFFT_INPLACE) {
              mybuffers = clInputBuffers;
            } else {
              mybuffers = clOutputBuffers;
            }

            bool xyflag = (fftPlan->length[0] == fftPlan->length[1]) ? false : true;

            if (xyflag) {
              //First transpose
              hcfftEnqueueTransform( fftPlan->planTX, dir, mybuffers, localIntBuffer, NULL );

              if (fftPlan->transposeType == HCFFT_NOTRANSPOSE) {
                //Second Row transform
                hcfftEnqueueTransform( fftPlan->planY, dir, localIntBuffer, NULL, NULL );
                //Second transpose
                hcfftEnqueueTransform( fftPlan->planTY, dir, localIntBuffer, mybuffers, NULL );
              } else {
                //Second Row transform
                hcfftEnqueueTransform( fftPlan->planY, dir, localIntBuffer, mybuffers, NULL );
              }
            } else {
              // First Transpose
              hcfftEnqueueTransform( fftPlan->planTX, dir, mybuffers, NULL, NULL );

              if (fftPlan->transposeType == HCFFT_NOTRANSPOSE) {
                //Second Row transform
                hcfftEnqueueTransform( fftPlan->planY, dir, mybuffers, NULL, NULL );
                //Second transpose
                hcfftEnqueueTransform( fftPlan->planTY, dir, mybuffers, NULL, NULL );
              } else {
                //Second Row transform
                hcfftEnqueueTransform( fftPlan->planY, dir, mybuffers, NULL, NULL );
              }
            }

            return HCFFT_SUCCESS;
          } else {
            if ( (fftPlan->large2D || fftPlan->length.size() > 2) &&
                 (fftPlan->ipLayout != HCFFT_REAL) && (fftPlan->opLayout != HCFFT_REAL)) {
              if (fftPlan->location == HCFFT_INPLACE) {
                //deal with row first
                hcfftEnqueueTransform( fftPlan->planX, dir, clInputBuffers, NULL, localIntBuffer );
                //deal with column
                hcfftEnqueueTransform( fftPlan->planY, dir, clInputBuffers, NULL, localIntBuffer );
              } else {
                //deal with row first
                hcfftEnqueueTransform( fftPlan->planX, dir, clInputBuffers, clOutputBuffers, localIntBuffer );
                //deal with column
                hcfftEnqueueTransform( fftPlan->planY, dir, clOutputBuffers, NULL, localIntBuffer );
              }
            } else {
              if ( (fftPlan->large2D || fftPlan->length.size() > 2) &&
                   (fftPlan->ipLayout != HCFFT_REAL) && (fftPlan->opLayout != HCFFT_REAL)) {
                if (fftPlan->location == HCFFT_INPLACE) {
                  //deal with row first
                  hcfftEnqueueTransform( fftPlan->planX, dir, clInputBuffers, NULL, localIntBuffer );
                  //deal with column
                  hcfftEnqueueTransform( fftPlan->planY, dir, clInputBuffers, NULL, localIntBuffer );
                } else {
                  //deal with row first
                  hcfftEnqueueTransform( fftPlan->planX, dir, clInputBuffers, clOutputBuffers, localIntBuffer );
                  //deal with column
                  hcfftEnqueueTransform( fftPlan->planY, dir, clOutputBuffers, NULL, localIntBuffer );
                }
              } else {
                if(fftPlan->ipLayout == HCFFT_REAL) {
                  if(fftPlan->planTX) {
                    //First row
                    hcfftEnqueueTransform( fftPlan->planX, dir, clInputBuffers, clOutputBuffers, NULL );
                    Concurrency::array_view<float, 1>* mybuffers;

                    if (fftPlan->location == HCFFT_INPLACE) {
                      mybuffers = clInputBuffers;
                    } else {
                      mybuffers = clOutputBuffers;
                    }

                    //First transpose
                    hcfftEnqueueTransform( fftPlan->planTX, dir, mybuffers, localIntBuffer, NULL );
                    //Second Row transform
                    hcfftEnqueueTransform( fftPlan->planY, dir, localIntBuffer, NULL, NULL );
                    //Second transpose
                    hcfftEnqueueTransform( fftPlan->planTY, dir, localIntBuffer, mybuffers, NULL );
                  } else {
                    if (fftPlan->location == HCFFT_INPLACE) {
                      // deal with row
                      hcfftEnqueueTransform( fftPlan->planX, HCFFT_FORWARD, clInputBuffers, NULL, localIntBuffer );
                      // deal with column
                      hcfftEnqueueTransform( fftPlan->planY, HCFFT_FORWARD, clInputBuffers, NULL, localIntBuffer );
                    } else {
                      // deal with row
                      hcfftEnqueueTransform( fftPlan->planX, HCFFT_FORWARD, clInputBuffers, clOutputBuffers, localIntBuffer );
                      // deal with column
                      hcfftEnqueueTransform( fftPlan->planY, HCFFT_FORWARD, clOutputBuffers, NULL, localIntBuffer );
                    }
                  }
                } else if(fftPlan->opLayout == HCFFT_REAL) {
                  if(fftPlan->planTY) {
                    Concurrency::array_view<float, 1>* mybuffers;

                    if ( (fftPlan->location == HCFFT_INPLACE) ||
                         ((fftPlan->location == HCFFT_OUTOFPLACE) && (fftPlan->length.size() > 2)) ) {
                      mybuffers = clInputBuffers;
                    } else {
                      mybuffers = fftPlan->intBufferC2R;
                    }

                    //First transpose
                    hcfftEnqueueTransform( fftPlan->planTY, dir, clInputBuffers, localIntBuffer, NULL );
                    //First row
                    hcfftEnqueueTransform( fftPlan->planY, dir, localIntBuffer, NULL, NULL );
                    //Second transpose
                    hcfftEnqueueTransform( fftPlan->planTX, dir, localIntBuffer, mybuffers, NULL );

                    //Second Row transform
                    if(fftPlan->location == HCFFT_INPLACE) {
                      hcfftEnqueueTransform( fftPlan->planX, dir, clInputBuffers, NULL, NULL );
                    } else {
                      hcfftEnqueueTransform( fftPlan->planX, dir, mybuffers, clOutputBuffers, NULL );
                    }
                  } else {
                    Concurrency::array_view<float, 1>* out_local, *int_local, *out_y;

                    if(fftPlan->location == HCFFT_INPLACE) {
                      out_local = NULL;
                      int_local = NULL;
                      out_y = clInputBuffers;
                    } else {
                      if(fftPlan->length.size() > 2) {
                        out_local = clOutputBuffers;
                        int_local = NULL;
                        out_y = clInputBuffers;
                      } else {
                        out_local = clOutputBuffers;
                        int_local = fftPlan->intBufferC2R;
                        out_y = int_local;
                      }
                    }

                    // deal with column
                    hcfftEnqueueTransform( fftPlan->planY, HCFFT_BACKWARD, clInputBuffers, int_local, localIntBuffer );
                    // deal with row
                    hcfftEnqueueTransform( fftPlan->planX, HCFFT_BACKWARD, out_y, out_local, localIntBuffer );
                  }
                } else {
                  //deal with row first
                  hcfftEnqueueTransform( fftPlan->planX, dir, clInputBuffers, localIntBuffer, localIntBuffer );

                  if (fftPlan->location == HCFFT_INPLACE) {
                    //deal with column
                    hcfftEnqueueTransform( fftPlan->planY, dir, localIntBuffer, clInputBuffers, localIntBuffer );
                  } else {
                    //deal with column
                    hcfftEnqueueTransform( fftPlan->planY, dir, localIntBuffer, clOutputBuffers, localIntBuffer );
                  }
                }
              }
            }
          }

          return  status;
        }
        break;

      case HCFFT_3D: {
          return status;
          break;
        }
    }

  FFTKernelGenKeyParams fftParams;
  //  Translate the user plan into the structure that we use to map plans to clPrograms
  fftPlan->GetKernelGenKey( fftParams );
  std::string kernel;
  fftRepo.getProgramCode( fftPlan->gen, plHandle, fftParams, kernel);
  /* constant buffer */
  unsigned int uarg = 0;

  if (!fftPlan->transflag && !(fftPlan->gen == Copy)) {
    vectArr.insert(std::make_pair(uarg++, fftPlan->const_buffer));
  }

  //  Decode the relevant properties from the plan paramter to figure out how many input/output buffers we have
  switch( fftPlan->ipLayout ) {
    case HCFFT_COMPLEX_INTERLEAVED: {
        switch( fftPlan->opLayout ) {
          case HCFFT_COMPLEX_INTERLEAVED: {
              if( fftPlan->location == HCFFT_INPLACE ) {
                vectArr.insert(std::make_pair(uarg++, &(clInputBuffers[0])));
              } else {
                vectArr.insert(std::make_pair(uarg++, &(clInputBuffers[0])));
                vectArr.insert(std::make_pair(uarg++, &(clOutputBuffers[0])));
              }

              break;
            }

          case HCFFT_COMPLEX_PLANAR: {
              if( fftPlan->location == HCFFT_INPLACE ) {
                //  Invalid to be an inplace transform, and go from 1 to 2 buffers
                return HCFFT_ERROR;
              } else {
                vectArr.insert(std::make_pair(uarg++, &(clInputBuffers[0])));
                vectArr.insert(std::make_pair(uarg++, &(clOutputBuffers[0])));
                vectArr.insert(std::make_pair(uarg++, &(clOutputBuffers[1])));
              }

              break;
            }

          case HCFFT_HERMITIAN_INTERLEAVED: {
              if( fftPlan->location == HCFFT_INPLACE ) {
                return HCFFT_ERROR;
              } else {
                vectArr.insert(std::make_pair(uarg++, &(clInputBuffers[0])));
                vectArr.insert(std::make_pair(uarg++, &(clOutputBuffers[0])));
              }

              break;
            }

          case HCFFT_HERMITIAN_PLANAR: {
              if( fftPlan->location == HCFFT_INPLACE ) {
                return HCFFT_ERROR;
              } else {
                vectArr.insert(std::make_pair(uarg++, &(clInputBuffers[0])));
                vectArr.insert(std::make_pair(uarg++, &(clOutputBuffers[0])));
                vectArr.insert(std::make_pair(uarg++, &(clOutputBuffers[1])));
              }

              break;
            }

          case HCFFT_REAL: {
              if( fftPlan->location == HCFFT_INPLACE ) {
                vectArr.insert(std::make_pair(uarg++, &(clInputBuffers[0])));
              } else {
                vectArr.insert(std::make_pair(uarg++, &(clInputBuffers[0])));
                vectArr.insert(std::make_pair(uarg++, &(clOutputBuffers[0])));
              }

              break;
            }

          default: {
              //  Don't recognize output layout
              return HCFFT_ERROR;
            }
        }

        break;
      }

    case HCFFT_COMPLEX_PLANAR: {
        switch( fftPlan->opLayout ) {
          case HCFFT_COMPLEX_INTERLEAVED: {
              if( fftPlan->location == HCFFT_INPLACE ) {
                return HCFFT_ERROR;
              } else {
                vectArr.insert(std::make_pair(uarg++, &(clInputBuffers[0])));
                vectArr.insert(std::make_pair(uarg++, &(clInputBuffers[1])));
                vectArr.insert(std::make_pair(uarg++, &(clOutputBuffers[0])));
              }

              break;
            }

          case HCFFT_COMPLEX_PLANAR: {
              if( fftPlan->location == HCFFT_INPLACE ) {
                vectArr.insert(std::make_pair(uarg++, &(clInputBuffers[0])));
                vectArr.insert(std::make_pair(uarg++, &(clInputBuffers[1])));
              } else {
                vectArr.insert(std::make_pair(uarg++, &(clInputBuffers[0])));
                vectArr.insert(std::make_pair(uarg++, &(clInputBuffers[1])));
                vectArr.insert(std::make_pair(uarg++, &(clOutputBuffers[0])));
                vectArr.insert(std::make_pair(uarg++, &(clOutputBuffers[1])));
              }

              break;
            }

          case HCFFT_HERMITIAN_INTERLEAVED: {
              if( fftPlan->location == HCFFT_INPLACE ) {
                return HCFFT_ERROR;
              } else {
                vectArr.insert(std::make_pair(uarg++, &(clInputBuffers[0])));
                vectArr.insert(std::make_pair(uarg++, &(clInputBuffers[1])));
                vectArr.insert(std::make_pair(uarg++, &(clOutputBuffers[0])));
              }

              break;
            }

          case HCFFT_HERMITIAN_PLANAR: {
              if( fftPlan->location == HCFFT_INPLACE ) {
                return HCFFT_ERROR;
              } else {
                vectArr.insert(std::make_pair(uarg++, &(clInputBuffers[0])));
                vectArr.insert(std::make_pair(uarg++, &(clInputBuffers[1])));
                vectArr.insert(std::make_pair(uarg++, &(clOutputBuffers[0])));
                vectArr.insert(std::make_pair(uarg++, &(clOutputBuffers[1])));
              }

              break;
            }

          case HCFFT_REAL: {
              if(fftPlan->location == HCFFT_INPLACE ) {
                return HCFFT_ERROR;
              } else {
                vectArr.insert(std::make_pair(uarg++, &(clInputBuffers[0])));
                vectArr.insert(std::make_pair(uarg++, &(clInputBuffers[1])));
                vectArr.insert(std::make_pair(uarg++, &(clOutputBuffers[0])));
              }

              break;
            }

          default: {
//  Don't recognize output layout
              return HCFFT_ERROR;
            }
        }

        break;
      }

    case HCFFT_HERMITIAN_INTERLEAVED: {
        switch( fftPlan->opLayout ) {
          case HCFFT_COMPLEX_INTERLEAVED: {
              if( fftPlan->location == HCFFT_INPLACE ) {
                return HCFFT_ERROR;
              } else {
                vectArr.insert(std::make_pair(uarg++, &(clInputBuffers[0])));
                vectArr.insert(std::make_pair(uarg++, &(clOutputBuffers[0])));
              }

              break;
            }

          case HCFFT_COMPLEX_PLANAR: {
              if( fftPlan->location == HCFFT_INPLACE ) {
                return HCFFT_ERROR;
              } else {
                vectArr.insert(std::make_pair(uarg++, &(clInputBuffers[0])));
                vectArr.insert(std::make_pair(uarg++, &(clOutputBuffers[0])));
                vectArr.insert(std::make_pair(uarg++, &(clOutputBuffers[1])));
              }

              break;
            }

          case HCFFT_HERMITIAN_INTERLEAVED: {
              return HCFFT_ERROR;
            }

          case HCFFT_HERMITIAN_PLANAR: {
              return HCFFT_ERROR;
            }

          case HCFFT_REAL: {
              if( fftPlan->location == HCFFT_INPLACE ) {
                vectArr.insert(std::make_pair(uarg++, &(clInputBuffers[0])));
              } else {
                vectArr.insert(std::make_pair(uarg++, &(clInputBuffers[0])));
                vectArr.insert(std::make_pair(uarg++, &(clOutputBuffers[0])));
              }

              break;
            }

          default: {
              //  Don't recognize output layout
              return HCFFT_ERROR;
            }
        }

        break;
      }

    case HCFFT_HERMITIAN_PLANAR: {
        switch( fftPlan->opLayout ) {
          case HCFFT_COMPLEX_INTERLEAVED: {
              if( fftPlan->location == HCFFT_INPLACE ) {
                return HCFFT_ERROR;
              } else {
                vectArr.insert(std::make_pair(uarg++, &(clInputBuffers[0])));
                vectArr.insert(std::make_pair(uarg++, &(clInputBuffers[1])));
                vectArr.insert(std::make_pair(uarg++, &(clOutputBuffers[0])));
              }

              break;
            }

          case HCFFT_COMPLEX_PLANAR: {
              if( fftPlan->location == HCFFT_INPLACE ) {
                return HCFFT_ERROR;
              } else {
                vectArr.insert(std::make_pair(uarg++, &(clInputBuffers[0])));
                vectArr.insert(std::make_pair(uarg++, &(clInputBuffers[1])));
                vectArr.insert(std::make_pair(uarg++, &(clOutputBuffers[0])));
                vectArr.insert(std::make_pair(uarg++, &(clOutputBuffers[1])));
              }

              break;
            }

          case HCFFT_HERMITIAN_INTERLEAVED: {
              return HCFFT_ERROR;
            }

          case HCFFT_HERMITIAN_PLANAR: {
              return HCFFT_ERROR;
            }

          case HCFFT_REAL: {
              if( fftPlan->location == HCFFT_INPLACE ) {
                return HCFFT_ERROR;
              } else {
                vectArr.insert(std::make_pair(uarg++, &(clInputBuffers[0])));
                vectArr.insert(std::make_pair(uarg++, &(clInputBuffers[1])));
                vectArr.insert(std::make_pair(uarg++, &(clOutputBuffers[0])));
              }

              break;
            }

          default: {
              //  Don't recognize output layout
              return HCFFT_ERROR;
            }
        }

        break;
      }

    case HCFFT_REAL: {
        switch( fftPlan->opLayout ) {
          case HCFFT_COMPLEX_INTERLEAVED: {
              if( fftPlan->location == HCFFT_INPLACE ) {
                vectArr.insert(std::make_pair(uarg++, &(clInputBuffers[0])));
              } else {
                vectArr.insert(std::make_pair(uarg++, &(clInputBuffers[0])));
                vectArr.insert(std::make_pair(uarg++, &(clOutputBuffers[0])));
              }

              break;
            }

          case HCFFT_COMPLEX_PLANAR: {
              if( fftPlan->location == HCFFT_INPLACE ) {
                return HCFFT_ERROR;
              } else {
                vectArr.insert(std::make_pair(uarg++, &(clInputBuffers[0])));
                vectArr.insert(std::make_pair(uarg++, &(clOutputBuffers[0])));
                vectArr.insert(std::make_pair(uarg++, &(clOutputBuffers[1])));
              }

              break;
            }

          case HCFFT_HERMITIAN_INTERLEAVED: {
              if( fftPlan->location == HCFFT_INPLACE ) {
                vectArr.insert(std::make_pair(uarg++, &(clInputBuffers[0])));
              } else {
                vectArr.insert(std::make_pair(uarg++, &(clInputBuffers[0])));
                vectArr.insert(std::make_pair(uarg++, &(clOutputBuffers[0])));
              }

              break;
            }

          case HCFFT_HERMITIAN_PLANAR: {
              if( fftPlan->location == HCFFT_INPLACE ) {
                return HCFFT_ERROR;
              } else {
                vectArr.insert(std::make_pair(uarg++, &(clInputBuffers[0])));
                vectArr.insert(std::make_pair(uarg++, &(clOutputBuffers[0])));
                vectArr.insert(std::make_pair(uarg++, &(clOutputBuffers[1])));
              }

              break;
            }

          default: {
              if(fftPlan->transflag) {
                if( fftPlan->location == HCFFT_INPLACE ) {
                  return HCFFT_ERROR;
                } else {
                  vectArr.insert(std::make_pair(uarg++, &(clInputBuffers[0])));
                  vectArr.insert(std::make_pair(uarg++, &(clOutputBuffers[0])));
                }
              } else {
                //  Don't recognize output layout
                return HCFFT_ERROR;
              }
            }
        }

        break;
      }

    default: {
        //  Don't recognize output layout
        return HCFFT_ERROR;
      }
  }

  vector< size_t > gWorkSize;
  vector< size_t > lWorkSize;
  hcfftStatus result = fftPlan->GetWorkSizes (gWorkSize, lWorkSize);

  if (HCFFT_ERROR == result) {
    std::cout << "Work size too large for clEnqueNDRangeKernel()" << std::endl;
  }

  BUG_CHECK (gWorkSize.size() == lWorkSize.size());
  remove(filename.c_str());
  void* kernelHandle = NULL;
  typedef void (FUNC_FFTFwd)(std::map<int, void*>* vectArr);
  FUNC_FFTFwd* FFTcall = NULL;
  char cwd[1024];

  if (getcwd(cwd, sizeof(cwd)) == NULL) {
    std::cout << "getcwd() error" << std::endl;
  }

  std::string pwd(cwd);
  char* err = (char*) calloc(128, 2);
  kernelHandle = dlopen(kernellib.c_str(), RTLD_NOW);

  if(!kernelHandle) {
    std::cout << "Failed to load Kernel: " << kernellib.c_str() << std::endl;
    return HCFFT_ERROR;
  }

  if(beforeTransform != fftPlan->plHandleOrigin) {
    countKernel = 0;
    beforeTransform = fftPlan->plHandleOrigin;
  } else {
    countKernel++;
  }

  if(fftPlan->gen == Copy) {
    bool h2c = ((fftPlan->ipLayout == HCFFT_HERMITIAN_PLANAR) ||
                (fftPlan->ipLayout == HCFFT_HERMITIAN_INTERLEAVED) ) ? true : false;
    std::string funcName = "copy_";

    if(h2c) {
      funcName += "h2c";
    } else {
      funcName += "c2h";
    }

    funcName +=  std::to_string(countKernel);
    FFTcall = (FUNC_FFTFwd*) dlsym(kernelHandle, funcName.c_str());

    if (!FFTcall) {
      std::cout << "Loading copy() fails " << std::endl;
    }

    err = dlerror();

    if (err) {
      std::cout << "failed to locate copy(): " << err;
      exit(1);
    }
  } else if(fftPlan->gen == Stockham) {
    if(dir == HCFFT_FORWARD) {
      std::string funcName = "fft_fwd";
      funcName +=  std::to_string(countKernel);
      FFTcall = (FUNC_FFTFwd*) dlsym(kernelHandle, funcName.c_str());

      if (!FFTcall) {
        std::cout << "Loading fft_fwd fails " << std::endl;
      }

      err = dlerror();

      if (err) {
        std::cout << "failed to locate fft_fwd(): " << err;
        exit(1);
      }
    } else if(dir == HCFFT_BACKWARD) {
      std::string funcName = "fft_back";
      funcName +=  std::to_string(countKernel);
      FFTcall = (FUNC_FFTFwd*) dlsym(kernelHandle, funcName.c_str());

      if (!FFTcall) {
        std::cout << "Loading fft_back fails " << std::endl;
      }

      err = dlerror();

      if (err) {
        std::cout << "failed to locate fft_back(): " << err;
        exit(1);
      }
    }
  } else if(fftPlan->gen == Transpose) {
    std::string funcName = "transpose";
    funcName +=  std::to_string(countKernel);
    FFTcall = (FUNC_FFTFwd*) dlsym(kernelHandle, funcName.c_str());

    if (!FFTcall) {
      std::cout << "Loading transpose fails " << std::endl;
    }

    err = dlerror();

    if (err) {
      std::cout << "failed to locate transpose(): " << err;
      exit(1);
    }
  }

  FFTcall(&vectArr);
  dlclose(kernelHandle);
  kernelHandle = NULL;
  return status;
}

hcfftStatus FFTPlan::hcfftBakePlan(hcfftPlanHandle plHandle) {
  FFTRepo& fftRepo = FFTRepo::getInstance( );
  FFTPlan* fftPlan = NULL;
  lockRAII* planLock = NULL;
  fftRepo.getPlan(plHandle, fftPlan, planLock);
  scopedLock sLock( *planLock, _T( "hcfftBakePlan" ) );

  // if we have already baked the plan and nothing has changed since, we're done here
  if( fftPlan->baked == true ) {
    return HCFFT_SUCCESS;
  }

  //find product of lengths
  size_t maxLengthInAnyDim = 1;

  switch(fftPlan->dimension) {
    case HCFFT_3D:
      maxLengthInAnyDim = maxLengthInAnyDim > fftPlan->length[2] ? maxLengthInAnyDim : fftPlan->length[2];

    case HCFFT_2D:
      maxLengthInAnyDim = maxLengthInAnyDim > fftPlan->length[1] ? maxLengthInAnyDim : fftPlan->length[1];

    case HCFFT_1D:
      maxLengthInAnyDim = maxLengthInAnyDim > fftPlan->length[0] ? maxLengthInAnyDim : fftPlan->length[0];
  }

  bool rc = (fftPlan->ipLayout == HCFFT_REAL) || (fftPlan->opLayout == HCFFT_REAL);
  // upper bounds on transfrom lengths - address this in the next release
  size_t SP_MAX_LEN = 1 << 24;
  size_t DP_MAX_LEN = 1 << 22;

  if((fftPlan->precision == HCFFT_SINGLE) && (maxLengthInAnyDim > SP_MAX_LEN) && rc) {
    return HCFFT_INVALID;
  }

  if((fftPlan->precision == HCFFT_DOUBLE) && (maxLengthInAnyDim > DP_MAX_LEN) && rc) {
    return HCFFT_INVALID;
  }

  // release buffers, as these will be created only in EnqueueTransform
  if( NULL != fftPlan->intBuffer ) {
    delete fftPlan->intBuffer;
  }

  if( NULL != fftPlan->intBufferRC ) {
    delete fftPlan->intBufferRC;
  }

  if( NULL != fftPlan->intBufferC2R ) {
    delete fftPlan->intBufferC2R;
  }

  if( fftPlan->userPlan ) { // confirm it is top-level plan (user plan)
    if(fftPlan->location == HCFFT_INPLACE) {
      if( (fftPlan->ipLayout == HCFFT_HERMITIAN_PLANAR) || (fftPlan->opLayout == HCFFT_HERMITIAN_PLANAR) ) {
        return HCFFT_INVALID;
      }
    }

    // Make sure strides & distance are same for C-C transforms
    if(fftPlan->location == HCFFT_INPLACE) {
      if( (fftPlan->ipLayout != HCFFT_REAL) && (fftPlan->opLayout != HCFFT_REAL) ) {
        // check strides
        for(size_t i = 0; i < fftPlan->dimension; i++)
          if(fftPlan->inStride[i] != fftPlan->outStride[i]) {
            return HCFFT_INVALID;
          }

        // check distance
        if(fftPlan->iDist != fftPlan->oDist) {
          return HCFFT_INVALID;
        }
      }
    }
  }

  if(fftPlan->gen == Copy) {
    if(!exist) {
      fftPlan->GenerateKernel(plHandle, fftRepo, count);
      count++;
    }

    CompileKernels(plHandle, fftPlan->gen, fftPlan, plHandleOrigin, exist);
    fftPlan->baked = true;
    return HCFFT_SUCCESS;
  }

  // Compress the plan by discarding length '1' dimensions
  // decision to pick generator
  if( fftPlan->userPlan && !rc ) { // confirm it is top-level plan (user plan)
    size_t dmnsn = fftPlan->dimension;
    bool pow2flag = true;

    // switch case flows with no 'break' statements
    switch(fftPlan->dimension) {
      case HCFFT_3D:
        if(fftPlan->length[2] == 1) {
          dmnsn -= 1;
          fftPlan-> inStride.erase(fftPlan-> inStride.begin() + 2);
          fftPlan->outStride.erase(fftPlan->outStride.begin() + 2);
          fftPlan->   length.erase(fftPlan->   length.begin() + 2);
        } else {
          if( !IsPo2(fftPlan->length[2])) {
            pow2flag = false;
          }
        }

      case HCFFT_2D:
        if(fftPlan->length[1] == 1) {
          dmnsn -= 1;
          fftPlan-> inStride.erase(fftPlan-> inStride.begin() + 1);
          fftPlan->outStride.erase(fftPlan->outStride.begin() + 1);
          fftPlan->   length.erase(fftPlan->   length.begin() + 1);
        } else {
          if( !IsPo2(fftPlan->length[1])) {
            pow2flag = false;
          }
        }

      case HCFFT_1D:
        if( (fftPlan->length[0] == 1) && (dmnsn > 1) ) {
          dmnsn -= 1;
          fftPlan-> inStride.erase(fftPlan-> inStride.begin());
          fftPlan->outStride.erase(fftPlan->outStride.begin());
          fftPlan->   length.erase(fftPlan->   length.begin());
        } else {
          if( !IsPo2(fftPlan->length[0])) {
            pow2flag = false;
          }
        }
    }

//#TODO Check dimension value
    fftPlan->dimension = (hcfftDim)dmnsn;
  }

  // first time check transposed
  if (fftPlan->transposeType != HCFFT_NOTRANSPOSE && fftPlan->dimension != HCFFT_2D &&
      fftPlan->dimension == fftPlan->length.size()) {
    return HCFFT_ERROR;
  }

  // The largest vector we can transform in a single pass
  // depends on the GPU caps -- especially the amount of LDS
  // available
  //
  size_t Large1DThreshold = 0;
  fftPlan->GetMax1DLength(&Large1DThreshold);
  BUG_CHECK(Large1DThreshold > 1);

  //  Verify that the data passed to us is packed
  switch( fftPlan->dimension ) {
    case HCFFT_1D: {
        if ( fftPlan->length[0] > Large1DThreshold ) {
          size_t clLengths[] = { 1, 1, 0 };
          size_t in_1d, in_x, count;
          BUG_CHECK (IsPo2 (Large1DThreshold))

          if( IsPo2(fftPlan->length[0]) ) {
            // Enable block compute under these conditions
            if( (fftPlan->inStride[0] == 1) && (fftPlan->outStride[0] == 1) && !rc
                && (fftPlan->length[0] <= 262144 / width(fftPlan->precision)) ) {
              fftPlan->blockCompute = true;

              if(1 == width(fftPlan->precision)) {
                switch(fftPlan->length[0]) {
                  case 8192:
                    clLengths[1] = 64;
                    break;

                  case 16384:
                    clLengths[1] = 64;
                    break;

                  case 32768:
                    clLengths[1] = 128;
                    break;

                  case 65536:
                    clLengths[1] = 256;
                    break;

                  case 131072:
                    clLengths[1] = 64;
                    break;

                  case 262144:
                    clLengths[1] = 64;
                    break;

                  case 524288:
                    clLengths[1] = 256;
                    break;

                  case 1048576:
                    clLengths[1] = 256;
                    break;

                  default:
                    assert(false);
                }
              } else {
                switch(fftPlan->length[0]) {
                  case 4096:
                    clLengths[1] = 64;
                    break;

                  case 8192:
                    clLengths[1] = 64;
                    break;

                  case 16384:
                    clLengths[1] = 64;
                    break;

                  case 32768:
                    clLengths[1] = 128;
                    break;

                  case 65536:
                    clLengths[1] = 64;
                    break;

                  case 131072:
                    clLengths[1] = 64;
                    break;

                  case 262144:
                    clLengths[1] = 128;
                    break;

                  case 524288:
                    clLengths[1] = 256;
                    break;

                  default:
                    assert(false);
                }
              }
            } else {
              if(fftPlan->length[0] > (Large1DThreshold * Large1DThreshold) ) {
                clLengths[1] = fftPlan->length[0] / Large1DThreshold;
              } else {
                in_1d = BitScanF (Large1DThreshold);  // this is log2(LARGE1D_THRESHOLD)
                in_x  = BitScanF (fftPlan->length[0]);  // this is log2(length)
                BUG_CHECK (in_1d > 0)
                count = in_x / in_1d;

                if (count * in_1d < in_x) {
                  count++;
                  in_1d = in_x / count;

                  if (in_1d * count < in_x) {
                    in_1d++;
                  }
                }

                clLengths[1] = (size_t)1 << in_1d;
              }
            }
          } else {
            // This array must be kept sorted in the ascending order
#if 0
            size_t supported[] = {  1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36, 40,
                                    45, 48, 50, 54, 60, 64, 72, 75, 80, 81, 90, 96, 100, 108, 120, 125, 128, 135,
                                    144, 150, 160, 162, 180, 192, 200, 216, 225, 240, 243, 250, 256, 270, 288,
                                    300, 320, 324, 360, 375, 384, 400, 405, 432, 450, 480, 486, 500, 512, 540,
                                    576, 600, 625, 640, 648, 675, 720, 729, 750, 768, 800, 810, 864, 900, 960,
                                    972, 1000, 1024, 1080, 1125, 1152, 1200, 1215, 1250, 1280, 1296, 1350, 1440,
                                    1458, 1500, 1536, 1600, 1620, 1728, 1800, 1875, 1920, 1944, 2000, 2025, 2048,
                                    2160, 2187, 2250, 2304, 2400, 2430, 2500, 2560, 2592, 2700, 2880, 2916, 3000,
                                    3072, 3125, 3200, 3240, 3375, 3456, 3600, 3645, 3750, 3840, 3888, 4000, 4050, 4096
                                 };
#else
            size_t supported[] =
            { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 24, 25, 27, 28, 30, 32, 35, 36, 40, 42, 45, 48, 49, 50, 54, 56, 60, 63, 64, 70, 72, 75, 80, 81, 84, 90, 96, 98, 100, 105, 108, 112, 120, 125, 126, 128, 135, 140, 144, 147, 150, 160, 162, 168, 175, 180, 189, 192, 196, 200, 210, 216, 224, 225, 240, 243, 245, 250, 252, 256, 270, 280, 288, 294, 300, 315, 320, 324, 336, 343, 350, 360, 375, 378, 384, 392, 400, 405, 420, 432, 441, 448, 450, 480, 486, 490, 500, 504, 512, 525, 540, 560, 567, 576, 588, 600, 625, 630, 640, 648, 672, 675, 686, 700, 720, 729, 735, 750, 756, 768, 784, 800, 810, 840, 864, 875, 882, 896, 900, 945, 960, 972, 980, 1000, 1008, 1024, 1029, 1050, 1080, 1120, 1125, 1134, 1152, 1176, 1200, 1215, 1225, 1250, 1260, 1280, 1296, 1323, 1344, 1350, 1372, 1400, 1440, 1458, 1470, 1500, 1512, 1536, 1568, 1575, 1600, 1620, 1680, 1701, 1715, 1728, 1750, 1764, 1792, 1800, 1875, 1890, 1920, 1944, 1960, 2000, 2016, 2025, 2048, 2058, 2100, 2160, 2187, 2205, 2240, 2250, 2268, 2304, 2352, 2400, 2401, 2430, 2450, 2500, 2520, 2560, 2592, 2625, 2646, 2688, 2700, 2744, 2800, 2835, 2880, 2916, 2940, 3000, 3024, 3072, 3087, 3125, 3136, 3150, 3200, 3240, 3360, 3375, 3402, 3430, 3456, 3500, 3528, 3584, 3600, 3645, 3675, 3750, 3780, 3840, 3888, 3920, 3969, 4000, 4032, 4050, 4096};
#endif
            size_t lenSupported = sizeof(supported) / sizeof(supported[0]);
            size_t maxFactoredLength = (supported[lenSupported - 1] < Large1DThreshold) ? supported[lenSupported - 1] : Large1DThreshold;
            size_t halfPowerLength = (size_t)1 << ( (CeilPo2(fftPlan->length[0]) + 1) / 2 );
            size_t factoredLengthStart =  (halfPowerLength < maxFactoredLength) ? halfPowerLength : maxFactoredLength;
            size_t indexStart = 0;

            while(supported[indexStart] < factoredLengthStart) {
              indexStart++;
            }

            for(size_t i = indexStart; i >= 1; i--) {
              if( fftPlan->length[0] % supported[i] == 0 ) {
                clLengths[1] = supported[i];
                break;
              }
            }
          }

          clLengths[0] = fftPlan->length[0] / clLengths[1];

          // Start of block where transposes are generated; 1D FFT
          while (1 && (fftPlan->ipLayout != HCFFT_REAL) && (fftPlan->opLayout != HCFFT_REAL)) {
            //if (!IsPo2(fftPlan->length[0])) break;

            //TBD, only one dimension?
            if (fftPlan->length.size() > 1) {
              break;
            }

            if (fftPlan->inStride[0] != 1 || fftPlan->outStride[0] != 1) {
              break;
            }

            if ( IsPo2(fftPlan->length[0])
                 && (fftPlan->length[0] <= 262144 / width(fftPlan->precision)) ) {
              break;
            }

            if ( clLengths[0] <= 32 && clLengths[1] <= 32) {
              break;
            }

            ARG_CHECK(clLengths[0] <= Large1DThreshold);
            size_t biggerDim = clLengths[0] > clLengths[1] ? clLengths[0] : clLengths[1];
            size_t smallerDim = biggerDim == clLengths[0] ? clLengths[1] : clLengths[0];
            size_t padding = 0;

            if( (smallerDim % 64 == 0) || (biggerDim % 64 == 0) ) {
              padding = 64;
            }

            if (fftPlan->tmpBufSize == 0 ) {
              fftPlan->tmpBufSize = (smallerDim + padding) * biggerDim *
                                    fftPlan->batchSize * fftPlan->ElementSize();
            }

            //Transpose
            //Input --> tmp buffer
            hcfftCreateDefaultPlanInternal( &fftPlan->planTX, HCFFT_2D, clLengths );
            FFTPlan* trans1Plan = NULL;
            lockRAII* trans1Lock  = NULL;
            fftRepo.getPlan( fftPlan->planTX, trans1Plan, trans1Lock );
            trans1Plan->location     = HCFFT_OUTOFPLACE;
            trans1Plan->precision     = fftPlan->precision;
            trans1Plan->tmpBufSize    = 0;
            trans1Plan->batchSize     = fftPlan->batchSize;
            trans1Plan->envelope    = fftPlan->envelope;
            trans1Plan->ipLayout   = fftPlan->ipLayout;
            trans1Plan->opLayout  = HCFFT_COMPLEX_INTERLEAVED;
            trans1Plan->inStride[0]   = fftPlan->inStride[0];
            trans1Plan->inStride[1]   = clLengths[0];
            trans1Plan->outStride[0]  = 1;
            trans1Plan->outStride[1]  = clLengths[1] + padding;
            trans1Plan->iDist         = fftPlan->iDist;
            trans1Plan->oDist         = clLengths[0] * trans1Plan->outStride[1];
            trans1Plan->gen           = Transpose;
            trans1Plan->transflag     = true;
            hcfftBakePlan(fftPlan->planTX);
            //Row transform
            //tmp->output
            //size clLengths[1], batch clLengths[0], with length[0] twiddle factor multiplication
            hcfftCreateDefaultPlanInternal( &fftPlan->planX, HCFFT_1D, &clLengths[1] );
            FFTPlan* row1Plan = NULL;
            lockRAII* row1Lock  = NULL;
            fftRepo.getPlan( fftPlan->planX, row1Plan, row1Lock );
            row1Plan->location     = HCFFT_OUTOFPLACE;
            row1Plan->precision     = fftPlan->precision;
            row1Plan->forwardScale  = 1.0f;
            row1Plan->backwardScale = 1.0f;
            row1Plan->tmpBufSize    = 0;
            row1Plan->batchSize     = fftPlan->batchSize;
            row1Plan->gen     = fftPlan->gen;
            row1Plan->envelope    = fftPlan->envelope;
            // twiddling is done in row2
            row1Plan->large1D   = 0;
            row1Plan->length.push_back(clLengths[0]);
            row1Plan->ipLayout   = HCFFT_COMPLEX_INTERLEAVED;
            row1Plan->opLayout  = fftPlan->opLayout;
            row1Plan->inStride[0]   = 1;
            row1Plan->outStride[0]  = fftPlan->outStride[0];
            row1Plan->inStride.push_back(clLengths[1] + padding);
            row1Plan->outStride.push_back(clLengths[1]);
            row1Plan->iDist         = clLengths[0] * row1Plan->inStride[1];
            row1Plan->oDist         = fftPlan->oDist;
            hcfftBakePlan(fftPlan->planX);
            //Transpose 2
            //Output --> tmp buffer
            clLengths[2] = clLengths[0];
            hcfftCreateDefaultPlanInternal( &fftPlan->planTY, HCFFT_2D, &clLengths[1] );
            FFTPlan* trans2Plan = NULL;
            lockRAII* trans2Lock  = NULL;
            fftRepo.getPlan( fftPlan->planTY, trans2Plan, trans2Lock );
            trans2Plan->location     = HCFFT_OUTOFPLACE;
            trans2Plan->precision     = fftPlan->precision;
            trans2Plan->tmpBufSize    = 0;
            trans2Plan->batchSize     = fftPlan->batchSize;
            trans2Plan->envelope    = fftPlan->envelope;
            trans2Plan->ipLayout   = fftPlan->opLayout;
            trans2Plan->opLayout  = HCFFT_COMPLEX_INTERLEAVED;
            trans2Plan->inStride[0]   = fftPlan->outStride[0];
            trans2Plan->inStride[1]   = clLengths[1];
            trans2Plan->outStride[0]  = 1;
            trans2Plan->outStride[1]  = clLengths[0] + padding;
            trans2Plan->iDist         = fftPlan->oDist;
            trans2Plan->oDist         = clLengths[1] * trans2Plan->outStride[1];
            trans2Plan->gen           = Transpose;
            trans2Plan->large1D   = fftPlan->length[0];
            trans2Plan->transflag     = true;
            hcfftBakePlan(fftPlan->planTY);
            //Row transform 2
            //tmp->tmp
            //size clLengths[0], batch clLengths[1]
            hcfftCreateDefaultPlanInternal( &fftPlan->planY, HCFFT_1D, &clLengths[0] );
            FFTPlan* row2Plan = NULL;
            lockRAII* row2Lock  = NULL;
            fftRepo.getPlan( fftPlan->planY, row2Plan, row2Lock );
            row2Plan->location     = HCFFT_INPLACE;
            row2Plan->precision     = fftPlan->precision;
            row2Plan->forwardScale  = fftPlan->forwardScale;
            row2Plan->backwardScale = fftPlan->backwardScale;
            row2Plan->tmpBufSize    = 0;
            row2Plan->batchSize     = fftPlan->batchSize;
            row2Plan->gen     = fftPlan->gen;
            row2Plan->envelope    = fftPlan->envelope;
            row2Plan->length.push_back(clLengths[1]);
            row2Plan->ipLayout   = HCFFT_COMPLEX_INTERLEAVED;
            row2Plan->opLayout  = HCFFT_COMPLEX_INTERLEAVED;
            row2Plan->inStride[0]   = 1;
            row2Plan->outStride[0]  = 1;
            row2Plan->inStride.push_back(clLengths[0] + padding);
            row2Plan->outStride.push_back(clLengths[0] + padding);
            row2Plan->iDist         = clLengths[1] * row2Plan->inStride[1];
            row2Plan->oDist         = clLengths[1] * row2Plan->outStride[1];
            hcfftBakePlan(fftPlan->planY);
            //Transpose 3
            //tmp --> output
            hcfftCreateDefaultPlanInternal( &fftPlan->planTZ, HCFFT_2D, clLengths );
            FFTPlan* trans3Plan = NULL;
            lockRAII* trans3Lock  = NULL;
            fftRepo.getPlan( fftPlan->planTZ, trans3Plan, trans3Lock);
            trans3Plan->location     = HCFFT_OUTOFPLACE;
            trans3Plan->precision     = fftPlan->precision;
            trans3Plan->tmpBufSize    = 0;
            trans3Plan->batchSize     = fftPlan->batchSize;
            trans3Plan->envelope    = fftPlan->envelope;
            trans3Plan->ipLayout   = HCFFT_COMPLEX_INTERLEAVED;
            trans3Plan->opLayout  = fftPlan->opLayout;
            trans3Plan->inStride[0]   = 1;
            trans3Plan->inStride[1]   = clLengths[0] + padding;
            trans3Plan->outStride[0]  = fftPlan->outStride[0];
            trans3Plan->outStride[1]  = clLengths[1];
            trans3Plan->iDist         = clLengths[1] * trans3Plan->inStride[1];
            trans3Plan->oDist         = fftPlan->oDist;
            trans3Plan->gen           = Transpose;
            trans3Plan->transflag     = true;
            trans3Plan->transOutHorizontal = true;
            hcfftBakePlan(fftPlan->planTZ);
            fftPlan->transflag = true;
            fftPlan->baked = true;
            return  HCFFT_SUCCESS;
          }

          size_t length0 = clLengths[0];
          size_t length1 = clLengths[1];

          // For real transforms
          // Special case optimization with 5-step algorithm
          if( (fftPlan->ipLayout == HCFFT_REAL) && IsPo2(fftPlan->length[0])
              && (fftPlan->inStride[0] == 1) && (fftPlan->outStride[0] == 1)
              && (fftPlan->length[0] > 4096) && (fftPlan->length.size() == 1) ) {
            if(fftPlan->length[0] == 8192) {
              size_t tmp = length0;
              clLengths[0] = length0 = length1;
              clLengths[1] = length1 = tmp;
            }

            ARG_CHECK(clLengths[0] <= Large1DThreshold);
            size_t biggerDim = clLengths[0] > clLengths[1] ? clLengths[0] : clLengths[1];
            size_t smallerDim = biggerDim == clLengths[0] ? clLengths[1] : clLengths[0];
            size_t padding = 0;

            if( (smallerDim % 64 == 0) || (biggerDim % 64 == 0) ) {
              padding = 64;
            }

            if (fftPlan->tmpBufSize == 0 ) {
              size_t Nf = (1 + smallerDim / 2) * biggerDim;
              fftPlan->tmpBufSize = (smallerDim + padding) * biggerDim / 2;

              if(fftPlan->tmpBufSize < Nf) {
                fftPlan->tmpBufSize = Nf;
              }

              fftPlan->tmpBufSize *= ( fftPlan->batchSize * fftPlan->ElementSize() );

              for (size_t index = 1; index < fftPlan->length.size(); index++) {
                fftPlan->tmpBufSize *= fftPlan->length[index];
              }
            }

            if (fftPlan->tmpBufSizeRC == 0 ) {
              fftPlan->tmpBufSizeRC = fftPlan->tmpBufSize;
            }

            //Transpose
            //Input --> tmp buffer
            hcfftCreateDefaultPlanInternal( &fftPlan->planTX, HCFFT_2D, clLengths );
            FFTPlan* trans1Plan = NULL;
            lockRAII* trans1Lock  = NULL;
            fftRepo.getPlan( fftPlan->planTX, trans1Plan, trans1Lock );
            trans1Plan->location     = HCFFT_OUTOFPLACE;
            trans1Plan->precision     = fftPlan->precision;
            trans1Plan->tmpBufSize    = 0;
            trans1Plan->batchSize     = fftPlan->batchSize;
            trans1Plan->envelope    = fftPlan->envelope;
            trans1Plan->ipLayout   = fftPlan->ipLayout;
            trans1Plan->opLayout  = HCFFT_REAL;
            trans1Plan->inStride[0]   = fftPlan->inStride[0];
            trans1Plan->inStride[1]   = clLengths[0];
            trans1Plan->outStride[0]  = 1;
            trans1Plan->outStride[1]  = clLengths[1] + padding;
            trans1Plan->iDist         = fftPlan->iDist;
            trans1Plan->oDist         = clLengths[0] * trans1Plan->outStride[1];
            trans1Plan->gen           = Transpose;
            trans1Plan->transflag     = true;
            hcfftBakePlan(fftPlan->planTX);
            //Row transform
            //tmp->output
            //size clLengths[1], batch clLengths[0], with length[0] twiddle factor multiplication
            hcfftCreateDefaultPlanInternal( &fftPlan->planX, HCFFT_1D, &clLengths[1] );
            FFTPlan* row1Plan = NULL;
            lockRAII* row1Lock  = NULL;
            fftRepo.getPlan( fftPlan->planX, row1Plan, row1Lock );
            row1Plan->location     = HCFFT_OUTOFPLACE;
            row1Plan->precision     = fftPlan->precision;
            row1Plan->forwardScale  = 1.0f;
            row1Plan->backwardScale = 1.0f;
            row1Plan->tmpBufSize    = 0;
            row1Plan->batchSize     = fftPlan->batchSize;
            row1Plan->gen     = fftPlan->gen;
            row1Plan->envelope    = fftPlan->envelope;
            // twiddling is done in row2
            row1Plan->large1D   = 0;
            row1Plan->length.push_back(clLengths[0]);
            row1Plan->ipLayout   = HCFFT_REAL;
            row1Plan->opLayout  = HCFFT_HERMITIAN_INTERLEAVED;
            row1Plan->inStride[0]   = 1;
            row1Plan->outStride[0]  = 1;
            row1Plan->inStride.push_back(clLengths[1] + padding);
            row1Plan->outStride.push_back(1 + clLengths[1] / 2);
            row1Plan->iDist         = clLengths[0] * row1Plan->inStride[1];
            row1Plan->oDist         = clLengths[0] * row1Plan->outStride[1];
            hcfftBakePlan(fftPlan->planX);
            //Transpose 2
            //Output --> tmp buffer
            clLengths[2] = clLengths[0];
            hcfftCreateDefaultPlanInternal( &fftPlan->planTY, HCFFT_2D, &clLengths[1] );
            FFTPlan* trans2Plan = NULL;
            lockRAII* trans2Lock  = NULL;
            fftRepo.getPlan( fftPlan->planTY, trans2Plan, trans2Lock );
            trans2Plan->transflag = true;
            size_t transLengths[2];
            transLengths[0] = 1 + clLengths[1] / 2;
            transLengths[1] = clLengths[0];
            hcfftSetPlanLength( fftPlan->planTY, HCFFT_2D, transLengths );
            trans2Plan->location     = HCFFT_OUTOFPLACE;
            trans2Plan->precision     = fftPlan->precision;
            trans2Plan->tmpBufSize    = 0;
            trans2Plan->batchSize     = fftPlan->batchSize;
            trans2Plan->envelope    = fftPlan->envelope;
            trans2Plan->ipLayout   = HCFFT_COMPLEX_INTERLEAVED;
            trans2Plan->opLayout  = HCFFT_COMPLEX_INTERLEAVED;
            trans2Plan->inStride[0]   = 1;
            trans2Plan->inStride[1]   = 1 + clLengths[1] / 2;
            trans2Plan->outStride[0]  = 1;
            trans2Plan->outStride[1]  = clLengths[0];
            trans2Plan->iDist         = clLengths[0] * trans2Plan->inStride[1];
            trans2Plan->oDist         = (1 + clLengths[1] / 2) * trans2Plan->outStride[1];
            trans2Plan->gen           = Transpose;
            trans2Plan->transflag     = true;
            trans2Plan->transOutHorizontal = true;
            hcfftBakePlan(fftPlan->planTY);
            //Row transform 2
            //tmp->tmp
            //size clLengths[0], batch clLengths[1]
            hcfftCreateDefaultPlanInternal( &fftPlan->planY, HCFFT_1D, &clLengths[0] );
            FFTPlan* row2Plan = NULL;
            lockRAII* row2Lock  = NULL;
            fftRepo.getPlan( fftPlan->planY, row2Plan, row2Lock );
            row2Plan->location     = HCFFT_OUTOFPLACE;
            row2Plan->precision     = fftPlan->precision;
            row2Plan->forwardScale  = fftPlan->forwardScale;
            row2Plan->backwardScale = fftPlan->backwardScale;
            row2Plan->tmpBufSize    = 0;
            row2Plan->batchSize     = fftPlan->batchSize;
            row2Plan->gen     = fftPlan->gen;
            row2Plan->envelope    = fftPlan->envelope;
            row2Plan->length.push_back(1 + clLengths[1] / 2);
            row2Plan->ipLayout   = HCFFT_COMPLEX_INTERLEAVED;
            row2Plan->opLayout  = HCFFT_COMPLEX_INTERLEAVED;
            row2Plan->inStride[0]   = 1;
            row2Plan->outStride[0]  = 1;
            row2Plan->inStride.push_back(clLengths[0]);
            row2Plan->outStride.push_back(1 + clLengths[0] / 2);
            row2Plan->iDist         = (1 + clLengths[1] / 2) * row2Plan->inStride[1];
            row2Plan->oDist         = clLengths[1] * row2Plan->outStride[1];
            row2Plan->large1D   = fftPlan->length[0];
            row2Plan->twiddleFront  = true;
            row2Plan->realSpecial = true;
            row2Plan->realSpecial_Nr = clLengths[1];
            hcfftBakePlan(fftPlan->planY);
            //Transpose 3
            //tmp --> output
            hcfftCreateDefaultPlanInternal( &fftPlan->planTZ, HCFFT_2D, clLengths );
            FFTPlan* trans3Plan = NULL;
            lockRAII* trans3Lock  = NULL;
            fftRepo.getPlan( fftPlan->planTZ, trans3Plan, trans3Lock );
            trans3Plan->transflag = true;
            transLengths[0] = 1 + clLengths[0] / 2;
            transLengths[1] = clLengths[1];
            hcfftSetPlanLength( fftPlan->planTZ, HCFFT_2D, transLengths );
            trans3Plan->location     = HCFFT_OUTOFPLACE;
            trans3Plan->precision     = fftPlan->precision;
            trans3Plan->tmpBufSize    = 0;
            trans3Plan->batchSize     = fftPlan->batchSize;
            trans3Plan->envelope    = fftPlan->envelope;
            trans3Plan->ipLayout   = HCFFT_COMPLEX_INTERLEAVED;

            if(fftPlan->opLayout == HCFFT_HERMITIAN_PLANAR) {
              trans3Plan->opLayout  = HCFFT_COMPLEX_PLANAR;
            } else {
              trans3Plan->opLayout  = HCFFT_COMPLEX_INTERLEAVED;
            }

            trans3Plan->inStride[0]   = 1;
            trans3Plan->inStride[1]   = 1 + clLengths[0] / 2;
            trans3Plan->outStride[0]  = 1;
            trans3Plan->outStride[1]  = clLengths[1];
            trans3Plan->iDist         = clLengths[1] * trans3Plan->inStride[1];
            trans3Plan->oDist         = fftPlan->oDist;
            trans3Plan->gen           = Transpose;
            trans3Plan->transflag     = true;
            trans3Plan->realSpecial   = true;
            trans3Plan->transOutHorizontal = true;
            hcfftBakePlan(fftPlan->planTZ);
            fftPlan->transflag = true;
            fftPlan->baked = true;
            return  HCFFT_SUCCESS;
          } else if(fftPlan->ipLayout == HCFFT_REAL) {
            if (fftPlan->tmpBufSizeRC == 0 ) {
              fftPlan->tmpBufSizeRC = length0 * length1 *
                                      fftPlan->batchSize * fftPlan->ElementSize();

              for (size_t index = 1; index < fftPlan->length.size(); index++) {
                fftPlan->tmpBufSizeRC *= fftPlan->length[index];
              }
            }

            // column FFT, size clLengths[1], batch clLengths[0], with length[0] twiddle factor multiplication
            // transposed output
            hcfftCreateDefaultPlanInternal( &fftPlan->planX, HCFFT_1D, &clLengths[1] );
            FFTPlan* colTPlan = NULL;
            lockRAII* colLock = NULL;
            fftRepo.getPlan( fftPlan->planX, colTPlan, colLock );
            // current plan is to create intermediate buffer, packed and interleave
            // This is a column FFT, the first elements distance between each FFT is the distance of the first two
            // elements in the original buffer. Like a transpose of the matrix
            // we need to pass clLengths[0] and instride size to kernel, so kernel can tell the difference
            //this part are common for both passes
            colTPlan->location     = HCFFT_OUTOFPLACE;
            colTPlan->precision     = fftPlan->precision;
            colTPlan->forwardScale  = 1.0f;
            colTPlan->backwardScale = 1.0f;
            colTPlan->tmpBufSize    = 0;
            colTPlan->batchSize     = fftPlan->batchSize;
            colTPlan->gen     = fftPlan->gen;
            colTPlan->envelope      = fftPlan->envelope;
            //Pass large1D flag to confirm we need multiply twiddle factor
            colTPlan->large1D       = fftPlan->length[0];
            colTPlan->RCsimple    = true;
            colTPlan->length.push_back(clLengths[0]);
            // first Pass
            colTPlan->ipLayout   = fftPlan->ipLayout;
            colTPlan->opLayout  = HCFFT_COMPLEX_INTERLEAVED;
            colTPlan->inStride[0]   = fftPlan->inStride[0] * clLengths[0];
            colTPlan->outStride[0]  = 1;
            colTPlan->iDist         = fftPlan->iDist;
            colTPlan->oDist         = length0 * length1;//fftPlan->length[0];
            colTPlan->inStride.push_back(fftPlan->inStride[0]);
            colTPlan->outStride.push_back(length1);//clLengths[1]);

            for (size_t index = 1; index < fftPlan->length.size(); index++) {
              colTPlan->length.push_back(fftPlan->length[index]);
              colTPlan->inStride.push_back(fftPlan->inStride[index]);
              // tmp buffer is tightly packed
              colTPlan->outStride.push_back(colTPlan->oDist);
              colTPlan->oDist        *= fftPlan->length[index];
            }

            hcfftBakePlan(fftPlan->planX);
            //another column FFT, size clLengths[0], batch clLengths[1], output without transpose
            hcfftCreateDefaultPlanInternal( &fftPlan->planY, HCFFT_1D,  &clLengths[0] );
            FFTPlan* col2Plan = NULL;
            lockRAII* rowLock = NULL;
            fftRepo.getPlan( fftPlan->planY, col2Plan, rowLock );
            // This is second column fft, intermediate buffer is packed and interleaved
            // we need to pass clLengths[1] and instride size to kernel, so kernel can tell the difference
            // common part for both passes
            col2Plan->location     = HCFFT_INPLACE;
            col2Plan->ipLayout   = HCFFT_COMPLEX_INTERLEAVED;
            col2Plan->opLayout  = HCFFT_COMPLEX_INTERLEAVED;
            col2Plan->precision     = fftPlan->precision;
            col2Plan->forwardScale  = fftPlan->forwardScale;
            col2Plan->backwardScale = fftPlan->backwardScale;
            col2Plan->tmpBufSize    = 0;
            col2Plan->batchSize     = fftPlan->batchSize;
            col2Plan->gen     = fftPlan->gen;
            col2Plan->envelope      = fftPlan->envelope;
            col2Plan->length.push_back(length1);
            col2Plan->inStride[0]  = length1;
            col2Plan->inStride.push_back(1);
            col2Plan->iDist        = length0 * length1;
            col2Plan->outStride[0] = length1;
            col2Plan->outStride.push_back(1);
            col2Plan->oDist         = length0 * length1;

            for (size_t index = 1; index < fftPlan->length.size(); index++) {
              col2Plan->length.push_back(fftPlan->length[index]);
              col2Plan->inStride.push_back(col2Plan->iDist);
              col2Plan->outStride.push_back(col2Plan->oDist);
              col2Plan->iDist   *= fftPlan->length[index];
              col2Plan->oDist   *= fftPlan->length[index];
            }

            hcfftBakePlan(fftPlan->planY);
            // copy plan to get back to hermitian
            hcfftCreateDefaultPlanInternal( &fftPlan->planRCcopy, HCFFT_1D,  &fftPlan->length[0]);
            FFTPlan* copyPlan = NULL;
            lockRAII* copyLock  = NULL;
            fftRepo.getPlan( fftPlan->planRCcopy, copyPlan, copyLock );
            // This is second column fft, intermediate buffer is packed and interleaved
            // we need to pass clLengths[1] and instride size to kernel, so kernel can tell the difference
            // common part for both passes
            copyPlan->location     = HCFFT_OUTOFPLACE;
            copyPlan->ipLayout   = HCFFT_COMPLEX_INTERLEAVED;
            copyPlan->opLayout  = fftPlan->opLayout;
            copyPlan->precision     = fftPlan->precision;
            copyPlan->forwardScale  = 1.0f;
            copyPlan->backwardScale = 1.0f;
            copyPlan->tmpBufSize    = 0;
            copyPlan->batchSize     = fftPlan->batchSize;
            copyPlan->gen     = Copy;
            copyPlan->envelope    = fftPlan->envelope;
            copyPlan->inStride[0]  = 1;
            copyPlan->iDist        = fftPlan->length[0];
            copyPlan->outStride[0] = fftPlan->outStride[0];
            copyPlan->oDist         = fftPlan->oDist;

            for (size_t index = 1; index < fftPlan->length.size(); index++) {
              copyPlan->length.push_back(fftPlan->length[index]);
              copyPlan->inStride.push_back(copyPlan->inStride[index - 1] * fftPlan->length[index - 1]);
              copyPlan->iDist   *= fftPlan->length[index];
              copyPlan->outStride.push_back(fftPlan->outStride[index]);
            }

            hcfftBakePlan(fftPlan->planRCcopy);
          } else if(fftPlan->opLayout == HCFFT_REAL) {
            if (fftPlan->tmpBufSizeRC == 0 ) {
              fftPlan->tmpBufSizeRC = length0 * length1 *
                                      fftPlan->batchSize * fftPlan->ElementSize();

              for (size_t index = 1; index < fftPlan->length.size(); index++) {
                fftPlan->tmpBufSizeRC *= fftPlan->length[index];
              }
            }

            // copy plan to from hermitian to full complex
            hcfftCreateDefaultPlanInternal( &fftPlan->planRCcopy, HCFFT_1D,  &fftPlan->length[0] );
            FFTPlan* copyPlan = NULL;
            lockRAII* copyLock  = NULL;
            fftRepo.getPlan( fftPlan->planRCcopy, copyPlan, copyLock );
            // This is second column fft, intermediate buffer is packed and interleaved
            // we need to pass clLengths[1] and instride size to kernel, so kernel can tell the difference
            // common part for both passes
            copyPlan->location     = HCFFT_OUTOFPLACE;
            copyPlan->ipLayout   = fftPlan->ipLayout;
            copyPlan->opLayout  = HCFFT_COMPLEX_INTERLEAVED;
            copyPlan->precision     = fftPlan->precision;
            copyPlan->forwardScale  = 1.0f;
            copyPlan->backwardScale = 1.0f;
            copyPlan->tmpBufSize    = 0;
            copyPlan->batchSize     = fftPlan->batchSize;
            copyPlan->gen     = Copy;
            copyPlan->envelope    = fftPlan->envelope;
            copyPlan->inStride[0]  = fftPlan->inStride[0];
            copyPlan->iDist        = fftPlan->iDist;
            copyPlan->outStride[0]  = 1;
            copyPlan->oDist        = fftPlan->length[0];

            for (size_t index = 1; index < fftPlan->length.size(); index++) {
              copyPlan->length.push_back(fftPlan->length[index]);
              copyPlan->outStride.push_back(copyPlan->outStride[index - 1] * fftPlan->length[index - 1]);
              copyPlan->oDist   *= fftPlan->length[index];
              copyPlan->inStride.push_back(fftPlan->inStride[index]);
            }

            hcfftBakePlan(fftPlan->planRCcopy);
            // column FFT, size clLengths[1], batch clLengths[0], with length[0] twiddle factor multiplication
            // transposed output
            hcfftCreateDefaultPlanInternal( &fftPlan->planX, HCFFT_1D, &clLengths[1] );
            FFTPlan* colTPlan = NULL;
            lockRAII* colLock = NULL;
            fftRepo.getPlan( fftPlan->planX, colTPlan, colLock );
            // current plan is to create intermediate buffer, packed and interleave
            // This is a column FFT, the first elements distance between each FFT is the distance of the first two
            // elements in the original buffer. Like a transpose of the matrix
            // we need to pass clLengths[0] and instride size to kernel, so kernel can tell the difference
            //this part are common for both passes
            colTPlan->location     = HCFFT_INPLACE;
            colTPlan->precision     = fftPlan->precision;
            colTPlan->forwardScale  = 1.0f;
            colTPlan->backwardScale = 1.0f;
            colTPlan->tmpBufSize    = 0;
            colTPlan->batchSize     = fftPlan->batchSize;
            colTPlan->gen     = fftPlan->gen;
            colTPlan->envelope      = fftPlan->envelope;
            //Pass large1D flag to confirm we need multiply twiddle factor
            colTPlan->large1D       = fftPlan->length[0];
            colTPlan->length.push_back(clLengths[0]);
            // first Pass
            colTPlan->ipLayout   = HCFFT_COMPLEX_INTERLEAVED;
            colTPlan->opLayout  = HCFFT_COMPLEX_INTERLEAVED;
            colTPlan->inStride[0]  = length0;
            colTPlan->inStride.push_back(1);
            colTPlan->iDist        = length0 * length1;
            colTPlan->outStride[0] = length0;
            colTPlan->outStride.push_back(1);
            colTPlan->oDist         = length0 * length1;

            for (size_t index = 1; index < fftPlan->length.size(); index++) {
              colTPlan->length.push_back(fftPlan->length[index]);
              colTPlan->inStride.push_back(colTPlan->iDist);
              colTPlan->outStride.push_back(colTPlan->oDist);
              colTPlan->iDist   *= fftPlan->length[index];
              colTPlan->oDist   *= fftPlan->length[index];
            }

            hcfftBakePlan(fftPlan->planX);
            //another column FFT, size clLengths[0], batch clLengths[1], output without transpose
            hcfftCreateDefaultPlanInternal( &fftPlan->planY, HCFFT_1D,  &clLengths[0] );
            FFTPlan* col2Plan = NULL;
            lockRAII* rowLock = NULL;
            fftRepo.getPlan( fftPlan->planY, col2Plan, rowLock );
            // This is second column fft, intermediate buffer is packed and interleaved
            // we need to pass clLengths[1] and instride size to kernel, so kernel can tell the difference
            // common part for both passes
            col2Plan->location     = HCFFT_OUTOFPLACE;
            col2Plan->ipLayout   = HCFFT_COMPLEX_INTERLEAVED;
            col2Plan->opLayout  = fftPlan->opLayout;
            col2Plan->precision     = fftPlan->precision;
            col2Plan->forwardScale  = fftPlan->forwardScale;
            col2Plan->backwardScale = fftPlan->backwardScale;
            col2Plan->tmpBufSize    = 0;
            col2Plan->batchSize     = fftPlan->batchSize;
            col2Plan->gen     = fftPlan->gen;
            col2Plan->envelope      = fftPlan->envelope;
            col2Plan->RCsimple = true;
            col2Plan->length.push_back(length1);
            col2Plan->inStride[0]  = 1;
            col2Plan->inStride.push_back(length0);
            col2Plan->iDist        = length0 * length1;
            col2Plan->outStride[0] = length1 * fftPlan->outStride[0];
            col2Plan->outStride.push_back(fftPlan->outStride[0]);
            col2Plan->oDist         = fftPlan->oDist;

            for (size_t index = 1; index < fftPlan->length.size(); index++) {
              col2Plan->length.push_back(fftPlan->length[index]);
              col2Plan->inStride.push_back(col2Plan->iDist);
              col2Plan->iDist   *= fftPlan->length[index];
              col2Plan->outStride.push_back(fftPlan->outStride[index]);
            }

            hcfftBakePlan(fftPlan->planY);
          } else {
            if( (fftPlan->length[0] > 262144 / width(fftPlan->precision)) && fftPlan->blockCompute ) {
              assert(fftPlan->length[0] <= 1048576);
              size_t padding = 64;

              if (fftPlan->tmpBufSize == 0 ) {
                fftPlan->tmpBufSize = (length1 + padding) * length0 *
                                      fftPlan->batchSize * fftPlan->ElementSize();

                for (size_t index = 1; index < fftPlan->length.size(); index++) {
                  fftPlan->tmpBufSize *= fftPlan->length[index];
                }
              }

              // Algorithm in this case is
              // T(with pad, out_of_place), R (in_place), C(in_place), Unpad(out_of_place)
              size_t len[3] = { clLengths[1], clLengths[0], 1 };
              hcfftCreateDefaultPlanInternal( &fftPlan->planTX, HCFFT_2D, len );
              FFTPlan* trans1Plan = NULL;
              lockRAII* trans1Lock  = NULL;
              fftRepo.getPlan( fftPlan->planTX, trans1Plan, trans1Lock );
              trans1Plan->location     = HCFFT_OUTOFPLACE;
              trans1Plan->precision     = fftPlan->precision;
              trans1Plan->tmpBufSize    = 0;
              trans1Plan->batchSize     = fftPlan->batchSize;
              trans1Plan->envelope    = fftPlan->envelope;
              trans1Plan->ipLayout   = fftPlan->ipLayout;
              trans1Plan->opLayout  = HCFFT_COMPLEX_INTERLEAVED;
              trans1Plan->inStride[0]   = fftPlan->inStride[0];
              trans1Plan->inStride[1]   = length1;
              trans1Plan->outStride[0]  = 1;
              trans1Plan->outStride[1]  = length0 + padding;
              trans1Plan->iDist         = fftPlan->iDist;
              trans1Plan->oDist         = length1 * trans1Plan->outStride[1];
              trans1Plan->gen           = Transpose;
              trans1Plan->transflag     = true;

              for (size_t index = 1; index < fftPlan->length.size(); index++) {
                trans1Plan->length.push_back(fftPlan->length[index]);
                trans1Plan->inStride.push_back(fftPlan->inStride[index]);
                trans1Plan->outStride.push_back(trans1Plan->oDist);
                trans1Plan->oDist *= fftPlan->length[index];
              }

              hcfftBakePlan(fftPlan->planTX);
              // row FFT
              hcfftCreateDefaultPlanInternal( &fftPlan->planX, HCFFT_1D, &clLengths[0] );
              FFTPlan* rowPlan  = NULL;
              lockRAII* rowLock = NULL;
              fftRepo.getPlan( fftPlan->planX, rowPlan, rowLock );
              assert(fftPlan->large1D == 0);
              rowPlan->location     = HCFFT_INPLACE;
              rowPlan->precision     = fftPlan->precision;
              rowPlan->forwardScale  = 1.0f;
              rowPlan->backwardScale = 1.0f;
              rowPlan->tmpBufSize    = 0;
              rowPlan->batchSize     = fftPlan->batchSize;
              rowPlan->gen      = fftPlan->gen;
              rowPlan->envelope   = fftPlan->envelope;
              rowPlan->length.push_back(length1);
              rowPlan->ipLayout   = HCFFT_COMPLEX_INTERLEAVED;
              rowPlan->opLayout  = HCFFT_COMPLEX_INTERLEAVED;
              rowPlan->inStride[0]   = 1;
              rowPlan->outStride[0]  = 1;
              rowPlan->inStride.push_back(length0 + padding);
              rowPlan->outStride.push_back(length0 + padding);
              rowPlan->iDist         = (length0 + padding) * length1;
              rowPlan->oDist         = (length0 + padding) * length1;

              for (size_t index = 1; index < fftPlan->length.size(); index++) {
                rowPlan->length.push_back(fftPlan->length[index]);
                rowPlan->inStride.push_back(rowPlan->iDist);
                rowPlan->iDist *= fftPlan->length[index];
                rowPlan->outStride.push_back(rowPlan->oDist);
                rowPlan->oDist *= fftPlan->length[index];
              }

              hcfftBakePlan(fftPlan->planX);
              //column FFT
              hcfftCreateDefaultPlanInternal( &fftPlan->planY, HCFFT_1D,  &clLengths[1] );
              FFTPlan* col2Plan = NULL;
              lockRAII* colLock = NULL;
              fftRepo.getPlan( fftPlan->planY, col2Plan, colLock );
              col2Plan->location     = HCFFT_INPLACE;
              col2Plan->ipLayout   = HCFFT_COMPLEX_INTERLEAVED;
              col2Plan->opLayout  = HCFFT_COMPLEX_INTERLEAVED;
              col2Plan->precision     = fftPlan->precision;
              col2Plan->forwardScale  = fftPlan->forwardScale;
              col2Plan->backwardScale = fftPlan->backwardScale;
              col2Plan->tmpBufSize    = 0;
              col2Plan->batchSize     = fftPlan->batchSize;
              col2Plan->gen     = fftPlan->gen;
              col2Plan->envelope    = fftPlan->envelope;
              col2Plan->large1D       = fftPlan->length[0];
              col2Plan->twiddleFront  = true;
              col2Plan->length.push_back(clLengths[0]);
              col2Plan->blockCompute = true;
              col2Plan->blockComputeType = BCT_C2C;
              col2Plan->inStride[0]  = length0 + padding;
              col2Plan->outStride[0] = length0 + padding;
              col2Plan->iDist        = (length0 + padding) * length1;
              col2Plan->oDist        = (length0 + padding) * length1;
              col2Plan->inStride.push_back(1);
              col2Plan->outStride.push_back(1);

              for (size_t index = 1; index < fftPlan->length.size(); index++) {
                col2Plan->length.push_back(fftPlan->length[index]);
                col2Plan->inStride.push_back(col2Plan->iDist);
                col2Plan->outStride.push_back(col2Plan->oDist);
                col2Plan->iDist   *= fftPlan->length[index];
                col2Plan->oDist   *= fftPlan->length[index];
              }

              hcfftBakePlan(fftPlan->planY);
              // copy plan to get results back to packed output
              hcfftCreateDefaultPlanInternal( &fftPlan->planCopy, HCFFT_1D,  &clLengths[0] );
              FFTPlan* copyPlan = NULL;
              lockRAII* copyLock  = NULL;
              fftRepo.getPlan( fftPlan->planCopy, copyPlan, copyLock );
              copyPlan->location     = HCFFT_OUTOFPLACE;
              copyPlan->ipLayout   = HCFFT_COMPLEX_INTERLEAVED;
              copyPlan->opLayout  = fftPlan->opLayout;
              copyPlan->precision     = fftPlan->precision;
              copyPlan->forwardScale  = 1.0f;
              copyPlan->backwardScale = 1.0f;
              copyPlan->tmpBufSize    = 0;
              copyPlan->batchSize     = fftPlan->batchSize;
              copyPlan->gen     = Copy;
              copyPlan->envelope    = fftPlan->envelope;
              copyPlan->length.push_back(length1);
              copyPlan->inStride[0]  = 1;
              copyPlan->inStride.push_back(length0 + padding);
              copyPlan->iDist        = length1 * (length0 + padding);
              copyPlan->outStride[0] = fftPlan->outStride[0];
              copyPlan->outStride.push_back(length0);
              copyPlan->oDist         = fftPlan->oDist;

              for (size_t index = 1; index < fftPlan->length.size(); index++) {
                copyPlan->length.push_back(fftPlan->length[index]);
                copyPlan->inStride.push_back(copyPlan->inStride[index] * copyPlan->length[index]);
                copyPlan->iDist   *= fftPlan->length[index];
                copyPlan->outStride.push_back(fftPlan->outStride[index]);
              }

              hcfftBakePlan(fftPlan->planCopy);
            } else {
              if (fftPlan->tmpBufSize == 0 ) {
                fftPlan->tmpBufSize = length0 * length1 *
                                      fftPlan->batchSize * fftPlan->ElementSize();

                for (size_t index = 1; index < fftPlan->length.size(); index++) {
                  fftPlan->tmpBufSize *= fftPlan->length[index];
                }
              }

              // column FFT, size clLengths[1], batch clLengths[0], with length[0] twiddle factor multiplication
              // transposed output
              hcfftCreateDefaultPlanInternal( &fftPlan->planX, HCFFT_1D, &clLengths[1] );
              FFTPlan* colTPlan = NULL;
              lockRAII* colLock = NULL;
              fftRepo.getPlan( fftPlan->planX, colTPlan, colLock );
              assert(fftPlan->large1D == 0);
              // current plan is to create intermediate buffer, packed and interleave
              // This is a column FFT, the first elements distance between each FFT is the distance of the first two
              // elements in the original buffer. Like a transpose of the matrix
              // we need to pass clLengths[0] and instride size to kernel, so kernel can tell the difference
              //this part are common for both passes
              colTPlan->location     = HCFFT_OUTOFPLACE;
              colTPlan->precision     = fftPlan->precision;
              colTPlan->forwardScale  = 1.0f;
              colTPlan->backwardScale = 1.0f;
              colTPlan->tmpBufSize    = 0;
              colTPlan->batchSize     = fftPlan->batchSize;
              colTPlan->gen     = fftPlan->gen;
              colTPlan->envelope      = fftPlan->envelope;
              //Pass large1D flag to confirm we need multiply twiddle factor
              colTPlan->large1D       = fftPlan->length[0];
              colTPlan->length.push_back(length0);
              colTPlan->ipLayout   = fftPlan->ipLayout;
              colTPlan->opLayout  = HCFFT_COMPLEX_INTERLEAVED;
              colTPlan->inStride[0]   = fftPlan->inStride[0] * length0;
              colTPlan->outStride[0]  = length0;
              colTPlan->iDist         = fftPlan->iDist;
              colTPlan->oDist         = length0 * length1;
              colTPlan->inStride.push_back(fftPlan->inStride[0]);
              colTPlan->outStride.push_back(1);

              // Enabling block column compute
              if( (colTPlan->inStride[0] == length0) && IsPo2(fftPlan->length[0]) && (fftPlan->length[0] < 524288) ) {
                colTPlan->blockCompute = true;
                colTPlan->blockComputeType = BCT_C2C;
              }

              for (size_t index = 1; index < fftPlan->length.size(); index++) {
                colTPlan->length.push_back(fftPlan->length[index]);
                colTPlan->inStride.push_back(fftPlan->inStride[index]);
                // tmp buffer is tightly packed
                colTPlan->outStride.push_back(colTPlan->oDist);
                colTPlan->oDist        *= fftPlan->length[index];
              }

              hcfftBakePlan(fftPlan->planX);
              //another column FFT, size clLengths[0], batch clLengths[1], output without transpose
              hcfftCreateDefaultPlanInternal( &fftPlan->planY, HCFFT_1D,  &clLengths[0] );
              FFTPlan* col2Plan = NULL;
              lockRAII* rowLock = NULL;
              fftRepo.getPlan( fftPlan->planY, col2Plan, rowLock );
              // This is second column fft, intermediate buffer is packed and interleaved
              // we need to pass clLengths[1] and instride size to kernel, so kernel can tell the difference
              // common part for both passes
              col2Plan->opLayout  = fftPlan->opLayout;
              col2Plan->precision     = fftPlan->precision;
              col2Plan->forwardScale  = fftPlan->forwardScale;
              col2Plan->backwardScale = fftPlan->backwardScale;
              col2Plan->tmpBufSize    = 0;
              col2Plan->batchSize     = fftPlan->batchSize;
              col2Plan->oDist         = fftPlan->oDist;
              col2Plan->gen     = fftPlan->gen;
              col2Plan->envelope    = fftPlan->envelope;
              col2Plan->length.push_back(clLengths[1]);
              bool integratedTranposes = true;

              if( colTPlan->blockCompute && (fftPlan->outStride[0] == 1) && clLengths[0] <= 256) {
                col2Plan->blockCompute = true;
                col2Plan->blockComputeType = BCT_R2C;
                col2Plan->location    = HCFFT_OUTOFPLACE;
                col2Plan->ipLayout  = HCFFT_COMPLEX_INTERLEAVED;
                col2Plan->inStride[0]  = 1;
                col2Plan->outStride[0] = length1;
                col2Plan->iDist        = length0 * length1;
                col2Plan->inStride.push_back(length0);
                col2Plan->outStride.push_back(1);
              } else if( colTPlan->blockCompute && (fftPlan->outStride[0] == 1) ) {
                integratedTranposes = false;
                col2Plan->location    = HCFFT_INPLACE;
                col2Plan->ipLayout  = HCFFT_COMPLEX_INTERLEAVED;
                col2Plan->opLayout = HCFFT_COMPLEX_INTERLEAVED;
                col2Plan->inStride[0]  = 1;
                col2Plan->outStride[0] = 1;
                col2Plan->iDist        = length0 * length1;
                col2Plan->oDist        = length0 * length1;
                col2Plan->inStride.push_back(length0);
                col2Plan->outStride.push_back(length0);
              } else {
                //first layer, large 1D from tmp buffer to output buffer
                col2Plan->location    = HCFFT_OUTOFPLACE;
                col2Plan->ipLayout  = HCFFT_COMPLEX_INTERLEAVED;
                col2Plan->inStride[0]  = 1;
                col2Plan->outStride[0] = fftPlan->outStride[0] * clLengths[1];
                col2Plan->iDist        = length0 * length1; //fftPlan->length[0];
                col2Plan->inStride.push_back(length0);
                col2Plan->outStride.push_back(fftPlan->outStride[0]);
              }

              if(!integratedTranposes) {
                for (size_t index = 1; index < fftPlan->length.size(); index++) {
                  col2Plan->length.push_back(fftPlan->length[index]);
                  col2Plan->inStride.push_back(col2Plan->iDist);
                  col2Plan->outStride.push_back(col2Plan->oDist);
                  col2Plan->iDist        *= fftPlan->length[index];
                  col2Plan->oDist        *= fftPlan->length[index];
                }
              } else {
                for (size_t index = 1; index < fftPlan->length.size(); index++) {
                  col2Plan->length.push_back(fftPlan->length[index]);
                  col2Plan->inStride.push_back(col2Plan->iDist);
                  col2Plan->outStride.push_back(fftPlan->outStride[index]);
                  col2Plan->iDist   *= fftPlan->length[index];
                }
              }

              hcfftBakePlan(fftPlan->planY);

              if(!integratedTranposes) {
                //Transpose
                //tmp --> output
                hcfftCreateDefaultPlanInternal( &fftPlan->planTZ, HCFFT_2D, clLengths );
                FFTPlan* trans3Plan = NULL;
                lockRAII* trans3Lock  = NULL;
                fftRepo.getPlan( fftPlan->planTZ, trans3Plan, trans3Lock );
                trans3Plan->location     = HCFFT_OUTOFPLACE;
                trans3Plan->precision     = fftPlan->precision;
                trans3Plan->tmpBufSize    = 0;
                trans3Plan->batchSize     = fftPlan->batchSize;
                trans3Plan->envelope    = fftPlan->envelope;
                trans3Plan->ipLayout   = HCFFT_COMPLEX_INTERLEAVED;
                trans3Plan->opLayout  = fftPlan->opLayout;
                trans3Plan->inStride[0]   = 1;
                trans3Plan->inStride[1]   = clLengths[0];
                trans3Plan->outStride[0]  = fftPlan->outStride[0];
                trans3Plan->outStride[1]  = clLengths[1] * fftPlan->outStride[0];
                trans3Plan->iDist         = fftPlan->length[0];
                trans3Plan->oDist         = fftPlan->oDist;
                trans3Plan->gen           = Transpose;
                trans3Plan->transflag     = true;

                for (size_t index = 1; index < fftPlan->length.size(); index++) {
                  trans3Plan->length.push_back(fftPlan->length[index]);
                  trans3Plan->inStride.push_back(trans3Plan->iDist);
                  trans3Plan->iDist *= fftPlan->length[index];
                  trans3Plan->outStride.push_back(fftPlan->outStride[index]);
                }

                hcfftBakePlan(fftPlan->planTZ);
              }
            }
          }

          fftPlan->baked = true;
          return  HCFFT_SUCCESS;
        }
      }
      break;

    case HCFFT_2D: {
        if (fftPlan->transflag) { //Transpose for 2D
          if(!exist) {
            fftPlan->GenerateKernel(plHandle, fftRepo, count);
            count++;
          }

          CompileKernels(plHandle, fftPlan->gen, fftPlan, fftPlan->plHandleOrigin, exist);
          fftPlan->baked    = true;
          return  HCFFT_SUCCESS;
        }

        size_t length0 = fftPlan->length[0];
        size_t length1 = fftPlan->length[1];

        if (fftPlan->length[0] == 256 && fftPlan->length[1] == 256) {
          length0 += 8;
          length1 += 1;
        } else if (fftPlan->length[0] == 512 && fftPlan->length[1] == 512) {
          length0 += 1;
          length1 += 1;//length1 += 0;
        } else if (fftPlan->length[0] == 1024 && fftPlan->length[1] == 512) {
          length0 += 2;
          length1 += 2;//length1 += 0;
        } else if (fftPlan->length[0] == 1024 && fftPlan->length[1] == 1024) {
          length0 += 1;
          length1 += 1;//length1 += 0;
        }

        if (fftPlan->length[0] > Large1DThreshold ||
            fftPlan->length[1] > Large1DThreshold) {
          fftPlan->large2D = true;
        }

        while (1 && (fftPlan->ipLayout != HCFFT_REAL) && (fftPlan->opLayout != HCFFT_REAL)) {
          //break;
          if (fftPlan->length.size() != 2) {
            break;
          }

          if (!(IsPo2(fftPlan->length[0])) || !(IsPo2(fftPlan->length[1]))) {
            break;
          }

          if (fftPlan->length[1] < 32) {
            break;
          }

          //TBD: restrict the use large2D in x!=y case becase we will need two temp buffers
          //     (1) for 2D usage (2) for 1D large usage
          //if (fftPlan->large2D) break;
          //Performance show 512 is the good case with transpose
          //if user want the result to be transposed, then we will.
          if (fftPlan->length[0] < 512 && fftPlan->transposeType == HCFFT_NOTRANSPOSE) {
            break;
          }

          if (fftPlan->length[0] < 32) {
            break;
          }

          //x!=y case, we need tmp buffer, currently temp buffer only support interleaved format
          //if (fftPlan->length[0] != fftPlan->length[1] && fftPlan->opLayout == HCFFT_COMPLEX_PLANAR) break;
          if (fftPlan->inStride[0] != 1 || fftPlan->outStride[0] != 1 ||
              fftPlan->inStride[1] != fftPlan->length[0] || fftPlan->outStride[1] != fftPlan->length[0]) {
            break;
          }

          //if (fftPlan->location != HCFFT_INPLACE || fftPlan->ipLayout != HCFFT_COMPLEX_PLANAR)
          //  break;
          //if (fftPlan->batchSize != 1) break;
          //if (fftPlan->precision != HCFFT_SINGLE) break;
          fftPlan->transflag = true;
          //create row plan,
          // x=y & x!=y, In->In for inplace, In->out for outofplace
          hcfftCreateDefaultPlanInternal( &fftPlan->planX, HCFFT_1D, &fftPlan->length[ 0 ]);
          FFTPlan* rowPlan  = NULL;
          lockRAII* rowLock = NULL;
          fftRepo.getPlan( fftPlan->planX, rowPlan, rowLock );
          rowPlan->ipLayout     = fftPlan->ipLayout;
          rowPlan->opLayout    = fftPlan->opLayout;
          rowPlan->location       = fftPlan->location;
          rowPlan->outStride[0]    = fftPlan->outStride[0];
          rowPlan->outStride.push_back(fftPlan->outStride[1]);
          rowPlan->oDist           = fftPlan->oDist;
          rowPlan->precision       = fftPlan->precision;
          rowPlan->forwardScale    = 1.0f;
          rowPlan->backwardScale   = 1.0f;
          rowPlan->tmpBufSize      = 0;
          rowPlan->gen       = fftPlan->gen;
          rowPlan->envelope    = fftPlan->envelope;
          rowPlan->batchSize       = fftPlan->batchSize;
          rowPlan->inStride[0]     = fftPlan->inStride[0];
          rowPlan->length.push_back(fftPlan->length[1]);
          rowPlan->inStride.push_back(fftPlan->inStride[1]);
          rowPlan->iDist           = fftPlan->iDist;
          hcfftBakePlan(fftPlan->planX);
          //Create transpose plan for first transpose
          //x=y: inplace. x!=y inplace: in->tmp, outofplace out->tmp
          size_t clLengths[] = { 1, 1, 0 };
          clLengths[0] = fftPlan->length[0];
          clLengths[1] = fftPlan->length[1];
          bool xyflag = (clLengths[0] == clLengths[1]) ? false : true;

          if (xyflag && fftPlan->tmpBufSize == 0 && fftPlan->length.size() <= 2) {
            // we need tmp buffer for x!=y case
            // we assume the tmp buffer is packed interleaved
            fftPlan->tmpBufSize = length0 * length1 *
                                  fftPlan->batchSize * fftPlan->ElementSize();
          }

          hcfftCreateDefaultPlanInternal( &fftPlan->planTX, HCFFT_2D, clLengths);
          FFTPlan* transPlanX = NULL;
          lockRAII* transLockX  = NULL;
          fftRepo.getPlan( fftPlan->planTX, transPlanX, transLockX );
          transPlanX->ipLayout     = fftPlan->opLayout;
          transPlanX->gen         = Transpose;
          transPlanX->precision       = fftPlan->precision;
          transPlanX->tmpBufSize      = 0;
          transPlanX->envelope    = fftPlan->envelope;
          transPlanX->batchSize       = fftPlan->batchSize;
          transPlanX->inStride[0]     = fftPlan->outStride[0];
          transPlanX->inStride[1]     = fftPlan->outStride[1];
          transPlanX->iDist           = fftPlan->oDist;
          transPlanX->transflag       = true;

          if (xyflag) {
            transPlanX->opLayout    = HCFFT_COMPLEX_INTERLEAVED;
            transPlanX->location       = HCFFT_OUTOFPLACE;
            transPlanX->outStride[0]    = 1;
            transPlanX->outStride[1]    = clLengths[0];
            transPlanX->oDist           = clLengths[0] * clLengths[1];
          } else {
            transPlanX->opLayout    = fftPlan->opLayout;
            transPlanX->location       = HCFFT_INPLACE;
            transPlanX->outStride[0]    = fftPlan->outStride[0];
            transPlanX->outStride[1]    = fftPlan->outStride[1];
            transPlanX->oDist           = fftPlan->oDist;
          }

          hcfftBakePlan(fftPlan->planTX);
          //create second row plan
          //x!=y: tmp->tmp, x=y case: In->In or Out->Out
          //if Transposed result is a choice x!=y: tmp->In or out
          hcfftCreateDefaultPlanInternal( &fftPlan->planY, HCFFT_1D, &fftPlan->length[ 1 ]);
          FFTPlan* colPlan  = NULL;
          lockRAII* colLock = NULL;
          fftRepo.getPlan( fftPlan->planY, colPlan, colLock );

          if (xyflag) {
            colPlan->ipLayout     = HCFFT_COMPLEX_INTERLEAVED;
            colPlan->inStride[0]     = 1;
            colPlan->inStride.push_back(clLengths[1]);
            colPlan->iDist           = clLengths[0] * clLengths[1];

            if (fftPlan->transposeType == HCFFT_NOTRANSPOSE) {
              colPlan->opLayout    = HCFFT_COMPLEX_INTERLEAVED;
              colPlan->outStride[0]    = 1;
              colPlan->outStride.push_back(clLengths[1]);
              colPlan->oDist           = clLengths[0] * clLengths[1];
              colPlan->location       = HCFFT_INPLACE;
            } else {
              colPlan->opLayout    = fftPlan->opLayout;
              colPlan->outStride[0]    = fftPlan->outStride[0];
              colPlan->outStride.push_back(clLengths[1] * fftPlan->outStride[0]);
              colPlan->oDist           = fftPlan->oDist;
              colPlan->location       = HCFFT_OUTOFPLACE;
            }
          } else {
            colPlan->ipLayout     = fftPlan->opLayout;
            colPlan->opLayout    = fftPlan->opLayout;
            colPlan->outStride[0]    = fftPlan->outStride[0];
            colPlan->outStride.push_back(fftPlan->outStride[1]);
            colPlan->oDist           = fftPlan->oDist;
            colPlan->inStride[0]     = fftPlan->outStride[0];
            colPlan->inStride.push_back(fftPlan->outStride[1]);
            colPlan->iDist           = fftPlan->oDist;
            colPlan->location       = HCFFT_INPLACE;
          }

          colPlan->precision       = fftPlan->precision;
          colPlan->forwardScale    = fftPlan->forwardScale;
          colPlan->backwardScale   = fftPlan->backwardScale;
          colPlan->tmpBufSize      = 0;
          colPlan->gen       = fftPlan->gen;
          colPlan->envelope    = fftPlan->envelope;
          colPlan->batchSize       = fftPlan->batchSize;
          colPlan->length.push_back(fftPlan->length[0]);
          hcfftBakePlan(fftPlan->planY);

          if (fftPlan->transposeType == HCFFT_TRANSPOSED) {
            fftPlan->baked = true;
            return  HCFFT_SUCCESS;
          }

          //Create transpose plan for second transpose
          //x!=y case tmp->In or Out, x=y case In->In or Out->out
          clLengths[0] = fftPlan->length[1];
          clLengths[1] = fftPlan->length[0];
          hcfftCreateDefaultPlanInternal( &fftPlan->planTY, HCFFT_2D, clLengths );
          FFTPlan* transPlanY = NULL;
          lockRAII* transLockY  = NULL;
          fftRepo.getPlan( fftPlan->planTY, transPlanY, transLockY );

          if (xyflag) {
            transPlanY->ipLayout     = HCFFT_COMPLEX_INTERLEAVED;
            transPlanY->location       = HCFFT_OUTOFPLACE;
            transPlanY->inStride[0]     = 1;
            transPlanY->inStride[1]     = clLengths[0];
            transPlanY->iDist           = clLengths[0] * clLengths[1];
          } else {
            transPlanY->ipLayout     = fftPlan->opLayout;
            transPlanY->location       = HCFFT_INPLACE;
            transPlanY->inStride[0]     = fftPlan->outStride[0];
            transPlanY->inStride[1]     = fftPlan->outStride[1];
            transPlanY->iDist           = fftPlan->oDist;
          }

          transPlanY->opLayout    = fftPlan->opLayout;
          transPlanY->outStride[0]    = fftPlan->outStride[0];
          transPlanY->outStride[1]    = fftPlan->outStride[1];
          transPlanY->oDist           = fftPlan->oDist;
          transPlanY->precision       = fftPlan->precision;
          transPlanY->tmpBufSize      = 0;
          transPlanY->gen             = Transpose;
          transPlanY->envelope      = fftPlan->envelope;
          transPlanY->batchSize       = fftPlan->batchSize;
          transPlanY->transflag       = true;
          hcfftBakePlan(fftPlan->planTY);
          fftPlan->baked = true;
          return  HCFFT_SUCCESS;
        }

        //check transposed
        if (fftPlan->transposeType != HCFFT_NOTRANSPOSE) {
          return HCFFT_ERROR;
        }

        if(fftPlan->ipLayout == HCFFT_REAL) {
          length0 = fftPlan->length[0];
          length1 = fftPlan->length[1];
          size_t Nt = (1 + length0 / 2);
          // create row plan
          // real to hermitian
          //create row plan
          hcfftCreateDefaultPlanInternal( &fftPlan->planX, HCFFT_1D, &fftPlan->length[ 0 ]);
          FFTPlan* rowPlan  = NULL;
          lockRAII* rowLock = NULL;
          fftRepo.getPlan( fftPlan->planX, rowPlan, rowLock );
          rowPlan->plHandleOrigin  = fftPlan->plHandleOrigin;
          rowPlan->opLayout  = fftPlan->opLayout;
          rowPlan->ipLayout  = fftPlan->ipLayout;
          rowPlan->location     = fftPlan->location;
          rowPlan->length.push_back(length1);
          rowPlan->inStride[0]  = fftPlan->inStride[0];
          rowPlan->inStride.push_back(fftPlan->inStride[1]);
          rowPlan->iDist         = fftPlan->iDist;
          rowPlan->precision     = fftPlan->precision;
          rowPlan->forwardScale  = 1.0f;
          rowPlan->backwardScale = 1.0f;
          rowPlan->tmpBufSize    = 0;
          rowPlan->gen      = fftPlan->gen;
          rowPlan->envelope   = fftPlan->envelope;
          rowPlan->batchSize    = fftPlan->batchSize;
          rowPlan->outStride[0]  = fftPlan->outStride[0];
          rowPlan->outStride.push_back(fftPlan->outStride[1]);
          rowPlan->oDist         = fftPlan->oDist;

          //this 2d is decomposed from 3d
          for (size_t index = 2; index < fftPlan->length.size(); index++) {
            rowPlan->length.push_back(fftPlan->length[index]);
            rowPlan->inStride.push_back(fftPlan->inStride[index]);
            rowPlan->outStride.push_back(fftPlan->outStride[index]);
          }

          hcfftBakePlan(fftPlan->planX);

          if( (rowPlan->inStride[0] == 1) && (rowPlan->outStride[0] == 1) &&
              ( ((rowPlan->inStride[1] == Nt * 2) && (rowPlan->location == HCFFT_INPLACE)) ||
                ((rowPlan->inStride[1] == length0) && (rowPlan->location == HCFFT_OUTOFPLACE)) )
              && (rowPlan->outStride[1] == Nt) ) {
            // calc temp buf size
            if (fftPlan->tmpBufSize == 0) {
              fftPlan->tmpBufSize = Nt * length1 * fftPlan->batchSize * fftPlan->ElementSize();

              for (size_t index = 2; index < fftPlan->length.size(); index++) {
                fftPlan->tmpBufSize *= fftPlan->length[index];
              }
            }

            // create first transpose plan
            //Transpose
            // output --> tmp
            size_t transLengths[2] = { length0, length1 };
            hcfftCreateDefaultPlanInternal( &fftPlan->planTX, HCFFT_2D, transLengths );
            FFTPlan* trans1Plan = NULL;
            lockRAII* trans1Lock  = NULL;
            fftRepo.getPlan( fftPlan->planTX, trans1Plan, trans1Lock );
            trans1Plan->transflag = true;
            transLengths[0] = Nt;
            hcfftSetPlanLength( fftPlan->planTX, HCFFT_2D, transLengths );

            switch(fftPlan->opLayout) {
              case HCFFT_HERMITIAN_INTERLEAVED: {
                  trans1Plan->opLayout = HCFFT_COMPLEX_INTERLEAVED;
                  trans1Plan->ipLayout  = HCFFT_COMPLEX_INTERLEAVED;
                }
                break;

              case HCFFT_HERMITIAN_PLANAR: {
                  trans1Plan->opLayout = HCFFT_COMPLEX_INTERLEAVED;
                  trans1Plan->ipLayout  = HCFFT_COMPLEX_PLANAR;
                }
                break;

              default:
                assert(false);
            }

            trans1Plan->plHandleOrigin  = fftPlan->plHandleOrigin;
            trans1Plan->location     = HCFFT_OUTOFPLACE;
            trans1Plan->precision     = fftPlan->precision;
            trans1Plan->tmpBufSize    = 0;
            trans1Plan->batchSize     = fftPlan->batchSize;
            trans1Plan->envelope    = fftPlan->envelope;
            trans1Plan->forwardScale  = 1.0f;
            trans1Plan->backwardScale = 1.0f;
            trans1Plan->inStride[0]   = 1;
            trans1Plan->inStride[1]   = Nt;
            trans1Plan->outStride[0]  = 1;
            trans1Plan->outStride[1]  = length1;
            trans1Plan->iDist         = rowPlan->oDist;
            trans1Plan->oDist     = Nt * length1;
            trans1Plan->transOutHorizontal = true;
            trans1Plan->gen           = Transpose;

            for (size_t index = 2; index < fftPlan->length.size(); index++) {
              trans1Plan->length.push_back(fftPlan->length[index]);
              trans1Plan->inStride.push_back(rowPlan->outStride[index]);
              trans1Plan->outStride.push_back(trans1Plan->oDist);
              trans1Plan->oDist *= fftPlan->length[index];
            }

            hcfftBakePlan(fftPlan->planTX);
            // Create column plan as a row plan
            hcfftCreateDefaultPlanInternal( &fftPlan->planY, HCFFT_1D, &fftPlan->length[ 1 ]);
            FFTPlan* colPlan  = NULL;
            lockRAII* colLock = NULL;
            fftRepo.getPlan( fftPlan->planY, colPlan, colLock );
            colPlan->plHandleOrigin  = fftPlan->plHandleOrigin;
            colPlan->opLayout  = trans1Plan->opLayout;
            colPlan->ipLayout   = trans1Plan->opLayout;
            colPlan->location     = HCFFT_INPLACE;
            colPlan->length.push_back(Nt);
            colPlan->inStride[0]  = 1;
            colPlan->inStride.push_back(length1);
            colPlan->iDist         = Nt * length1;
            colPlan->outStride[0]  = 1;
            colPlan->outStride.push_back(length1);
            colPlan->oDist         = Nt * length1;
            colPlan->precision     = fftPlan->precision;
            colPlan->forwardScale  = fftPlan->forwardScale;
            colPlan->backwardScale = fftPlan->backwardScale;
            colPlan->tmpBufSize    = 0;
            colPlan->gen      = fftPlan->gen;
            colPlan->envelope   = fftPlan->envelope;
            colPlan->batchSize    = fftPlan->batchSize;

            //this 2d is decomposed from 3d
            for (size_t index = 2; index < fftPlan->length.size(); index++) {
              colPlan->length.push_back(fftPlan->length[index]);
              colPlan->inStride.push_back(colPlan->iDist);
              colPlan->outStride.push_back(colPlan->oDist);
              colPlan->iDist *= fftPlan->length[index];
              colPlan->oDist *= fftPlan->length[index];
            }

            hcfftBakePlan(fftPlan->planY);

            if (fftPlan->transposeType == HCFFT_TRANSPOSED) {
              fftPlan->baked = true;
              return  HCFFT_SUCCESS;
            }

            // create second transpose plan
            //Transpose
            //output --> tmp
            size_t trans2Lengths[2] = { length1, length0 };
            hcfftCreateDefaultPlanInternal( &fftPlan->planTY, HCFFT_2D, trans2Lengths );
            FFTPlan* trans2Plan = NULL;
            lockRAII* trans2Lock  = NULL;
            fftRepo.getPlan( fftPlan->planTY, trans2Plan, trans2Lock );
            trans2Plan->transflag = true;
            trans2Lengths[1] = Nt;
            hcfftSetPlanLength( fftPlan->planTY, HCFFT_2D, trans2Lengths );

            switch(fftPlan->opLayout) {
              case HCFFT_HERMITIAN_INTERLEAVED: {
                  trans2Plan->opLayout = HCFFT_COMPLEX_INTERLEAVED;
                  trans2Plan->ipLayout  = HCFFT_COMPLEX_INTERLEAVED;
                }
                break;

              case HCFFT_HERMITIAN_PLANAR: {
                  trans2Plan->opLayout = HCFFT_COMPLEX_PLANAR;
                  trans2Plan->ipLayout  = HCFFT_COMPLEX_INTERLEAVED;
                }
                break;

              default:
                assert(false);
            }

            trans2Plan->plHandleOrigin  = fftPlan->plHandleOrigin;
            trans2Plan->location     = HCFFT_OUTOFPLACE;
            trans2Plan->precision     = fftPlan->precision;
            trans2Plan->tmpBufSize    = 0;
            trans2Plan->batchSize     = fftPlan->batchSize;
            trans2Plan->envelope    = fftPlan->envelope;
            trans2Plan->forwardScale  = 1.0f;
            trans2Plan->backwardScale = 1.0f;
            trans2Plan->inStride[0]   = 1;
            trans2Plan->inStride[1]   = length1;
            trans2Plan->outStride[0]  = 1;
            trans2Plan->outStride[1]  = Nt;
            trans2Plan->iDist         = Nt * length1;
            trans2Plan->oDist     = fftPlan->oDist;
            trans2Plan->gen           = Transpose;
            trans2Plan->transflag     = true;

            for (size_t index = 2; index < fftPlan->length.size(); index++) {
              trans2Plan->length.push_back(fftPlan->length[index]);
              trans2Plan->inStride.push_back(trans2Plan->iDist);
              trans2Plan->iDist *= fftPlan->length[index];
              trans2Plan->outStride.push_back(fftPlan->outStride[index]);
            }

            hcfftBakePlan(fftPlan->planTY);
          } else {
            // create col plan
            // complex to complex
            hcfftCreateDefaultPlanInternal( &fftPlan->planY, HCFFT_1D, &fftPlan->length[ 1 ] );
            FFTPlan* colPlan  = NULL;
            lockRAII* colLock = NULL;
            fftRepo.getPlan( fftPlan->planY, colPlan, colLock );

            switch(fftPlan->opLayout) {
              case HCFFT_HERMITIAN_INTERLEAVED: {
                  colPlan->opLayout = HCFFT_COMPLEX_INTERLEAVED;
                  colPlan->ipLayout  = HCFFT_COMPLEX_INTERLEAVED;
                }
                break;

              case HCFFT_HERMITIAN_PLANAR: {
                  colPlan->opLayout = HCFFT_COMPLEX_PLANAR;
                  colPlan->ipLayout  = HCFFT_COMPLEX_PLANAR;
                }
                break;

              default:
                assert(false);
            }

            colPlan->location     = HCFFT_INPLACE;
            colPlan->length.push_back(Nt);
            colPlan->outStride[0]  = fftPlan->outStride[1];
            colPlan->outStride.push_back(fftPlan->outStride[0]);
            colPlan->oDist         = fftPlan->oDist;
            colPlan->plHandleOrigin  = fftPlan->plHandleOrigin;
            colPlan->precision     = fftPlan->precision;
            colPlan->forwardScale  = fftPlan->forwardScale;
            colPlan->backwardScale = fftPlan->backwardScale;
            colPlan->tmpBufSize    = fftPlan->tmpBufSize;
            colPlan->gen      = fftPlan->gen;
            colPlan->envelope     = fftPlan->envelope;
            colPlan->batchSize = fftPlan->batchSize;
            colPlan->inStride[0]  = rowPlan->outStride[1];
            colPlan->inStride.push_back(rowPlan->outStride[0]);
            colPlan->iDist         = rowPlan->oDist;

            //this 2d is decomposed from 3d
            for (size_t index = 2; index < fftPlan->length.size(); index++) {
              colPlan->length.push_back(fftPlan->length[index]);
              colPlan->outStride.push_back(fftPlan->outStride[index]);
              colPlan->inStride.push_back(rowPlan->outStride[index]);
            }

            hcfftBakePlan(fftPlan->planY);
          }
        } else if(fftPlan->opLayout == HCFFT_REAL) {
          length0 = fftPlan->length[0];
          length1 = fftPlan->length[1];
          size_t Nt = (1 + length0 / 2);

          if (fftPlan->tmpBufSize == 0) {
            fftPlan->tmpBufSize = Nt * length1 * fftPlan->batchSize * fftPlan->ElementSize();

            for (size_t index = 2; index < fftPlan->length.size(); index++) {
              fftPlan->tmpBufSize *= fftPlan->length[index];
            }
          }

          if ((fftPlan->tmpBufSizeC2R == 0) && (fftPlan->location == HCFFT_OUTOFPLACE) && (fftPlan->length.size() == 2)) {
            fftPlan->tmpBufSizeC2R = fftPlan->tmpBufSize;
          }

          if( (fftPlan->inStride[0] == 1) && (fftPlan->outStride[0] == 1) &&
              ( ((fftPlan->outStride[1] == Nt * 2) && (fftPlan->oDist == Nt * 2 * length1) && (fftPlan->location == HCFFT_INPLACE)) ||
                ((fftPlan->outStride[1] == length0) && (fftPlan->oDist == length0 * length1) && (fftPlan->location == HCFFT_OUTOFPLACE)) )
              && (fftPlan->inStride[1] == Nt) && (fftPlan->iDist == Nt * length1) ) {
            // create first transpose plan
            //Transpose
            // input --> tmp
            size_t transLengths[2] = { length0, length1 };
            hcfftCreateDefaultPlanInternal( &fftPlan->planTY, HCFFT_2D, transLengths);
            FFTPlan* trans1Plan = NULL;
            lockRAII* trans1Lock  = NULL;
            fftRepo.getPlan( fftPlan->planTY, trans1Plan, trans1Lock );
            trans1Plan->transflag = true;
            transLengths[0] = Nt;
            hcfftSetPlanLength( fftPlan->planTY, HCFFT_2D, transLengths );

            switch(fftPlan->ipLayout) {
              case HCFFT_HERMITIAN_INTERLEAVED: {
                  trans1Plan->opLayout = HCFFT_COMPLEX_INTERLEAVED;
                  trans1Plan->ipLayout  = HCFFT_COMPLEX_INTERLEAVED;
                }
                break;

              case HCFFT_HERMITIAN_PLANAR: {
                  trans1Plan->opLayout = HCFFT_COMPLEX_INTERLEAVED;
                  trans1Plan->ipLayout  = HCFFT_COMPLEX_PLANAR;
                }
                break;

              default:
                assert(false);
            }

            trans1Plan->plHandleOrigin  = fftPlan->plHandleOrigin;
            trans1Plan->location     = HCFFT_OUTOFPLACE;
            trans1Plan->precision     = fftPlan->precision;
            trans1Plan->tmpBufSize    = 0;
            trans1Plan->batchSize     = fftPlan->batchSize;
            trans1Plan->envelope    = fftPlan->envelope;
            trans1Plan->forwardScale  = 1.0f;
            trans1Plan->backwardScale = 1.0f;
            trans1Plan->inStride[0]   = 1;
            trans1Plan->inStride[1]   = Nt;
            trans1Plan->outStride[0]  = 1;
            trans1Plan->outStride[1]  = length1;
            trans1Plan->iDist         = fftPlan->iDist;
            trans1Plan->oDist   = Nt * length1;
            trans1Plan->transOutHorizontal = true;
            trans1Plan->gen           = Transpose;

            for (size_t index = 2; index < fftPlan->length.size(); index++) {
              trans1Plan->length.push_back(fftPlan->length[index]);
              trans1Plan->inStride.push_back(fftPlan->inStride[index]);
              trans1Plan->outStride.push_back(trans1Plan->oDist);
              trans1Plan->oDist *= fftPlan->length[index];
            }

            hcfftBakePlan(fftPlan->planTY);
            // create col plan
            // complex to complex
            hcfftCreateDefaultPlanInternal( &fftPlan->planY, HCFFT_1D, &fftPlan->length[ 1 ] );
            FFTPlan* colPlan  = NULL;
            lockRAII* colLock = NULL;
            fftRepo.getPlan( fftPlan->planY, colPlan, colLock );
            colPlan->length.push_back(Nt);
            colPlan->inStride[0]  = 1;
            colPlan->inStride.push_back(length1);
            colPlan->iDist         = trans1Plan->oDist;
            colPlan->plHandleOrigin  = fftPlan->plHandleOrigin;
            colPlan->location = HCFFT_INPLACE;
            colPlan->ipLayout = HCFFT_COMPLEX_INTERLEAVED;
            colPlan->opLayout = HCFFT_COMPLEX_INTERLEAVED;
            colPlan->outStride[0]  = colPlan->inStride[0];
            colPlan->outStride.push_back(colPlan->inStride[1]);
            colPlan->oDist         = colPlan->iDist;

            for (size_t index = 2; index < fftPlan->length.size(); index++) {
              colPlan->length.push_back(fftPlan->length[index]);
              colPlan->inStride.push_back(trans1Plan->outStride[index]);
              colPlan->outStride.push_back(trans1Plan->outStride[index]);
            }

            colPlan->precision     = fftPlan->precision;
            colPlan->forwardScale  = 1.0f;
            colPlan->backwardScale = 1.0f;
            colPlan->tmpBufSize    = 0;
            colPlan->gen      = fftPlan->gen;
            colPlan->envelope   = fftPlan->envelope;
            colPlan->batchSize = fftPlan->batchSize;
            hcfftBakePlan(fftPlan->planY);
            // create second transpose plan
            //Transpose
            //tmp --> output
            size_t trans2Lengths[2] = { length1, length0 };
            hcfftCreateDefaultPlanInternal( &fftPlan->planTX, HCFFT_2D, trans2Lengths );
            FFTPlan* trans2Plan = NULL;
            lockRAII* trans2Lock  = NULL;
            fftRepo.getPlan( fftPlan->planTX, trans2Plan, trans2Lock );
            trans2Plan->transflag = true;
            trans2Lengths[1] = Nt;
            hcfftSetPlanLength( fftPlan->planTX, HCFFT_2D, trans2Lengths );
            trans2Plan->opLayout = HCFFT_COMPLEX_INTERLEAVED;
            trans2Plan->ipLayout  = HCFFT_COMPLEX_INTERLEAVED;
            trans2Plan->plHandleOrigin  = fftPlan->plHandleOrigin;
            trans2Plan->location     = HCFFT_OUTOFPLACE;
            trans2Plan->precision     = fftPlan->precision;
            trans2Plan->tmpBufSize    = 0;
            trans2Plan->batchSize     = fftPlan->batchSize;
            trans2Plan->envelope    = fftPlan->envelope;
            trans2Plan->forwardScale  = 1.0f;
            trans2Plan->backwardScale = 1.0f;
            trans2Plan->inStride[0]   = 1;
            trans2Plan->inStride[1]   = length1;
            trans2Plan->outStride[0]  = 1;
            trans2Plan->outStride[1]  = Nt;
            trans2Plan->iDist         = colPlan->oDist;
            trans2Plan->oDist     = Nt * length1;
            trans2Plan->transflag     = true;
            trans2Plan->gen           = Transpose;

            for (size_t index = 2; index < fftPlan->length.size(); index++) {
              trans2Plan->length.push_back(fftPlan->length[index]);
              trans2Plan->inStride.push_back(colPlan->outStride[index]);
              trans2Plan->outStride.push_back(trans2Plan->oDist);
              trans2Plan->oDist *= fftPlan->length[index];
            }

            hcfftBakePlan(fftPlan->planTX);
            // create row plan
            // hermitian to real
            //create row plan
            hcfftCreateDefaultPlanInternal( &fftPlan->planX, HCFFT_1D, &fftPlan->length[ 0 ]);
            FFTPlan* rowPlan  = NULL;
            lockRAII* rowLock = NULL;
            fftRepo.getPlan( fftPlan->planX, rowPlan, rowLock );
            rowPlan->plHandleOrigin  = fftPlan->plHandleOrigin;
            rowPlan->opLayout  = fftPlan->opLayout;
            rowPlan->ipLayout   = HCFFT_HERMITIAN_INTERLEAVED;
            rowPlan->length.push_back(length1);
            rowPlan->outStride[0]  = fftPlan->outStride[0];
            rowPlan->outStride.push_back(fftPlan->outStride[1]);
            rowPlan->oDist         = fftPlan->oDist;
            rowPlan->inStride[0]  = trans2Plan->outStride[0];
            rowPlan->inStride.push_back(trans2Plan->outStride[1]);
            rowPlan->iDist         = trans2Plan->oDist;

            for (size_t index = 2; index < fftPlan->length.size(); index++) {
              rowPlan->length.push_back(fftPlan->length[index]);
              rowPlan->inStride.push_back(trans2Plan->outStride[index]);
              rowPlan->outStride.push_back(fftPlan->outStride[index]);
            }

            if (fftPlan->location == HCFFT_INPLACE) {
              rowPlan->location     = HCFFT_INPLACE;
            } else {
              rowPlan->location     = HCFFT_OUTOFPLACE;
            }

            rowPlan->precision     = fftPlan->precision;
            rowPlan->forwardScale  = fftPlan->forwardScale;
            rowPlan->backwardScale = fftPlan->backwardScale;
            rowPlan->tmpBufSize    = 0;
            rowPlan->gen      = fftPlan->gen;
            rowPlan->envelope   = fftPlan->envelope;
            rowPlan->batchSize    = fftPlan->batchSize;
            hcfftBakePlan(fftPlan->planX);
          } else {
            // create col plan
            // complex to complex
            hcfftCreateDefaultPlanInternal( &fftPlan->planY, HCFFT_1D, &fftPlan->length[ 1 ]);
            FFTPlan* colPlan  = NULL;
            lockRAII* colLock = NULL;
            fftRepo.getPlan( fftPlan->planY, colPlan, colLock );

            switch(fftPlan->ipLayout) {
              case HCFFT_HERMITIAN_INTERLEAVED: {
                  colPlan->opLayout = HCFFT_COMPLEX_INTERLEAVED;
                  colPlan->ipLayout  = HCFFT_COMPLEX_INTERLEAVED;
                }
                break;

              case HCFFT_HERMITIAN_PLANAR: {
                  colPlan->opLayout = HCFFT_COMPLEX_INTERLEAVED;
                  colPlan->ipLayout  = HCFFT_COMPLEX_PLANAR;
                }
                break;

              default:
                assert(false);
            }

            colPlan->length.push_back(Nt);
            colPlan->inStride[0]  = fftPlan->inStride[1];
            colPlan->inStride.push_back(fftPlan->inStride[0]);
            colPlan->iDist         = fftPlan->iDist;

            if (fftPlan->location == HCFFT_INPLACE) {
              colPlan->location = HCFFT_INPLACE;
            } else {
              if(fftPlan->length.size() > 2) {
                colPlan->location = HCFFT_INPLACE;
              } else {
                colPlan->location = HCFFT_OUTOFPLACE;
              }
            }

            if(colPlan->location == HCFFT_INPLACE) {
              colPlan->outStride[0]  = colPlan->inStride[0];
              colPlan->outStride.push_back(colPlan->inStride[1]);
              colPlan->oDist         = colPlan->iDist;

              for (size_t index = 2; index < fftPlan->length.size(); index++) {
                colPlan->length.push_back(fftPlan->length[index]);
                colPlan->inStride.push_back(fftPlan->inStride[index]);
                colPlan->outStride.push_back(fftPlan->inStride[index]);
              }
            } else {
              colPlan->outStride[0]  = Nt;
              colPlan->outStride.push_back(1);
              colPlan->oDist         = Nt * length1;

              for (size_t index = 2; index < fftPlan->length.size(); index++) {
                colPlan->length.push_back(fftPlan->length[index]);
                colPlan->inStride.push_back(fftPlan->inStride[index]);
                colPlan->outStride.push_back(colPlan->oDist);
                colPlan->oDist *= fftPlan->length[index];
              }
            }

            colPlan->plHandleOrigin  = fftPlan->plHandleOrigin;
            colPlan->precision     = fftPlan->precision;
            colPlan->forwardScale  = 1.0f;
            colPlan->backwardScale = 1.0f;
            colPlan->tmpBufSize    = 0;
            colPlan->gen    = fftPlan->gen;
            colPlan->envelope = fftPlan->envelope;
            colPlan->batchSize = fftPlan->batchSize;
            hcfftBakePlan(fftPlan->planY);
            // create row plan
            // hermitian to real
            //create row plan
            hcfftCreateDefaultPlanInternal( &fftPlan->planX, HCFFT_1D, &fftPlan->length[ 0 ]);
            FFTPlan* rowPlan  = NULL;
            lockRAII* rowLock = NULL;
            fftRepo.getPlan( fftPlan->planX, rowPlan, rowLock );
            rowPlan->opLayout  = fftPlan->opLayout;
            rowPlan->ipLayout   = HCFFT_HERMITIAN_INTERLEAVED;
            rowPlan->length.push_back(length1);
            rowPlan->outStride[0]  = fftPlan->outStride[0];
            rowPlan->outStride.push_back(fftPlan->outStride[1]);
            rowPlan->oDist         = fftPlan->oDist;

            if (fftPlan->location == HCFFT_INPLACE) {
              rowPlan->location     = HCFFT_INPLACE;
              rowPlan->inStride[0]  = colPlan->outStride[1];
              rowPlan->inStride.push_back(colPlan->outStride[0]);
              rowPlan->iDist         = colPlan->oDist;

              for (size_t index = 2; index < fftPlan->length.size(); index++) {
                rowPlan->length.push_back(fftPlan->length[index]);
                rowPlan->inStride.push_back(colPlan->outStride[index]);
                rowPlan->outStride.push_back(fftPlan->outStride[index]);
              }
            } else {
              rowPlan->location     = HCFFT_OUTOFPLACE;
              rowPlan->inStride[0]   = 1;
              rowPlan->inStride.push_back(Nt);
              rowPlan->iDist         = Nt * length1;

              for (size_t index = 2; index < fftPlan->length.size(); index++) {
                rowPlan->length.push_back(fftPlan->length[index]);
                rowPlan->outStride.push_back(fftPlan->outStride[index]);
                rowPlan->inStride.push_back(rowPlan->iDist);
                rowPlan->iDist *= fftPlan->length[index];
              }
            }

            rowPlan->plHandleOrigin  = fftPlan->plHandleOrigin;
            rowPlan->precision     = fftPlan->precision;
            rowPlan->forwardScale  = fftPlan->forwardScale;
            rowPlan->backwardScale = fftPlan->backwardScale;
            rowPlan->tmpBufSize    = 0;
            rowPlan->gen      = fftPlan->gen;
            rowPlan->envelope   = fftPlan->envelope;
            rowPlan->batchSize    = fftPlan->batchSize;
            hcfftBakePlan(fftPlan->planX);
          }
        } else {
          if (fftPlan->tmpBufSize == 0 && fftPlan->length.size() <= 2) {
            fftPlan->tmpBufSize = length0 * length1 *
                                  fftPlan->batchSize * fftPlan->ElementSize();
          }

          //create row plan
          hcfftCreateDefaultPlanInternal( &fftPlan->planX, HCFFT_1D, &fftPlan->length[ 0 ]);
          FFTPlan* rowPlan  = NULL;
          lockRAII* rowLock = NULL;
          fftRepo.getPlan( fftPlan->planX, rowPlan, rowLock );
          rowPlan->ipLayout   = fftPlan->ipLayout;

          if (fftPlan->large2D || fftPlan->length.size() > 2) {
            rowPlan->opLayout  = fftPlan->opLayout;
            rowPlan->location     = fftPlan->location;
            rowPlan->outStride[0]  = fftPlan->outStride[0];
            rowPlan->outStride.push_back(fftPlan->outStride[1]);
            rowPlan->oDist         = fftPlan->oDist;
          } else {
            rowPlan->opLayout  = HCFFT_COMPLEX_INTERLEAVED;
            rowPlan->location     = HCFFT_OUTOFPLACE;
            rowPlan->outStride[0]  = length1;//1;
            rowPlan->outStride.push_back(1);//length0);
            rowPlan->oDist         = length0 * length1;
          }

          rowPlan->precision     = fftPlan->precision;
          rowPlan->forwardScale  = 1.0f;
          rowPlan->backwardScale = 1.0f;
          rowPlan->tmpBufSize    = fftPlan->tmpBufSize;
          rowPlan->plHandleOrigin  = fftPlan->plHandleOrigin;
          rowPlan->gen    = fftPlan->gen;
          rowPlan->envelope = fftPlan->envelope;
          // This is the row fft, the first elements distance between the first two FFTs is the distance of the first elements
          // of the first two rows in the original buffer.
          rowPlan->batchSize    = fftPlan->batchSize;
          rowPlan->inStride[0]  = fftPlan->inStride[0];
          //pass length and other info to kernel, so the kernel knows this is decomposed from higher dimension
          rowPlan->length.push_back(fftPlan->length[1]);
          rowPlan->inStride.push_back(fftPlan->inStride[1]);

          //this 2d is decomposed from 3d
          if (fftPlan->length.size() > 2) {
            rowPlan->length.push_back(fftPlan->length[2]);
            rowPlan->inStride.push_back(fftPlan->inStride[2]);
            rowPlan->outStride.push_back(fftPlan->outStride[2]);
          }

          rowPlan->iDist    = fftPlan->iDist;
          hcfftBakePlan(fftPlan->planX);
          //create col plan
          hcfftCreateDefaultPlanInternal( &fftPlan->planY, HCFFT_1D, &fftPlan->length[ 1 ] );
          FFTPlan* colPlan  = NULL;
          lockRAII* colLock = NULL;
          fftRepo.getPlan( fftPlan->planY, colPlan, colLock );

          if (fftPlan->large2D || fftPlan->length.size() > 2) {
            colPlan->ipLayout   = fftPlan->opLayout;
            colPlan->location     = HCFFT_INPLACE;
            colPlan->inStride[0]   = fftPlan->outStride[1];
            colPlan->inStride.push_back(fftPlan->outStride[0]);
            colPlan->iDist         = fftPlan->oDist;
          } else {
            colPlan->ipLayout   = HCFFT_COMPLEX_INTERLEAVED;
            colPlan->location     = HCFFT_OUTOFPLACE;
            colPlan->inStride[0]   = 1;//length0;
            colPlan->inStride.push_back(length1);//1);
            colPlan->iDist         = length0 * length1;
          }

          colPlan->plHandleOrigin  = fftPlan->plHandleOrigin;
          colPlan->opLayout  = fftPlan->opLayout;
          colPlan->precision     = fftPlan->precision;
          colPlan->forwardScale  = fftPlan->forwardScale;
          colPlan->backwardScale = fftPlan->backwardScale;
          colPlan->tmpBufSize    = fftPlan->tmpBufSize;
          colPlan->gen  = fftPlan->gen;
          colPlan->envelope = fftPlan->envelope;
          // This is a column FFT, the first elements distance between each FFT is the distance of the first two
          // elements in the original buffer. Like a transpose of the matrix
          colPlan->batchSize = fftPlan->batchSize;
          colPlan->outStride[0] = fftPlan->outStride[1];
          //pass length and other info to kernel, so the kernel knows this is decomposed from higher dimension
          colPlan->length.push_back(fftPlan->length[0]);
          colPlan->outStride.push_back(fftPlan->outStride[0]);
          colPlan->oDist    = fftPlan->oDist;

          //this 2d is decomposed from 3d
          if (fftPlan->length.size() > 2) {
            //assert(fftPlan->large2D);
            colPlan->length.push_back(fftPlan->length[2]);
            colPlan->inStride.push_back(fftPlan->outStride[2]);
            colPlan->outStride.push_back(fftPlan->outStride[2]);
          }

          hcfftBakePlan(fftPlan->planY);
        }

        fftPlan->baked = true;
        return  HCFFT_SUCCESS;
      }

    case HCFFT_3D: {
        if(fftPlan->ipLayout == HCFFT_REAL) {
          size_t length0 = fftPlan->length[ 0 ];
          size_t length1 = fftPlan->length[ 1 ];
          size_t length2 = fftPlan->length[ 2 ];
          size_t Nt = (1 + length0 / 2);
          //create 2D xy plan
          size_t clLengths[] = { length0, length1, 0 };
          hcfftCreateDefaultPlanInternal( &fftPlan->planX, HCFFT_2D, clLengths );
          FFTPlan* xyPlan = NULL;
          lockRAII* rowLock = NULL;
          fftRepo.getPlan( fftPlan->planX, xyPlan, rowLock );
          xyPlan->plHandleOrigin  = fftPlan->plHandleOrigin;
          xyPlan->ipLayout   = fftPlan->ipLayout;
          xyPlan->opLayout  = fftPlan->opLayout;
          xyPlan->location     = fftPlan->location;
          xyPlan->precision     = fftPlan->precision;
          xyPlan->forwardScale  = 1.0f;
          xyPlan->backwardScale = 1.0f;
          xyPlan->tmpBufSize    = fftPlan->tmpBufSize;
          xyPlan->gen      = fftPlan->gen;
          xyPlan->envelope       = fftPlan->envelope;
          // This is the xy fft, the first elements distance between the first two FFTs is the distance of the first elements
          // of the first two rows in the original buffer.
          xyPlan->batchSize    = fftPlan->batchSize;
          xyPlan->inStride[0]  = fftPlan->inStride[0];
          xyPlan->inStride[1]  = fftPlan->inStride[1];
          xyPlan->outStride[0] = fftPlan->outStride[0];
          xyPlan->outStride[1] = fftPlan->outStride[1];
          //pass length and other info to kernel, so the kernel knows this is decomposed from higher dimension
          xyPlan->length.push_back(fftPlan->length[2]);
          xyPlan->inStride.push_back(fftPlan->inStride[2]);
          xyPlan->outStride.push_back(fftPlan->outStride[2]);
          xyPlan->iDist    = fftPlan->iDist;
          xyPlan->oDist    = fftPlan->oDist;

          //this 3d is decomposed from 4d
          for (size_t index = 3; index < fftPlan->length.size(); index++) {
            xyPlan->length.push_back(fftPlan->length[index]);
            xyPlan->inStride.push_back(fftPlan->inStride[index]);
            xyPlan->outStride.push_back(fftPlan->outStride[index]);
          }

          hcfftBakePlan(fftPlan->planX);

          if( (xyPlan->inStride[0] == 1) && (xyPlan->outStride[0] == 1) &&
              (xyPlan->outStride[2] == Nt * length1) &&
              ( ((xyPlan->inStride[2] == Nt * 2 * length1) && (xyPlan->location == HCFFT_INPLACE)) ||
                ((xyPlan->inStride[2] == length0 * length1) && (xyPlan->location == HCFFT_OUTOFPLACE)) ) ) {
            if (fftPlan->tmpBufSize == 0) {
              fftPlan->tmpBufSize = Nt * length1 * length2 * fftPlan->batchSize * fftPlan->ElementSize();

              for (size_t index = 3; index < fftPlan->length.size(); index++) {
                fftPlan->tmpBufSize *= fftPlan->length[index];
              }
            }

            // create first transpose plan
            //Transpose
            // output --> tmp
            size_t transLengths[2] = { length0 * length1, length2 };
            hcfftCreateDefaultPlanInternal( &fftPlan->planTX, HCFFT_2D, transLengths );
            FFTPlan* trans1Plan = NULL;
            lockRAII* trans1Lock  = NULL;
            fftRepo.getPlan( fftPlan->planTX, trans1Plan, trans1Lock );
            trans1Plan->transflag = true;
            transLengths[0] = Nt * length1;
            hcfftSetPlanLength( fftPlan->planTX, HCFFT_2D, transLengths );

            switch(fftPlan->opLayout) {
              case HCFFT_HERMITIAN_INTERLEAVED: {
                  trans1Plan->opLayout = HCFFT_COMPLEX_INTERLEAVED;
                  trans1Plan->ipLayout  = HCFFT_COMPLEX_INTERLEAVED;
                }
                break;

              case HCFFT_HERMITIAN_PLANAR: {
                  trans1Plan->opLayout = HCFFT_COMPLEX_INTERLEAVED;
                  trans1Plan->ipLayout  = HCFFT_COMPLEX_PLANAR;
                }
                break;

              default:
                assert(false);
            }

            trans1Plan->plHandleOrigin  = fftPlan->plHandleOrigin;
            trans1Plan->location     = HCFFT_OUTOFPLACE;
            trans1Plan->precision     = fftPlan->precision;
            trans1Plan->tmpBufSize    = 0;
            trans1Plan->batchSize     = fftPlan->batchSize;
            trans1Plan->envelope    = fftPlan->envelope;
            trans1Plan->forwardScale  = 1.0f;
            trans1Plan->backwardScale = 1.0f;
            trans1Plan->inStride[0]   = 1;
            trans1Plan->inStride[1]   = Nt * length1;
            trans1Plan->outStride[0]  = 1;
            trans1Plan->outStride[1]  = length2;
            trans1Plan->iDist         = xyPlan->oDist;
            trans1Plan->oDist     = Nt * length1 * length2;
            trans1Plan->transOutHorizontal = true;
            trans1Plan->gen           = Transpose;

            for (size_t index = 3; index < fftPlan->length.size(); index++) {
              trans1Plan->length.push_back(fftPlan->length[index]);
              trans1Plan->inStride.push_back(xyPlan->outStride[index]);
              trans1Plan->outStride.push_back(trans1Plan->oDist);
              trans1Plan->oDist *= fftPlan->length[index];
            }

            hcfftBakePlan(fftPlan->planTX);
            // Create column plan as a row plan
            hcfftCreateDefaultPlanInternal( &fftPlan->planZ, HCFFT_1D, &fftPlan->length[2] );
            FFTPlan* colPlan  = NULL;
            lockRAII* colLock = NULL;
            fftRepo.getPlan( fftPlan->planZ, colPlan, colLock );
            colPlan->plHandleOrigin  = fftPlan->plHandleOrigin;
            colPlan->opLayout  = trans1Plan->opLayout;
            colPlan->ipLayout   = trans1Plan->opLayout;
            colPlan->location     = HCFFT_INPLACE;
            colPlan->length.push_back(Nt * length1);
            colPlan->inStride[0]  = 1;
            colPlan->inStride.push_back(length2);
            colPlan->iDist         = Nt * length1 * length2;
            colPlan->outStride[0]  = 1;
            colPlan->outStride.push_back(length2);
            colPlan->oDist         = Nt * length1 * length2;
            colPlan->precision     = fftPlan->precision;
            colPlan->forwardScale  = fftPlan->forwardScale;
            colPlan->backwardScale = fftPlan->backwardScale;
            colPlan->tmpBufSize    = 0;
            colPlan->gen      = fftPlan->gen;
            colPlan->envelope   = fftPlan->envelope;
            colPlan->batchSize    = fftPlan->batchSize;

            //this 2d is decomposed from 3d
            for (size_t index = 3; index < fftPlan->length.size(); index++) {
              colPlan->length.push_back(fftPlan->length[index]);
              colPlan->inStride.push_back(colPlan->iDist);
              colPlan->outStride.push_back(colPlan->oDist);
              colPlan->iDist *= fftPlan->length[index];
              colPlan->oDist *= fftPlan->length[index];
            }

            hcfftBakePlan(fftPlan->planZ);

            if (fftPlan->transposeType == HCFFT_TRANSPOSED) {
              fftPlan->baked = true;
              return  HCFFT_SUCCESS;
            }

            // create second transpose plan
            //Transpose
            //output --> tmp
            size_t trans2Lengths[2] = { length2, length0 * length1 };
            hcfftCreateDefaultPlanInternal( &fftPlan->planTY, HCFFT_2D, trans2Lengths );
            FFTPlan* trans2Plan = NULL;
            lockRAII* trans2Lock  = NULL;
            fftRepo.getPlan( fftPlan->planTY, trans2Plan, trans2Lock );
            trans2Plan->transflag = true;
            trans2Lengths[1] = Nt * length1;
            hcfftSetPlanLength( fftPlan->planTY, HCFFT_2D, trans2Lengths );

            switch(fftPlan->opLayout) {
              case HCFFT_HERMITIAN_INTERLEAVED: {
                  trans2Plan->opLayout = HCFFT_COMPLEX_INTERLEAVED;
                  trans2Plan->ipLayout  = HCFFT_COMPLEX_INTERLEAVED;
                }
                break;

              case HCFFT_HERMITIAN_PLANAR: {
                  trans2Plan->opLayout = HCFFT_COMPLEX_PLANAR;
                  trans2Plan->ipLayout  = HCFFT_COMPLEX_INTERLEAVED;
                }
                break;

              default:
                assert(false);
            }

            trans2Plan->plHandleOrigin  = fftPlan->plHandleOrigin;
            trans2Plan->location     = HCFFT_OUTOFPLACE;
            trans2Plan->precision     = fftPlan->precision;
            trans2Plan->tmpBufSize    = 0;
            trans2Plan->batchSize     = fftPlan->batchSize;
            trans2Plan->envelope    = fftPlan->envelope;
            trans2Plan->forwardScale  = 1.0f;
            trans2Plan->backwardScale = 1.0f;
            trans2Plan->inStride[0]   = 1;
            trans2Plan->inStride[1]   = length2;
            trans2Plan->outStride[0]  = 1;
            trans2Plan->outStride[1]  = Nt * length1;
            trans2Plan->iDist         = Nt * length1 * length2;
            trans2Plan->oDist     = fftPlan->oDist;
            trans2Plan->gen           = Transpose;
            trans2Plan->transflag     = true;

            for (size_t index = 3; index < fftPlan->length.size(); index++) {
              trans2Plan->length.push_back(fftPlan->length[index]);
              trans2Plan->inStride.push_back(trans2Plan->iDist);
              trans2Plan->iDist *= fftPlan->length[index];
              trans2Plan->outStride.push_back(fftPlan->outStride[index]);
            }

            hcfftBakePlan(fftPlan->planTY);
          } else {
            clLengths[0] = fftPlan->length[ 2 ];
            clLengths[1] = clLengths[2] = 0;
            //create 1D col plan
            hcfftCreateDefaultPlanInternal( &fftPlan->planZ, HCFFT_1D, clLengths );
            FFTPlan* colPlan  = NULL;
            lockRAII* colLock = NULL;
            fftRepo.getPlan( fftPlan->planZ, colPlan, colLock );

            switch(fftPlan->opLayout) {
              case HCFFT_HERMITIAN_INTERLEAVED: {
                  colPlan->opLayout = HCFFT_COMPLEX_INTERLEAVED;
                  colPlan->ipLayout  = HCFFT_COMPLEX_INTERLEAVED;
                }
                break;

              case HCFFT_HERMITIAN_PLANAR: {
                  colPlan->opLayout = HCFFT_COMPLEX_PLANAR;
                  colPlan->ipLayout  = HCFFT_COMPLEX_PLANAR;
                }
                break;

              default:
                assert(false);
            }

            colPlan->plHandleOrigin  = fftPlan->plHandleOrigin;
            colPlan->location     = HCFFT_INPLACE;
            colPlan->precision     = fftPlan->precision;
            colPlan->forwardScale  = fftPlan->forwardScale;
            colPlan->backwardScale = fftPlan->backwardScale;
            colPlan->tmpBufSize    = fftPlan->tmpBufSize;
            colPlan->gen       = fftPlan->gen;
            colPlan->envelope      = fftPlan->envelope;
            // This is a column FFT, the first elements distance between each FFT is the distance of the first two
            // elements in the original buffer. Like a transpose of the matrix
            colPlan->batchSize = fftPlan->batchSize;
            colPlan->inStride[0] = fftPlan->outStride[2];
            colPlan->outStride[0] = fftPlan->outStride[2];
            //pass length and other info to kernel, so the kernel knows this is decomposed from higher dimension
            colPlan->length.push_back(1 + fftPlan->length[0] / 2);
            colPlan->length.push_back(fftPlan->length[1]);
            colPlan->inStride.push_back(fftPlan->outStride[0]);
            colPlan->inStride.push_back(fftPlan->outStride[1]);
            colPlan->outStride.push_back(fftPlan->outStride[0]);
            colPlan->outStride.push_back(fftPlan->outStride[1]);
            colPlan->iDist    = fftPlan->oDist;
            colPlan->oDist    = fftPlan->oDist;

            //this 3d is decomposed from 4d
            for (size_t index = 3; index < fftPlan->length.size(); index++) {
              colPlan->length.push_back(fftPlan->length[index]);
              colPlan->inStride.push_back(xyPlan->outStride[index]);
              colPlan->outStride.push_back(fftPlan->outStride[index]);
            }

            hcfftBakePlan(fftPlan->planZ);
          }
        } else if(fftPlan->opLayout == HCFFT_REAL) {
          size_t length0 = fftPlan->length[ 0 ];
          size_t length1 = fftPlan->length[ 1 ];
          size_t length2 = fftPlan->length[ 2 ];
          size_t Nt = (1 + length0 / 2);

          if (fftPlan->tmpBufSize == 0) {
            fftPlan->tmpBufSize = Nt * length1 * length2 * fftPlan->batchSize * fftPlan->ElementSize();

            for (size_t index = 2; index < fftPlan->length.size(); index++) {
              fftPlan->tmpBufSize *= fftPlan->length[index];
            }
          }

          if ((fftPlan->tmpBufSizeC2R == 0) && (fftPlan->location == HCFFT_OUTOFPLACE)) {
            fftPlan->tmpBufSizeC2R = fftPlan->tmpBufSize;
          }

          if( (fftPlan->inStride[0] == 1) && (fftPlan->outStride[0] == 1) &&
              ( ((fftPlan->outStride[2] == Nt * 2 * length1) && (fftPlan->oDist == Nt * 2 * length1 * length2) && (fftPlan->location == HCFFT_INPLACE)) ||
                ((fftPlan->outStride[2] == length0 * length1) && (fftPlan->oDist == length0 * length1 * length2) && (fftPlan->location == HCFFT_OUTOFPLACE)) )
              && (fftPlan->inStride[2] == Nt * length1) && (fftPlan->iDist == Nt * length1 * length2)) {
            // create first transpose plan
            //Transpose
            // input --> tmp
            size_t transLengths[2] = { length0 * length1, length2 };
            hcfftCreateDefaultPlanInternal( &fftPlan->planTZ, HCFFT_2D, transLengths );
            FFTPlan* trans1Plan = NULL;
            lockRAII* trans1Lock  = NULL;
            fftRepo.getPlan( fftPlan->planTZ, trans1Plan, trans1Lock );
            trans1Plan->transflag = true;
            transLengths[0] = Nt * length1;
            hcfftSetPlanLength( fftPlan->planTZ, HCFFT_2D, transLengths );

            switch(fftPlan->ipLayout) {
              case HCFFT_HERMITIAN_INTERLEAVED: {
                  trans1Plan->opLayout = HCFFT_COMPLEX_INTERLEAVED;
                  trans1Plan->ipLayout  = HCFFT_COMPLEX_INTERLEAVED;
                }
                break;

              case HCFFT_HERMITIAN_PLANAR: {
                  trans1Plan->opLayout = HCFFT_COMPLEX_INTERLEAVED;
                  trans1Plan->ipLayout  = HCFFT_COMPLEX_PLANAR;
                }
                break;

              default:
                assert(false);
            }

            trans1Plan->plHandleOrigin  = fftPlan->plHandleOrigin;
            trans1Plan->location     = HCFFT_OUTOFPLACE;
            trans1Plan->precision     = fftPlan->precision;
            trans1Plan->tmpBufSize    = 0;
            trans1Plan->batchSize     = fftPlan->batchSize;
            trans1Plan->envelope    = fftPlan->envelope;
            trans1Plan->forwardScale  = 1.0f;
            trans1Plan->backwardScale = 1.0f;
            trans1Plan->inStride[0]   = 1;
            trans1Plan->inStride[1]   = Nt * length1;
            trans1Plan->outStride[0]  = 1;
            trans1Plan->outStride[1]  = length2;
            trans1Plan->iDist         = fftPlan->iDist;
            trans1Plan->oDist   = Nt * length1 * length2;
            trans1Plan->transOutHorizontal = true;
            trans1Plan->gen           = Transpose;

            for (size_t index = 3; index < fftPlan->length.size(); index++) {
              trans1Plan->length.push_back(fftPlan->length[index]);
              trans1Plan->inStride.push_back(fftPlan->inStride[index]);
              trans1Plan->outStride.push_back(trans1Plan->oDist);
              trans1Plan->oDist *= fftPlan->length[index];
            }

            hcfftBakePlan(fftPlan->planTZ);
            // create col plan
            // complex to complex
            hcfftCreateDefaultPlanInternal( &fftPlan->planZ, HCFFT_1D, &fftPlan->length[ 2 ] );
            FFTPlan* colPlan  = NULL;
            lockRAII* colLock = NULL;
            fftRepo.getPlan( fftPlan->planZ, colPlan, colLock );
            colPlan->length.push_back(Nt * length1);
            colPlan->plHandleOrigin  = fftPlan->plHandleOrigin;
            colPlan->inStride[0]  = 1;
            colPlan->inStride.push_back(length2);
            colPlan->iDist        = trans1Plan->oDist;
            colPlan->location = HCFFT_INPLACE;
            colPlan->ipLayout = HCFFT_COMPLEX_INTERLEAVED;
            colPlan->opLayout = HCFFT_COMPLEX_INTERLEAVED;
            colPlan->outStride[0]  = colPlan->inStride[0];
            colPlan->outStride.push_back(colPlan->inStride[1]);
            colPlan->oDist         = colPlan->iDist;

            for (size_t index = 3; index < fftPlan->length.size(); index++) {
              colPlan->length.push_back(fftPlan->length[index]);
              colPlan->inStride.push_back(trans1Plan->outStride[index - 1]);
              colPlan->outStride.push_back(trans1Plan->outStride[index - 1]);
            }

            colPlan->precision     = fftPlan->precision;
            colPlan->forwardScale  = 1.0f;
            colPlan->backwardScale = 1.0f;
            colPlan->tmpBufSize    = 0;
            colPlan->gen      = fftPlan->gen;
            colPlan->envelope   = fftPlan->envelope;
            colPlan->batchSize = fftPlan->batchSize;
            hcfftBakePlan(fftPlan->planZ);
            // create second transpose plan
            //Transpose
            //tmp --> output
            size_t trans2Lengths[2] = { length2, length0 * length1 };
            hcfftCreateDefaultPlanInternal( &fftPlan->planTX, HCFFT_2D, trans2Lengths );
            FFTPlan* trans2Plan = NULL;
            lockRAII* trans2Lock  = NULL;
            fftRepo.getPlan( fftPlan->planTX, trans2Plan, trans2Lock );
            trans2Plan->transflag = true;
            trans2Lengths[1] = Nt * length1;
            hcfftSetPlanLength( fftPlan->planTX, HCFFT_2D, trans2Lengths );
            trans2Plan->opLayout = HCFFT_COMPLEX_INTERLEAVED;
            trans2Plan->ipLayout  = HCFFT_COMPLEX_INTERLEAVED;
            trans2Plan->plHandleOrigin  = fftPlan->plHandleOrigin;
            trans2Plan->location     = HCFFT_OUTOFPLACE;
            trans2Plan->precision     = fftPlan->precision;
            trans2Plan->tmpBufSize    = 0;
            trans2Plan->batchSize     = fftPlan->batchSize;
            trans2Plan->envelope    = fftPlan->envelope;
            trans2Plan->forwardScale  = 1.0f;
            trans2Plan->backwardScale = 1.0f;
            trans2Plan->inStride[0]   = 1;
            trans2Plan->inStride[1]   = length2;
            trans2Plan->outStride[0]  = 1;
            trans2Plan->outStride[1]  = Nt * length1;
            trans2Plan->iDist         = colPlan->oDist;
            trans2Plan->oDist   = Nt * length1 * length2;
            trans2Plan->gen           = Transpose;
            trans2Plan->transflag     = true;

            for (size_t index = 3; index < fftPlan->length.size(); index++) {
              trans2Plan->length.push_back(fftPlan->length[index]);
              trans2Plan->inStride.push_back(colPlan->outStride[index - 1]);
              trans2Plan->outStride.push_back(trans2Plan->oDist);
              trans2Plan->oDist *= fftPlan->length[index];
            }

            hcfftBakePlan(fftPlan->planTX);
            // create row plan
            // hermitian to real
            //create 2D xy plan
            size_t clLengths[] = { length0, length1, 0 };
            hcfftCreateDefaultPlanInternal( &fftPlan->planX, HCFFT_2D, clLengths );
            FFTPlan* rowPlan  = NULL;
            lockRAII* rowLock = NULL;
            fftRepo.getPlan( fftPlan->planX, rowPlan, rowLock );
            rowPlan->opLayout  = fftPlan->opLayout;
            rowPlan->ipLayout   = HCFFT_HERMITIAN_INTERLEAVED;
            rowPlan->length.push_back(length2);
            rowPlan->plHandleOrigin  = fftPlan->plHandleOrigin;
            rowPlan->outStride[0]  = fftPlan->outStride[0];
            rowPlan->outStride[1]  = fftPlan->outStride[1];
            rowPlan->outStride.push_back(fftPlan->outStride[2]);
            rowPlan->oDist         = fftPlan->oDist;
            rowPlan->inStride[0]  = trans2Plan->outStride[0];
            rowPlan->inStride[1]  = Nt;
            rowPlan->inStride.push_back(Nt * length1);
            rowPlan->iDist         = trans2Plan->oDist;

            for (size_t index = 3; index < fftPlan->length.size(); index++) {
              rowPlan->length.push_back(fftPlan->length[index]);
              rowPlan->inStride.push_back(trans2Plan->outStride[index - 1]);
              rowPlan->outStride.push_back(fftPlan->outStride[index]);
            }

            if (fftPlan->location == HCFFT_INPLACE) {
              rowPlan->location     = HCFFT_INPLACE;
            } else {
              rowPlan->location     = HCFFT_OUTOFPLACE;
            }

            rowPlan->precision     = fftPlan->precision;
            rowPlan->forwardScale  = fftPlan->forwardScale;
            rowPlan->backwardScale = fftPlan->backwardScale;
            rowPlan->tmpBufSize    = 0;
            rowPlan->gen      = fftPlan->gen;
            rowPlan->envelope   = fftPlan->envelope;
            rowPlan->batchSize    = fftPlan->batchSize;
            hcfftBakePlan(fftPlan->planX);
          } else {
            size_t clLengths[] = { 1, 0, 0 };
            clLengths[0] = fftPlan->length[ 2 ];
            //create 1D col plan
            hcfftCreateDefaultPlanInternal( &fftPlan->planZ, HCFFT_1D, clLengths );
            FFTPlan* colPlan  = NULL;
            lockRAII* colLock = NULL;
            fftRepo.getPlan( fftPlan->planZ, colPlan, colLock );

            switch(fftPlan->ipLayout) {
              case HCFFT_HERMITIAN_INTERLEAVED: {
                  colPlan->opLayout = HCFFT_COMPLEX_INTERLEAVED;
                  colPlan->ipLayout  = HCFFT_COMPLEX_INTERLEAVED;
                }
                break;

              case HCFFT_HERMITIAN_PLANAR: {
                  colPlan->opLayout = HCFFT_COMPLEX_INTERLEAVED;
                  colPlan->ipLayout  = HCFFT_COMPLEX_PLANAR;
                }
                break;

              default:
                assert(false);
            }

            colPlan->plHandleOrigin  = fftPlan->plHandleOrigin;
            colPlan->length.push_back(Nt);
            colPlan->length.push_back(length1);
            colPlan->inStride[0]  = fftPlan->inStride[2];
            colPlan->inStride.push_back(fftPlan->inStride[0]);
            colPlan->inStride.push_back(fftPlan->inStride[1]);
            colPlan->iDist         = fftPlan->iDist;

            if (fftPlan->location == HCFFT_INPLACE) {
              colPlan->location = HCFFT_INPLACE;
              colPlan->outStride[0]  = colPlan->inStride[0];
              colPlan->outStride.push_back(colPlan->inStride[1]);
              colPlan->outStride.push_back(colPlan->inStride[2]);
              colPlan->oDist         = colPlan->iDist;

              for (size_t index = 3; index < fftPlan->length.size(); index++) {
                colPlan->length.push_back(fftPlan->length[index]);
                colPlan->inStride.push_back(fftPlan->inStride[index]);
                colPlan->outStride.push_back(fftPlan->inStride[index]);
              }
            } else {
              colPlan->location = HCFFT_OUTOFPLACE;
              colPlan->outStride[0]  = Nt * length1;
              colPlan->outStride.push_back(1);
              colPlan->outStride.push_back(Nt);
              colPlan->oDist         = Nt * length1 * length2;

              for (size_t index = 3; index < fftPlan->length.size(); index++) {
                colPlan->length.push_back(fftPlan->length[index]);
                colPlan->inStride.push_back(fftPlan->inStride[index]);
                colPlan->outStride.push_back(colPlan->oDist);
                colPlan->oDist *= fftPlan->length[index];
              }
            }

            colPlan->precision     = fftPlan->precision;
            colPlan->forwardScale  = 1.0f;
            colPlan->backwardScale = 1.0f;
            colPlan->tmpBufSize    = 0;
            colPlan->gen       = fftPlan->gen;
            colPlan->envelope    = fftPlan->envelope;
            colPlan->batchSize = fftPlan->batchSize;
            hcfftBakePlan(fftPlan->planZ);
            clLengths[0] = fftPlan->length[ 0 ];
            clLengths[1] = fftPlan->length[ 1 ];
            //create 2D xy plan
            hcfftCreateDefaultPlanInternal( &fftPlan->planX, HCFFT_2D, clLengths );
            FFTPlan* xyPlan = NULL;
            lockRAII* rowLock = NULL;
            fftRepo.getPlan( fftPlan->planX, xyPlan, rowLock );
            xyPlan->ipLayout   = HCFFT_HERMITIAN_INTERLEAVED;
            xyPlan->opLayout  = fftPlan->opLayout;
            xyPlan->length.push_back(length2);
            xyPlan->plHandleOrigin  = fftPlan->plHandleOrigin;
            xyPlan->outStride[0]  = fftPlan->outStride[0];
            xyPlan->outStride[1]  = fftPlan->outStride[1];
            xyPlan->outStride.push_back(fftPlan->outStride[2]);
            xyPlan->oDist         = fftPlan->oDist;

            if (fftPlan->location == HCFFT_INPLACE) {
              xyPlan->location     = HCFFT_INPLACE;
              xyPlan->inStride[0]  = colPlan->outStride[1];
              xyPlan->inStride[1]  = colPlan->outStride[2];
              xyPlan->inStride.push_back(colPlan->outStride[0]);
              xyPlan->iDist         = colPlan->oDist;

              for (size_t index = 3; index < fftPlan->length.size(); index++) {
                xyPlan->length.push_back(fftPlan->length[index]);
                xyPlan->inStride.push_back(colPlan->outStride[index]);
                xyPlan->outStride.push_back(fftPlan->outStride[index]);
              }
            } else {
              xyPlan->location     = HCFFT_OUTOFPLACE;
              xyPlan->inStride[0]   = 1;
              xyPlan->inStride[1]   = Nt;
              xyPlan->inStride.push_back(Nt * length1);
              xyPlan->iDist         = Nt * length1 * length2;

              for (size_t index = 3; index < fftPlan->length.size(); index++) {
                xyPlan->length.push_back(fftPlan->length[index]);
                xyPlan->outStride.push_back(fftPlan->outStride[index]);
                xyPlan->inStride.push_back(xyPlan->iDist);
                xyPlan->iDist *= fftPlan->length[index];
              }
            }

            xyPlan->precision     = fftPlan->precision;
            xyPlan->forwardScale  = fftPlan->forwardScale;
            xyPlan->backwardScale = fftPlan->backwardScale;
            xyPlan->tmpBufSize    = fftPlan->tmpBufSize;
            xyPlan->gen      = fftPlan->gen;
            xyPlan->envelope   = fftPlan->envelope;
            xyPlan->batchSize    = fftPlan->batchSize;
            hcfftBakePlan(fftPlan->planX);
          }
        } else {
          if (fftPlan->tmpBufSize == 0 && (
                fftPlan->length[0] > Large1DThreshold ||
                fftPlan->length[1] > Large1DThreshold ||
                fftPlan->length[2] > Large1DThreshold
              )) {
            fftPlan->tmpBufSize = fftPlan->length[0] * fftPlan->length[1] * fftPlan->length[2] *
                                  fftPlan->batchSize * fftPlan->ElementSize();
          }

          size_t clLengths[] = { 1, 1, 0 };
          clLengths[0] = fftPlan->length[ 0 ];
          clLengths[1] = fftPlan->length[ 1 ];
          //create 2D xy plan
          hcfftCreateDefaultPlanInternal( &fftPlan->planX, HCFFT_2D, clLengths );
          FFTPlan* xyPlan = NULL;
          lockRAII* rowLock = NULL;
          fftRepo.getPlan( fftPlan->planX, xyPlan, rowLock );
          xyPlan->plHandleOrigin  = fftPlan->plHandleOrigin;
          xyPlan->ipLayout   = fftPlan->ipLayout;
          xyPlan->opLayout  = fftPlan->opLayout;
          xyPlan->location     = fftPlan->location;
          xyPlan->precision     = fftPlan->precision;
          xyPlan->forwardScale  = 1.0f;
          xyPlan->backwardScale = 1.0f;
          xyPlan->tmpBufSize    = fftPlan->tmpBufSize;
          xyPlan->gen      = fftPlan->gen;
          xyPlan->envelope       = fftPlan->envelope;
          // This is the xy fft, the first elements distance between the first two FFTs is the distance of the first elements
          // of the first two rows in the original buffer.
          xyPlan->batchSize    = fftPlan->batchSize;
          xyPlan->inStride[0]  = fftPlan->inStride[0];
          xyPlan->inStride[1]  = fftPlan->inStride[1];
          xyPlan->outStride[0] = fftPlan->outStride[0];
          xyPlan->outStride[1] = fftPlan->outStride[1];
          //pass length and other info to kernel, so the kernel knows this is decomposed from higher dimension
          xyPlan->length.push_back(fftPlan->length[2]);
          xyPlan->inStride.push_back(fftPlan->inStride[2]);
          xyPlan->outStride.push_back(fftPlan->outStride[2]);
          xyPlan->iDist    = fftPlan->iDist;
          xyPlan->oDist    = fftPlan->oDist;
          hcfftBakePlan(fftPlan->planX);
          clLengths[0] = fftPlan->length[ 2 ];
          clLengths[1] = clLengths[2] = 0;
          //create 1D col plan
          hcfftCreateDefaultPlanInternal( &fftPlan->planZ, HCFFT_1D, clLengths );
          FFTPlan* colPlan  = NULL;
          lockRAII* colLock = NULL;
          fftRepo.getPlan( fftPlan->planZ, colPlan, colLock );
          colPlan->plHandleOrigin  = fftPlan->plHandleOrigin;
          colPlan->ipLayout   = fftPlan->opLayout;
          colPlan->opLayout  = fftPlan->opLayout;
          colPlan->location     = HCFFT_INPLACE;
          colPlan->precision     = fftPlan->precision;
          colPlan->forwardScale  = fftPlan->forwardScale;
          colPlan->backwardScale = fftPlan->backwardScale;
          colPlan->tmpBufSize    = fftPlan->tmpBufSize;
          colPlan->gen         = fftPlan->gen;
          colPlan->envelope      = fftPlan->envelope;
          // This is a column FFT, the first elements distance between each FFT is the distance of the first two
          // elements in the original buffer. Like a transpose of the matrix
          colPlan->batchSize = fftPlan->batchSize;
          colPlan->inStride[0] = fftPlan->outStride[2];
          colPlan->outStride[0] = fftPlan->outStride[2];
          //pass length and other info to kernel, so the kernel knows this is decomposed from higher dimension
          colPlan->length.push_back(fftPlan->length[0]);
          colPlan->length.push_back(fftPlan->length[1]);
          colPlan->inStride.push_back(fftPlan->outStride[0]);
          colPlan->inStride.push_back(fftPlan->outStride[1]);
          colPlan->outStride.push_back(fftPlan->outStride[0]);
          colPlan->outStride.push_back(fftPlan->outStride[1]);
          colPlan->iDist    = fftPlan->oDist;
          colPlan->oDist    = fftPlan->oDist;
          hcfftBakePlan(fftPlan->planZ);
        }

        fftPlan->baked = true;
        return  HCFFT_SUCCESS;
      }
  }

  if(!exist) {
    //  For the radices that we have factored, we need to load/compile and build the appropriate OpenCL kernels
    fftPlan->GenerateKernel( plHandle, fftRepo, count);
    //  For the radices that we have factored, we need to load/compile and build the appropriate OpenCL kernels
    count++;
  }

  CompileKernels( plHandle, fftPlan->gen, fftPlan, fftPlan->plHandleOrigin, exist);
  //  Allocate resources
  fftPlan->AllocateWriteBuffers ();
  //  Record that we baked the plan
  fftPlan->baked    = true;
  return  HCFFT_SUCCESS;
}

hcfftStatus FFTPlan::hcfftGetPlanPrecision( const  hcfftPlanHandle plHandle,  hcfftPrecision* precision ) {
  FFTRepo& fftRepo = FFTRepo::getInstance( );
  FFTPlan* fftPlan = NULL;
  lockRAII* planLock = NULL;
  fftRepo.getPlan(plHandle, fftPlan, planLock );
  scopedLock sLock(*planLock, _T( " hcfftGetPlanPrecision" ) );
  *precision = fftPlan->precision;
  return HCFFT_SUCCESS;
}

hcfftStatus FFTPlan::hcfftSetPlanPrecision(  hcfftPlanHandle plHandle,  hcfftPrecision precision ) {
  FFTRepo& fftRepo = FFTRepo::getInstance( );
  FFTPlan* fftPlan = NULL;
  lockRAII* planLock = NULL;
  fftRepo.getPlan(plHandle, fftPlan, planLock );
  scopedLock sLock(*planLock, _T( " hcfftSetPlanPrecision" ) );
  //  If we modify the state of the plan, we assume that we can't trust any pre-calculated contents anymore
  fftPlan->baked = false;
  fftPlan->precision = precision;
  return HCFFT_SUCCESS;
}

hcfftStatus FFTPlan::hcfftGetPlanScale( const  hcfftPlanHandle plHandle,  hcfftDirection dir, float* scale ) {
  FFTRepo& fftRepo = FFTRepo::getInstance( );
  FFTPlan* fftPlan = NULL;
  lockRAII* planLock = NULL;
  fftRepo.getPlan(plHandle, fftPlan, planLock );
  scopedLock sLock(*planLock, _T( " hcfftGetPlanScale" ) );

  if( dir == HCFFT_FORWARD) {
    *scale = (float)(fftPlan->forwardScale);
  } else {
    *scale = (float)(fftPlan->backwardScale);
  }

  return HCFFT_SUCCESS;
}

hcfftStatus FFTPlan::hcfftSetPlanScale(  hcfftPlanHandle plHandle,  hcfftDirection dir, float scale ) {
  FFTRepo& fftRepo = FFTRepo::getInstance( );
  FFTPlan* fftPlan = NULL;
  lockRAII* planLock = NULL;
  fftRepo.getPlan(plHandle, fftPlan, planLock );
  scopedLock sLock(*planLock, _T( " hcfftSetPlanScale" ) );
  //  If we modify the state of the plan, we assume that we can't trust any pre-calculated contents anymore
  fftPlan->baked = false;

  if( dir == HCFFT_FORWARD) {
    fftPlan->forwardScale = scale;
  } else {
    fftPlan->backwardScale = scale;
  }

  return HCFFT_SUCCESS;
}

hcfftStatus FFTPlan::hcfftGetPlanBatchSize( const  hcfftPlanHandle plHandle, size_t* batchsize ) {
  FFTRepo& fftRepo = FFTRepo::getInstance( );
  FFTPlan* fftPlan = NULL;
  lockRAII* planLock = NULL;
  fftRepo.getPlan(plHandle, fftPlan, planLock );
  scopedLock sLock(*planLock, _T( " hcfftGetPlanBatchSize" ) );
  *batchsize = fftPlan->batchSize;
  return HCFFT_SUCCESS;
}

hcfftStatus FFTPlan::hcfftSetPlanBatchSize( hcfftPlanHandle plHandle, size_t batchsize ) {
  FFTRepo& fftRepo = FFTRepo::getInstance( );
  FFTPlan* fftPlan = NULL;
  lockRAII* planLock = NULL;
  fftRepo.getPlan(plHandle, fftPlan, planLock );
  scopedLock sLock(*planLock, _T( " hcfftSetPlanBatchSize" ) );
  //  If we modify the state of the plan, we assume that we can't trust any pre-calculated contents anymore
  fftPlan->baked = false;
  fftPlan->batchSize = batchsize;
  return HCFFT_SUCCESS;
}

hcfftStatus FFTPlan::hcfftGetPlanDim( const hcfftPlanHandle plHandle,  hcfftDim* dim, int* size ) {
  FFTRepo& fftRepo = FFTRepo::getInstance( );
  FFTPlan* fftPlan = NULL;
  lockRAII* planLock = NULL;
  fftRepo.getPlan( plHandle, fftPlan, planLock );
  scopedLock sLock( *planLock, _T( " hcfftGetPlanDim" ) );
  *dim = fftPlan->dimension;

  switch( fftPlan->dimension ) {
    case HCFFT_1D: {
        *size = 1;
      }
      break;

    case HCFFT_2D: {
        *size = 2;
      }
      break;

    case HCFFT_3D: {
        *size = 3;
      }
      break;

    default:
      return HCFFT_ERROR;
      break;
  }

  return HCFFT_SUCCESS;
}

hcfftStatus FFTPlan::hcfftSetPlanDim(  hcfftPlanHandle plHandle, const  hcfftDim dim ) {
  FFTRepo& fftRepo = FFTRepo::getInstance( );
  FFTPlan* fftPlan = NULL;
  lockRAII* planLock = NULL;
  fftRepo.getPlan( plHandle, fftPlan, planLock );
  scopedLock sLock( *planLock, _T( " hcfftGetPlanDim" ) );

  // We resize the vectors in the plan to keep their sizes consistent with the value of the dimension
  switch( dim ) {
    case HCFFT_1D: {
        fftPlan->length.resize( 1 );
        fftPlan->inStride.resize( 1 );
        fftPlan->outStride.resize( 1 );
      }
      break;

    case HCFFT_2D: {
        fftPlan->length.resize( 2 );
        fftPlan->inStride.resize( 2 );
        fftPlan->outStride.resize( 2 );
      }
      break;

    case HCFFT_3D: {
        fftPlan->length.resize( 3 );
        fftPlan->inStride.resize( 3 );
        fftPlan->outStride.resize( 3 );
      }
      break;

    default:
      return HCFFT_ERROR;
      break;
  }

  //  If we modify the state of the plan, we assume that we can't trust any pre-calculated contents anymore
  fftPlan->baked = false;
  fftPlan->dimension = dim;
  return HCFFT_SUCCESS;
}

hcfftStatus FFTPlan::hcfftGetPlanLength( const  hcfftPlanHandle plHandle, const  hcfftDim dim, size_t* clLengths ) {
  FFTRepo& fftRepo = FFTRepo::getInstance( );
  FFTPlan* fftPlan = NULL;
  lockRAII* planLock = NULL;
  fftRepo.getPlan( plHandle, fftPlan, planLock );
  scopedLock sLock( *planLock, _T( " hcfftGetPlanLength" ) );

  if( clLengths == NULL ) {
    return HCFFT_ERROR;
  }

  if( fftPlan->length.empty( ) ) {
    return HCFFT_ERROR;
  }

  switch( dim ) {
    case HCFFT_1D: {
        clLengths[0] = fftPlan->length[0];
      }
      break;

    case HCFFT_2D: {
        if( fftPlan->length.size() < 2 ) {
          return HCFFT_ERROR;
        }

        clLengths[0] = fftPlan->length[0];
        clLengths[1 ] = fftPlan->length[1];
      }
      break;

    case HCFFT_3D: {
        if( fftPlan->length.size() < 3 ) {
          return HCFFT_ERROR;
        }

        clLengths[0] = fftPlan->length[0];
        clLengths[1 ] = fftPlan->length[1];
        clLengths[2] = fftPlan->length[2];
      }
      break;

    default:
      return HCFFT_ERROR;
      break;
  }

  return HCFFT_SUCCESS;
}

hcfftStatus FFTPlan::hcfftSetPlanLength(  hcfftPlanHandle plHandle, const  hcfftDim dim, const size_t* clLengths ) {
  FFTRepo& fftRepo = FFTRepo::getInstance( );
  FFTPlan* fftPlan = NULL;
  lockRAII* planLock = NULL;
  fftRepo.getPlan( plHandle, fftPlan, planLock );
  scopedLock sLock( *planLock, _T( " hcfftSetPlanLength" ) );

  if( clLengths == NULL ) {
    return HCFFT_ERROR;
  }

  //  Simplest to clear any previous contents, because it's valid for user to shrink dimension
  fftPlan->length.clear( );

  switch( dim ) {
    case HCFFT_1D: {
        //  Minimum length size is 1
        if( clLengths[0] == 0 ) {
          return HCFFT_ERROR;
        }

        fftPlan->length.push_back( clLengths[0] );
      }
      break;

    case HCFFT_2D: {
        //  Minimum length size is 1
        if(clLengths[0] == 0 || clLengths[1] == 0 ) {
          return HCFFT_ERROR;
        }

        fftPlan->length.push_back( clLengths[0] );
        fftPlan->length.push_back( clLengths[1] );
      }
      break;

    case HCFFT_3D: {
        //  Minimum length size is 1
        if(clLengths[0 ] == 0 || clLengths[1] == 0 || clLengths[2] == 0) {
          return HCFFT_ERROR;
        }

        fftPlan->length.push_back( clLengths[0] );
        fftPlan->length.push_back( clLengths[1] );
        fftPlan->length.push_back( clLengths[2] );
      }
      break;

    default:
      return HCFFT_ERROR;
      break;
  }

  fftPlan->dimension = dim;
  //  If we modify the state of the plan, we assume that we can't trust any pre-calculated contents anymore
  fftPlan->baked = false;
  return HCFFT_SUCCESS;
}

hcfftStatus FFTPlan::hcfftGetPlanInStride( const  hcfftPlanHandle plHandle, const  hcfftDim dim, size_t* clStrides ) {
  FFTRepo& fftRepo = FFTRepo::getInstance( );
  FFTPlan* fftPlan = NULL;
  lockRAII* planLock = NULL;
  fftRepo.getPlan( plHandle, fftPlan, planLock );
  scopedLock sLock( *planLock, _T( " hcfftGetPlanInStride" ) );

  if(clStrides == NULL ) {
    return HCFFT_ERROR;
  }

  switch( dim ) {
    case HCFFT_1D: {
        if(fftPlan->inStride.size( ) > 0 ) {
          clStrides[0] = fftPlan->inStride[0];
        } else {
          return HCFFT_ERROR;
        }
      }
      break;

    case HCFFT_2D: {
        if( fftPlan->inStride.size( ) > 1 ) {
          clStrides[0] = fftPlan->inStride[0];
          clStrides[1] = fftPlan->inStride[1];
        } else {
          return HCFFT_ERROR;
        }
      }
      break;

    case HCFFT_3D: {
        if( fftPlan->inStride.size( ) > 2 ) {
          clStrides[0] = fftPlan->inStride[0];
          clStrides[1] = fftPlan->inStride[1];
          clStrides[2] = fftPlan->inStride[2];
        } else {
          return HCFFT_ERROR;
        }
      }
      break;

    default:
      return HCFFT_ERROR;
      break;
  }

  return HCFFT_SUCCESS;
}

hcfftStatus FFTPlan::hcfftSetPlanInStride(  hcfftPlanHandle plHandle, const  hcfftDim dim, size_t* clStrides ) {
  FFTRepo& fftRepo = FFTRepo::getInstance( );
  FFTPlan* fftPlan = NULL;
  lockRAII* planLock = NULL;
  fftRepo.getPlan( plHandle, fftPlan, planLock );
  scopedLock sLock( *planLock, _T( " hcfftSetPlanInStride" ) );

  if( clStrides == NULL ) {
    return HCFFT_ERROR;
  }

  //  Simplest to clear any previous contents, because it's valid for user to shrink dimension
  fftPlan->inStride.clear( );

  switch( dim ) {
    case HCFFT_1D: {
        fftPlan->inStride.push_back( clStrides[0] );
      }
      break;

    case HCFFT_2D: {
        fftPlan->inStride.push_back( clStrides[0] );
        fftPlan->inStride.push_back( clStrides[1] );
      }
      break;

    case HCFFT_3D: {
        fftPlan->inStride.push_back( clStrides[0] );
        fftPlan->inStride.push_back( clStrides[1] );
        fftPlan->inStride.push_back( clStrides[2] );
      }
      break;

    default:
      return HCFFT_ERROR;
      break;
  }

  //  If we modify the state of the plan, we assume that we can't trust any pre-calculated contents anymore
  fftPlan->baked = false;
  return HCFFT_SUCCESS;
}

hcfftStatus FFTPlan::hcfftGetPlanOutStride( const  hcfftPlanHandle plHandle, const  hcfftDim dim, size_t* clStrides ) {
  FFTRepo& fftRepo  = FFTRepo::getInstance( );
  FFTPlan* fftPlan  = NULL;
  lockRAII* planLock  = NULL;
  fftRepo.getPlan( plHandle, fftPlan, planLock );
  scopedLock sLock( *planLock, _T( " hcfftGetPlanOutStride" ) );

  if( clStrides == NULL ) {
    return HCFFT_ERROR;
  }

  switch( dim ) {
    case HCFFT_1D: {
        if( fftPlan->outStride.size() > 0 ) {
          clStrides[0] = fftPlan->outStride[0];
        } else {
          return HCFFT_ERROR;
        }
      }
      break;

    case HCFFT_2D: {
        if( fftPlan->outStride.size() > 1 ) {
          clStrides[0] = fftPlan->outStride[0];
          clStrides[1] = fftPlan->outStride[1];
        } else {
          return HCFFT_ERROR;
        }
      }
      break;

    case HCFFT_3D: {
        if( fftPlan->outStride.size() > 2 ) {
          clStrides[0] = fftPlan->outStride[0];
          clStrides[1] = fftPlan->outStride[1];
          clStrides[2] = fftPlan->outStride[2];
        } else {
          return HCFFT_ERROR;
        }
      }
      break;

    default:
      return HCFFT_ERROR;
      break;
  }

  return HCFFT_SUCCESS;
}

hcfftStatus FFTPlan::hcfftSetPlanOutStride(  hcfftPlanHandle plHandle, const  hcfftDim dim, size_t* clStrides ) {
  FFTRepo& fftRepo  = FFTRepo::getInstance( );
  FFTPlan* fftPlan  = NULL;
  lockRAII* planLock  = NULL;
  fftRepo.getPlan( plHandle, fftPlan, planLock );
  scopedLock sLock( *planLock, _T( " hcfftSetPlanOutStride" ) );

  if( clStrides == NULL ) {
    return HCFFT_ERROR;
  }

  switch( dim ) {
    case HCFFT_1D: {
        fftPlan->outStride[0] = clStrides[0];
      }
      break;

    case HCFFT_2D: {
        fftPlan->outStride[0] = clStrides[0];
        fftPlan->outStride[1] = clStrides[1];
      }
      break;

    case HCFFT_3D: {
        fftPlan->outStride[0] = clStrides[0];
        fftPlan->outStride[1] = clStrides[1];
        fftPlan->outStride[2] = clStrides[2];
      }
      break;

    default:
      return HCFFT_ERROR;
      break;
  }

  //  If we modify the state of the plan, we assume that we can't trust any pre-calculated contents anymore
  fftPlan->baked  = false;
  return HCFFT_SUCCESS;
}

hcfftStatus FFTPlan::hcfftGetPlanDistance( const  hcfftPlanHandle plHandle, size_t* iDist, size_t* oDist ) {
  FFTRepo& fftRepo  = FFTRepo::getInstance( );
  FFTPlan* fftPlan  = NULL;
  lockRAII* planLock  = NULL;
  fftRepo.getPlan( plHandle, fftPlan, planLock );
  scopedLock sLock( *planLock, _T( " hcfftGetPlanDistance" ) );
  *iDist = fftPlan->iDist;
  *oDist = fftPlan->oDist;
  return HCFFT_SUCCESS;
}

hcfftStatus FFTPlan::hcfftSetPlanDistance(  hcfftPlanHandle plHandle, size_t iDist, size_t oDist ) {
  FFTRepo& fftRepo = FFTRepo::getInstance( );
  FFTPlan* fftPlan = NULL;
  lockRAII* planLock = NULL;
  fftRepo.getPlan(plHandle, fftPlan, planLock );
  scopedLock sLock(*planLock, _T( " hcfftSetPlanDistance" ) );
  //  If we modify the state of the plan, we assume that we can't trust any pre-calculated contents anymore
  fftPlan->baked = false;
  fftPlan->iDist = iDist;
  fftPlan->oDist = oDist;
  return HCFFT_SUCCESS;
}

hcfftStatus FFTPlan::hcfftGetLayout( const  hcfftPlanHandle plHandle,  hcfftIpLayout* iLayout,  hcfftOpLayout* oLayout ) {
  FFTRepo& fftRepo = FFTRepo::getInstance( );
  FFTPlan* fftPlan = NULL;
  lockRAII* planLock = NULL;
  fftRepo.getPlan(plHandle, fftPlan, planLock );
  scopedLock sLock(*planLock, _T( " hcfftGetLayout" ) );
  *iLayout = fftPlan->ipLayout;
  *oLayout = fftPlan->opLayout;
  return HCFFT_SUCCESS;
}

hcfftStatus FFTPlan::hcfftSetLayout(  hcfftPlanHandle plHandle,  hcfftIpLayout iLayout,  hcfftOpLayout oLayout ) {
  FFTRepo& fftRepo = FFTRepo::getInstance( );
  FFTPlan* fftPlan = NULL;
  lockRAII* planLock = NULL;
  fftRepo.getPlan( plHandle, fftPlan, planLock );
  scopedLock sLock( *planLock, _T( " hcfftSetLayout" ) );

  //  We currently only support a subset of formats
  switch( iLayout ) {
    case HCFFT_COMPLEX_INTERLEAVED: {
        if( (oLayout == HCFFT_HERMITIAN_INTERLEAVED) || (oLayout == HCFFT_HERMITIAN_PLANAR) || (oLayout == HCFFT_REAL)) {
          return HCFFT_ERROR;
        }
      }
      break;

    case HCFFT_COMPLEX_PLANAR: {
        if( (oLayout == HCFFT_HERMITIAN_INTERLEAVED) || (oLayout == HCFFT_HERMITIAN_PLANAR) || (oLayout == HCFFT_REAL)) {
          return HCFFT_ERROR;
        }
      }
      break;

    case HCFFT_HERMITIAN_INTERLEAVED: {
        if(oLayout != HCFFT_REAL) {
          return HCFFT_ERROR;
        }
      }
      break;

    case HCFFT_HERMITIAN_PLANAR: {
        if(oLayout != HCFFT_REAL) {
          return HCFFT_ERROR;
        }
      }
      break;

    case HCFFT_REAL: {
        if((oLayout == HCFFT_REAL) || (oLayout == HCFFT_COMPLEX_INTERLEAVED) || (oLayout == HCFFT_COMPLEX_PLANAR)) {
          return HCFFT_ERROR;
        }
      }
      break;

    default:
      return HCFFT_ERROR;
      break;
  }

  //  We currently only support a subset of formats
  switch( oLayout ) {
    case HCFFT_COMPLEX_PLANAR:
    case HCFFT_COMPLEX_INTERLEAVED:
    case HCFFT_HERMITIAN_INTERLEAVED:
    case HCFFT_HERMITIAN_PLANAR:
    case HCFFT_REAL:
      break;

    default:
      return HCFFT_ERROR;
      break;
  }

  //  If we modify the state of the plan, we assume that we can't trust any pre-calculated contents anymore
  fftPlan->baked  = false;
  fftPlan->ipLayout = iLayout;
  fftPlan->opLayout = oLayout;
  return HCFFT_SUCCESS;
}

hcfftStatus FFTPlan::hcfftGetResultLocation( const  hcfftPlanHandle plHandle,  hcfftResLocation* placeness ) {
  FFTRepo& fftRepo  = FFTRepo::getInstance( );
  FFTPlan* fftPlan  = NULL;
  lockRAII* planLock  = NULL;
  fftRepo.getPlan( plHandle, fftPlan, planLock );
  scopedLock sLock( *planLock, _T( " hcfftGetResultLocation" ) );
  *placeness  = fftPlan->location;
  return  HCFFT_SUCCESS;
}

hcfftStatus FFTPlan::hcfftSetResultLocation(  hcfftPlanHandle plHandle,  hcfftResLocation placeness ) {
  FFTRepo& fftRepo  = FFTRepo::getInstance( );
  FFTPlan* fftPlan  = NULL;
  lockRAII* planLock  = NULL;
  fftRepo.getPlan( plHandle, fftPlan, planLock );
  scopedLock sLock( *planLock, _T( " hcfftSetResultLocation" ) );
  //  If we modify the state of the plan, we assume that we can't trust any pre-calculated contents anymore
  fftPlan->baked    = false;
  fftPlan->location = placeness;
  return HCFFT_SUCCESS;
}

hcfftStatus FFTPlan::hcfftGetPlanTransposeResult( const  hcfftPlanHandle plHandle,  hcfftResTransposed* transposed ) {
  FFTRepo& fftRepo  = FFTRepo::getInstance( );
  FFTPlan* fftPlan  = NULL;
  lockRAII* planLock  = NULL;
  fftRepo.getPlan( plHandle, fftPlan, planLock );
  scopedLock sLock( *planLock, _T( " hcfftGetResultLocation" ) );
  *transposed = fftPlan->transposeType;
  return HCFFT_SUCCESS;
}

hcfftStatus FFTPlan::hcfftSetPlanTransposeResult(  hcfftPlanHandle plHandle,  hcfftResTransposed transposed ) {
  FFTRepo& fftRepo  = FFTRepo::getInstance( );
  FFTPlan* fftPlan  = NULL;
  lockRAII* planLock  = NULL;
  fftRepo.getPlan( plHandle, fftPlan, planLock );
  scopedLock sLock( *planLock, _T( " hcfftSetResultLocation" ) );
  //  If we modify the state of the plan, we assume that we can't trust any pre-calculated contents anymore
  fftPlan->baked    = false;
  fftPlan->transposeType  = transposed;
  return HCFFT_SUCCESS;
}

hcfftStatus FFTPlan::GetMax1DLength (size_t* longest ) const {
  switch(gen) {
    case Stockham:
      return GetMax1DLengthPvt<Stockham>(longest);

    case Copy: {
        *longest = 4096;
        return HCFFT_SUCCESS;
      }

    case Transpose: {
        *longest = 4096;
        return HCFFT_SUCCESS;
      }

    default:
      return HCFFT_ERROR;
  }
}

hcfftStatus  FFTPlan::GetKernelGenKey (FFTKernelGenKeyParams & params) const {
  switch(gen) {
    case Stockham:
      return GetKernelGenKeyPvt<Stockham>(params);

    case Copy:
      return GetKernelGenKeyPvt<Copy>(params);

    case Transpose:
      return GetKernelGenKeyPvt<Transpose>(params);

    default:
      return HCFFT_ERROR;
  }
}

hcfftStatus  FFTPlan::GetWorkSizes (std::vector<size_t> & globalws, std::vector<size_t> & localws) const {
  switch(gen) {
    case Stockham:
      return GetWorkSizesPvt<Stockham>(globalws, localws);

    case Copy:
      return GetWorkSizesPvt<Copy>(globalws, localws);

    case Transpose:
      return GetWorkSizesPvt<Transpose>(globalws, localws);

    default:
      return HCFFT_ERROR;
  }
}

hcfftStatus  FFTPlan::GenerateKernel (const hcfftPlanHandle plHandle, FFTRepo & fftRepo, size_t count) const {
  switch(gen) {
    case Stockham:
      return GenerateKernelPvt<Stockham>(plHandle, fftRepo, count);

    case Copy:
      return GenerateKernelPvt<Copy>(plHandle, fftRepo, count);

    case Transpose:
      return GenerateKernelPvt<Transpose>(plHandle, fftRepo, count);

    default:
      return HCFFT_ERROR;
  }
}
hcfftStatus FFTPlan::hcfftDestroyPlan( hcfftPlanHandle* plHandle ) {
  FFTRepo& fftRepo  = FFTRepo::getInstance( );
  FFTPlan* fftPlan  = NULL;
  lockRAII* planLock  = NULL;
  fftRepo.getPlan( *plHandle, fftPlan, planLock );

  //  Recursively destroy subplans, that are used for higher dimensional FFT's
  if( fftPlan->planX ) {
    hcfftDestroyPlan( &fftPlan->planX );
  }

  if( fftPlan->planY ) {
    hcfftDestroyPlan( &fftPlan->planY );
  }

  if( fftPlan->planZ ) {
    hcfftDestroyPlan( &fftPlan->planZ );
  }

  if( fftPlan->planTX ) {
    hcfftDestroyPlan( &fftPlan->planTX );
  }

  if( fftPlan->planTY ) {
    hcfftDestroyPlan( &fftPlan->planTY );
  }

  if( fftPlan->planTZ ) {
    hcfftDestroyPlan( &fftPlan->planTZ );
  }

  if( fftPlan->planRCcopy ) {
    hcfftDestroyPlan( &fftPlan->planRCcopy );
  }

  fftPlan->ReleaseBuffers();
  fftRepo.deletePlan( plHandle );
  return HCFFT_SUCCESS;
}

hcfftStatus FFTPlan::AllocateWriteBuffers () {
  hcfftStatus status = HCFFT_SUCCESS;
  assert (NULL == const_buffer);
  assert(4 == sizeof(int));
  //  Construct the constant buffer and call clEnqueueWriteBuffer
  float ConstantBufferParams[HCFFT_CB_SIZE];
  memset (& ConstantBufferParams, 0, sizeof (ConstantBufferParams));
  ConstantBufferParams[1] = std::max<uint> (1, uint(batchSize));
  Concurrency::array<float, 1> arr = Concurrency::array<float, 1>(Concurrency::extent<1>(HCFFT_CB_SIZE), ConstantBufferParams);
  const_buffer = new Concurrency::array_view<float>(arr);
  return status;
}

hcfftStatus FFTPlan::ReleaseBuffers () {
  hcfftStatus result = HCFFT_SUCCESS;

  if( NULL != const_buffer ) {
    delete const_buffer;
  }

  if( NULL != intBuffer ) {
    delete intBuffer;
  }

  if( NULL != intBufferRC ) {
    delete intBufferRC;
  }

  if( NULL != intBufferC2R ) {
    delete intBufferC2R;
  }

  return result;
}

size_t FFTPlan::ElementSize() const {
  return ((precision == HCFFT_DOUBLE) ? sizeof(std::complex<double> ) : sizeof(std::complex<float>));
}

/*----------------------------------------------------FFTPlan-----------------------------------------------------------------------------*/

/*---------------------------------------------------FFTRepo--------------------------------------------------------------------------------*/
hcfftStatus FFTRepo::createPlan( hcfftPlanHandle* plHandle, FFTPlan*& fftPlan ) {
  scopedLock sLock( lockRepo, _T( "insertPlan" ) );
  //  We keep track of this memory in our own collection class, to make sure it's freed in releaseResources
  //  The lifetime of a plan is tracked by the client and is freed when the client calls ::hcfftDestroyPlan()
  fftPlan = new FFTPlan;
  //  We allocate a new lock here, and expect it to be freed in ::hcfftDestroyPlan();
  //  The lifetime of the lock is the same as the lifetime of the plan
  lockRAII* lockPlan  = new lockRAII;
  //  Add and remember the fftPlan in our map
  repoPlans[ planCount ] = make_pair( fftPlan, lockPlan );
  //  Assign the user handle the plan count (unique identifier), and bump the count for the next plan
  *plHandle = planCount++;
  return  HCFFT_SUCCESS;
}


hcfftStatus FFTRepo::getPlan( hcfftPlanHandle plHandle, FFTPlan*& fftPlan, lockRAII*& planLock ) {
  scopedLock sLock( lockRepo, _T( "getPlan" ) );
  //  First, check if we have already created a plan with this exact same FFTPlan
  repoPlansType::iterator iter  = repoPlans.find( plHandle );

  if( iter == repoPlans.end( ) ) {
    return  HCFFT_ERROR;
  }

  //  If plan is valid, return fill out the output pointers
  fftPlan   = iter->second.first;
  planLock  = iter->second.second;
  return  HCFFT_SUCCESS;
}

hcfftStatus FFTRepo::deletePlan( hcfftPlanHandle* plHandle ) {
  scopedLock sLock( lockRepo, _T( "deletePlan" ) );
  //  First, check if we have already created a plan with this exact same FFTPlan
  repoPlansType::iterator iter  = repoPlans.find( *plHandle );

  if( iter == repoPlans.end( ) ) {
    return  HCFFT_ERROR;
  }

  //  We lock the plan object while we are in the process of deleting it
  {
    scopedLock sLock( *iter->second.second, _T( "hcfftDestroyPlan" ) );
    //  Delete the FFTPlan
    delete iter->second.first;
  }
  //  Delete the lockRAII
  delete iter->second.second;
  //  Remove entry from our map object
  repoPlans.erase( iter );
  //  Clear the client's handle to signify that the plan is gone
  *plHandle = 0;
  return  HCFFT_SUCCESS;
}

hcfftStatus FFTRepo::setProgramEntryPoints( const hcfftGenerators gen, const hcfftPlanHandle& handle,
    const FFTKernelGenKeyParams& fftParam, const char* kernel_fwd,
    const char* kernel_back) {
  scopedLock sLock( lockRepo, _T( "setProgramEntryPoints" ) );
  fftRepoKey key = std::make_pair( gen, handle );
  fftRepoValue& fft = mapFFTs[ key ];
  fft.EntryPoint_fwd  = kernel_fwd;
  fft.EntryPoint_back = kernel_back;
  return  HCFFT_SUCCESS;
}

hcfftStatus FFTRepo::getProgramEntryPoint( const hcfftGenerators gen, const hcfftPlanHandle& handle,
    const FFTKernelGenKeyParams& fftParam, hcfftDirection dir,
    std::string& kernel) {
  scopedLock sLock( lockRepo, _T( "getProgramEntryPoint" ) );
  fftRepoKey key = std::make_pair( gen, handle );
  fftRepo_iterator pos = mapFFTs.find( key );

  if( pos == mapFFTs.end( ) ) {
    return  HCFFT_ERROR;
  }

  switch (dir) {
    case HCFFT_FORWARD:
      kernel = pos->second.EntryPoint_fwd;
      break;

    case HCFFT_BACKWARD:
      kernel = pos->second.EntryPoint_back;
      break;

    default:
      assert (false);
      return HCFFT_ERROR;
  }

  if (0 == kernel.size()) {
    return  HCFFT_ERROR;
  }

  return  HCFFT_SUCCESS;
}

hcfftStatus FFTRepo::setProgramCode( const hcfftGenerators gen, const hcfftPlanHandle& handle, const FFTKernelGenKeyParams& fftParam, const std::string& kernel) {
  scopedLock sLock( lockRepo, _T( "setProgramCode" ) );
  fftRepoKey key = std::make_pair( gen, handle );
  // Prefix copyright statement at the top of generated kernels
  std::stringstream ss;
  ss <<
     "/* ************************************************************************\n"
     " * Copyright 2013 MCW, Inc.\n"
     " *\n"
     " * ************************************************************************/"
     << std::endl << std::endl;
  std::string prefixCopyright = ss.str();
  mapFFTs[ key ].ProgramString = prefixCopyright + kernel;
  return  HCFFT_SUCCESS;
}

hcfftStatus FFTRepo::getProgramCode( const hcfftGenerators gen, const hcfftPlanHandle& handle, const FFTKernelGenKeyParams& fftParam, std::string& kernel) {
  scopedLock sLock( lockRepo, _T( "getProgramCode" ) );
  fftRepoKey key = std::make_pair( gen, handle );
  fftRepo_iterator pos = mapFFTs.find( key);

  if( pos == mapFFTs.end( ) ) {
    return  HCFFT_ERROR;
  }

  kernel = pos->second.ProgramString;
  return  HCFFT_SUCCESS;
}

hcfftStatus FFTRepo::releaseResources( ) {
  scopedLock sLock( lockRepo, _T( "releaseResources" ) );

  //  Free all memory allocated in the repoPlans; represents cached plans that were not destroyed by the client
  //
  for( repoPlansType::iterator iter = repoPlans.begin( ); iter != repoPlans.end( ); ++iter ) {
    FFTPlan* plan = iter->second.first;
    lockRAII* lock  = iter->second.second;

    if( plan != NULL ) {
      delete plan;
    }

    if( lock != NULL ) {
      delete lock;
    }
  }

  //  Reset the plan count to zero because we are guaranteed to have destroyed all plans
  planCount = 1;
  //  Release all strings
  mapFFTs.clear( );
  return  HCFFT_SUCCESS;
}
/*------------------------------------------------FFTRepo----------------------------------------------------------------------------------*/

#include <dlfcn.h>
#include "ampfftlib.h"

//	Static initialization of the repo lock variable
lockRAII FFTRepo::lockRepo( _T( "FFTRepo" ) );

//	Static initialization of the plan count variable
size_t FFTRepo::planCount	= 1;

/*----------------------------------------------------FFTPlan-----------------------------------------------------------------------------*/

//	Read the kernels that this plan uses from file, and store into the plan
ampfftStatus WriteKernel( const ampfftPlanHandle plHandle, const ampfftGenerators gen, const FFTKernelGenKeyParams& fftParams)
{
	FFTRepo& fftRepo	= FFTRepo::getInstance( );

	std::string kernel;
	fftRepo.getProgramCode( gen, plHandle, fftParams, kernel);

        std::string filename;
        filename = "../kernel0.cpp";
        FILE *fp = fopen (filename.c_str(),"a+");
        if (!fp)
        {
          std::cout<<" File kernel.cpp open failed for writing "<<std::endl;
          return AMPFFT_ERROR;
        }

	size_t written = fwrite(kernel.c_str(), kernel.size(), 1, fp);
        if(!written)
        {
           std::cout<< "Kernel Write Failed "<<std::endl;
           exit(1);
        }

        fflush(fp);
        fclose(fp);

	return	AMPFFT_SUCCESS;
}

//	Compile the kernels that this plan uses, and store into the plan
ampfftStatus CompileKernels(const ampfftPlanHandle plHandle, const ampfftGenerators gen, FFTPlan* fftPlan )
{
	FILE *fp = fopen("kernel_program_code.cl", "w+");

	FFTRepo& fftRepo	= FFTRepo::getInstance( );

	FFTKernelGenKeyParams fftParams;
	fftPlan->GetKernelGenKey( fftParams );

	WriteKernel( plHandle, gen, fftParams);

        char cwd[1024];
        if (getcwd(cwd, sizeof(cwd)) != NULL)
          std::cout << "Current working dir: "<<cwd<<std::endl;
        else
	  std::cout<< "getcwd() error"<<std::endl;

        std::string pwd(cwd);
        std::string kernellib = pwd + "/../libFFTKernel0.so";

        char *compilerPath = (char*)calloc(100, 1);
        compilerPath = getenv ("MCWCPPAMPROOT");
        if(!compilerPath)
          std::cout<<"No Compiler Path Variable found. Please export MCWCPPAMPROOT "<<std::endl;
        else
          std::cout<< "The Compiler path is: "<<compilerPath<<std::endl;

        char *CLPath = (char*)calloc(100, 1);
        CLPath = getenv ("AMDAPPSDKROOT");
        if(!CLPath)
          std::cout<<"No OpenCL path Variable found. Please export AMDAPPSDKROOT "<<std::endl;
        else
          std::cout<<"The  OpenCL path is: "<<CLPath<<std::endl;

        string fftLibPath = pwd + "/../../Build/linux/";

        std::string Path(compilerPath);
        std::string OpenCLPath(CLPath);

        std::string execCmd = Path + "/build/compiler/bin/clang++ `" + Path + "/build/build/Release/bin/clamp-config --build --cxxflags --ldflags --shared` -I/opt/AMDAPP/include  ../kernel0.cpp -o " + kernellib ;

        system(execCmd.c_str());

	// For real transforms we comppile either forward or backward kernel
	bool r2c_transform = (fftParams.fft_inputLayout == AMPFFT_REAL);
	bool c2r_transform = (fftParams.fft_outputLayout == AMPFFT_REAL);
	bool real_transform = (gen == Copy) ? true : (r2c_transform || c2r_transform);
	bool h2c = (gen == Copy) && ((fftParams.fft_inputLayout == AMPFFT_COMPLEX) || (fftParams.fft_inputLayout == AMPFFT_COMPLEX));
	bool c2h = (gen == Copy) && ((fftParams.fft_outputLayout == AMPFFT_COMPLEX) || (fftParams.fft_outputLayout == AMPFFT_COMPLEX));

	// get a kernel object handle for a kernel with the given name
	if( (!real_transform) || r2c_transform || c2h )
	{
		std::string entryPoint;
		fftRepo.getProgramEntryPoint( gen, plHandle, fftParams, AMPFFT_FORWARD, entryPoint);
	}

	if( (!real_transform) || c2r_transform || h2c )
	{
		std::string entryPoint;
		fftRepo.getProgramEntryPoint( gen, plHandle, fftParams, AMPFFT_BACKWARD, entryPoint);
	}
}

//	This routine will query the OpenCL context for it's devices
//	and their hardware limitations, which we synthesize into a
//	hardware "envelope".
//	We only query the devices the first time we're called after
//	the object's context is set.  On 2nd and subsequent calls,
//	we just return the pointer.
//
ampfftStatus FFTPlan::SetEnvelope ()
{

	// TODO  The caller has already acquired the lock on *this
	//	However, we shouldn't depend on it.

        envelope.limit_LocalMemSize = 32768;
        envelope.limit_WorkGroupSize = 256;
        envelope.limit_Dimensions = 3;
        for(int i = 0 ;i < envelope.limit_Dimensions; i++)
          envelope.limit_Size[i] = 256;

	return AMPFFT_SUCCESS;
}

ampfftStatus FFTPlan::GetEnvelope (const FFTEnvelope ** ppEnvelope) const
{
	if(&envelope == NULL) assert(false);
	*ppEnvelope = &envelope;
	return AMPFFT_SUCCESS;
}

ampfftStatus FFTPlan::ampfftCreateDefaultPlan (ampfftPlanHandle* plHandle,ampfftDim dimension, const size_t *length)
{
  if( length == NULL )
    return AMPFFT_ERROR;

  size_t lenX = 1, lenY = 1, lenZ = 1;
  switch( dimension )
  {
    case AMPFFT_1D:
    {
      if( length[ 0 ] == 0 )
	return AMPFFT_ERROR;
      lenX = length[ 0 ];
    }
    break;

    case AMPFFT_2D:
    {
      if( length[ 0 ] == 0 || length[ 1 ] == 0 )
	return AMPFFT_ERROR;
      lenX = length[ 0 ];
      lenY = length[ 1 ];
    }
    break;

    case AMPFFT_3D:
    {
      if( length[ 0 ] == 0 || length[ 1 ] == 0 || length[ 2 ] == 0 )
	return AMPFFT_ERROR;
      lenX = length[ 0 ];
      lenY = length[ 1 ];
      lenZ = length[ 2 ];
    }
    break;

    default:
      return AMPFFT_ERROR;
    break;
  }

  FFTPlan* fftPlan = NULL;
  FFTRepo& fftRepo = FFTRepo::getInstance( );
  fftRepo.createPlan( plHandle, fftPlan );
  fftPlan->baked = false;
  fftPlan->dimension = dimension;
  fftPlan->location = AMPFFT_INPLACE;
  fftPlan->ipLayout = AMPFFT_COMPLEX;
  fftPlan->opLayout = AMPFFT_COMPLEX;
  fftPlan->precision = AMPFFT_SINGLE;
  fftPlan->forwardScale	= 1.0;
  fftPlan->backwardScale = 1.0 / static_cast< double >( lenX * lenY * lenZ );
  fftPlan->batchSize = 1;

  fftPlan->gen = Stockham; //default setting

  fftPlan->SetEnvelope();

  switch( dimension )
  {
    case AMPFFT_1D:
    {
      fftPlan->length.push_back( lenX );
      fftPlan->inStride.push_back( 1 );
      fftPlan->outStride.push_back( 1 );
      fftPlan->iDist		= lenX;
      fftPlan->oDist		= lenX;
    }
    break;

    case AMPFFT_2D:
    {
      fftPlan->length.push_back( lenX );
      fftPlan->length.push_back( lenY );
      fftPlan->inStride.push_back( 1 );
      fftPlan->inStride.push_back( lenX );
      fftPlan->outStride.push_back( 1 );
      fftPlan->outStride.push_back( lenX );
      fftPlan->iDist		= lenX*lenY;
      fftPlan->oDist		= lenX*lenY;
    }
    break;

    case AMPFFT_3D:
    {
      fftPlan->length.push_back( lenX );
      fftPlan->length.push_back( lenY );
      fftPlan->length.push_back( lenZ );
      fftPlan->inStride.push_back( 1 );
      fftPlan->inStride.push_back( lenX );
      fftPlan->inStride.push_back( lenX*lenY );
      fftPlan->outStride.push_back( 1 );
      fftPlan->outStride.push_back( lenX );
      fftPlan->outStride.push_back( lenX*lenY );
      fftPlan->iDist		= lenX*lenY*lenZ;
      fftPlan->oDist		= lenX*lenY*lenZ;
    }
    break;
  }
  return AMPFFT_SUCCESS;
}

ampfftStatus FFTPlan::ampfftEnqueueTransform(ampfftPlanHandle plHandle, ampfftDirection dir, Concurrency::array_view<float, 1> *clInputBuffers,
				             Concurrency::array_view<float, 1> *clOutputBuffers, Concurrency::array_view<float, 1> *clTmpBuffers)
{
	ampfftStatus status = AMPFFT_SUCCESS;

        std::map<int, void*> vectArr;

	FFTRepo& fftRepo	= FFTRepo::getInstance( );
	FFTPlan* fftPlan	= NULL;
	lockRAII* planLock	= NULL;

	//	At this point, the user wants to enqueue a plan to execute.  We lock the plan down now, such that
	//	after we finish baking the plan (if the user did not do that explicitely before), the plan cannot
	//	change again through the action of other thread before we enqueue this plan for execution.
	fftRepo.getPlan( plHandle, fftPlan, planLock );
	scopedLock sLock( *planLock, _T( "ampfftGetPlanBatchSize" ) );

	if( fftPlan->baked == false )
	{
		ampfftBakePlan( plHandle);
	}

	if (fftPlan->ipLayout == AMPFFT_REAL)
	  dir = AMPFFT_FORWARD;
	else if	(fftPlan->opLayout == AMPFFT_REAL)
	  dir = AMPFFT_BACKWARD;

        // we do not check the user provided buffer at this release
	Concurrency::array_view<float, 1> *localIntBuffer = clTmpBuffers;

	if( clTmpBuffers == NULL && fftPlan->tmpBufSize > 0 && fftPlan->intBuffer == NULL)
	{
		// create the intermediate buffers
		// The intermediate buffer is always interleave and packed
		// For outofplace operation, we have the choice not to create intermediate buffer
		// input ->(col+Transpose) output ->(col) output
                float *init = (float*)calloc(fftPlan->tmpBufSize, sizeof(float));
                Concurrency::array<float, 1> arr = Concurrency::array<float, 1>(Concurrency::extent<1>(fftPlan->tmpBufSize), init);
		fftPlan->intBuffer = new Concurrency::array_view<float>(arr);
	}

	if( localIntBuffer == NULL && fftPlan->intBuffer != NULL )
		localIntBuffer = fftPlan->intBuffer;

	if( fftPlan->intBufferRC == NULL && fftPlan->tmpBufSizeRC > 0 )
	{
                float *init = (float*)calloc(fftPlan->tmpBufSizeRC, sizeof(float));
                Concurrency::array<float, 1> arr = Concurrency::array<float, 1>(Concurrency::extent<1>(fftPlan->tmpBufSizeRC), init);
		fftPlan->intBufferRC = new Concurrency::array_view<float>(arr);
	}

	if( fftPlan->intBufferC2R == NULL && fftPlan->tmpBufSizeC2R > 0 )
	{
                float *init = (float*)calloc(fftPlan->tmpBufSizeC2R, sizeof(float));
                Concurrency::array<float, 1> arr = Concurrency::array<float, 1>(Concurrency::extent<1>(fftPlan->tmpBufSizeC2R), init);
		fftPlan->intBufferC2R = new Concurrency::array_view<float>(arr);
	}

	//	The largest vector we can transform in a single pass
	//	depends on the GPU caps -- especially the amount of LDS
	//	available
	//
	size_t Large1DThreshold = 0;
	fftPlan->GetMax1DLength (&Large1DThreshold);
	BUG_CHECK (Large1DThreshold > 1);

	if(fftPlan->gen != Copy)
	switch( fftPlan->dimension )
	{
		case AMPFFT_1D:
		{
			if (fftPlan->length[0] <= Large1DThreshold)
				break;

			if( fftPlan->ipLayout == AMPFFT_REAL )
			{
				// First pass
				ampfftEnqueueTransform( fftPlan->planX, AMPFFT_FORWARD, clInputBuffers, fftPlan->intBufferRC, localIntBuffer);

				ampfftEnqueueTransform( fftPlan->planY, AMPFFT_FORWARD, fftPlan->intBufferRC, fftPlan->intBufferRC, localIntBuffer );

				Concurrency::array_view<float, 1> *out_local;
				out_local = (fftPlan->location==AMPFFT_INPLACE) ? clInputBuffers : clOutputBuffers;

				ampfftEnqueueTransform( fftPlan->planRCcopy, AMPFFT_FORWARD, fftPlan->intBufferRC, out_local, localIntBuffer );

				return	AMPFFT_SUCCESS;

			}
			else if( fftPlan->opLayout == AMPFFT_REAL )
			{
				// copy from hermitian to full complex
				ampfftEnqueueTransform( fftPlan->planRCcopy, AMPFFT_BACKWARD, clInputBuffers, fftPlan->intBufferRC, localIntBuffer );

				// First pass
				// column with twiddle first, INPLACE,
				ampfftEnqueueTransform( fftPlan->planX, AMPFFT_BACKWARD, fftPlan->intBufferRC, fftPlan->intBufferRC, localIntBuffer);

				Concurrency::array_view<float, 1> *out_local;
				out_local = (fftPlan->location==AMPFFT_INPLACE) ? clInputBuffers : clOutputBuffers;

				// another column FFT output, OUTOFPLACE + transpose
				ampfftEnqueueTransform( fftPlan->planY, AMPFFT_BACKWARD, fftPlan->intBufferRC, out_local, localIntBuffer );

				return	AMPFFT_SUCCESS;
			}
			else
			{
				if (fftPlan->transflag)
				{
					//First transpose
					// Input->tmp
					ampfftEnqueueTransform( fftPlan->planTX, dir, clInputBuffers, localIntBuffer, NULL );

					Concurrency::array_view<float, 1> *mybuffers;
					if (fftPlan->location == AMPFFT_INPLACE)
						mybuffers = clInputBuffers;
					else
						mybuffers = clOutputBuffers;

					//First Row
					//tmp->output
					ampfftEnqueueTransform( fftPlan->planX, dir, localIntBuffer, mybuffers, NULL );

					//Second Transpose
					// output->tmp
					ampfftEnqueueTransform( fftPlan->planTY, dir, mybuffers, localIntBuffer, NULL );

					//Second Row
					//tmp->tmp, inplace
					ampfftEnqueueTransform( fftPlan->planY, dir, localIntBuffer, NULL, NULL );

					//Third Transpose
					// tmp->output
					ampfftEnqueueTransform( fftPlan->planTZ, dir, localIntBuffer, mybuffers, NULL );

					return	AMPFFT_SUCCESS;
				}

				if (fftPlan->large1D == 0)
				{
					// First pass
					// column with twiddle first, OUTOFPLACE, + transpose
					ampfftEnqueueTransform( fftPlan->planX, dir, clInputBuffers, localIntBuffer, localIntBuffer);

					//another column FFT output, OUTOFPLACE
					if (fftPlan->location == AMPFFT_INPLACE)
					{
						ampfftEnqueueTransform( fftPlan->planY, dir, localIntBuffer, clInputBuffers, localIntBuffer );

					}
					else
					{
						ampfftEnqueueTransform( fftPlan->planY, dir, localIntBuffer, clOutputBuffers, localIntBuffer );

					}
				}
				else
				{
					// second pass for huge 1D
					// column with twiddle first, OUTOFPLACE, + transpose
					ampfftEnqueueTransform( fftPlan->planX, dir, localIntBuffer, clOutputBuffers, localIntBuffer);

					ampfftEnqueueTransform( fftPlan->planY, dir,clOutputBuffers, clOutputBuffers, localIntBuffer );

				}

				return	AMPFFT_SUCCESS;
			}
			break;
		}
		case AMPFFT_2D:
		{
			// if transpose kernel, we will fall below
			if (fftPlan->transflag && !(fftPlan->planTX)) break;

			//cl_event rowOutEvents = NULL;

			if (fftPlan->transflag)
			{//first time set up transpose kernel for 2D
				//First row
				ampfftEnqueueTransform( fftPlan->planX, dir, clInputBuffers, clOutputBuffers, NULL );

				Concurrency::array_view<float, 1> *mybuffers;

				if (fftPlan->location==AMPFFT_INPLACE)
					mybuffers = clInputBuffers;
				else
					mybuffers = clOutputBuffers;

				bool xyflag = (fftPlan->length[0] == fftPlan->length[1]) ? false : true;

				if (xyflag)
				{
					//First transpose
					ampfftEnqueueTransform( fftPlan->planTX, dir, mybuffers, localIntBuffer, NULL );

					if (fftPlan->transposeType == AMPFFT_NOTRANSPOSE)
					{
						//Second Row transform
						ampfftEnqueueTransform( fftPlan->planY, dir, localIntBuffer, NULL, NULL );

						//Second transpose
						ampfftEnqueueTransform( fftPlan->planTY, dir, localIntBuffer, mybuffers, NULL );

					}
					else
					{
						//Second Row transform
						ampfftEnqueueTransform( fftPlan->planY, dir, localIntBuffer, mybuffers, NULL );
					}
				}
				else
				{
					// First Transpose
					ampfftEnqueueTransform( fftPlan->planTX, dir, mybuffers, NULL, NULL );

					if (fftPlan->transposeType == AMPFFT_NOTRANSPOSE)
					{
						//Second Row transform
						ampfftEnqueueTransform( fftPlan->planY, dir, mybuffers, NULL, NULL );

						//Second transpose
						ampfftEnqueueTransform( fftPlan->planTY, dir, mybuffers, NULL, NULL );
					}
					else
					{
						//Second Row transform
						ampfftEnqueueTransform( fftPlan->planY, dir, mybuffers, NULL, NULL );
					}
				}

				return AMPFFT_SUCCESS;
			}

			if ( (fftPlan->large2D || fftPlan->length.size()>2) &&
				(fftPlan->ipLayout != AMPFFT_REAL) && (fftPlan->opLayout != AMPFFT_REAL))
			{
				if (fftPlan->location==AMPFFT_INPLACE)
				{
					//deal with row first
					ampfftEnqueueTransform( fftPlan->planX, dir, clInputBuffers, NULL, localIntBuffer );

					//deal with column
					ampfftEnqueueTransform( fftPlan->planY, dir, clInputBuffers, NULL, localIntBuffer );
				}
				else
				{
					//deal with row first
					ampfftEnqueueTransform( fftPlan->planX, dir, clInputBuffers, clOutputBuffers, localIntBuffer );

					//deal with column
					ampfftEnqueueTransform( fftPlan->planY, dir, clOutputBuffers, NULL, localIntBuffer );
				}
			}
			else
			{
				if(fftPlan->ipLayout == AMPFFT_REAL)
				{
					if (fftPlan->location==AMPFFT_INPLACE)
					{
						// deal with row
						ampfftEnqueueTransform( fftPlan->planX, AMPFFT_FORWARD, clInputBuffers, NULL, localIntBuffer );

						// deal with column
						ampfftEnqueueTransform( fftPlan->planY, AMPFFT_FORWARD, clInputBuffers, NULL, localIntBuffer );
					}
					else
					{
						// deal with row
						ampfftEnqueueTransform( fftPlan->planX, AMPFFT_FORWARD, clInputBuffers, clOutputBuffers, localIntBuffer );

						// deal with column
						ampfftEnqueueTransform( fftPlan->planY, AMPFFT_FORWARD, clOutputBuffers, NULL, localIntBuffer );
					}
				}
				else if(fftPlan->opLayout == AMPFFT_REAL)
				{
					Concurrency::array_view<float, 1> *out_local, *int_local, *out_y;

					if(fftPlan->length.size() > 2)
					{
						out_local = clOutputBuffers;
						int_local = NULL;
						out_y = clInputBuffers;
					}
					else
					{
						out_local = (fftPlan->location == AMPFFT_INPLACE) ? clInputBuffers : clOutputBuffers;
						int_local = fftPlan->tmpBufSizeC2R ? fftPlan->intBufferC2R : localIntBuffer;
						out_y = int_local;
					}
					// deal with column
					ampfftEnqueueTransform( fftPlan->planY, AMPFFT_BACKWARD, clInputBuffers, int_local, localIntBuffer );

					// deal with row
					ampfftEnqueueTransform( fftPlan->planX, AMPFFT_BACKWARD, out_y, out_local, localIntBuffer );

				}
				else
				{
					//deal with row first
					ampfftEnqueueTransform( fftPlan->planX, dir, clInputBuffers, localIntBuffer, localIntBuffer );

					if (fftPlan->location == AMPFFT_INPLACE)
					{
						//deal with column
						ampfftEnqueueTransform( fftPlan->planY, dir, localIntBuffer, clInputBuffers, localIntBuffer );
					}
					else
					{
						//deal with column
						ampfftEnqueueTransform( fftPlan->planY, dir, localIntBuffer, clOutputBuffers, localIntBuffer );
					}
				}
			}

			return	AMPFFT_SUCCESS;
		}
	}

	FFTKernelGenKeyParams fftParams;
	//	Translate the user plan into the structure that we use to map plans to clPrograms
	fftPlan->GetKernelGenKey( fftParams );

        std::string kernel;
        fftRepo.getProgramCode( fftPlan->gen, plHandle, fftParams, kernel);

        /* constant buffer */
	unsigned int uarg = 0;
	vectArr.insert(std::make_pair(uarg++,fftPlan->const_buffer));

	//	Decode the relevant properties from the plan paramter to figure out how many input/output buffers we have
	switch( fftPlan->ipLayout )
	{
		case AMPFFT_COMPLEX:
		{
			switch( fftPlan->opLayout )
			{
                                case AMPFFT_COMPLEX:
				case AMPFFT_REAL:
				{
					if( fftPlan->location == AMPFFT_INPLACE )
					{
						vectArr.insert(std::make_pair(uarg++, clInputBuffers));
					}
					else
					{
						vectArr.insert(std::make_pair(uarg++, clInputBuffers));
						vectArr.insert(std::make_pair(uarg++, clOutputBuffers));
					}
					break;
				}
				default:
				{
					//	Don't recognize output layout
					return AMPFFT_ERROR;
				}
			}

			break;
		}
		case AMPFFT_REAL:
		{
			switch( fftPlan->opLayout )
			{
				case AMPFFT_COMPLEX:
				{
					if( fftPlan->location == AMPFFT_INPLACE )
					{
						vectArr.insert(std::make_pair(uarg++,clInputBuffers));
					}
					else
					{
						vectArr.insert(std::make_pair(uarg++, clInputBuffers));
						vectArr.insert(std::make_pair(uarg++, clOutputBuffers));
					}
					break;
				}
			}

			break;
		}
		default:
		{
			//	Don't recognize output layout
			return AMPFFT_ERROR;
		}
	}

        vector< size_t > gWorkSize;
	vector< size_t > lWorkSize;
	ampfftStatus result = fftPlan->GetWorkSizes (gWorkSize, lWorkSize);

	if (AMPFFT_ERROR == result)
	{
		std::cout<<"Work size too large for clEnqueNDRangeKernel()"<<std::endl;
	}
	BUG_CHECK (gWorkSize.size() == lWorkSize.size());

        void * kernelHandle = NULL;
        typedef void (FUNC_FFTFwd)(std::map<int, void*> *vectArr);
        FUNC_FFTFwd * FFTcall;

        char cwd[1024];
        if (getcwd(cwd, sizeof(cwd)) != NULL)
          std::cout << "Current working dir: "<<cwd<<std::endl;
        else
	  std::cout<< "getcwd() error"<<std::endl;

        std::string pwd(cwd);
        std::string kernellib = pwd + "/../libFFTKernel0.so";

        char *err = (char*) calloc(128,2);
        kernelHandle = dlopen(kernellib.c_str(),RTLD_NOW);
         if(!kernelHandle)
        {
          std::cout << "Failed to load Kernel: "<< kernellib.c_str()<<std::endl;
          return AMPFFT_ERROR;
        }
        else
        {
          std::cout<<"Loaded Kernel: "<<kernellib.c_str()<<std::endl;
        }

        if(dir == AMPFFT_FORWARD)
        {
        std::string funcName = "fft_fwd";
        funcName +=  std::to_string(plHandle);
        FFTcall= (FUNC_FFTFwd*) dlsym(kernelHandle, funcName.c_str());
        if (!FFTcall)
          std::cout<<"Loading fft_fwd fails "<<std::endl;
        err=dlerror();
        if (err)
        {
          std::cout<<"failed to locate fft_fwd(): "<< err;
          exit(1);
        }
        }
        else if(dir == AMPFFT_BACKWARD)
        {
        std::string funcName = "fft_back";
        funcName +=  std::to_string(plHandle);
        FFTcall= (FUNC_FFTFwd*) dlsym(kernelHandle, funcName.c_str());
        if (!FFTcall)
          std::cout<<"Loading fft_back fails "<<std::endl;
        err=dlerror();
        if (err)
        {
          std::cout<<"failed to locate fft_back(): "<< err;
          exit(1);
        }
        }

        FFTcall(&vectArr);

        dlclose(kernelHandle);
        kernelHandle = NULL;
}

ampfftStatus FFTPlan::ampfftBakePlan(ampfftPlanHandle plHandle)
{
  FFTRepo& fftRepo = FFTRepo::getInstance( );
  FFTPlan* fftPlan = NULL;
  lockRAII* planLock = NULL;

  fftRepo.getPlan(plHandle, fftPlan, planLock);
  scopedLock sLock( *planLock, _T( "ampfftBakePlan" ) );

  // if we have already baked the plan and nothing has changed since, we're done here
  if( fftPlan->baked == true )
  {
    return AMPFFT_SUCCESS;
  }

  //find product of lengths
  size_t pLength = 1;
  switch(fftPlan->dimension)
  {
    case AMPFFT_3D: pLength *= fftPlan->length[2];
    case AMPFFT_2D: pLength *= fftPlan->length[1];
    case AMPFFT_1D: pLength *= fftPlan->length[0];
  }

  if(fftPlan->dimension == fftPlan->length.size() && fftPlan->gen != Transpose && fftPlan->gen != Copy) // confirm it is top-level plan (user plan)
  {
    if(fftPlan->location == AMPFFT_INPLACE)
    {
      if((fftPlan->ipLayout == AMPFFT_COMPLEX) || (fftPlan->opLayout == AMPFFT_COMPLEX))
        return AMPFFT_ERROR;
    }

    // Make sure strides & distance are same for C-C transforms
    if(fftPlan->location == AMPFFT_INPLACE)
    {
      if((fftPlan->ipLayout != AMPFFT_REAL) && (fftPlan->opLayout != AMPFFT_REAL))
        {
	  // check strides
	  for(size_t i=0; i<fftPlan->dimension; i++)
	    if(fftPlan->inStride[i] != fftPlan->outStride[i])
	      return AMPFFT_ERROR;

	  // check distance
	  if(fftPlan->iDist != fftPlan->oDist)
	     return AMPFFT_ERROR;
        }
     }
  }
     if(fftPlan->gen == Copy)
     {
       fftPlan->GenerateKernel(plHandle, fftRepo);
       CompileKernels(plHandle, fftPlan->gen, fftPlan);
       fftPlan->baked = true;
       return AMPFFT_SUCCESS;
     }

     bool rc = (fftPlan->ipLayout == AMPFFT_REAL) || (fftPlan->opLayout == AMPFFT_REAL);
     // Compress the plan by discarding length '1' dimensions
     // decision to pick generator
     if(fftPlan->dimension == fftPlan->length.size() && fftPlan->gen != Transpose && !rc) // confirm it is top-level plan (user plan)
     {
       size_t dmnsn = fftPlan->dimension;
       bool pow2flag = true;

       // 	 case flows with no 'break' statements
       switch(fftPlan->dimension)
       {
         case AMPFFT_3D:
           if(fftPlan->length[2] == 1)
           {
             dmnsn -= 1;
             fftPlan-> inStride.erase(fftPlan-> inStride.begin() + 2);
             fftPlan->outStride.erase(fftPlan->outStride.begin() + 2);
             fftPlan->   length.erase(fftPlan->   length.begin() + 2);
           }
	   else
	   {
	     if(!IsPo2(fftPlan->length[2]))
               pow2flag=false;
	   }
	 case AMPFFT_2D:
           if(fftPlan->length[1] == 1)
	   {
	     dmnsn -= 1;
	     fftPlan-> inStride.erase(fftPlan-> inStride.begin() + 1);
	     fftPlan->outStride.erase(fftPlan->outStride.begin() + 1);
	     fftPlan->   length.erase(fftPlan->   length.begin() + 1);
	   }
	   else
	   {
	     if(!IsPo2(fftPlan->length[1]))
               pow2flag=false;
	   }
	 case AMPFFT_1D:
           if( (fftPlan->length[0] == 1) && (dmnsn > 1) )
	   {
	     dmnsn -= 1;
	     fftPlan-> inStride.erase(fftPlan-> inStride.begin());
	     fftPlan->outStride.erase(fftPlan->outStride.begin());
	     fftPlan->   length.erase(fftPlan->   length.begin());
	   }
	   else
	   {
	     if(!IsPo2(fftPlan->length[0]))
               pow2flag=false;
	   }
	 }
         fftPlan->dimension = (ampfftDim)dmnsn;
       }

       // first time check transposed
       if (fftPlan->transposeType != AMPFFT_NOTRANSPOSE && fftPlan->dimension != AMPFFT_2D &&
	   fftPlan->dimension == fftPlan->length.size())
	     return AMPFFT_ERROR;

       //	The largest vector we can transform in a single pass
       //	depends on the GPU caps -- especially the amount of LDS
       //	available
       //
       size_t Large1DThreshold = 0;

       //First time check or see if LDS paramters are set-up.
       if (fftPlan->uLdsFraction == 0)
       {
         switch( fftPlan->dimension )
         {
           case AMPFFT_1D:
           {
	     if(fftPlan->length[0] < 32768 || fftPlan->length[0] > 1048576)
	       fftPlan->uLdsFraction = 8;
	     else
	       fftPlan->uLdsFraction = 4;

	     if(fftPlan->length[0] < 1024 )
	       fftPlan->bLdsComplex = true;
	     else
	       fftPlan->bLdsComplex = false;
	   }
	   break;
	   case AMPFFT_2D:
	   {
	     fftPlan->uLdsFraction = 4;
	     fftPlan->bLdsComplex = false;
	   }
	   break;
	   case AMPFFT_3D:
	   {
	     //for case 128*128*128 and 1024*128*128, fraction = 8 is faster.
	     fftPlan->uLdsFraction = 4;
	     fftPlan->bLdsComplex = false;
	   }
	   break;
	   }
	}
	fftPlan->GetMax1DLength(&Large1DThreshold);
	BUG_CHECK(Large1DThreshold > 1);

	//	Verify that the data passed to us is packed
	switch( fftPlan->dimension )
	{
	  case AMPFFT_1D:
	  {
	    if(fftPlan->length[0] > Large1DThreshold)
	    {
              size_t clLengths[] = { 1, 1, 0 };
	      size_t in_1d, in_x, count;

              BUG_CHECK (IsPo2 (Large1DThreshold))

              // see whether large1D_Xfactor are fixed or not
	      if (fftPlan->large1D_Xfactor == 0 )
	      {
	        if(IsPo2(fftPlan->length[0]) )
	        {
		  in_1d = BitScanF (Large1DThreshold);	// this is log2(LARGE1D_THRESHOLD)
		  in_x  = BitScanF (fftPlan->length[0]);	// this is log2(length)
		  BUG_CHECK (in_1d > 0)
		  count = in_x/in_1d;
		  if (count*in_1d < in_x)
		  {
		    count++;
		    in_1d = in_x / count;
		    if (in_1d * count < in_x) in_1d++;
		  }
		  clLengths[1] = (size_t)1 << in_1d;
                }
		else
		{
		  //This array must be kept sorted in the ascending order
		  size_t supported[] = {1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36, 40,
					45, 48, 50, 54, 60, 64, 72, 75, 80, 81, 90, 96, 100, 108, 120, 125, 128, 135,
					144, 150, 160, 162, 180, 192, 200, 216, 225, 240, 243, 250, 256, 270, 288,
					300, 320, 324, 360, 375, 384, 400, 405, 432, 450, 480, 486, 500, 512, 540,
					576, 600, 625, 640, 648, 675, 720, 729, 750, 768, 800, 810, 864, 900, 960,
					972, 1000, 1024, 1080, 1125, 1152, 1200, 1215, 1250, 1280, 1296, 1350, 1440,
					1458, 1500, 1536, 1600, 1620, 1728, 1800, 1875, 1920, 1944, 2000, 2025, 2048,
					2160, 2187, 2250, 2304, 2400, 2430, 2500, 2560, 2592, 2700, 2880, 2916, 3000,
					3072, 3125, 3200, 3240, 3375, 3456, 3600, 3645, 3750, 3840, 3888, 4000, 4050, 4096 };

		  size_t lenSupported = sizeof(supported)/sizeof(supported[0]);
		  size_t maxFactoredLength = (supported[lenSupported-1] < Large1DThreshold) ? supported[lenSupported-1] : Large1DThreshold;
		  size_t halfPowerLength = (size_t)1 << ( (CeilPo2(fftPlan->length[0]) + 1) / 2 );
		  size_t factoredLengthStart =  (halfPowerLength < maxFactoredLength) ? halfPowerLength : maxFactoredLength;

		  size_t indexStart = 0;
		  while(supported[indexStart] < factoredLengthStart) indexStart++;

		  for(size_t i = indexStart; i >= 1; i--)
		  {
		    if(fftPlan->length[0] % supported[i] == 0 )
		    {
		      clLengths[1] = supported[i];
		      break;
		    }
		  }
		}
                clLengths[0] = fftPlan->length[0]/clLengths[1];
	      }
	      else
	      {
	        //large1D_Xfactor will not pass to the second level of call
		clLengths[0] = fftPlan->large1D_Xfactor;
		clLengths[1] = fftPlan->length[0]/clLengths[0];
		ARG_CHECK (fftPlan->length[0] == clLengths[0] * clLengths[1]);
	      }

              while (1 && (fftPlan->ipLayout != AMPFFT_REAL) && (fftPlan->opLayout != AMPFFT_REAL))
	      {
	        if (!IsPo2(fftPlan->length[0])) break;
		if (fftPlan->length.size() > 1) break;
		if (fftPlan->inStride[0] != 1 || fftPlan->outStride[0] != 1) break;
		//This length is good for using transpose
		if (fftPlan->length[0] < 131072) break;
		//first version not support huge1D, TBD
		if (clLengths[0] > Large1DThreshold) break;
		ARG_CHECK(clLengths[0]>=32 && clLengths[1]>=32);
		if (fftPlan->tmpBufSize==0 )
		{
		  fftPlan->tmpBufSize = clLengths[0] * clLengths[1] * fftPlan->batchSize * fftPlan->ElementSize();
		}

		//Transpose
		//Input --> tmp buffer
		ampfftCreateDefaultPlan(&fftPlan->planTX, AMPFFT_2D, clLengths);

	        FFTPlan* trans1Plan	= NULL;
		lockRAII* trans1Lock	= NULL;
		fftRepo.getPlan(fftPlan->planTX, trans1Plan, trans1Lock);

                trans1Plan->location = AMPFFT_OUTOFPLACE;
		trans1Plan->precision = fftPlan->precision;
		trans1Plan->tmpBufSize = 0;
		trans1Plan->batchSize = fftPlan->batchSize;
		trans1Plan->envelope = fftPlan->envelope;
		trans1Plan->ipLayout = fftPlan->ipLayout;
		trans1Plan->opLayout = AMPFFT_COMPLEX;
		trans1Plan->inStride[0]   = fftPlan->inStride[0];
                trans1Plan->inStride[1]   = clLengths[0];
		trans1Plan->outStride[0]  = 1;
		trans1Plan->outStride[1]  = clLengths[1];
		trans1Plan->iDist         = fftPlan->iDist;
		trans1Plan->oDist         = fftPlan->length[0];
		trans1Plan->gen           = Transpose;
		trans1Plan->transflag     = true;

		ampfftBakePlan(fftPlan->planTX);

		//Row transform
		//tmp->output
		//size clLengths[1], batch clLengths[0], with length[0] twiddle factor multiplication
		ampfftCreateDefaultPlan(&fftPlan->planX, AMPFFT_1D, &clLengths[1]);

		FFTPlan* row1Plan	= NULL;
		lockRAII* row1Lock	= NULL;
		fftRepo.getPlan( fftPlan->planX, row1Plan, row1Lock);

		row1Plan->location     = AMPFFT_OUTOFPLACE;
		row1Plan->precision     = fftPlan->precision;
		row1Plan->forwardScale  = 1.0f;
		row1Plan->backwardScale = 1.0f;
		row1Plan->tmpBufSize    = 0;
		row1Plan->batchSize     = fftPlan->batchSize;
		row1Plan->bLdsComplex   = fftPlan->bLdsComplex;
		row1Plan->uLdsFraction  = fftPlan->uLdsFraction;
		row1Plan->ldsPadding    = fftPlan->ldsPadding;
		row1Plan->gen		= fftPlan->gen;
		row1Plan->envelope	= fftPlan->envelope;

		//Pass large1D flag to confirm we need multiply twiddle factor
		row1Plan->large1D       = fftPlan->length[0];

		row1Plan->length.push_back(clLengths[0]);
		row1Plan->ipLayout   = AMPFFT_COMPLEX;
		row1Plan->opLayout  = fftPlan->opLayout;
		row1Plan->inStride[0]   = 1;
		row1Plan->outStride[0]  = fftPlan->outStride[0];
		row1Plan->iDist         = fftPlan->length[0];
		row1Plan->oDist         = fftPlan->oDist;
		row1Plan->inStride.push_back(clLengths[1]);
		row1Plan->outStride.push_back(clLengths[1]);

		ampfftBakePlan(fftPlan->planX);

		//Transpose 2
		//Output --> tmp buffer
		clLengths[2] = clLengths[0];
		ampfftCreateDefaultPlan(&fftPlan->planTY, AMPFFT_2D, &clLengths[1]);
		FFTPlan* trans2Plan	= NULL;
		lockRAII* trans2Lock	= NULL;
		fftRepo.getPlan( fftPlan->planTY, trans2Plan, trans2Lock );

                trans2Plan->location     = AMPFFT_OUTOFPLACE;
		trans2Plan->precision     = fftPlan->precision;
		trans2Plan->tmpBufSize    = 0;
		trans2Plan->batchSize     = fftPlan->batchSize;
		trans2Plan->envelope	  = fftPlan->envelope;
		trans2Plan->opLayout   = fftPlan->opLayout;
		trans2Plan->opLayout  = AMPFFT_COMPLEX;
		trans2Plan->inStride[0]   = fftPlan->outStride[0];
		trans2Plan->inStride[1]   = clLengths[1];
		trans2Plan->outStride[0]  = 1;
		trans2Plan->outStride[1]  = clLengths[0];
		trans2Plan->iDist         = fftPlan->oDist;
		trans2Plan->oDist         = fftPlan->length[0];
		trans2Plan->gen           = Transpose;
		trans2Plan->transflag     = true;

		ampfftBakePlan(fftPlan->planTY);

		//Row transform 2
		//tmp->tmp
		//size clLengths[0], batch clLengths[1]
		ampfftCreateDefaultPlan( &fftPlan->planY, AMPFFT_1D, &clLengths[0]);

		FFTPlan* row2Plan	= NULL;
		lockRAII* row2Lock	= NULL;
		fftRepo.getPlan( fftPlan->planY, row2Plan, row2Lock );

		row2Plan->location     = AMPFFT_INPLACE;
		row2Plan->precision     = fftPlan->precision;
		row2Plan->forwardScale  = fftPlan->forwardScale;
		row2Plan->backwardScale = fftPlan->backwardScale;
		row2Plan->tmpBufSize    = 0;
		row2Plan->batchSize     = fftPlan->batchSize;
		row2Plan->bLdsComplex   = fftPlan->bLdsComplex;
		row2Plan->uLdsFraction  = fftPlan->uLdsFraction;
		row2Plan->ldsPadding    = fftPlan->ldsPadding;
		row2Plan->gen		= fftPlan->gen;
		row2Plan->envelope	= fftPlan->envelope;

		//No twiddle factor is needed.
		row2Plan->large1D       = 0;

		row2Plan->length.push_back(clLengths[1]);
		row2Plan->ipLayout   = AMPFFT_COMPLEX;
		row2Plan->opLayout  = AMPFFT_COMPLEX;
		row2Plan->inStride[0]   = 1;
		row2Plan->outStride[0]  = 1;
		row2Plan->iDist         = fftPlan->length[0];
		row2Plan->oDist         = fftPlan->length[0];
		row2Plan->inStride.push_back(clLengths[0]);
		row2Plan->outStride.push_back(clLengths[0]);

		ampfftBakePlan(fftPlan->planY);

		//Transpose 3
		//tmp --> output
		ampfftCreateDefaultPlan( &fftPlan->planTZ, AMPFFT_2D, clLengths);

		FFTPlan* trans3Plan	= NULL;
		lockRAII* trans3Lock	= NULL;
		fftRepo.getPlan( fftPlan->planTZ, trans3Plan, trans3Lock);

		trans3Plan->location     = AMPFFT_OUTOFPLACE;
		trans3Plan->precision     = fftPlan->precision;
		trans3Plan->tmpBufSize    = 0;
		trans3Plan->batchSize     = fftPlan->batchSize;
		trans3Plan->envelope	  = fftPlan->envelope;
		trans3Plan->ipLayout   = AMPFFT_COMPLEX;
		trans3Plan->opLayout  = fftPlan->opLayout;
		trans3Plan->inStride[0]   = 1;
		trans3Plan->inStride[1]   = clLengths[0];
		trans3Plan->outStride[0]  = fftPlan->outStride[0];
		trans3Plan->outStride[1]  = clLengths[1];
		trans3Plan->iDist         = fftPlan->length[0];
		trans3Plan->oDist         = fftPlan->oDist;
		trans3Plan->gen           = Transpose;
		trans3Plan->transflag     = true;

		ampfftBakePlan(fftPlan->planTZ);

                fftPlan->transflag = true;
		fftPlan->baked = true;
		return	AMPFFT_SUCCESS;
	      }
              size_t length0 = clLengths[0];
	      size_t length1 = clLengths[1];

	      if(fftPlan->ipLayout == AMPFFT_REAL)
	      {
	        if (fftPlan->tmpBufSizeRC==0 )
		{
		  fftPlan->tmpBufSizeRC = length0 * length1 * fftPlan->batchSize * fftPlan->ElementSize();
                  for (size_t index=1; index < fftPlan->length.size(); index++)
		  {
		    fftPlan->tmpBufSizeRC *= fftPlan->length[index];
		  }
		}

	        // column FFT, size clLengths[1], batch clLengths[0], with length[0] twiddle factor multiplication
		// transposed output
		ampfftCreateDefaultPlan( &fftPlan->planX, AMPFFT_1D, &clLengths[1]);

		FFTPlan* colTPlan	= NULL;
		lockRAII* colLock	= NULL;
		fftRepo.getPlan( fftPlan->planX, colTPlan, colLock);

		// current plan is to create intermediate buffer, packed and interleave
		// This is a column FFT, the first elements distance between each FFT is the distance of the first two
		// elements in the original buffer. Like a transpose of the matrix
		// we need to pass clLengths[0] and instride size to kernel, so kernel can tell the difference
		//this part are common for both passes
	        colTPlan->location     = AMPFFT_OUTOFPLACE;
		colTPlan->precision     = fftPlan->precision;
		colTPlan->forwardScale  = 1.0f;
		colTPlan->backwardScale = 1.0f;
		colTPlan->tmpBufSize    = 0;
		colTPlan->batchSize     = fftPlan->batchSize;
		colTPlan->bLdsComplex   = fftPlan->bLdsComplex;
		colTPlan->uLdsFraction  = fftPlan->uLdsFraction;
		colTPlan->ldsPadding    = fftPlan->ldsPadding;
		colTPlan->gen		= fftPlan->gen;
		colTPlan->envelope	= fftPlan->envelope;

		//Pass large1D flag to confirm we need multiply twiddle factor
		colTPlan->large1D       = fftPlan->length[0];
		colTPlan->RCsimple	= true;

		colTPlan->length.push_back(clLengths[0]);

		// first Pass
		colTPlan->ipLayout   = fftPlan->ipLayout;
		colTPlan->opLayout  = AMPFFT_COMPLEX;
		colTPlan->inStride[0]   = fftPlan->inStride[0] * clLengths[0];
		colTPlan->outStride[0]  = 1;
		colTPlan->iDist         = fftPlan->iDist;
		colTPlan->oDist         = length0 * length1;
		colTPlan->inStride.push_back(fftPlan->inStride[0]);
		colTPlan->outStride.push_back(length1);

		for (size_t index=1; index < fftPlan->length.size(); index++)
		{
		  colTPlan->length.push_back(fftPlan->length[index]);
		  colTPlan->inStride.push_back(fftPlan->inStride[index]);
		  // tmp buffer is tightly packed
		  colTPlan->outStride.push_back(colTPlan->oDist);
		  colTPlan->oDist        *= fftPlan->length[index];
		}

		ampfftBakePlan(fftPlan->planX);

		//another column FFT, size clLengths[0], batch clLengths[1], output without transpose
		ampfftCreateDefaultPlan( &fftPlan->planY, AMPFFT_1D,  &clLengths[0]);

		FFTPlan* col2Plan	= NULL;
		lockRAII* rowLock	= NULL;
		fftRepo.getPlan( fftPlan->planY, col2Plan, rowLock);

		// This is second column fft, intermediate buffer is packed and interleaved
		// we need to pass clLengths[1] and instride size to kernel, so kernel can tell the difference
		// common part for both passes
		col2Plan->location     = AMPFFT_INPLACE;
		col2Plan->ipLayout   = AMPFFT_COMPLEX;
		col2Plan->opLayout  = AMPFFT_COMPLEX;

		col2Plan->precision     = fftPlan->precision;
		col2Plan->forwardScale  = fftPlan->forwardScale;
		col2Plan->backwardScale = fftPlan->backwardScale;
		col2Plan->tmpBufSize    = 0;
		col2Plan->batchSize     = fftPlan->batchSize;
		col2Plan->bLdsComplex   = fftPlan->bLdsComplex;
		col2Plan->uLdsFraction  = fftPlan->uLdsFraction;
		col2Plan->ldsPadding    = fftPlan->ldsPadding;
		col2Plan->gen	= fftPlan->gen;
		col2Plan->envelope = fftPlan->envelope;

		col2Plan->length.push_back(length1);

		col2Plan->inStride[0]  = length1;
		col2Plan->inStride.push_back(1);
		col2Plan->iDist        = length0 * length1;

		col2Plan->outStride[0] = length1;
		col2Plan->outStride.push_back(1);
		col2Plan->oDist         = length0 * length1;

		for (size_t index=1; index < fftPlan->length.size(); index++)
		{
		  col2Plan->length.push_back(fftPlan->length[index]);
		  col2Plan->inStride.push_back(col2Plan->iDist);
		  col2Plan->outStride.push_back(col2Plan->oDist);
		  col2Plan->iDist   *= fftPlan->length[index];
		  col2Plan->oDist   *= fftPlan->length[index];
		}

		ampfftBakePlan(fftPlan->planY);

		// copy plan to get back to hermitian
		ampfftCreateDefaultPlan( &fftPlan->planRCcopy, AMPFFT_1D,  &fftPlan->length[0]);

		FFTPlan* copyPlan	= NULL;
		lockRAII* copyLock	= NULL;
		fftRepo.getPlan( fftPlan->planRCcopy, copyPlan, copyLock);

		// This is second column fft, intermediate buffer is packed and interleaved
		// we need to pass clLengths[1] and instride size to kernel, so kernel can tell the difference
		// common part for both passes
		copyPlan->location     = AMPFFT_OUTOFPLACE;
		copyPlan->ipLayout   = AMPFFT_COMPLEX;
		copyPlan->opLayout  = fftPlan->opLayout;
		copyPlan->precision     = fftPlan->precision;
		copyPlan->forwardScale  = 1.0f;
		copyPlan->backwardScale = 1.0f;
		copyPlan->tmpBufSize    = 0;
		copyPlan->batchSize     = fftPlan->batchSize;
		copyPlan->bLdsComplex   = fftPlan->bLdsComplex;
		copyPlan->uLdsFraction  = fftPlan->uLdsFraction;
		copyPlan->ldsPadding    = fftPlan->ldsPadding;
		copyPlan->gen		= Copy;
		copyPlan->envelope	= fftPlan->envelope;
		copyPlan->inStride[0]  = 1;
		copyPlan->iDist        = fftPlan->length[0];
		copyPlan->outStride[0] = fftPlan->outStride[0];
		copyPlan->oDist         = fftPlan->oDist;

		for (size_t index=1; index < fftPlan->length.size(); index++)
		{
		  copyPlan->length.push_back(fftPlan->length[index]);
		  copyPlan->inStride.push_back(copyPlan->inStride[index-1] * fftPlan->length[index-1]);
		  copyPlan->iDist   *= fftPlan->length[index];
		  copyPlan->outStride.push_back(fftPlan->outStride[index]);
		}

		ampfftBakePlan(fftPlan->planRCcopy);
	      }
	      else if(fftPlan->opLayout == AMPFFT_REAL)
	      {
	        if (fftPlan->tmpBufSizeRC==0 )
		{
		  fftPlan->tmpBufSizeRC = length0 * length1 * fftPlan->batchSize * fftPlan->ElementSize();
	          for (size_t index=1; index < fftPlan->length.size(); index++)
		  {
		    fftPlan->tmpBufSizeRC *= fftPlan->length[index];
		  }
		}
                // copy plan to from hermitian to full complex
		ampfftCreateDefaultPlan( &fftPlan->planRCcopy, AMPFFT_1D,  &fftPlan->length[0]);

		FFTPlan* copyPlan	= NULL;
		lockRAII* copyLock	= NULL;
		fftRepo.getPlan( fftPlan->planRCcopy, copyPlan, copyLock);

		// This is second column fft, intermediate buffer is packed and interleaved
		// we need to pass clLengths[1] and instride size to kernel, so kernel can tell the difference
		// common part for both passes
		copyPlan->location     = AMPFFT_OUTOFPLACE;
		copyPlan->ipLayout   = fftPlan->ipLayout;
		copyPlan->opLayout  = AMPFFT_COMPLEX;
		copyPlan->precision     = fftPlan->precision;
		copyPlan->forwardScale  = 1.0f;
		copyPlan->backwardScale = 1.0f;
		copyPlan->tmpBufSize    = 0;
		copyPlan->batchSize     = fftPlan->batchSize;
		copyPlan->bLdsComplex   = fftPlan->bLdsComplex;
		copyPlan->uLdsFraction  = fftPlan->uLdsFraction;
		copyPlan->ldsPadding    = fftPlan->ldsPadding;
		copyPlan->gen		= Copy;
		copyPlan->envelope	= fftPlan->envelope;
                copyPlan->inStride[0]  = fftPlan->inStride[0];
		copyPlan->iDist        = fftPlan->iDist;
                copyPlan->outStride[0]  = 1;
		copyPlan->oDist        = fftPlan->length[0];

		for (size_t index=1; index < fftPlan->length.size(); index++)
		{
		  copyPlan->length.push_back(fftPlan->length[index]);
		  copyPlan->outStride.push_back(copyPlan->outStride[index-1] * fftPlan->length[index-1]);
		  copyPlan->oDist   *= fftPlan->length[index];
		  copyPlan->inStride.push_back(fftPlan->inStride[index]);
		}

		ampfftBakePlan(fftPlan->planRCcopy);

		// column FFT, size clLengths[1], batch clLengths[0], with length[0] twiddle factor multiplication
		// transposed output
		ampfftCreateDefaultPlan(&fftPlan->planX, AMPFFT_1D, &clLengths[1]);

		FFTPlan* colTPlan	= NULL;
		lockRAII* colLock	= NULL;
		fftRepo.getPlan( fftPlan->planX, colTPlan, colLock);

		// current plan is to create intermediate buffer, packed and interleave
		// This is a column FFT, the first elements distance between each FFT is the distance of the first two
		// elements in the original buffer. Like a transpose of the matrix
		// we need to pass clLengths[0] and instride size to kernel, so kernel can tell the difference
		//this part are common for both passes
		colTPlan->location     = AMPFFT_INPLACE;
		colTPlan->precision     = fftPlan->precision;
		colTPlan->forwardScale  = 1.0f;
		colTPlan->backwardScale = 1.0f;
		colTPlan->tmpBufSize    = 0;
		colTPlan->batchSize     = fftPlan->batchSize;
		colTPlan->bLdsComplex   = fftPlan->bLdsComplex;
		colTPlan->uLdsFraction  = fftPlan->uLdsFraction;
		colTPlan->ldsPadding    = fftPlan->ldsPadding;
		colTPlan->gen		= fftPlan->gen;
		colTPlan->envelope	= fftPlan->envelope;
		//Pass large1D flag to confirm we need multiply twiddle factor
		colTPlan->large1D       = fftPlan->length[0];
		colTPlan->length.push_back(clLengths[0]);
		// first Pass
		colTPlan->ipLayout   = AMPFFT_COMPLEX;
		colTPlan->opLayout  = AMPFFT_COMPLEX;
		colTPlan->inStride[0]  = length0;
		colTPlan->inStride.push_back(1);
		colTPlan->iDist        = length0 * length1;
		colTPlan->outStride[0] = length0;
		colTPlan->outStride.push_back(1);
		colTPlan->oDist         = length0 * length1;
		for (size_t index=1; index < fftPlan->length.size(); index++)
		{
		  colTPlan->length.push_back(fftPlan->length[index]);
		  colTPlan->inStride.push_back(colTPlan->iDist);
		  colTPlan->outStride.push_back(colTPlan->oDist);
		  colTPlan->iDist   *= fftPlan->length[index];
		  colTPlan->oDist   *= fftPlan->length[index];
		}

                ampfftBakePlan(fftPlan->planX);

		//another column FFT, size clLengths[0], batch clLengths[1], output without transpose
		ampfftCreateDefaultPlan( &fftPlan->planY, AMPFFT_1D,  &clLengths[0]);

		FFTPlan* col2Plan	= NULL;
		lockRAII* rowLock	= NULL;
		fftRepo.getPlan( fftPlan->planY, col2Plan, rowLock);

		// This is second column fft, intermediate buffer is packed and interleaved
		// we need to pass clLengths[1] and instride size to kernel, so kernel can tell the difference
		// common part for both passes
		col2Plan->location     = AMPFFT_OUTOFPLACE;
		col2Plan->ipLayout   = AMPFFT_COMPLEX;
		col2Plan->opLayout  = fftPlan->opLayout;
		col2Plan->precision     = fftPlan->precision;
		col2Plan->forwardScale  = fftPlan->forwardScale;
		col2Plan->backwardScale = fftPlan->backwardScale;
		col2Plan->tmpBufSize    = 0;
		col2Plan->batchSize     = fftPlan->batchSize;
		col2Plan->bLdsComplex   = fftPlan->bLdsComplex;
		col2Plan->uLdsFraction  = fftPlan->uLdsFraction;
		col2Plan->ldsPadding    = fftPlan->ldsPadding;
		col2Plan->gen		= fftPlan->gen;
		col2Plan->envelope	= fftPlan->envelope;
		col2Plan->RCsimple = true;
		col2Plan->length.push_back(length1);
		col2Plan->inStride[0]  = 1;
		col2Plan->inStride.push_back(length0);
		col2Plan->iDist        = length0 * length1;
		col2Plan->outStride[0] = length1 * fftPlan->outStride[0];
		col2Plan->outStride.push_back(fftPlan->outStride[0]);
		col2Plan->oDist         = fftPlan->oDist;
		for (size_t index=1; index < fftPlan->length.size(); index++)
		{
		  col2Plan->length.push_back(fftPlan->length[index]);
		  col2Plan->inStride.push_back(col2Plan->iDist);
		  col2Plan->iDist   *= fftPlan->length[index];
		  col2Plan->outStride.push_back(fftPlan->outStride[index]);
		}

		ampfftBakePlan(fftPlan->planY);
		}
		else
		{
		  if (fftPlan->cacheSize)
                  {
		    length0 += fftPlan->cacheSize & 0xFF;
		    length1 += (fftPlan->cacheSize >> 8) & 0xFF;
		    if (length0 * length1 > 2 * fftPlan->length[0])
		    {
		      length0 = clLengths[0];
		      length1 = clLengths[1];
		    }
		  }
		  else
		  {
		    if(fftPlan->length[0] == 131072)
                      length1 += 1;     //x0=0, y0=1 good for Cayman card
		    else if (fftPlan->length[0] == 65536)
                      length1 += 8; //x0=0, y0=8 good for Cypress card
		  }

		  if (clLengths[0] > Large1DThreshold)
		  {//make no change for Huge 1D case
		    length0 = clLengths[0];
		    length1 = clLengths[1];
		  }

		  if (fftPlan->tmpBufSize==0 )
		  {
		    fftPlan->tmpBufSize = length0 * length1 *
		    fftPlan->batchSize * fftPlan->ElementSize();
		    for (size_t index=1; index < fftPlan->length.size(); index++)
		    {
		      fftPlan->tmpBufSize *= fftPlan->length[index];
		    }
		  }
		  else
		  {//make no change for cases passed from higher dimension
		    length0 = clLengths[0];
		    length1 = clLengths[1];
		  }

		  // column FFT, size clLengths[1], batch clLengths[0], with length[0] twiddle factor multiplication
		  // transposed output
		  ampfftCreateDefaultPlan( &fftPlan->planX, AMPFFT_1D, &clLengths[1]);

		  FFTPlan* colTPlan	= NULL;
		  lockRAII* colLock	= NULL;
		  fftRepo.getPlan( fftPlan->planX, colTPlan, colLock);

		  // current plan is to create intermediate buffer, packed and interleave
		  // This is a column FFT, the first elements distance between each FFT is the distance of the first two
		  // elements in the original buffer. Like a transpose of the matrix
		  // we need to pass clLengths[0] and instride size to kernel, so kernel can tell the difference
		  //this part are common for both passes
		  colTPlan->location     = AMPFFT_OUTOFPLACE;
		  colTPlan->precision     = fftPlan->precision;
		  colTPlan->forwardScale  = 1.0f;
		  colTPlan->backwardScale = 1.0f;
		  colTPlan->tmpBufSize    = 0;
		  colTPlan->batchSize     = fftPlan->batchSize;
		  colTPlan->bLdsComplex   = fftPlan->bLdsComplex;
		  colTPlan->uLdsFraction  = fftPlan->uLdsFraction;
		  colTPlan->ldsPadding    = fftPlan->ldsPadding;
		  colTPlan->gen		  = fftPlan->gen;
		  colTPlan->envelope	  = fftPlan->envelope;

                  //Pass large1D flag to confirm we need multiply twiddle factor
		  colTPlan->large1D       = fftPlan->length[0];

		  colTPlan->length.push_back(clLengths[0]);

		  if (fftPlan->large1D == 0)
		  {
		    // first Pass
		    colTPlan->ipLayout   = fftPlan->ipLayout;
		    colTPlan->opLayout  = AMPFFT_COMPLEX;
		    colTPlan->inStride[0]   = fftPlan->inStride[0] * clLengths[0];
		    colTPlan->outStride[0]  = 1;
		    colTPlan->iDist         = fftPlan->iDist;
		    colTPlan->oDist         = length0 * length1;
		    colTPlan->inStride.push_back(fftPlan->inStride[0]);
		    colTPlan->outStride.push_back(length1);

                    for (size_t index=1; index < fftPlan->length.size(); index++)
		    {
		      colTPlan->length.push_back(fftPlan->length[index]);
		      colTPlan->inStride.push_back(fftPlan->inStride[index]);
		      // tmp buffer is tightly packed
		      colTPlan->outStride.push_back(colTPlan->oDist);
		      colTPlan->oDist        *= fftPlan->length[index];
		     }
		   }
	           else
		   {
		     // second pass for huge 1D
		     colTPlan->ipLayout   = AMPFFT_COMPLEX;
		     colTPlan->opLayout  = fftPlan->opLayout;
		     colTPlan->inStride[0]   = fftPlan->length[1]*clLengths[0];
		     colTPlan->outStride[0]  = fftPlan->outStride[0];
		     colTPlan->iDist         = fftPlan->length[0];
		     colTPlan->oDist         = fftPlan->oDist;
		     colTPlan->inStride.push_back(fftPlan->length[1]);
		     colTPlan->outStride.push_back(fftPlan->outStride[0]*clLengths[1]);

	             for (size_t index=1; index < fftPlan->length.size(); index++)
		     {
		       colTPlan->length.push_back(fftPlan->length[index]);
		       colTPlan->inStride.push_back(fftPlan->inStride[index]);
		       colTPlan->outStride.push_back(fftPlan->outStride[index]);
		       colTPlan->iDist        *= fftPlan->length[index];
		     }
		   }

		   ampfftBakePlan(fftPlan->planX);

		   //another column FFT, size clLengths[0], batch clLengths[1], output without transpose
		   ampfftCreateDefaultPlan( &fftPlan->planY, AMPFFT_1D,  &clLengths[0]);

		   FFTPlan* col2Plan	= NULL;
		   lockRAII* rowLock	= NULL;
		   fftRepo.getPlan( fftPlan->planY, col2Plan, rowLock);

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
		   col2Plan->bLdsComplex   = fftPlan->bLdsComplex;
		   col2Plan->uLdsFraction  = fftPlan->uLdsFraction;
		   col2Plan->ldsPadding    = fftPlan->ldsPadding;
		   col2Plan->gen	   = fftPlan->gen;
		   col2Plan->envelope	   = fftPlan->envelope;

		   if (clLengths[0] > Large1DThreshold)
		   //prepare for huge 1D
		     col2Plan->large1D   = fftPlan->length[0];

		   col2Plan->length.push_back(clLengths[1]);
		   col2Plan->outStride.push_back(fftPlan->outStride[0]);

		   if (fftPlan->large1D == 0)
		   {
		     //first layer, large 1D from tmp buffer to output buffer
		     col2Plan->location    = AMPFFT_OUTOFPLACE;
		     col2Plan->ipLayout  = AMPFFT_COMPLEX;
		     col2Plan->inStride[0]  = length1;
		     col2Plan->outStride[0] = fftPlan->outStride[0] * clLengths[1];
		     col2Plan->iDist        = length0 * length1;
		     col2Plan->inStride.push_back(1);

		     for (size_t index=1; index < fftPlan->length.size(); index++)
		     {
		       col2Plan->length.push_back(fftPlan->length[index]);
		       col2Plan->inStride.push_back(col2Plan->iDist);
		       col2Plan->outStride.push_back(fftPlan->outStride[index]);
		       col2Plan->iDist   *= fftPlan->length[index];
		     }
		   }
		   else
		   {
		     //second layer, huge 1D from output buffer to output buffer
		     col2Plan->location    = AMPFFT_INPLACE;
		     col2Plan->ipLayout  = fftPlan->opLayout;
		     col2Plan->inStride[0]  = fftPlan->outStride[0] * clLengths[1];
		     col2Plan->outStride[0] = col2Plan->inStride[0];
		     col2Plan->iDist        = fftPlan->oDist;
		     col2Plan->inStride.push_back(fftPlan->outStride[0]);

                     for (size_t index=1; index < fftPlan->length.size(); index++)
		     {
		       col2Plan->length.push_back(fftPlan->length[index]);
		       col2Plan->inStride.push_back(fftPlan->outStride[index]);
		       col2Plan->outStride.push_back(fftPlan->outStride[index]);
		     }
		   }

                   ampfftBakePlan(fftPlan->planY);
		   }

                   fftPlan->baked = true;
		   return AMPFFT_SUCCESS;
		 }
	      }
	      break;
	    case AMPFFT_2D:
	    {
	      size_t length0 = fftPlan->length[0];
	      size_t length1 = fftPlan->length[1];

	      if (fftPlan->cacheSize)
	      {
	        length0 += fftPlan->cacheSize & 0xFF;
	        length1 += (fftPlan->cacheSize >> 8) & 0xFF;
		if (length0 * length1 > 2 * fftPlan->length[0] * fftPlan->length[1])
		{
		  length0 = fftPlan->length[0];
		  length1 = fftPlan->length[1];
		}
	      }
	      else
	      {
	        if(fftPlan->length[0]==256 && fftPlan->length[1]==256)
		{
		  length0 += 8;
		  length1 += 1;
		}
		else if (fftPlan->length[0]==512 && fftPlan->length[1]==512)
		{
		  length0 += 1;
		  length1 += 1;
		}
		else if (fftPlan->length[0]==1024 && fftPlan->length[1]==512)
		{
		  length0 += 2;
		  length1 += 2;
		}
		else if (fftPlan->length[0]==1024 && fftPlan->length[1]==1024)
		{
		  length0 += 1;
		  length1 += 1;
		}
	      }

	      if(fftPlan->length[0] > Large1DThreshold || fftPlan->length[1] > Large1DThreshold)
		 fftPlan->large2D = true;

	      while (1 && (fftPlan->ipLayout != AMPFFT_REAL) && (fftPlan->opLayout != AMPFFT_REAL))
	      {
	        if (fftPlan->transflag) //Transpose for 2D
		{
		  fftPlan->GenerateKernel( plHandle, fftRepo);
		  CompileKernels(plHandle, fftPlan->gen, fftPlan);
                  fftPlan->baked		= true;
		  return	AMPFFT_SUCCESS;
		}

                // TODO : Check for a better way to do this.
                bool isnvidia = false;

                // nvidia gpus are failing when doing transpose for 2D FFTs
                if (isnvidia) break;

		if (fftPlan->length.size() != 2) break;
		if (!(IsPo2(fftPlan->length[0])) || !(IsPo2(fftPlan->length[1])))
		  break;
		if (fftPlan->length[1] < 32) break;

                if (fftPlan->length[0] < 512 && fftPlan->transposeType == AMPFFT_NOTRANSPOSE) break;
		if (fftPlan->length[0] < 32) break;

                if (fftPlan->inStride[0] != 1 || fftPlan->outStride[0] != 1 ||
		fftPlan->inStride[1] != fftPlan->length[0] || fftPlan->outStride[1] != fftPlan->length[0])
		  break;
		fftPlan->transflag = true;
		//create row plan,
		// x=y & x!=y, In->In for inplace, In->out for outofplace
                ampfftCreateDefaultPlan( &fftPlan->planX, AMPFFT_1D, &fftPlan->length[ 0 ] );

		FFTPlan* rowPlan	= NULL;
		lockRAII* rowLock	= NULL;
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
		rowPlan->bLdsComplex     = fftPlan->bLdsComplex;
		rowPlan->uLdsFraction    = fftPlan->uLdsFraction;
		rowPlan->ldsPadding      = fftPlan->ldsPadding;
		rowPlan->gen		 = fftPlan->gen;
		rowPlan->envelope	 = fftPlan->envelope;
		rowPlan->batchSize       = fftPlan->batchSize;
		rowPlan->inStride[0]     = fftPlan->inStride[0];
		rowPlan->length.push_back(fftPlan->length[1]);
		rowPlan->inStride.push_back(fftPlan->inStride[1]);
		rowPlan->iDist           = fftPlan->iDist;

		ampfftBakePlan(fftPlan->planX);

		//Create transpose plan for first transpose
		//x=y: inplace. x!=y inplace: in->tmp, outofplace out->tmp
		size_t clLengths[] = { 1, 1, 0 };
		clLengths[0] = fftPlan->length[0];
		clLengths[1] = fftPlan->length[1];

		bool xyflag = (clLengths[0]==clLengths[1]) ? false : true;
		if (xyflag && fftPlan->tmpBufSize==0 && fftPlan->length.size()<=2)
		{
		  // we need tmp buffer for x!=y case
		  // we assume the tmp buffer is packed interleaved
		  fftPlan->tmpBufSize = length0 * length1 * fftPlan->batchSize * fftPlan->ElementSize();
		}
		ampfftCreateDefaultPlan( &fftPlan->planTX, AMPFFT_2D, clLengths );

		FFTPlan* transPlanX	= NULL;
		lockRAII* transLockX	= NULL;
		fftRepo.getPlan( fftPlan->planTX, transPlanX, transLockX );

		transPlanX->ipLayout     = fftPlan->opLayout;
		transPlanX->precision       = fftPlan->precision;
		transPlanX->tmpBufSize      = 0;
		transPlanX->gen		    = Transpose;
		transPlanX->envelope        = fftPlan->envelope;
		transPlanX->batchSize       = fftPlan->batchSize;
		transPlanX->inStride[0]     = fftPlan->outStride[0];
		transPlanX->inStride[1]     = fftPlan->outStride[1];
		transPlanX->iDist           = fftPlan->oDist;
		transPlanX->transflag       = true;

		if (xyflag)
		{
		  transPlanX->opLayout    = AMPFFT_COMPLEX;
		  transPlanX->location       = AMPFFT_OUTOFPLACE;
		  transPlanX->outStride[0]    = 1;
		  transPlanX->outStride[1]    = clLengths[0];
		  transPlanX->oDist           = clLengths[0] * clLengths[1];
		}
		else
		{
		  transPlanX->opLayout    = fftPlan->opLayout;
		  transPlanX->location       = AMPFFT_INPLACE;
		  transPlanX->outStride[0]    = fftPlan->outStride[0];
		  transPlanX->outStride[1]    = fftPlan->outStride[1];
		  transPlanX->oDist           = fftPlan->oDist;
		}

		ampfftBakePlan(fftPlan->planTX);

		//create second row plan
		//x!=y: tmp->tmp, x=y case: In->In or Out->Out
		//if Transposed result is a choice x!=y: tmp->In or out
		ampfftCreateDefaultPlan( &fftPlan->planY, AMPFFT_1D, &fftPlan->length[ 1 ] );

		FFTPlan* colPlan	= NULL;
		lockRAII* colLock	= NULL;
		fftRepo.getPlan( fftPlan->planY, colPlan, colLock );

		if (xyflag)
		{
		  colPlan->ipLayout     = AMPFFT_COMPLEX;
		  colPlan->inStride[0]     = 1;
		  colPlan->inStride.push_back(clLengths[1]);
		  colPlan->iDist           = clLengths[0] * clLengths[1];

		  if (fftPlan->transposeType == AMPFFT_NOTRANSPOSE)
		  {
		    colPlan->opLayout    = AMPFFT_COMPLEX;
		    colPlan->outStride[0]    = 1;
		    colPlan->outStride.push_back(clLengths[1]);
		    colPlan->oDist           = clLengths[0] * clLengths[1];
		    colPlan->location       = AMPFFT_INPLACE;
		  }
		  else
		  {
		    colPlan->opLayout    = fftPlan->opLayout;
		    colPlan->outStride[0]    = fftPlan->outStride[0];
		    colPlan->outStride.push_back(clLengths[1] * fftPlan->outStride[0]);
		    colPlan->oDist           = fftPlan->oDist;
		    colPlan->location       = AMPFFT_OUTOFPLACE;
		  }
		}
		else
		{
		  colPlan->ipLayout     = fftPlan->opLayout;
		  colPlan->opLayout    = fftPlan->opLayout;
		  colPlan->outStride[0]    = fftPlan->outStride[0];
		  colPlan->outStride.push_back(fftPlan->outStride[1]);
		  colPlan->oDist           = fftPlan->oDist;
		  colPlan->inStride[0]     = fftPlan->outStride[0];
		  colPlan->inStride.push_back(fftPlan->outStride[1]);
		  colPlan->iDist           = fftPlan->oDist;
		  colPlan->location       = AMPFFT_INPLACE;
		}
		colPlan->precision       = fftPlan->precision;
		colPlan->forwardScale    = fftPlan->forwardScale;
		colPlan->backwardScale   = fftPlan->backwardScale;
		colPlan->tmpBufSize      = 0;
		colPlan->bLdsComplex     = fftPlan->bLdsComplex;
		colPlan->uLdsFraction    = fftPlan->uLdsFraction;
		colPlan->ldsPadding      = fftPlan->ldsPadding;
		colPlan->gen		 = fftPlan->gen;
		colPlan->envelope	 = fftPlan->envelope;
		colPlan->batchSize       = fftPlan->batchSize;
		colPlan->length.push_back(fftPlan->length[0]);

	        ampfftBakePlan(fftPlan->planY);

		if (fftPlan->transposeType == AMPFFT_TRANSPOSED)
		{
		  fftPlan->baked = true;
		  return AMPFFT_SUCCESS;
		}

		//Create transpose plan for second transpose
		//x!=y case tmp->In or Out, x=y case In->In or Out->out
		clLengths[0] = fftPlan->length[1];
		clLengths[1] = fftPlan->length[0];
		ampfftCreateDefaultPlan( &fftPlan->planTY, AMPFFT_2D, clLengths );

		FFTPlan* transPlanY	= NULL;
		lockRAII* transLockY	= NULL;
		fftRepo.getPlan( fftPlan->planTY, transPlanY, transLockY);
		if (xyflag)
		{
		  transPlanY->ipLayout     = AMPFFT_COMPLEX;
		  transPlanY->location       = AMPFFT_OUTOFPLACE;
		  transPlanY->inStride[0]     = 1;
		  transPlanY->inStride[1]     = clLengths[0];
		  transPlanY->iDist           = clLengths[0] * clLengths[1];
		}
		else
		{
		  transPlanY->ipLayout     = fftPlan->opLayout;
		  transPlanY->location       = AMPFFT_INPLACE;
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
		transPlanY->gen		    = Transpose;
		transPlanY->envelope	    = fftPlan->envelope;
		transPlanY->batchSize       = fftPlan->batchSize;
		transPlanY->transflag       = true;
		ampfftBakePlan(fftPlan->planTY);

		fftPlan->baked = true;
		return	AMPFFT_SUCCESS;
		}

		//check transposed
		if (fftPlan->transposeType != AMPFFT_NOTRANSPOSE)
		  return AMPFFT_ERROR;

		if(fftPlan->ipLayout == AMPFFT_REAL)
		{
                  cout<<" inside real "<<endl;
		  length0 = fftPlan->length[0];
		  length1 = fftPlan->length[1];

		  size_t Nt = (1 + length0/2);
		  if (fftPlan->tmpBufSize==0)
		  {
		    fftPlan->tmpBufSize = Nt * length1 * fftPlan->batchSize * fftPlan->ElementSize();
		    if(fftPlan->length.size() > 2) fftPlan->tmpBufSize *= fftPlan->length[2];
		  }

		  // create row plan
		  // real to hermitian
		  ampfftCreateDefaultPlan( &fftPlan->planX, AMPFFT_1D, &fftPlan->length[ 0 ] );

		  FFTPlan* rowPlan	= NULL;
		  lockRAII* rowLock	= NULL;
		  fftRepo.getPlan( fftPlan->planX, rowPlan, rowLock );

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
		  rowPlan->tmpBufSize    = fftPlan->tmpBufSize;
		  rowPlan->bLdsComplex   = fftPlan->bLdsComplex;
		  rowPlan->uLdsFraction  = fftPlan->uLdsFraction;
		  rowPlan->ldsPadding    = fftPlan->ldsPadding;
		  rowPlan->gen		 = fftPlan->gen;
		  rowPlan->envelope	 = fftPlan->envelope;

		  rowPlan->batchSize    = fftPlan->batchSize;

		  rowPlan->outStride[0]  = fftPlan->outStride[0];
		  rowPlan->outStride.push_back(fftPlan->outStride[1]);
		  rowPlan->oDist         = fftPlan->oDist;

		  //this 2d is decomposed from 3d
		  if (fftPlan->length.size()>2)
		  {
		    rowPlan->length.push_back(fftPlan->length[2]);
		    rowPlan->inStride.push_back(fftPlan->inStride[2]);
		    rowPlan->outStride.push_back(fftPlan->outStride[2]);
		  }

		  ampfftBakePlan(fftPlan->planX);

	          // create col plan
		  // complex to complex

		  ampfftCreateDefaultPlan( &fftPlan->planY, AMPFFT_1D, &fftPlan->length[ 1 ] );

		  FFTPlan* colPlan	= NULL;
		  lockRAII* colLock	= NULL;
		  fftRepo.getPlan( fftPlan->planY, colPlan, colLock );

		  colPlan->location     = AMPFFT_INPLACE;
		  colPlan->length.push_back(Nt);

		  colPlan->outStride[0]  = fftPlan->outStride[1];
		  colPlan->outStride.push_back(fftPlan->outStride[0]);
		  colPlan->oDist         = fftPlan->oDist;

		  colPlan->precision     = fftPlan->precision;
		  colPlan->forwardScale  = fftPlan->forwardScale;
		  colPlan->backwardScale = fftPlan->backwardScale;
		  colPlan->tmpBufSize    = fftPlan->tmpBufSize;
		  colPlan->bLdsComplex   = fftPlan->bLdsComplex;
		  colPlan->uLdsFraction  = fftPlan->uLdsFraction;
		  colPlan->ldsPadding    = fftPlan->ldsPadding;
		  colPlan->gen		 = fftPlan->gen;
		  colPlan->envelope	 = fftPlan->envelope;

		  colPlan->batchSize = fftPlan->batchSize;
		  colPlan->inStride[0]  = rowPlan->outStride[1];
		  colPlan->inStride.push_back(rowPlan->outStride[0]);
		  colPlan->iDist         = rowPlan->oDist;

		  //this 2d is decomposed from 3d
		  if (fftPlan->length.size()>2)
		  {
		    colPlan->length.push_back(fftPlan->length[2]);
		    colPlan->outStride.push_back(fftPlan->outStride[2]);
		    colPlan->inStride.push_back(rowPlan->outStride[2]);
		  }

		  ampfftBakePlan(fftPlan->planY);

		}
		else if(fftPlan->opLayout == AMPFFT_REAL)
		{
		  length0 = fftPlan->length[0];
		  length1 = fftPlan->length[1];

		  size_t Nt = (1 + length0/2);
		  if (fftPlan->tmpBufSize==0)
		  {
		    fftPlan->tmpBufSize = Nt * length1 * fftPlan->batchSize * fftPlan->ElementSize();
		    if(fftPlan->length.size() > 2) fftPlan->tmpBufSize *= fftPlan->length[2];
		  }

		  // create col plan
		  // complex to complex

		  ampfftCreateDefaultPlan( &fftPlan->planY, AMPFFT_1D, &fftPlan->length[ 1 ] );
                  FFTPlan* colPlan	= NULL;
		  lockRAII* colLock	= NULL;
		  fftRepo.getPlan( fftPlan->planY, colPlan, colLock );

		  colPlan->length.push_back(Nt);
		  colPlan->inStride[0]  = fftPlan->inStride[1];
		  colPlan->inStride.push_back(fftPlan->inStride[0]);
		  colPlan->iDist         = fftPlan->iDist;

		  //this 2d is decomposed from 3d
		  if (fftPlan->length.size()>2)
		  {
		    colPlan->location = AMPFFT_INPLACE;
		    colPlan->length.push_back(fftPlan->length[2]);
		    colPlan->inStride.push_back(fftPlan->inStride[2]);
		    colPlan->outStride[0]  = colPlan->inStride[0];
		    colPlan->outStride.push_back(colPlan->inStride[1]);
		    colPlan->outStride.push_back(colPlan->inStride[2]);
		    colPlan->oDist         = fftPlan->iDist;
		  }
		  else
		  {
		    colPlan->location = AMPFFT_OUTOFPLACE;
		    colPlan->outStride[0]  = Nt;
		    colPlan->outStride.push_back(1);
		    colPlan->oDist         = Nt*length1;
		  }

		  colPlan->precision     = fftPlan->precision;
		  colPlan->forwardScale  = 1.0f;
		  colPlan->backwardScale = 1.0f;
		  colPlan->tmpBufSize    = fftPlan->tmpBufSize;
		  colPlan->bLdsComplex   = fftPlan->bLdsComplex;
		  colPlan->uLdsFraction  = fftPlan->uLdsFraction;
		  colPlan->ldsPadding    = fftPlan->ldsPadding;
		  colPlan->gen		 = fftPlan->gen;
		  colPlan->envelope	 = fftPlan->envelope;
		  colPlan->batchSize = fftPlan->batchSize;

		  if ((fftPlan->tmpBufSizeC2R==0) && (length1 > Large1DThreshold) && (fftPlan->length.size()<=2))
		  {
		    fftPlan->tmpBufSizeC2R = Nt * length1 * fftPlan->batchSize * fftPlan->ElementSize();
		    if(fftPlan->length.size() > 2) fftPlan->tmpBufSizeC2R *= fftPlan->length[2];
		  }

		  ampfftBakePlan(fftPlan->planY);

		  // create row plan
		  // hermitian to real
		  ampfftCreateDefaultPlan( &fftPlan->planX, AMPFFT_1D, &fftPlan->length[ 0 ]);

		  FFTPlan* rowPlan	= NULL;
		  lockRAII* rowLock	= NULL;
		  fftRepo.getPlan( fftPlan->planX, rowPlan, rowLock );

                  rowPlan->opLayout  = fftPlan->opLayout;
		  rowPlan->ipLayout   = AMPFFT_COMPLEX;
		  rowPlan->location     = AMPFFT_OUTOFPLACE;
		  rowPlan->length.push_back(length1);

		  rowPlan->inStride[0]   = 1;
		  rowPlan->inStride.push_back(Nt);
		  rowPlan->iDist         = colPlan->oDist;

		  rowPlan->precision     = fftPlan->precision;
		  rowPlan->forwardScale  = fftPlan->forwardScale;
		  rowPlan->backwardScale = fftPlan->backwardScale;
		  rowPlan->tmpBufSize    = fftPlan->tmpBufSize;
		  rowPlan->bLdsComplex   = fftPlan->bLdsComplex;
		  rowPlan->uLdsFraction  = fftPlan->uLdsFraction;
		  rowPlan->ldsPadding    = fftPlan->ldsPadding;
		  rowPlan->gen		 = fftPlan->gen;
		  rowPlan->envelope	= fftPlan->envelope;
                  rowPlan->batchSize    = fftPlan->batchSize;
                  rowPlan->outStride[0]  = fftPlan->outStride[0];
		  rowPlan->outStride.push_back(fftPlan->outStride[1]);
		  rowPlan->oDist         = fftPlan->oDist;

		  //this 2d is decomposed from 3d
		  if (fftPlan->length.size()>2)
		  {
		    rowPlan->length.push_back(fftPlan->length[2]);
		    rowPlan->inStride.push_back(Nt*length1);
		    rowPlan->outStride.push_back(fftPlan->outStride[2]);
		  }
	          ampfftBakePlan(fftPlan->planX);
		}
		else
		{
		  if (fftPlan->tmpBufSize==0 && fftPlan->length.size()<=2)
		  {
		    fftPlan->tmpBufSize = length0 * length1 *
		    fftPlan->batchSize * fftPlan->ElementSize();
		  }

		  //create row plan
		  ampfftCreateDefaultPlan( &fftPlan->planX, AMPFFT_1D, &fftPlan->length[ 0 ] );

		  FFTPlan* rowPlan	= NULL;
		  lockRAII* rowLock	= NULL;
		  fftRepo.getPlan( fftPlan->planX, rowPlan, rowLock );

		  rowPlan->ipLayout   = fftPlan->ipLayout;
		  if (fftPlan->large2D || fftPlan->length.size()>2)
		  {
		    rowPlan->opLayout  = fftPlan->opLayout;
		    rowPlan->location     = fftPlan->location;
		    rowPlan->outStride[0]  = fftPlan->outStride[0];
		    rowPlan->outStride.push_back(fftPlan->outStride[1]);
		    rowPlan->oDist         = fftPlan->oDist;
		  }
		  else
		  {
		    rowPlan->opLayout  = AMPFFT_COMPLEX;
		    rowPlan->location     = AMPFFT_OUTOFPLACE;
		    rowPlan->outStride[0]  = length1;
		    rowPlan->outStride.push_back(1);
		    rowPlan->oDist         = length0 * length1;
		  }
		  rowPlan->precision     = fftPlan->precision;
		  rowPlan->forwardScale  = 1.0f;
		  rowPlan->backwardScale = 1.0f;
		  rowPlan->tmpBufSize    = fftPlan->tmpBufSize;
		  rowPlan->bLdsComplex   = fftPlan->bLdsComplex;
		  rowPlan->uLdsFraction  = fftPlan->uLdsFraction;
		  rowPlan->ldsPadding    = fftPlan->ldsPadding;
		  rowPlan->gen		 = fftPlan->gen;
		  rowPlan->envelope	 = fftPlan->envelope;

		  // This is the row fft, the first elements distance between the first two FFTs is the distance of the first elements
		  // of the first two rows in the original buffer.
		  rowPlan->batchSize    = fftPlan->batchSize;
		  rowPlan->inStride[0]  = fftPlan->inStride[0];

		  //pass length and other info to kernel, so the kernel knows this is decomposed from higher dimension
		  rowPlan->length.push_back(fftPlan->length[1]);
		  rowPlan->inStride.push_back(fftPlan->inStride[1]);

		  //this 2d is decomposed from 3d
		  if (fftPlan->length.size()>2)
		  {
		    rowPlan->length.push_back(fftPlan->length[2]);
		    rowPlan->inStride.push_back(fftPlan->inStride[2]);
		    rowPlan->outStride.push_back(fftPlan->outStride[2]);
		  }

		  rowPlan->iDist    = fftPlan->iDist;
                  ampfftBakePlan(fftPlan->planX);

                  //create col plan
		  ampfftCreateDefaultPlan( &fftPlan->planY, AMPFFT_1D, &fftPlan->length[ 1 ] );

                  FFTPlan* colPlan	= NULL;
		  lockRAII* colLock	= NULL;
		  fftRepo.getPlan( fftPlan->planY, colPlan, colLock );

		  if (fftPlan->large2D || fftPlan->length.size()>2)
		  {
		    colPlan->ipLayout   = fftPlan->opLayout;
		    colPlan->location     = AMPFFT_INPLACE;
		    colPlan->inStride[0]   = fftPlan->outStride[1];
		    colPlan->inStride.push_back(fftPlan->outStride[0]);
		    colPlan->iDist         = fftPlan->oDist;
		  }
		  else
		  {
		    colPlan->ipLayout   = AMPFFT_COMPLEX;
		    colPlan->location = AMPFFT_OUTOFPLACE;
		    colPlan->inStride[0]   = 1;
		    colPlan->inStride.push_back(length1);
		    colPlan->iDist         = length0 * length1;
		  }

		  colPlan->opLayout  = fftPlan->opLayout;
		  colPlan->precision     = fftPlan->precision;
		  colPlan->forwardScale  = fftPlan->forwardScale;
		  colPlan->backwardScale = fftPlan->backwardScale;
		  colPlan->tmpBufSize    = fftPlan->tmpBufSize;
		  colPlan->bLdsComplex   = fftPlan->bLdsComplex;
		  colPlan->uLdsFraction  = fftPlan->uLdsFraction;
		  colPlan->ldsPadding    = fftPlan->ldsPadding;
		  colPlan->gen		 = fftPlan->gen;
		  colPlan->envelope	 = fftPlan->envelope;

		  // This is a column FFT, the first elements distance between each FFT is the distance of the first two
		  // elements in the original buffer. Like a transpose of the matrix
		  colPlan->batchSize = fftPlan->batchSize;
		  colPlan->outStride[0] = fftPlan->outStride[1];

		  //pass length and other info to kernel, so the kernel knows this is decomposed from higher dimension
		  colPlan->length.push_back(fftPlan->length[0]);
		  colPlan->outStride.push_back(fftPlan->outStride[0]);
		  colPlan->oDist    = fftPlan->oDist;

		  //this 2d is decomposed from 3d
		  if (fftPlan->length.size()>2)
		  {
	            //assert(fftPlan->large2D);
		    colPlan->length.push_back(fftPlan->length[2]);
		    colPlan->inStride.push_back(fftPlan->outStride[2]);
		    colPlan->outStride.push_back(fftPlan->outStride[2]);
		  }

		  ampfftBakePlan(fftPlan->planY);
		}

		fftPlan->baked = true;
		return	AMPFFT_SUCCESS;
	     }
	     case AMPFFT_3D:
	     {
	       if(fftPlan->ipLayout == AMPFFT_REAL)
	       {
		 size_t clLengths[] = { 1, 1, 0 };
		 clLengths[0] = fftPlan->length[ 0 ];
		 clLengths[1] = fftPlan->length[ 1 ];

		 //create 2D xy plan
		 ampfftCreateDefaultPlan( &fftPlan->planX, AMPFFT_2D, clLengths );

		 FFTPlan* xyPlan	= NULL;
		 lockRAII* rowLock	= NULL;
		 fftRepo.getPlan( fftPlan->planX, xyPlan, rowLock );

		 xyPlan->ipLayout   = fftPlan->ipLayout;
		 xyPlan->opLayout  = fftPlan->opLayout;
		 xyPlan->location     = fftPlan->location;
		 xyPlan->precision     = fftPlan->precision;
		 xyPlan->forwardScale  = 1.0f;
		 xyPlan->backwardScale = 1.0f;
		 xyPlan->tmpBufSize    = fftPlan->tmpBufSize;
		 xyPlan->bLdsComplex   = fftPlan->bLdsComplex;
		 xyPlan->uLdsFraction  = fftPlan->uLdsFraction;
		 xyPlan->ldsPadding    = fftPlan->ldsPadding;
		 xyPlan->gen	       = fftPlan->gen;
		 xyPlan->envelope      = fftPlan->envelope;

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
		 ampfftBakePlan(fftPlan->planX);

		 clLengths[0] = fftPlan->length[ 2 ];
		 clLengths[1] = clLengths[2] = 0;
		 //create 1D col plan
		 ampfftCreateDefaultPlan( &fftPlan->planZ, AMPFFT_1D, clLengths );

		 FFTPlan* colPlan	= NULL;
		 lockRAII* colLock	= NULL;
		 fftRepo.getPlan( fftPlan->planZ, colPlan, colLock );

		 colPlan->location     = AMPFFT_INPLACE;
		 colPlan->precision     = fftPlan->precision;
		 colPlan->forwardScale  = fftPlan->forwardScale;
		 colPlan->backwardScale = fftPlan->backwardScale;
		 colPlan->tmpBufSize    = fftPlan->tmpBufSize;
		 colPlan->bLdsComplex   = fftPlan->bLdsComplex;
		 colPlan->uLdsFraction  = fftPlan->uLdsFraction;
		 colPlan->ldsPadding    = fftPlan->ldsPadding;
		 colPlan->gen		= fftPlan->gen;
		 colPlan->envelope	= fftPlan->envelope;
		 // This is a column FFT, the first elements distance between each FFT is the distance of the first two
		 // elements in the original buffer. Like a transpose of the matrix
		 colPlan->batchSize = fftPlan->batchSize;
		 colPlan->inStride[0] = fftPlan->outStride[2];
		 colPlan->outStride[0] = fftPlan->outStride[2];

		 //pass length and other info to kernel, so the kernel knows this is decomposed from higher dimension
		 colPlan->length.push_back(1 + fftPlan->length[0]/2);
		 colPlan->length.push_back(fftPlan->length[1]);
		 colPlan->inStride.push_back(fftPlan->outStride[0]);
		 colPlan->inStride.push_back(fftPlan->outStride[1]);
		 colPlan->outStride.push_back(fftPlan->outStride[0]);
		 colPlan->outStride.push_back(fftPlan->outStride[1]);
		 colPlan->iDist    = fftPlan->oDist;
		 colPlan->oDist    = fftPlan->oDist;

		 ampfftBakePlan(fftPlan->planZ);
		 }
		 else if(fftPlan->opLayout == AMPFFT_REAL)
		 {
		   if (fftPlan->tmpBufSize == 0)
		   {
		     fftPlan->tmpBufSize = fftPlan->length[2] * fftPlan->length[1] * (1 + fftPlan->length[0]/2);
		     fftPlan->tmpBufSize *= fftPlan->batchSize * fftPlan->ElementSize();
		   }

		   size_t clLengths[] = { 1, 1, 0 };
                   clLengths[0] = fftPlan->length[ 2 ];
		   clLengths[1] = clLengths[2] = 0;

		   //create 1D col plan
		   ampfftCreateDefaultPlan( &fftPlan->planZ, AMPFFT_1D, clLengths );

		   FFTPlan* colPlan	= NULL;
		   lockRAII* colLock	= NULL;
		   fftRepo.getPlan( fftPlan->planZ, colPlan, colLock );

		   colPlan->location     = AMPFFT_OUTOFPLACE;
		   colPlan->precision     = fftPlan->precision;
		   colPlan->forwardScale  = 1.0f;
		   colPlan->backwardScale = 1.0f;
		   colPlan->tmpBufSize    = fftPlan->tmpBufSize;
		   colPlan->bLdsComplex   = fftPlan->bLdsComplex;
		   colPlan->uLdsFraction  = fftPlan->uLdsFraction;
		   colPlan->ldsPadding    = fftPlan->ldsPadding;
		   colPlan->gen		  = fftPlan->gen;
		   colPlan->envelope	  = fftPlan->envelope;

		   // This is a column FFT, the first elements distance between each FFT is the distance of the first two
		   // elements in the original buffer. Like a transpose of the matrix
		   colPlan->batchSize = fftPlan->batchSize;
		   colPlan->inStride[0] = fftPlan->inStride[2];
		   colPlan->outStride[0] = fftPlan->length[1] * (1 + fftPlan->length[0]/2);

		   //pass length and other info to kernel, so the kernel knows this is decomposed from higher dimension
		   colPlan->length.push_back(1 + fftPlan->length[0]/2);
		   colPlan->length.push_back(fftPlan->length[1]);
		   colPlan->inStride.push_back(fftPlan->inStride[0]);
		   colPlan->inStride.push_back(fftPlan->inStride[1]);
		   colPlan->outStride.push_back(1);
		   colPlan->outStride.push_back(1 + fftPlan->length[0]/2);
		   colPlan->iDist    = fftPlan->iDist;
		   colPlan->oDist    = fftPlan->length[2] * fftPlan->length[1] * (1 + fftPlan->length[0]/2);

		   if ((fftPlan->tmpBufSizeC2R==0) && ((fftPlan->length[2] > Large1DThreshold) || (fftPlan->length[1] > Large1DThreshold)))
		   {
		     fftPlan->tmpBufSizeC2R = (1 + fftPlan->length[0]/2) * (fftPlan->length[1]) * (fftPlan->length[2]) *
		     fftPlan->batchSize * fftPlan->ElementSize();
		   }

		   ampfftBakePlan(fftPlan->planZ);

		   clLengths[0] = fftPlan->length[ 0 ];
		   clLengths[1] = fftPlan->length[ 1 ];
		   //create 2D xy plan
		   ampfftCreateDefaultPlan( &fftPlan->planX, AMPFFT_2D, clLengths );

		   FFTPlan* xyPlan	= NULL;
		   lockRAII* rowLock	= NULL;
		   fftRepo.getPlan( fftPlan->planX, xyPlan, rowLock );

		   xyPlan->ipLayout   = AMPFFT_COMPLEX;
		   xyPlan->opLayout  = fftPlan->opLayout;
		   xyPlan->location     = AMPFFT_OUTOFPLACE;
		   xyPlan->precision     = fftPlan->precision;
		   xyPlan->forwardScale  = fftPlan->forwardScale;
		   xyPlan->backwardScale = fftPlan->backwardScale;
		   xyPlan->tmpBufSize    = fftPlan->tmpBufSize;
		   xyPlan->bLdsComplex   = fftPlan->bLdsComplex;
		   xyPlan->uLdsFraction  = fftPlan->uLdsFraction;
		   xyPlan->ldsPadding    = fftPlan->ldsPadding;
		   xyPlan->gen		 = fftPlan->gen;
		   xyPlan->envelope	 = fftPlan->envelope;

		   // This is the xy fft, the first elements distance between the first two FFTs is the distance of the first elements
		   // of the first two rows in the original buffer.
		   xyPlan->batchSize    = fftPlan->batchSize;
		   xyPlan->inStride[0]  = 1;
		   xyPlan->inStride[1]  = (1 + fftPlan->length[0]/2);
		   xyPlan->outStride[0] = fftPlan->outStride[0];
		   xyPlan->outStride[1] = fftPlan->outStride[1];

		   //pass length and other info to kernel, so the kernel knows this is decomposed from higher dimension
		   xyPlan->length.push_back(fftPlan->length[2]);
		   xyPlan->inStride.push_back(fftPlan->length[1] * (1 + fftPlan->length[0]/2));
		   xyPlan->outStride.push_back(fftPlan->outStride[2]);
		   xyPlan->iDist    = colPlan->oDist;
		   xyPlan->oDist    = fftPlan->oDist;

		   ampfftBakePlan(fftPlan->planX);
		 }
		 else
		 {
		   if(fftPlan->tmpBufSize==0 && (fftPlan->length[0] > Large1DThreshold ||
		       fftPlan->length[1] > Large1DThreshold || fftPlan->length[2] > Large1DThreshold))
	           {
		     fftPlan->tmpBufSize = fftPlan->length[0] * fftPlan->length[1] * fftPlan->length[2] *
		     fftPlan->batchSize * fftPlan->ElementSize();
		   }

		   size_t clLengths[] = { 1, 1, 0 };
		   clLengths[0] = fftPlan->length[ 0 ];
		   clLengths[1] = fftPlan->length[ 1 ];

		   //create 2D xy plan
		   ampfftCreateDefaultPlan( &fftPlan->planX, AMPFFT_2D, clLengths );

		   FFTPlan* xyPlan	= NULL;
		   lockRAII* rowLock	= NULL;
		   fftRepo.getPlan( fftPlan->planX, xyPlan, rowLock );
		   xyPlan->ipLayout   = fftPlan->ipLayout;
		   xyPlan->opLayout  = fftPlan->opLayout;
		   xyPlan->location     = fftPlan->location;
		   xyPlan->precision     = fftPlan->precision;
		   xyPlan->forwardScale  = 1.0f;
		   xyPlan->backwardScale = 1.0f;
		   xyPlan->tmpBufSize    = fftPlan->tmpBufSize;
		   xyPlan->bLdsComplex   = fftPlan->bLdsComplex;
		   xyPlan->uLdsFraction  = fftPlan->uLdsFraction;
		   xyPlan->ldsPadding    = fftPlan->ldsPadding;
		   xyPlan->gen		 = fftPlan->gen;
		   xyPlan->envelope	 = fftPlan->envelope;

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

		   ampfftBakePlan(fftPlan->planX);
		   clLengths[0] = fftPlan->length[ 2 ];
		   clLengths[1] = clLengths[2] = 0;
		   //create 1D col plan
		   ampfftCreateDefaultPlan( &fftPlan->planZ, AMPFFT_1D, clLengths );

		   FFTPlan* colPlan	= NULL;
		   lockRAII* colLock	= NULL;
		   fftRepo.getPlan( fftPlan->planZ, colPlan, colLock );

		   colPlan->ipLayout   = fftPlan->ipLayout;
		   colPlan->opLayout  = fftPlan->opLayout;
		   colPlan->location     = AMPFFT_INPLACE;
		   colPlan->precision     = fftPlan->precision;
		   colPlan->forwardScale  = fftPlan->forwardScale;
		   colPlan->backwardScale = fftPlan->backwardScale;
		   colPlan->tmpBufSize    = fftPlan->tmpBufSize;
		   colPlan->bLdsComplex   = fftPlan->bLdsComplex;
		   colPlan->uLdsFraction  = fftPlan->uLdsFraction;
		   colPlan->ldsPadding    = fftPlan->ldsPadding;
		   colPlan->gen		  = fftPlan->gen;
		   colPlan->envelope	  = fftPlan->envelope;

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

		   ampfftBakePlan(fftPlan->planZ);
		 }

		 fftPlan->baked = true;
		 return	AMPFFT_SUCCESS;
	      }
	   }
	//	For the radices that we have factored, we need to load/compile and build the appropriate OpenCL kernels
	fftPlan->GenerateKernel( plHandle, fftRepo);
	//	For the radices that we have factored, we need to load/compile and build the appropriate OpenCL kernels
	CompileKernels( plHandle, fftPlan->gen, fftPlan );

	//	Allocate resources
	fftPlan->AllocateWriteBuffers ();

	//	Record that we baked the plan
	fftPlan->baked		= true;

	return	AMPFFT_SUCCESS;
}

ampfftStatus FFTPlan::executePlan(FFTPlan* fftPlan)
{
  if(!fftPlan)
    return AMPFFT_INVALID;

  return AMPFFT_SUCCESS;
}

ampfftStatus FFTPlan::ampfftGetPlanPrecision( const  ampfftPlanHandle plHandle,  ampfftPrecision* precision )
{
  FFTRepo& fftRepo = FFTRepo::getInstance( );
  FFTPlan* fftPlan = NULL;
  lockRAII* planLock = NULL;

  fftRepo.getPlan(plHandle, fftPlan, planLock );
  scopedLock sLock(*planLock, _T( " ampfftGetPlanPrecision" ) );

  *precision = fftPlan->precision;

  return AMPFFT_SUCCESS;
}

ampfftStatus FFTPlan::ampfftSetPlanPrecision(  ampfftPlanHandle plHandle,  ampfftPrecision precision )
{
  FFTRepo& fftRepo = FFTRepo::getInstance( );
  FFTPlan* fftPlan = NULL;
  lockRAII* planLock = NULL;

  fftRepo.getPlan(plHandle, fftPlan, planLock );
  scopedLock sLock(*planLock, _T( " ampfftSetPlanPrecision" ) );

  //	If we modify the state of the plan, we assume that we can't trust any pre-calculated contents anymore
  fftPlan->baked = false;
  fftPlan->precision = precision;

  return AMPFFT_SUCCESS;
}

ampfftStatus FFTPlan::ampfftGetPlanScale( const  ampfftPlanHandle plHandle,  ampfftDirection dir, float* scale )
{
  FFTRepo& fftRepo = FFTRepo::getInstance( );
  FFTPlan* fftPlan = NULL;
  lockRAII* planLock = NULL;

  fftRepo.getPlan(plHandle, fftPlan, planLock );
  scopedLock sLock(*planLock, _T( " ampfftGetPlanScale" ) );

  if( dir == AMPFFT_FORWARD)
    *scale = (float)(fftPlan->forwardScale);
  else
    *scale = (float)(fftPlan->backwardScale);

  return AMPFFT_SUCCESS;
}

ampfftStatus FFTPlan::ampfftSetPlanScale(  ampfftPlanHandle plHandle,  ampfftDirection dir, float scale )
{
  FFTRepo& fftRepo = FFTRepo::getInstance( );
  FFTPlan* fftPlan = NULL;
  lockRAII* planLock = NULL;

  fftRepo.getPlan(plHandle, fftPlan, planLock );
  scopedLock sLock(*planLock, _T( " ampfftSetPlanScale" ) );

  //	If we modify the state of the plan, we assume that we can't trust any pre-calculated contents anymore
  fftPlan->baked = false;
  if( dir == AMPFFT_FORWARD)
    fftPlan->forwardScale = scale;
  else
    fftPlan->backwardScale = scale;

  return AMPFFT_SUCCESS;
}

ampfftStatus FFTPlan::ampfftGetPlanBatchSize( const  ampfftPlanHandle plHandle, size_t* batchsize )
{
  FFTRepo& fftRepo = FFTRepo::getInstance( );
  FFTPlan* fftPlan = NULL;
  lockRAII* planLock = NULL;

  fftRepo.getPlan(plHandle, fftPlan, planLock );
  scopedLock sLock(*planLock, _T( " ampfftGetPlanBatchSize" ) );

  *batchsize = fftPlan->batchSize;
  return AMPFFT_SUCCESS;
}

ampfftStatus FFTPlan::ampfftSetPlanBatchSize( ampfftPlanHandle plHandle, size_t batchsize )
{
 FFTRepo& fftRepo = FFTRepo::getInstance( );
 FFTPlan* fftPlan = NULL;
 lockRAII* planLock = NULL;

 fftRepo.getPlan(plHandle, fftPlan, planLock );
 scopedLock sLock(*planLock, _T( " ampfftSetPlanBatchSize" ) );

  //	If we modify the state of the plan, we assume that we can't trust any pre-calculated contents anymore
  fftPlan->baked = false;
  fftPlan->batchSize = batchsize;
  return AMPFFT_SUCCESS;
}

ampfftStatus FFTPlan::ampfftGetPlanDim( const ampfftPlanHandle plHandle,  ampfftDim* dim, int* size )
{
  FFTRepo& fftRepo = FFTRepo::getInstance( );
  FFTPlan* fftPlan = NULL;
  lockRAII* planLock = NULL;

  fftRepo.getPlan( plHandle, fftPlan, planLock );
  scopedLock sLock( *planLock, _T( " ampfftGetPlanDim" ) );

  *dim = fftPlan->dimension;

  switch( fftPlan->dimension )
  {
    case AMPFFT_1D:
    {
      *size = 1;
    }
    break;
    case AMPFFT_2D:
    {
      *size = 2;
    }
    break;
    case AMPFFT_3D:
    {
      *size = 3;
    }
    break;
    default:
      return AMPFFT_ERROR;
      break;
  }
  return AMPFFT_SUCCESS;
}

ampfftStatus FFTPlan::ampfftSetPlanDim(  ampfftPlanHandle plHandle, const  ampfftDim dim )
{
  FFTRepo& fftRepo = FFTRepo::getInstance( );
  FFTPlan* fftPlan = NULL;
  lockRAII* planLock = NULL;

  fftRepo.getPlan( plHandle, fftPlan, planLock );
  scopedLock sLock( *planLock, _T( " ampfftGetPlanDim" ) );

  // We resize the vectors in the plan to keep their sizes consistent with the value of the dimension
  switch( dim )
  {
    case AMPFFT_1D:
    {
      fftPlan->length.resize( 1 );
      fftPlan->inStride.resize( 1 );
      fftPlan->outStride.resize( 1 );
    }
    break;
    case AMPFFT_2D:
    {
      fftPlan->length.resize( 2 );
      fftPlan->inStride.resize( 2 );
      fftPlan->outStride.resize( 2 );
    }
    break;
    case AMPFFT_3D:
    {
      fftPlan->length.resize( 3 );
      fftPlan->inStride.resize( 3 );
      fftPlan->outStride.resize( 3 );
    }
    break;
    default:
      return AMPFFT_ERROR;
      break;
  }

  //	If we modify the state of the plan, we assume that we can't trust any pre-calculated contents anymore
  fftPlan->baked = false;
  fftPlan->dimension = dim;

  return AMPFFT_SUCCESS;
}

ampfftStatus FFTPlan::ampfftGetPlanLength( const  ampfftPlanHandle plHandle, const  ampfftDim dim, size_t* clLengths )
{
  FFTRepo& fftRepo = FFTRepo::getInstance( );
  FFTPlan* fftPlan = NULL;
  lockRAII* planLock = NULL;

  fftRepo.getPlan( plHandle, fftPlan, planLock );
  scopedLock sLock( *planLock, _T( " ampfftGetPlanLength" ) );

  if( clLengths == NULL )
    return AMPFFT_ERROR;

  if( fftPlan->length.empty( ) )
    return AMPFFT_ERROR;

  switch( dim )
  {
    case AMPFFT_1D:
    {
      clLengths[0] = fftPlan->length[0];
    }
    break;
    case AMPFFT_2D:
    {
      if( fftPlan->length.size() < 2 )
        return AMPFFT_ERROR;

      clLengths[0] = fftPlan->length[0];
      clLengths[1 ] = fftPlan->length[1];
    }
    break;
    case AMPFFT_3D:
    {
      if( fftPlan->length.size() < 3 )
	return AMPFFT_ERROR;

      clLengths[0] = fftPlan->length[0];
      clLengths[1 ] = fftPlan->length[1];
      clLengths[2] = fftPlan->length[2];
    }
    break;
    default:
      return AMPFFT_ERROR;
      break;
  }
  return AMPFFT_SUCCESS;
}

ampfftStatus FFTPlan::ampfftSetPlanLength(  ampfftPlanHandle plHandle, const  ampfftDim dim, const size_t* clLengths )
{
  FFTRepo& fftRepo = FFTRepo::getInstance( );
  FFTPlan* fftPlan = NULL;
  lockRAII* planLock = NULL;

  fftRepo.getPlan( plHandle, fftPlan, planLock );
  scopedLock sLock( *planLock, _T( " ampfftSetPlanLength" ) );

  if( clLengths == NULL )
    return AMPFFT_ERROR;

  //	Simplest to clear any previous contents, because it's valid for user to shrink dimension
  fftPlan->length.clear( );
  switch( dim )
  {
    case AMPFFT_1D:
    {
    //	Minimum length size is 1
    if( clLengths[0] == 0 )
      return AMPFFT_ERROR;

    fftPlan->length.push_back( clLengths[0] );
    }
    break;
    case AMPFFT_2D:
    {
      //	Minimum length size is 1
      if(clLengths[0] == 0 || clLengths[1] == 0 )
         return AMPFFT_ERROR;

      fftPlan->length.push_back( clLengths[0] );
      fftPlan->length.push_back( clLengths[1] );
    }
    break;
    case AMPFFT_3D:
    {
      //	Minimum length size is 1
      if(clLengths[0 ] == 0 || clLengths[1] == 0 || clLengths[2] == 0)
        return AMPFFT_ERROR;

      fftPlan->length.push_back( clLengths[0] );
      fftPlan->length.push_back( clLengths[1] );
      fftPlan->length.push_back( clLengths[2] );
    }
    break;
    default:
      return AMPFFT_ERROR;
      break;
  }
  fftPlan->dimension = dim;

  //	If we modify the state of the plan, we assume that we can't trust any pre-calculated contents anymore
  fftPlan->baked = false;

  return AMPFFT_SUCCESS;
}

ampfftStatus FFTPlan::ampfftGetPlanInStride( const  ampfftPlanHandle plHandle, const  ampfftDim dim, size_t* clStrides )
{
  FFTRepo& fftRepo = FFTRepo::getInstance( );
  FFTPlan* fftPlan = NULL;
  lockRAII* planLock = NULL;

  fftRepo.getPlan( plHandle, fftPlan, planLock );
  scopedLock sLock( *planLock, _T( " ampfftGetPlanInStride" ) );

  if(clStrides == NULL )
    return AMPFFT_ERROR;

  switch( dim )
  {
    case AMPFFT_1D:
    {
      if(fftPlan->inStride.size( ) > 0 )
        clStrides[0] = fftPlan->inStride[0];
      else
        return AMPFFT_ERROR;
    }
    break;
    case AMPFFT_2D:
    {
      if( fftPlan->inStride.size( ) > 1 )
      {
        clStrides[0] = fftPlan->inStride[0];
	clStrides[1] = fftPlan->inStride[1];
      }
      else
        return AMPFFT_ERROR;
    }
    break;
    case AMPFFT_3D:
    {
      if( fftPlan->inStride.size( ) > 2 )
      {
        clStrides[0] = fftPlan->inStride[0];
        clStrides[1] = fftPlan->inStride[1];
	clStrides[2] = fftPlan->inStride[2];
      }
      else
        return AMPFFT_ERROR;
    }
    break;
    default:
      return AMPFFT_ERROR;
      break;
  }
  return AMPFFT_SUCCESS;
}

ampfftStatus FFTPlan::ampfftSetPlanInStride(  ampfftPlanHandle plHandle, const  ampfftDim dim, size_t* clStrides )
{
  FFTRepo& fftRepo = FFTRepo::getInstance( );
  FFTPlan* fftPlan = NULL;
  lockRAII* planLock = NULL;

  fftRepo.getPlan( plHandle, fftPlan, planLock );
  scopedLock sLock( *planLock, _T( " ampfftSetPlanInStride" ) );

  if( clStrides == NULL )
    return AMPFFT_ERROR;

  //	Simplest to clear any previous contents, because it's valid for user to shrink dimension
  fftPlan->inStride.clear( );
  switch( dim )
  {
    case AMPFFT_1D:
    {
      fftPlan->inStride.push_back( clStrides[0] );
    }
    break;
    case AMPFFT_2D:
    {
      fftPlan->inStride.push_back( clStrides[0] );
      fftPlan->inStride.push_back( clStrides[1] );
    }
    break;
    case AMPFFT_3D:
    {
      fftPlan->inStride.push_back( clStrides[0] );
      fftPlan->inStride.push_back( clStrides[1] );
      fftPlan->inStride.push_back( clStrides[2] );
    }
    break;
    default:
      return AMPFFT_ERROR;
      break;
  }

  //	If we modify the state of the plan, we assume that we can't trust any pre-calculated contents anymore
  fftPlan->baked = false;

  return AMPFFT_SUCCESS;
}

ampfftStatus FFTPlan::ampfftGetPlanOutStride( const  ampfftPlanHandle plHandle, const  ampfftDim dim, size_t* clStrides )
{
  FFTRepo& fftRepo	= FFTRepo::getInstance( );
  FFTPlan* fftPlan	= NULL;
  lockRAII* planLock	= NULL;

  fftRepo.getPlan( plHandle, fftPlan, planLock );
  scopedLock sLock( *planLock, _T( " ampfftGetPlanOutStride" ) );

  if( clStrides == NULL )
    return AMPFFT_ERROR;

  switch( dim )
  {
    case AMPFFT_1D:
    {
      if( fftPlan->outStride.size() > 0 )
        clStrides[0] = fftPlan->outStride[0];
      else
        return AMPFFT_ERROR;
    }
    break;
    case AMPFFT_2D:
    {
      if( fftPlan->outStride.size() > 1 )
      {
        clStrides[0] = fftPlan->outStride[0];
        clStrides[1] = fftPlan->outStride[1];
      }
      else
        return AMPFFT_ERROR;
     }
     break;
     case AMPFFT_3D:
     {
       if( fftPlan->outStride.size() > 2 )
       {
         clStrides[0] = fftPlan->outStride[0];
         clStrides[1] = fftPlan->outStride[1];
         clStrides[2] = fftPlan->outStride[2];
       }
       else
         return AMPFFT_ERROR;
     }
     break;
     default:
       return AMPFFT_ERROR;
       break;
    }
    return AMPFFT_SUCCESS;
}

ampfftStatus FFTPlan::ampfftSetPlanOutStride(  ampfftPlanHandle plHandle, const  ampfftDim dim, size_t* clStrides )
{
  FFTRepo& fftRepo	= FFTRepo::getInstance( );
  FFTPlan* fftPlan	= NULL;
  lockRAII* planLock	= NULL;

  fftRepo.getPlan( plHandle, fftPlan, planLock );
  scopedLock sLock( *planLock, _T( " ampfftSetPlanOutStride" ) );

  if( clStrides == NULL )
    return AMPFFT_ERROR;

  switch( dim )
  {
    case AMPFFT_1D:
    {
      fftPlan->outStride[0] = clStrides[0];
    }
    break;
    case AMPFFT_2D:
    {
      fftPlan->outStride[0] = clStrides[0];
      fftPlan->outStride[1] = clStrides[1];
    }
    break;
    case AMPFFT_3D:
    {
      fftPlan->outStride[0] = clStrides[0];
      fftPlan->outStride[1] = clStrides[1];
      fftPlan->outStride[2] = clStrides[2];
    }
    break;
    default:
      return AMPFFT_ERROR;
      break;
  }

  //	If we modify the state of the plan, we assume that we can't trust any pre-calculated contents anymore
  fftPlan->baked	= false;

  return AMPFFT_SUCCESS;
}

ampfftStatus FFTPlan::ampfftGetPlanDistance( const  ampfftPlanHandle plHandle, size_t* iDist, size_t* oDist )
{
  FFTRepo& fftRepo	= FFTRepo::getInstance( );
  FFTPlan* fftPlan	= NULL;
  lockRAII* planLock	= NULL;

  fftRepo.getPlan( plHandle, fftPlan, planLock );
  scopedLock sLock( *planLock, _T( " ampfftGetPlanDistance" ) );

  *iDist = fftPlan->iDist;
  *oDist = fftPlan->oDist;

  return AMPFFT_SUCCESS;
}

ampfftStatus FFTPlan::ampfftSetPlanDistance(  ampfftPlanHandle plHandle, size_t iDist, size_t oDist )
{
  FFTRepo& fftRepo = FFTRepo::getInstance( );
  FFTPlan* fftPlan = NULL;
  lockRAII* planLock = NULL;

  fftRepo.getPlan(plHandle, fftPlan, planLock );
  scopedLock sLock(*planLock, _T( " ampfftSetPlanDistance" ) );

  //	If we modify the state of the plan, we assume that we can't trust any pre-calculated contents anymore
  fftPlan->baked = false;
  fftPlan->iDist = iDist;
  fftPlan->oDist = oDist;

  return AMPFFT_SUCCESS;
}

ampfftStatus FFTPlan::ampfftGetLayout( const  ampfftPlanHandle plHandle,  ampfftIpLayout* iLayout,  ampfftOpLayout* oLayout )
{
  FFTRepo& fftRepo = FFTRepo::getInstance( );
  FFTPlan* fftPlan = NULL;
  lockRAII* planLock = NULL;

  fftRepo.getPlan(plHandle, fftPlan, planLock );
  scopedLock sLock(*planLock, _T( " ampfftGetLayout" ) );

  *iLayout = fftPlan->ipLayout;
  *oLayout = fftPlan->opLayout;

  return AMPFFT_SUCCESS;
}

ampfftStatus FFTPlan::ampfftSetLayout(  ampfftPlanHandle plHandle,  ampfftIpLayout iLayout,  ampfftOpLayout oLayout )
{
  FFTRepo& fftRepo = FFTRepo::getInstance( );
  FFTPlan* fftPlan = NULL;
  lockRAII* planLock = NULL;

  fftRepo.getPlan( plHandle, fftPlan, planLock );
  scopedLock sLock( *planLock, _T( " ampfftSetLayout" ) );

  //	We currently only support a subset of formats
  switch( iLayout )
  {
    case AMPFFT_COMPLEX:
    {
      if( oLayout == AMPFFT_COMPLEX)
	return AMPFFT_ERROR;
    }
    break;
    case AMPFFT_REAL:
    {
      if(oLayout == AMPFFT_REAL)
	return AMPFFT_ERROR;
    }
    break;
    default:
      return AMPFFT_ERROR;
      break;
  }

  //	We currently only support a subset of formats
  switch( oLayout )
  {
    case AMPFFT_COMPLEX:
    {
      if(iLayout == AMPFFT_COMPLEX)
        return AMPFFT_ERROR;
    }
    break;

    case AMPFFT_REAL:
    {
      if(iLayout == AMPFFT_REAL)
        return AMPFFT_ERROR;
    }
    break;
    default:
      return AMPFFT_ERROR;
      break;
  }

  //	If we modify the state of the plan, we assume that we can't trust any pre-calculated contents anymore
  fftPlan->baked	= false;
  fftPlan->ipLayout	= iLayout;
  fftPlan->opLayout	= oLayout;
  return AMPFFT_SUCCESS;
}

ampfftStatus FFTPlan::ampfftGetResultLocation( const  ampfftPlanHandle plHandle,  ampfftResLocation* placeness )
{
  FFTRepo& fftRepo	= FFTRepo::getInstance( );
  FFTPlan* fftPlan	= NULL;
  lockRAII* planLock	= NULL;

  fftRepo.getPlan( plHandle, fftPlan, planLock );
  scopedLock sLock( *planLock, _T( " ampfftGetResultLocation" ) );

  *placeness	= fftPlan->location;
  return	AMPFFT_SUCCESS;
}

ampfftStatus FFTPlan::ampfftSetResultLocation(  ampfftPlanHandle plHandle,  ampfftResLocation placeness )
{
  FFTRepo& fftRepo	= FFTRepo::getInstance( );
  FFTPlan* fftPlan	= NULL;
  lockRAII* planLock	= NULL;

  fftRepo.getPlan( plHandle, fftPlan, planLock );
  scopedLock sLock( *planLock, _T( " ampfftSetResultLocation" ) );

  //	If we modify the state of the plan, we assume that we can't trust any pre-calculated contents anymore
  fftPlan->baked		= false;
  fftPlan->location	= placeness;

  return AMPFFT_SUCCESS;
}

ampfftStatus FFTPlan::ampfftGetPlanTransposeResult( const  ampfftPlanHandle plHandle,  ampfftResTransposed * transposed )
{
  FFTRepo& fftRepo	= FFTRepo::getInstance( );
  FFTPlan* fftPlan	= NULL;
  lockRAII* planLock	= NULL;

  fftRepo.getPlan( plHandle, fftPlan, planLock );
  scopedLock sLock( *planLock, _T( " ampfftGetResultLocation" ) );

  *transposed	= fftPlan->transposeType;

  return AMPFFT_SUCCESS;
}

ampfftStatus FFTPlan::ampfftSetPlanTransposeResult(  ampfftPlanHandle plHandle,  ampfftResTransposed transposed )
{
  FFTRepo& fftRepo	= FFTRepo::getInstance( );
  FFTPlan* fftPlan	= NULL;
  lockRAII* planLock	= NULL;

  fftRepo.getPlan( plHandle, fftPlan, planLock );
  scopedLock sLock( *planLock, _T( " ampfftSetResultLocation" ) );

  //	If we modify the state of the plan, we assume that we can't trust any pre-calculated contents anymore
  fftPlan->baked		= false;
  fftPlan->transposeType	= transposed;

  return AMPFFT_SUCCESS;
}

ampfftStatus FFTPlan::GetMax1DLength (size_t *longest ) const
{
	switch(gen)
	{
	case Stockham:
          return GetMax1DLengthPvt<Stockham>(longest);
	default:
          return AMPFFT_ERROR;
	}
}

ampfftStatus  FFTPlan::GetKernelGenKey (FFTKernelGenKeyParams & params) const
{
	switch(gen)
	{
	case Stockham:
          return GetKernelGenKeyPvt<Stockham>(params);
	default:
	  return AMPFFT_ERROR;
	}
}

ampfftStatus  FFTPlan::GetWorkSizes (std::vector<size_t> & globalws, std::vector<size_t> & localws) const
{
	switch(gen)
	{
	case Stockham:
	  return GetWorkSizesPvt<Stockham>(globalws, localws);
	default:
	  return AMPFFT_ERROR;
	}
}

ampfftStatus  FFTPlan::GenerateKernel (const ampfftPlanHandle plHandle, FFTRepo & fftRepo) const
{
        switch(gen)
	{
	case Stockham:
          return GenerateKernelPvt<Stockham>(plHandle, fftRepo);
	default:
          return AMPFFT_ERROR;
	}
}
ampfftStatus FFTPlan::ampfftDestroyPlan( ampfftPlanHandle* plHandle )
{
  FFTRepo& fftRepo	= FFTRepo::getInstance( );
  FFTPlan* fftPlan	= NULL;
  lockRAII* planLock	= NULL;

  fftRepo.getPlan( *plHandle, fftPlan, planLock );

  //	Recursively destroy subplans, that are used for higher dimensional FFT's
  if( fftPlan->planX )
    ampfftDestroyPlan( &fftPlan->planX );

  if( fftPlan->planY )
    ampfftDestroyPlan( &fftPlan->planY );

   if( fftPlan->planZ )
    ampfftDestroyPlan( &fftPlan->planZ );

   if( fftPlan->planTX )
    ampfftDestroyPlan( &fftPlan->planTX );

   if( fftPlan->planTY )
    ampfftDestroyPlan( &fftPlan->planTY );

   if( fftPlan->planTZ )
    ampfftDestroyPlan( &fftPlan->planTZ );

   if( fftPlan->planRCcopy )
    ampfftDestroyPlan( &fftPlan->planRCcopy );

  fftRepo.deletePlan( plHandle );

  return AMPFFT_SUCCESS;
}

ampfftStatus FFTPlan::AllocateWriteBuffers ()
{
	ampfftStatus status = AMPFFT_SUCCESS;

	assert (NULL == const_buffer);

	assert(4 == sizeof(int));

	//	Construct the constant buffer and call clEnqueueWriteBuffer
	float ConstantBufferParams[AMPFFT_CB_SIZE];
	memset (& ConstantBufferParams, 0, sizeof (ConstantBufferParams));

	float nY = 1;
	float nZ = 0;
	float nW = 0;
	float n5 = 0;

	switch( length.size() )
	{
	case 1:
		nY = (float)batchSize;
		break;

	case 2:
		nY = (float)length[1];
		nZ = (float)batchSize;
		break;

	case 3:
		nY = (float)length[1];
		nZ = (float)length[2];
		nW = (float)batchSize;
		break;

	case 4:
		nY = (float)length[1];
		nZ = (float)length[2];
		nW = (float)length[3];
		n5 = (float)batchSize;
		break;
	}
	ConstantBufferParams[AMPFFT_CB_NY ] = nY;
	ConstantBufferParams[AMPFFT_CB_NZ ] = nZ;
	ConstantBufferParams[AMPFFT_CB_NW ] = nW;
	ConstantBufferParams[AMPFFT_CB_N5 ] = n5;

	assert (/*fftPlan->*/inStride.size() == /*fftPlan->*/outStride.size());

	switch (/*fftPlan->*/inStride.size()) {
	case 1:
		ConstantBufferParams[AMPFFT_CB_ISX] = (float)inStride[0];
		ConstantBufferParams[AMPFFT_CB_ISY] = (float)iDist;
		break;

	case 2:
		ConstantBufferParams[AMPFFT_CB_ISX] = (float)inStride[0];
		ConstantBufferParams[AMPFFT_CB_ISY] = (float)inStride[1];
		ConstantBufferParams[AMPFFT_CB_ISZ] = (float)iDist;
		break;

	case 3:
		ConstantBufferParams[AMPFFT_CB_ISX] = (float)inStride[0];
		ConstantBufferParams[AMPFFT_CB_ISY] = (float)inStride[1];
		ConstantBufferParams[AMPFFT_CB_ISZ] = (float)inStride[2];
		ConstantBufferParams[AMPFFT_CB_ISW] = (float)iDist;
		break;

	case 4:
		ConstantBufferParams[AMPFFT_CB_ISX] = (float)inStride[0];
		ConstantBufferParams[AMPFFT_CB_ISY] = (float)inStride[1];
		ConstantBufferParams[AMPFFT_CB_ISZ] = (float)inStride[2];
		ConstantBufferParams[AMPFFT_CB_ISW] = (float)inStride[3];
		ConstantBufferParams[AMPFFT_CB_IS5] = (float)iDist;
		break;
	}

	switch (/*fftPlan->*/outStride.size()) {
	case 1:
		ConstantBufferParams[AMPFFT_CB_OSX] = (float)outStride[0];
		ConstantBufferParams[AMPFFT_CB_OSY] = (float)oDist;
		break;

	case 2:
		ConstantBufferParams[AMPFFT_CB_OSX] = (float)outStride[0];
		ConstantBufferParams[AMPFFT_CB_OSY] = (float)outStride[1];
		ConstantBufferParams[AMPFFT_CB_OSZ] = (float)oDist;
		break;

	case 3:
		ConstantBufferParams[AMPFFT_CB_OSX] = (float)outStride[0];
		ConstantBufferParams[AMPFFT_CB_OSY] = (float)outStride[1];
		ConstantBufferParams[AMPFFT_CB_OSZ] = (float)outStride[2];
		ConstantBufferParams[AMPFFT_CB_OSW] = (float)oDist;
		break;

	case 4:
		ConstantBufferParams[AMPFFT_CB_OSX] = (float)outStride[0];
		ConstantBufferParams[AMPFFT_CB_OSY] = (float)outStride[1];
		ConstantBufferParams[AMPFFT_CB_OSZ] = (float)outStride[2];
		ConstantBufferParams[AMPFFT_CB_OSW] = (float)outStride[3];
		ConstantBufferParams[AMPFFT_CB_OS5] = (float)oDist;
		break;
	}

        Concurrency::array<float, 1> arr = Concurrency::array<float, 1>(Concurrency::extent<1>(AMPFFT_CB_SIZE), ConstantBufferParams);
        const_buffer = new Concurrency::array_view<float>(arr);
	return AMPFFT_SUCCESS;
}

ampfftStatus FFTPlan::ReleaseBuffers ()
{
	ampfftStatus result = AMPFFT_SUCCESS;

	if( NULL != const_buffer )
	{
                delete const_buffer;
	}

	if( NULL != intBuffer )
	{
                delete intBuffer;
	}

	if( NULL != intBufferRC )
	{
                delete intBufferRC;
	}

	return	AMPFFT_SUCCESS;
}

size_t FFTPlan::ElementSize() const
{
  return ((precision == AMPFFT_DOUBLE) ? sizeof(std::complex<double> ) : sizeof(std::complex<float>));
}

/*----------------------------------------------------FFTPlan-----------------------------------------------------------------------------*/

/*---------------------------------------------------FFTRepo--------------------------------------------------------------------------------*/
ampfftStatus FFTRepo::createPlan( ampfftPlanHandle* plHandle, FFTPlan*& fftPlan )
{
	scopedLock sLock( lockRepo, _T( "insertPlan" ) );

	//	We keep track of this memory in our own collection class, to make sure it's freed in releaseResources
	//	The lifetime of a plan is tracked by the client and is freed when the client calls ::ampfftDestroyPlan()
	fftPlan	= new FFTPlan;

	//	We allocate a new lock here, and expect it to be freed in ::ampfftDestroyPlan();
	//	The lifetime of the lock is the same as the lifetime of the plan
	lockRAII* lockPlan	= new lockRAII;

	//	Add and remember the fftPlan in our map
	repoPlans[ planCount ] = make_pair( fftPlan, lockPlan );

	//	Assign the user handle the plan count (unique identifier), and bump the count for the next plan
	*plHandle	= planCount++;

	return	AMPFFT_SUCCESS;
}


ampfftStatus FFTRepo::getPlan( ampfftPlanHandle plHandle, FFTPlan*& fftPlan, lockRAII*& planLock )
{
	scopedLock sLock( lockRepo, _T( "getPlan" ) );

	//	First, check if we have already created a plan with this exact same FFTPlan
	repoPlansType::iterator iter	= repoPlans.find( plHandle );
	if( iter == repoPlans.end( ) )
		return	AMPFFT_ERROR;

	//	If plan is valid, return fill out the output pointers
	fftPlan		= iter->second.first;
	planLock	= iter->second.second;

	return	AMPFFT_SUCCESS;
}

ampfftStatus FFTRepo::deletePlan( ampfftPlanHandle* plHandle )
{
	scopedLock sLock( lockRepo, _T( "deletePlan" ) );

	//	First, check if we have already created a plan with this exact same FFTPlan
	repoPlansType::iterator iter	= repoPlans.find( *plHandle );
	if( iter == repoPlans.end( ) )
		return	AMPFFT_ERROR;

	//	We lock the plan object while we are in the process of deleting it
	{
		scopedLock sLock( *iter->second.second, _T( "ampfftDestroyPlan" ) );

		//	Delete the FFTPlan
		delete iter->second.first;
	}

		//	Delete the lockRAII
	delete iter->second.second;

	//	Remove entry from our map object
	repoPlans.erase( iter );

	//	Clear the client's handle to signify that the plan is gone
	*plHandle = 0;

	return	AMPFFT_SUCCESS;
}

ampfftStatus FFTRepo::setProgramEntryPoints( const ampfftGenerators gen, const ampfftPlanHandle& handle,
                                             const FFTKernelGenKeyParams& fftParam, const char * kernel_fwd,
                                             const char * kernel_back)
{
	scopedLock sLock( lockRepo, _T( "setProgramEntryPoints" ) );

	fftRepoKey key = std::make_pair( gen, handle );

	fftRepoValue& fft = mapFFTs[ key ];
	fft.EntryPoint_fwd  = kernel_fwd;
	fft.EntryPoint_back = kernel_back;

	return	AMPFFT_SUCCESS;
}

ampfftStatus FFTRepo::getProgramEntryPoint( const ampfftGenerators gen, const ampfftPlanHandle& handle,
                                            const FFTKernelGenKeyParams& fftParam, ampfftDirection dir,
                                            std::string& kernel)
{
	scopedLock sLock( lockRepo, _T( "getProgramEntryPoint" ) );

	fftRepoKey key = std::make_pair( gen, handle );

	fftRepo_iterator pos = mapFFTs.find( key );

	if( pos == mapFFTs.end( ) )
		return	AMPFFT_ERROR;

	switch (dir) {
	case AMPFFT_FORWARD:
		kernel = pos->second.EntryPoint_fwd;
		break;
	case AMPFFT_BACKWARD:
		kernel = pos->second.EntryPoint_back;
		break;
	default:
		assert (false);
		return AMPFFT_ERROR;
	}

	if (0 == kernel.size())
		return	AMPFFT_ERROR;

	return	AMPFFT_SUCCESS;
}

ampfftStatus FFTRepo::setProgramCode( const ampfftGenerators gen, const ampfftPlanHandle& handle, const FFTKernelGenKeyParams& fftParam, const std::string& kernel)
{
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

	return	AMPFFT_SUCCESS;
}

ampfftStatus FFTRepo::getProgramCode( const ampfftGenerators gen, const ampfftPlanHandle& handle, const FFTKernelGenKeyParams& fftParam, std::string& kernel)
{

	scopedLock sLock( lockRepo, _T( "getProgramCode" ) );
	fftRepoKey key = std::make_pair( gen, handle );

	fftRepo_iterator pos = mapFFTs.find( key);
	if( pos == mapFFTs.end( ) )
		return	AMPFFT_ERROR;

        kernel = pos->second.ProgramString;
	return	AMPFFT_SUCCESS;
}

ampfftStatus FFTRepo::releaseResources( )
{
	scopedLock sLock( lockRepo, _T( "releaseResources" ) );

	//	Free all memory allocated in the repoPlans; represents cached plans that were not destroyed by the client
	//
	for( repoPlansType::iterator iter = repoPlans.begin( ); iter != repoPlans.end( ); ++iter )
	{
		FFTPlan* plan	= iter->second.first;
		lockRAII* lock	= iter->second.second;
		if( plan != NULL )
		{
			delete plan;
		}
		if( lock != NULL )
		{
			delete lock;
		}
	}

	//	Reset the plan count to zero because we are guaranteed to have destroyed all plans
	planCount	= 1;

	//	Release all strings
	mapFFTs.clear( );

	return	AMPFFT_SUCCESS;
}
/*------------------------------------------------FFTRepo----------------------------------------------------------------------------------*/

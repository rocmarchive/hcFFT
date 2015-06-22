#include "ampfftlib.h"

/*----------------------------------------------------FFTPlan-----------------------------------------------------------------------------*/

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

ampfftStatus FFTPlan::executePlan(FFTPlan* fftPlan)
{
  if(!fftPlan)
    return AMPFFT_INVALID;

  return AMPFFT_SUCCESS;
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

        std::cout<<" inStride.size() "<<inStride.size() << " outStride.size() "<<outStride.size() <<std::endl;
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
        for(int i = 0 ; i < 32 ; i++)
          std::cout<<" const_buffer["<<i<<"] "<<(*const_buffer)[i]<<std::endl;
	return AMPFFT_SUCCESS;
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

#include "ampfftlib.h"

ampfftStatus FFTPlan::ampfftCreateDefaultPlan (ampfftPlanHandle* plHandle,ampfftDim dimension, const size_t *length)
{
  return AMPFFT_SUCCESS;
}

ampfftStatus FFTPlan::executePlan(FFTPlan* fftPlan)
{
  if(!fftPlan)
    return AMPFFT_INVALID;

  return AMPFFT_SUCCESS;
}


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


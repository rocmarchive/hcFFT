#include <math.h>
#include "stockham.h"
#include <list>
// FFT Stockham Autosort Method
//
//   Each pass does one digit reverse in essence. Hence by the time all passes are done, complete
//   digit reversal is done and output FFT is in correct order. Intermediate FFTs are stored in natural order,
//   which is not the case with basic Cooley-Tukey algorithm. Natural order in intermediate data makes it
//   convenient for stitching together passes with different radices.
//
//  Basic FFT algorithm:
//
//        Pass loop
//        {
//            Outer loop
//            {
//                Inner loop
//                {
//                }
//            }
//        }
//
//  The sweeps of the outer and inner loop resemble matrix indexing, this matrix changes shape with every pass as noted below
//
//   FFT pass diagram (radix 2)
//
//                k            k+R                                    k
//            * * * * * * * * * * * * * * * *                     * * * * * * * *
//            *   |             |           *                     *   |         *
//            *   |             |           *                     *   |         *
//            *   |             |           * LS        -->       *   |         *
//            *   |             |           *                     *   |         *
//            *   |             |           *                     *   |         *
//            * * * * * * * * * * * * * * * *                     *   |         *
//                         RS                                     *   |         * L
//                                                                *   |         *
//                                                                *   |         *
//                                                                *   |         *
//                                                                *   |         *
//                                                                *   |         *
//                                                                *   |         *
//                                                                *   |         *
//                                                                * * * * * * * *
//                                                                       R
//
//
//    With every pass, the matrix doubles in height and halves in length
//
//
//  N = 2^T = Length of FFT
//  q = pass loop index
//  k = outer loop index = (0 ... R-1)
//  j = inner loop index = (0 ... LS-1)
//
//  Tables shows how values change as we go through the passes
//
//    q | LS   |  R   |  L  | RS
//   ___|______|______|_____|___
//    0 |  1   | N/2  |  2  | N
//    1 |  2   | N/4  |  4  | N/2
//    2 |  4   | N/8  |  8  | N/4
//    . |  .   | .    |  .  | .
//  T-1 |  N/2 | 1    |  N  | 2
//
//
//   Data Read Order
//     Radix 2: k*LS + j, (k+R)*LS + j
//     Radix 3: k*LS + j, (k+R)*LS + j, (k+2R)*LS + j
//     Radix 4: k*LS + j, (k+R)*LS + j, (k+2R)*LS + j, (k+3R)*LS + j
//     Radix 5: k*LS + j, (k+R)*LS + j, (k+2R)*LS + j, (k+3R)*LS + j, (k+4R)*LS + j
//
//   Data Write Order
//       Radix 2: k*L + j, k*L + j + LS
//       Radix 3: k*L + j, k*L + j + LS, k*L + j + 2*LS
//       Radix 4: k*L + j, k*L + j + LS, k*L + j + 2*LS, k*L + j + 3*LS
//       Radix 5: k*L + j, k*L + j + LS, k*L + j + 2*LS, k*L + j + 3*LS, k*L + j + 4*LS
//

namespace StockhamGenerator
{
// Experimental End ===========================================

#define RADIX_TABLE_COMMON 	{     2048,           256,             1,         4,     8, 8, 8, 4, 0, 0, 0, 0, 0, 0, 0, 0 },	\
							{      512,            64,             1,         3,     8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0 },	\
							{      256,            64,             1,         4,     4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0 },	\
							{       64,            64,             4,         3,     4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0 },	\
							{       32,            64,            16,         2,     8, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },	\
							{       16,            64,            16,         2,     4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },	\
							{        4,            64,            32,         2,     2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },	\
							{        2,            64,            64,         1,     2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },

    template <Precision PR>
	class KernelCoreSpecs
	{
		struct SpecRecord
		{
			size_t length;
			size_t workGroupSize;
			size_t numTransforms;
			size_t numPasses;
			size_t radices[12]; // Setting upper limit of number of passes to 12
		};

		typedef typename std::map<size_t, SpecRecord> SpecTable;
		SpecTable specTable;

	public:
		KernelCoreSpecs()
		{
			switch(PR)
			{
			case P_SINGLE:
				{
					SpecRecord specRecord[] = {

					RADIX_TABLE_COMMON

					//  Length, WorkGroupSize, NumTransforms, NumPasses,  Radices
					{     4096,           256,             1,         4,     8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0 },
					{     1024,           128,             1,         4,     8, 8, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0 },
					{      128,            64,             4,         3,     8, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
					{        8,            64,            32,         2,     4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },

					};

					size_t tableLength = sizeof(specRecord)/sizeof(specRecord[0]);
					for(size_t i=0; i<tableLength; i++) specTable[specRecord[i].length] = specRecord[i];

				} break;

			case P_DOUBLE:
				{
					SpecRecord specRecord[] = {

					RADIX_TABLE_COMMON

					//  Length, WorkGroupSize, NumTransforms, NumPasses,  Radices
					{     1024,           128,             1,         4,     8, 8, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0 },
					//{      128,            64,             1,         7,     2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0 },
					{      128,            64,             4,         3,     8, 8, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
					{        8,            64,            16,         3,     2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0 },

					};

					size_t tableLength = sizeof(specRecord)/sizeof(specRecord[0]);
					for(size_t i=0; i<tableLength; i++) specTable[specRecord[i].length] = specRecord[i];
				} break;

			default:
				assert(false);
			}
		}

		void GetRadices(size_t length, size_t &numPasses, const size_t * &pRadices) const
		{
			pRadices = NULL;
			numPasses = 0;

			typename SpecTable::const_iterator it = specTable.find(length);
			if(it != specTable.end())
			{
				pRadices = it->second.radices;
				numPasses = it->second.numPasses;
			}
		}

		void GetWGSAndNT(size_t length, size_t &workGroupSize, size_t &numTransforms) const
		{
			workGroupSize = 0;
			numTransforms = 0;

			typename SpecTable::const_iterator it = specTable.find(length);
			if(it != specTable.end())
			{
				workGroupSize = it->second.workGroupSize;
				numTransforms = it->second.numTransforms;
			}
		}
	};

	// Given the length of 1d fft, this function determines the appropriate work group size
	// and the number of transforms per work group
	// TODO for optimizations - experiment with different possibilities for work group sizes and num transforms for improving performance
	void DetermineSizes(const size_t &MAX_WGS, const size_t &length, size_t &workGroupSize, size_t &numTrans)
	{
		assert(MAX_WGS >= 64);

		if(length == 1) // special case
		{
			workGroupSize = 64;
			numTrans = 64;
			return;
		}

		size_t baseRadix[] = {5,3,2}; // list only supported primes
		size_t baseRadixSize = sizeof(baseRadix)/sizeof(baseRadix[0]);

		size_t l = length;
		std::map<size_t, size_t> primeFactors;
		std::map<size_t, size_t> primeFactorsExpanded;
		for(size_t r=0; r<baseRadixSize; r++)
		{
			size_t rad = baseRadix[r];
			size_t p = 0;
			size_t e = 1;
			while(!(l%rad))
			{
				l /= rad;
				e *= rad;
				p++;
			}

			primeFactors[rad] = p;
			primeFactorsExpanded[rad] = e;
		}

		assert(l == 1); // Makes sure the number is composed of only supported primes

		if (primeFactorsExpanded[2] == length)	// Length is pure power of 2
		{
			//if(length == 1024) { workGroupSize = 128;  numTrans = 1; }
			if		(length >= 1024)	{ workGroupSize = (MAX_WGS >= 256) ? 256 : MAX_WGS; numTrans = 1; }
			//else if (length == 512)		{ workGroupSize = (MAX_WGS >= 128) ? 128 : MAX_WGS; numTrans = 1; }
			else if (length == 512)		{ workGroupSize = 64; numTrans = 1; }
			else if	(length >= 16)		{ workGroupSize = 64;  numTrans = 256/length; }
			else						{ workGroupSize = 64;  numTrans = 128/length; }
		}
		else if	(primeFactorsExpanded[3] == length) // Length is pure power of 3
		{
			workGroupSize = (MAX_WGS >= 256) ? 243 : 27;
			if(length >= 3*workGroupSize)	numTrans = 1;
			else							numTrans = (3*workGroupSize)/length;
		}
		else if	(primeFactorsExpanded[5] == length) // Length is pure power of 5
		{
			workGroupSize = (MAX_WGS >= 128) ? 125 : 25;
			if(length >= 5*workGroupSize)	numTrans = 1;
			else							numTrans = (5*workGroupSize)/length;
		}
		else
		{
			size_t leastNumPerWI; // least number of elements in one work item
			size_t maxWorkGroupSize; // maximum work group size desired

			if		(primeFactorsExpanded[2] * primeFactorsExpanded[3] == length) // Length is mix of 2&3 only
			{
				if(!(length%12))	{ leastNumPerWI = 12; maxWorkGroupSize = (MAX_WGS >= 128) ? 128 : MAX_WGS; }
				else				{ leastNumPerWI = 6;  maxWorkGroupSize = (MAX_WGS >= 256) ? 256 : MAX_WGS; }
			}
			else if	(primeFactorsExpanded[2] * primeFactorsExpanded[5] == length) // Length is mix of 2&5 only
			{
				if(!(length%20))	{ leastNumPerWI = 20; maxWorkGroupSize = 64; }
				else				{ leastNumPerWI = 10; maxWorkGroupSize = (MAX_WGS >= 128) ? 128 : MAX_WGS; }
			}
			else if (primeFactorsExpanded[3] * primeFactorsExpanded[5] == length) // Length is mix of 3&5 only
			{
				leastNumPerWI = 15;
				maxWorkGroupSize = 64;
			}
			else
			{
				leastNumPerWI = 30;
				maxWorkGroupSize = 64;
			}


			// Make sure the work group size does not exceed MAX_WGS
			// for large problems sizes, this means doing more work per work-item
			size_t lnpi;
			size_t ft = 1;
			while(1)
			{
				lnpi = leastNumPerWI * ft++;
				if(length%lnpi) continue;

				if( (length/lnpi) <= MAX_WGS )
				{
					leastNumPerWI = lnpi;
					break;
				}
			}

			numTrans = 1;
			size_t n=1;
			while( ((n*length)/leastNumPerWI) <= maxWorkGroupSize )
			{
				numTrans = n;
				n++;
			}

			workGroupSize = (numTrans*length)/leastNumPerWI;
			assert(workGroupSize <= MAX_WGS);
		}
	}
}

template<>
ampfftStatus FFTPlan::GetMax1DLengthPvt<Stockham> (size_t * longest) const
{
	// TODO  The caller has already acquired the lock on *this
	//	However, we shouldn't depend on it.

	//	Query the devices in this context for their local memory sizes
	//	How large a kernel we can generate depends on the *minimum* LDS
	//	size for all devices.
	//
	const FFTEnvelope * pEnvelope = NULL;
	this->GetEnvelope (& pEnvelope);
	BUG_CHECK (NULL != pEnvelope);

	ARG_CHECK (NULL != longest)
	size_t LdsperElement = this->ElementSize();
	size_t result = pEnvelope->limit_LocalMemSize /
		(1 * LdsperElement);
	result = FloorPo2 (result);
	*longest = result;
	return AMPFFT_SUCCESS;
}


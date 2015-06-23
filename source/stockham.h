#include <iostream>
#include <cassert>
#include <vector>
#include <sstream>
#include <fstream>
#include <map>
#include "ampfftlib.h"

using namespace std;

namespace StockhamGenerator
{
	// Precision
	enum Precision
	{
		P_SINGLE,
		P_DOUBLE,
	};

	template <Precision PR>
	inline size_t PrecisionWidth()
	{
		switch(PR)
		{
		case P_SINGLE:	return 1;
		case P_DOUBLE:	return 2;
		default:		assert(false); return 1;
		}
	}

	// Convert unsigned integers to string
	inline std::string SztToStr(size_t i)
	{
		std::stringstream ss;
		ss << i;
		return ss.str();
	}

	inline std::string FloatToStr(double f)
	{
		std::stringstream ss;
		ss.precision(16);
		ss << std::scientific << f;
		return ss.str();
	}
}

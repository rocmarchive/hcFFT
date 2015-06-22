#include <pthread.h>
#include <sstream>
using namespace std;

#if defined( __GNUC__ )
  typedef char TCHAR;
  typedef char _TCHAR;
#define _tmain main

#if defined( UNICODE )
#define _T(x)	L ## x
#else
#define _T(x)	x
#endif
#endif

//	lockRAII provides an abstraction for the concept of a mutex; it wraps all  mutex functions in generic methods
//	Linux implementation not done yet
//	The template argument 'debugPrint' activates debugging information, but if not active the compiler optimizes
//	the print statements out

template< bool debugPrint >
class lockRAII
{
	pthread_mutex_t	mutex;
	pthread_mutexattr_t mAttr;
	string	mutexName;
	stringstream	tstream;

	//	Does not make sense to create a copy of a lock object; private method
	lockRAII( const lockRAII& rhs ): mutexName( rhs.mutexName )
	{
		tstream << std::hex << std::showbase;
	}

	public:
		lockRAII( )
		{
			tstream << std::hex << std::showbase;
			pthread_mutexattr_init( &mAttr );
			pthread_mutexattr_settype( &mAttr, PTHREAD_MUTEX_RECURSIVE );
			pthread_mutex_init( &mutex, &mAttr );
		}

		lockRAII( const string& name ): mutexName( name )
		{
			tstream << std::hex << std::showbase;
			pthread_mutexattr_init( &mAttr );
			pthread_mutexattr_settype( &mAttr, PTHREAD_MUTEX_RECURSIVE );
			pthread_mutex_init( &mutex, &mAttr );
		}

		~lockRAII( )
		{
			pthread_mutex_destroy( &mutex );
			pthread_mutexattr_destroy( &mAttr );
		}

		string& getName( )
		{
			return mutexName;
		}

		void setName( const string& name )
		{
			mutexName	= name;
		}

		void enter( )
		{
			if( debugPrint )
			{
				tstream.str( _T( "" ) );
				tstream << _T( "Attempting pthread_mutex_t( " ) << mutexName << _T( " )" ) << std::endl;
				std::cout << tstream.str( );
			}

			::pthread_mutex_lock( &mutex );

			if( debugPrint )
			{
				tstream.str( _T( "" ) );
				tstream << _T( "Acquired pthread_mutex_t( " ) << mutexName << _T( " )" ) << std::endl;
				std::cout << tstream.str( );
			}
		}

		void leave( )
		{
			if( debugPrint )
			{
				tstream.str( _T( "" ) );
				tstream << _T( "Releasing pthread_mutex_t( " ) << mutexName << _T( " )" ) << std::endl;
				std::cout << tstream.str( );
			}

			::pthread_mutex_unlock( &mutex );
		}
};

//	Convenience macro to enable/disable debugging print statements
#define lockRAII lockRAII< false >

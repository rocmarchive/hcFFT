#ifndef LIB_INCLUDE_LOCK_H_
#define LIB_INCLUDE_LOCK_H_

#include <pthread.h>
#include <sstream>
#include <string>

#if defined(__GNUC__)
typedef char TCHAR;
typedef char _TCHAR;
#define _tmain main

#if defined(UNICODE)
#define _T(x) L##x
#else
#define _T(x) x
#endif
#endif

//  lockRAII provides an abstraction for the concept of a mutex; it wraps all
//  mutex functions in generic methods
//  Linux implementation not done yet
//  The template argument 'debugPrint' activates debugging information, but if
//  not active the compiler optimizes
//  the print statements out

template <bool debugPrint>
class lockRAII {
  pthread_mutex_t mutex;
  pthread_mutexattr_t mAttr;
  std::string mutexName;
  std::stringstream tstream;

  //  Does not make sense to create a copy of a lock object; private method
  lockRAII(const lockRAII& rhs) : mutexName(rhs.mutexName) {
    tstream << std::hex << std::showbase;
  }

 public:
  lockRAII() {
    tstream << std::hex << std::showbase;
    pthread_mutexattr_init(&mAttr);
    pthread_mutexattr_settype(&mAttr, PTHREAD_MUTEX_RECURSIVE);
    pthread_mutex_init(&mutex, &mAttr);
  }

  explicit lockRAII(const std::string& name) : mutexName(name) {
    tstream << std::hex << std::showbase;
    pthread_mutexattr_init(&mAttr);
    pthread_mutexattr_settype(&mAttr, PTHREAD_MUTEX_RECURSIVE);
    pthread_mutex_init(&mutex, &mAttr);
  }

  ~lockRAII() {
    pthread_mutex_destroy(&mutex);
    pthread_mutexattr_destroy(&mAttr);
  }

  std::string& getName() { return mutexName; }

  void setName(const std::string& name) { mutexName = name; }

  void enter() {
    if (debugPrint) {
      tstream.str(_T(""));
      tstream << _T("Attempting pthread_mutex_t( ") << mutexName << _T(" )" )
              << std::endl;
      std::cout << tstream.str();
    }

    ::pthread_mutex_lock(&mutex);

    if (debugPrint) {
      tstream.str(_T(""));
      tstream << _T("Acquired pthread_mutex_t( ") << mutexName << _T(" )" )
              << std::endl;
      std::cout << tstream.str();
    }
  }

  void leave() {
    if (debugPrint) {
      tstream.str(_T(""));
      tstream << _T("Releasing pthread_mutex_t( ") << mutexName << _T(" )" )
              << std::endl;
      std::cout << tstream.str();
    }

    ::pthread_mutex_unlock(&mutex);
  }
};

//  Class used to make sure that we enter and leave critical sections in pairs
//  The template logic logs our CRITICAL_SECTION actions; if the template
//  parameter is false,
//  the branch is constant and the compiler will optimize the branch out
template <bool debugPrint>
class scopedLock {
  lockRAII<debugPrint>* sLock;
  std::string sLockName;
  std::stringstream tstream;

 public:
  scopedLock(lockRAII<debugPrint>& lock, const std::string& name)
      : sLock(&lock), sLockName(name) {
    if (debugPrint) {
      tstream.str(_T(""));
      tstream << _T("Entering scopedLock( ") << sLockName << _T(" )" )
              << std::endl
              << std::endl;
      std::cout << tstream.str();
    }

    sLock->enter();
  }

  ~scopedLock() {
    sLock->leave();

    if (debugPrint) {
      tstream.str(_T(""));
      tstream << _T("Left scopedLock( ") << sLockName << _T(" )" )
              << std::endl
              << std::endl;
      std::cout << tstream.str();
    }
  }
};
//  Convenience macro to enable/disable debugging print statements
#define lockRAII lockRAII<false>
#define scopedLock scopedLock<false>

#endif  // LIB_INCLUDE_LOCK_H_

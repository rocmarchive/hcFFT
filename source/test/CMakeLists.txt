# Object libraries require CMAKE 2.8.8 version 
CMAKE_MINIMUM_REQUIRED (VERSION 2.8.8) 
MESSAGE(STATUS "CMAKE VERSION ${CMAKE_VERSION}")

#Setting a Project name
SET(PROJECT_NAME "libhcffttest")
PROJECT (${PROJECT_NAME})
MESSAGE(STATUS "PROJECT NAME    : ${PROJECT_NAME}")


#Setting a variable for source files
SET (TESTSRCS
    fft_test.cpp  
    )

IF(WIN32)
  #include the additional libraries 
  #settings->c/c++->general->additional directories 
  SET(TESTSRCS ${TESTSRCS})
  ADD_DEFINITIONS(-D__LITTLE_ENDIAN__)
  ADD_DEFINITIONS(-DTARGET_OS_WIN32=1)
    
  IF(MSVC)
    ADD_DEFINITIONS(/W3)
    SET (DEBUG_PATH "../../Build/vc11-x86_64/Debug/")
    SET (RELEASE_PATH "../../Build/vc11-x86_64/Release/")
    include_directories( ../../source/)
    SET(CMAKE_CXX_STANDARD_LIBRARIES ${CMAKE_CXX_STANDARD_LIBRARIES} libhcfft.lib)
    IF(EXISTS "${DEBUG_PATH}/libhcfft.lib")
     LINK_DIRECTORIES (${DEBUG_PATH})
    ELSE(EXISTS "${DEBUG_PATH}/libhcfft.lib")
     LINK_DIRECTORIES (${RELEASE_PATH})
    ENDIF(EXISTS "${DEBUG_PATH}/libhcfft.lib")
    STRING( REPLACE ".cpp" "" testname ${TESTSRCS} )
    ADD_EXECUTABLE( ${testname} ${TESTSRCS})
  ELSE(MSVC)
    MESSAGE(FATAL_ERROR "TARGET IDE ISNT VISUAL STUDIO 11/12.... EXITING")
  ENDIF(MSVC)
ELSE(WIN32)

  # MCW CLAMP Specific. Version >= 0.3 is Must
  SET (CLANG_PATH "/opt/kalmar/bin/")
  IF(EXISTS "${CLANG_PATH}/clang++")
   SET (PREIX "${CLANG_PATH}")
   SET (CLANG_AMP "${CLANG_PATH}/clang++")
   SET (CLAMP_CONFIG "${CLANG_PATH}/clamp-config")
  ELSE(EXISTS "${CLANG_PATH}/clang++")
   SET (PREFIX "$ENV{MCWCPPAMPBUILD}")
   SET (CLANG_AMP "${PREFIX}/compiler/bin/clang++")
   SET (CLAMP_CONFIG "${PREFIX}/build/Release/bin/clamp-config")
  ENDIF(EXISTS "${CLANG_PATH}/clang++")

  IF(EXISTS "${CLANG_PATH}/clang++")
   execute_process(COMMAND ${CLAMP_CONFIG} --cxxflags
    OUTPUT_VARIABLE CLAMP_CXXFLAGS)
  ELSE(EXISTS "${CLANG_PATH}/clang++")
   execute_process(COMMAND ${CLAMP_CONFIG} --build --cxxflags
    OUTPUT_VARIABLE CLAMP_CXXFLAGS)
  ENDIF(EXISTS "${CLANG_PATH}/clang++")

  string(STRIP "${CLAMP_CXXFLAGS}" CLAMP_CXXFLAGS)
  set (CLAMP_CXXFLAGS "${CLAMP_CXXFLAGS}")

  IF(EXISTS "${CLANG_PATH}/clang++")
   execute_process(COMMAND ${CLAMP_CONFIG} --ldflags 
    OUTPUT_VARIABLE CLAMP_LDFLAGS)
  ELSE(EXISTS "${CLANG_PATH}/clang++")
   execute_process(COMMAND ${CLAMP_CONFIG} --build --ldflags 
    OUTPUT_VARIABLE CLAMP_LDFLAGS)
  ENDIF(EXISTS "${CLANG_PATH}/clang++")

  string(STRIP "${CLAMP_LDFLAGS}" CLAMP_LDFLAGS)
  set (CLAMP_LDFLAGS "${CLAMP_LDFLAGS}")

  SET (LINK "-lhcfft")
  SET_PROPERTY(SOURCE ${TESTSRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " ${CLAMP_CXXFLAGS}")
  STRING( REPLACE "_test.cpp" "" testname ${TESTSRCS} )
  ADD_EXECUTABLE( ${testname} ${TESTSRCS} )
  SET_PROPERTY(TARGET ${testname} APPEND_STRING PROPERTY LINK_FLAGS " ${CLAMP_LDFLAGS} ${LINK}")

ENDIF(WIN32)

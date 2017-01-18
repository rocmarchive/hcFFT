cmake_minimum_required(VERSION 2.8)

IF(${HIP_SHARED_OBJ} MATCHES "ON")
    MESSAGE(STATUS "HIPRNG UNIT TEST USING libhiprng.so")
    MESSAGE(STATUS "***********************************")
ELSE()
    MESSAGE(STATUS "HIPRNG UNIT TEST USING header containing inline static hiprng APIs")
    MESSAGE(STATUS "*******************************************************************")
ENDIF()

add_subdirectory(hipfft_transforms)
add_subdirectory(hipfft_Create_Destroy_Plan)

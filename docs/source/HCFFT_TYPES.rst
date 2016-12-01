############
HCFFT TYPES
############

Enumerations
^^^^^^^^^^^^

| enum hcfftResult_t {
| HCFFT_SUCCESS        = 0,
| HCFFT_INVALID_PLAN   = 1,
| HCFFT_ALLOC_FAILED   = 2,
| HCFFT_INVALID_TYPE   = 3,
| HCFFT_INVALID_VALUE  = 4,
| HCFFT_INTERNAL_ERROR = 5,
| HCFFT_EXEC_FAILED    = 6,
| HCFFT_SETUP_FAILED   = 7,
| HCFFT_INVALID_SIZE   = 8,
| HCFFT_UNALIGNED_DATA = 9,
| HCFFT_INCOMPLETE_PARAMETER_LIST = 10,
| HCFFT_INVALID_DEVICE = 11,
| HCFFT_PARSE_ERROR    = 12,
| HCFFT_NO_WORKSPACE   = 13 
| }
| enum hcfftType_t {
| HCFFT_R2C = 0x2a,
| HCFFT_C2R = 0x2c,
| HCFFT_C2C = 0x29,
| HCFFT_D2Z = 0x6a,
| HCFFT_Z2D = 0x6c,
| HCFFT_Z2Z = 0x69 
| }

| typedef float hcfftReal;
| typedef float_2 hcComplex;
| typedef double hcfftDoubleReal;
| typedef double_2 hcDoubleComplex;

Detailed Description
^^^^^^^^^^^^^^^^^^^^

HCFFT RESULT (hcfftResult_t)
------------------------------

| This enumeration is the set of HCFFT error codes.
+---------------------------------+------------------------------------------------------------------------------------+
| Enumerator                                                                                                           |
+=================================+====================================================================================+
| HCFFT_SUCCESS                   | The hcFFT operation was successful.                                                |
+---------------------------------+------------------------------------------------------------------------------------+    
| HCFFT_INVALID_PLAN              | hcFFT was passed an invalid plan handle.                                           |
+---------------------------------+------------------------------------------------------------------------------------+
| HCFFT_ALLOC_FAILED              | hcFFT failed to allocate GPU or CPU memory.                                        |
+---------------------------------+------------------------------------------------------------------------------------+
| HCFFT_INVALID_TYPE              | unsupported numerical value was passed to function.                                |
+---------------------------------+------------------------------------------------------------------------------------+
| HCFFT_INVALID_VALUE             | User specified an invalid pointer or parameter.                                    |
+---------------------------------+------------------------------------------------------------------------------------+
| HCFFT_INTERNAL_ERROR            | Driver or internal hcFFT library error.                                            |
+---------------------------------+------------------------------------------------------------------------------------+
| HCFFT_EXEC_FAILED               | Failed to execute an FFT on the GPU.                                               |
+---------------------------------+------------------------------------------------------------------------------------+
| HCFFT_SETUP_FAILED              | The hcFFT library failed to initialize.                                            |
+---------------------------------+------------------------------------------------------------------------------------+    
| HCFFT_INVALID_SIZE              | User specified an invalid transform size.                                          |
+---------------------------------+------------------------------------------------------------------------------------+
| HCFFT_UNALIGNED_DATA            | No longer used.                                                                    |
+---------------------------------+------------------------------------------------------------------------------------+
| HCFFT_INCOMPLETE_PARAMETER_LIST | Missing parameters in call.                                                        |
+---------------------------------+------------------------------------------------------------------------------------+
| HCFFT_INVALID_DEVICE            | Execution of a plan was on different GPU than plan creation.                       |
+---------------------------------+------------------------------------------------------------------------------------+
| HCFFT_PARSE_ERROR               | Internal plan database error.                                                      |
+---------------------------------+------------------------------------------------------------------------------------+
| HCFFT_NO_WORKSPACE              | No workspace has been provided prior to plan execution.                            |
+---------------------------------+------------------------------------------------------------------------------------+

|

HCFFT TYPE (hcfftType_t)
--------------------------

| types of transform data supported by hcFFT
+-------------+-------------------------------------------------------------------------------+
| Enumerator                                                                                  |
+=============+===============================================================================+
| HCFFT_R2C   | Real to complex (interleaved).                                                |
+-------------+-------------------------------------------------------------------------------+    
| HCFFT_C2R   | Complex (interleaved) to real.                                                |
+-------------+-------------------------------------------------------------------------------+
| HCFFT_C2C   | Complex to complex (interleaved).                                             |
+-------------+-------------------------------------------------------------------------------+    
| HCFFT_D2Z   | Double to double-complex (interleaved).                                       |
+-------------+-------------------------------------------------------------------------------+
| HCFFT_Z2D   | Double-complex (interleaved) to double.                                       |
+-------------+-------------------------------------------------------------------------------+    
| HCFFT_Z2Z   | Double-complex to double-complex (interleaved).                               |
+-------------+-------------------------------------------------------------------------------+

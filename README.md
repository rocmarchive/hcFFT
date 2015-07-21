# ** C++ AMP FFT ** #

##Introduction: ##

This repository hosts the C++ AMP implementation of FFT routines. The following are the sub-routines that are implemented

1. R2C : Transforms Real valued input in Time domain to Complex valued output in Frequency domain.
2. C2R : Transforms Complex valued input in Frequency domain to Real valued output in Real domain.


##Repository Structure: ##

##Prerequisites: ##
* **dGPU**:  AMD FirePro W9100 (FireGL V)
* **OS** : Ubuntu 14.04 LTS
* **Ubuntu Pack**: libc6-dev-i386
* **AMD APP SDK** : Ver 2.9.1 launched on 18/8/2014 from [here](http://developer.amd.com/tools-and-sdks/opencl-zone/amd-accelerated-parallel-processing-app-sdk/)
* **AMD Driver installer**: amd-driver-installer-15.20


##Building and set up:    
######Need to be a super user

(i)  ** C++ AMP Compiler installation**: Indepth details can be found [here](https://bitbucket.org/multicoreware/cppamp-driver-ng/wiki/Home)

Prepare a directory for work space.

   * mkdir mcw_cppamp

   * cd mcw_cppamp 
   
   * git clone https://bitbucket.org/multicoreware/cppamp-driver-ng.git src

   * git checkout master

Create a build directory and configure using CMake.

  * mkdir mcw_cppamp/build

  * cd mcw_cppamp/build

  * cmake ../src -DCMAKE_BUILD_TYPE=Release (The master branch expects the AMDAPP SDK in the path /opt/AMDAPP)

Build the whole system. This will build clang and other libraries that require one time build.

  * make [-j #] world           (# is the number of parallel builds. Generally it is # of CPU cores)

  * make                        (this builds llvm utilities)

Note that you might need to manually check updates from C++ AMP Compiler.
Please do the following and rebuild the Compiler if any update is available

```
#!python
 # check updates from C++AMP Compiler
 cd mcw_cppamp/src
 git fetch --all
 git checkout master

 # check updates from C++AMP Compiler's dependency
 cd mcw_cppamp/src/compiler/tools/clang
 git fetch --all
 git checkout master
```
Prior to building the library the following environment variables need to be set using export command

* AMDAPPSDKROOT=<path to AMD APP SDK>
* MCWCPPAMPROOT=<path to mcw_cppamp dir>

Steps to build AMPFFT:

   * git clone https://bitbucket.org/multicoreware/ampfft.git

   * cd ampfft

   For Linux :

     * cd Build/linux/
     * sh build.sh
     * make

   For 32-bit Windows : (It requires Visual Studio 12 version)

     * cd Build
     * cd vc11-x86
     * make-solutions.bat (This creates a Visual studio solution for ampfft Library)

   For 64-bit Windows : (It requires Visual Studio 12 version)

     * cd Build
     * cd vc11-x86_64
     * make-solutions.bat (This creates a Visual studio solution for ampfft Library)
# ** HCFFT ** #

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


## Installation Steps:

### A. C++ AMP Compiler Installation: 

Make sure the parent directory chosen is say ~/ or any other folder of your choice. Lets take ~/ as an example

  (a) Prepare a directory for work space

       * mkdir ~/mcw_cppamp35

       * cd ~/mcw_cppamp35

       * git clone https://bitbucket.org/multicoreware/cppamp-driver-ng-35.git src

       * cd ~/mcw_cppamp35/src/

       * git checkout 4fb5922

  (b) Create a build directory and configure using CMake.

       * mkdir ~/mcw_cppamp35/build

       * cd ~/mcw_cppamp35/build

       * export CLAMP_NOTILECHECK=ON

       * cmake ../src -DCMAKE_BUILD_TYPE=Release -DCXXAMP_ENABLE_BOLT=ON -DOPENCL_HEADER_DIR=<path to SDK's OpenCL headers> -DOPENCL_LIBRARY_DIR=<path to SDK's OpenCL library> 
  
       * For example, cmake ../src -DCMAKE_BUILD_TYPE=Release -DCXXAMP_ENABLE_BOLT=ON  -DOPENCL_HEADER_DIR=/opt/AMDAPPSDK-2.9.1/include/CL -DOPENCL_LIBRARY_DIR=/opt/AMDAPPSDK-2.9.1/lib/x86_64


  (c) Build AMP

       * cd ~/mcw_cppamp35/build

       * make [-j #] world && make          (# is the number of parallel builds. Generally it is # of CPU cores)

With this the C++ AMP Compiler installation is complete.

### B. HCFFT Installation

(i) Clone MCW HCFFT source codes

      * cd ~/
   
      * git clone https://bitbucket.org/multicoreware/hcfft.git 

      * cd ~/hcfft

(ii) Platform-specific build

(a) For Linux:

       * cd ~/hcfft/Build/linux
       
       * export MCWCPPAMPROOT=<path_to>/mcw_cppamp35/ (Here path_to points to parent folder of mcw_cppamp. ~/ in our case)

       * sh build.sh

       * make

(b)  For Windows: (Prerequisite: Visual Studio 12 version )

1. For 32 Bit:

     * cd Build/vc11-x86

     * make-solutions.bat (This creates a Visual studio solution for hcfft Library) 

 2. For 64-bit:

     * cd Build/vc11-x86_64

     * make-solutions.bat (This creates a Visual Studio solution for hcfft Library)


### C. Unit testing

1. FFT R2C and C2R Testing: 

     * export HCFFT_LIBRARY_PATH = ~/hcfft/Build/linux
     
     * export MCWCPPAMPROOT=~/mcw_cppamp35/
     
     * export LD_LIBRARY_PATH=$HCFFT_LIBRARY_PATH:$LD_LIBRARY_PATH
     
     * cd ~/hcfft/source/test/

     * export CLAMP_NOTILECHECK=ON
     
     * make
     
     * chmod +x run.sh

     * ./run.sh N1 N2
# ** HCFFT ** #

##Introduction: ##

This repository hosts the HCC based FFT Library, that targets GPU acceleration of FFT routines on AMD devices. To know what HCC compiler features, refer [here](https://bitbucket.org/multicoreware/hcc/wiki/Home).

The following are the sub-routines that are implemented

1. R2C : Transforms Real valued input in Time domain to Complex valued output in Frequency domain.
2. C2R : Transforms Complex valued input in Frequency domain to Real valued output in Real domain.

## Key Features: ##

* Support 1D, 2D and 3D Fast Fourier Transforms
* Support Out-Of-Place data storage
* Ability to Choose desired target accelerator
* Single and Double precision

##Prerequisites: ##

**A. Hardware Requirements:**

* CPU: mainstream brand, Better if with >=4 Cores Intel Haswell based CPU 
* System Memory >= 4GB (Better if >10GB for NN application over multiple GPUs)
* Hard Drive > 200GB (Better if SSD or NVMe driver  for NN application over multiple GPUs)
* Minimum GPU Memory (Global) > 2GB

**B. GPU SDK and driver Requirements:**

* AMD R9 Fury X, R9 Fur, R9 Nano
* AMD APU Kaveri or Carrizo

**C. System software requirements:**

* Ubuntu 14.04 trusty
* GCC 4.6 and later
* CPP 4.6 and later (come with GCC package)
* python 2.7 and later


**D. Tools and Misc Requirements:**

* git 1.9 and later
* cmake 2.6 and later (2.6 and 2.8 are tested)
* firewall off
* root privilege or user account in sudo group


**E. Ubuntu Packages requirements:**

* libc6-dev-i386
* liblapack-dev
* graphicsmagick


## Tested Environment so far: 

**A. Driver versions tested**  

* Boltzmann Early Release Driver 
* HSA driver

**B. GPU Cards tested:**

* Radeon R9 Nano
* Radeon R9 FuryX 
* Radeon R9 Fury 
* Kaveri and Carizo APU

**C. Desktop System Tested**

* Supermicro SYS-7048GR-TR  Tower 4 W9100 GPU
* ASUS X99-E WS motherboard with 4 AMD FirePro W9100
* Gigabyte GA-X79S 2 AMD FirePro W9100 GPU’s

**D. Server System Tested**

* Supermicro SYS 2028GR-THT  6 R9 NANO
* Supermicro SYS-1028GQ-TRT 4 R9 NANO
* Supermicro SYS-7048GR-TR Tower 4 R9 NANO


## Installation Steps:   

### A. HCC Compiler Installation: 

a) Download the compiler debian.

* Click [here](https://bitbucket.org/multicoreware/hcc/downloads/hcc-0.9.16041-0be508d-ff03947-5a1009a-Linux.deb)

   (or)

* via terminal: 

               wget https://bitbucket.org/multicoreware/hcc/downloads/hcc-0.9.16041-0be508d-ff03947-5a1009a-Linux.deb 


b) Install the compiler
 
      sudo dpkg -i hcc-0.9.16041-0be508d-ff03947-5a1009a-Linux.deb
      
### B. HCFFT Installation 
   
       * git clone https://bitbucket.org/multicoreware/hcfft.git 

       * cd ~/hcfft

       * export OPENCL_INCLUDE_PATH=/opt/AMDAPPSDK-x.y.z/include

       * export OPENCL_LIBRARY_PATH=/opt/AMDAPPSDK-x.y.z/lib/x86_64

       * export CLFFT_LIBRARY_PATH=/home/user/clFFT/build/library

       * export LD_LIBRARY_PATH=$CLFFT_LIBRARY_PATH:$OPENCL_LIBRARY_PATH:$LD_LIBRARY_PATH

       * ./install.sh test=OFF
         Where
           test=OFF    - Build library and tests
           test=ON     - Build library, tests and run test.sh

       
### C. Unit testing

### 1. Install clFFT library

     * git clone https://github.com/clMathLibraries/clFFT.git

     * cd clFFT

     * mkdir build && cd build

     * cmake ../src

     * make && make install

### 2. Testing:
    
a) Automated testing:

     * cd ~/hcfft/test/unit/
     
     * ./test.sh
     
b) Manual testing:

     * cd ~/hcfft/test/build/linux/bin/
     
     * choose the appropriate named binary

### 3. Examples:

     * cd ~/hcfft/build/lib/examples/bin/
     
     * choose the appropriate named binary
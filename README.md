# ** HCFFT ** #

##Introduction: ##

This repository hosts the HCC implementation of FFT routines. The following are the sub-routines that are implemented

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

### A. HCC Compiler Installation: 

**Install HCC compiler debian package:**

  Download the debian package from the link given below,
  
  [Compiler-Debians](https://multicorewareinc.egnyte.com/dl/TD5IwsNEx3)
  
  Install the package hcc-0.8.1544-a9f4d2f-ddba18d-Linux.deb 
  
  using the command,
  
    sudo dpkg -i <package_name>
      e.g. sudo dpkg -i  hcc-0.8.1544-a9f4d2f-ddba18d-Linux.deb 
      
  Note: 
      Ignore clamp-bolt, Bolt is not required for hcFFT.
    

### B. HCFFT Installation

(i) Clone MCW HCFFT source codes

      * cd ~/
   
      * git clone https://bitbucket.org/multicoreware/hcfft.git 

      * cd ~/hcfft

(ii) Platform-specific build

(a) For Linux:

       * sh install.sh
    

(b)  For Windows: (Prerequisite: Visual Studio 12 version )

1. For 32 Bit:

     * cd ~/hcfft/Build/vc11-x86

     * make-solutions.bat (This creates a Visual studio solution for hcfft Library) 

 2. For 64-bit:

     * cd ~/hcfft/Build/vc11-x86_64

     * make-solutions.bat (This creates a Visual Studio solution for hcfft Library)


### C. Unit testing

(a) For Linux:

1. FFT R2C and C2R Testing: 
     
     * cd ~/hcfft/source/test/build/linux/

     * sh build.sh
     
     * make
     
     * ./runme_ffttest.sh N1 N2
      where N1 and N2 are sizes.

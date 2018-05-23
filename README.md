# hcFFT has been deprecated and has been replaced by [rocFFT](https://github.com/ROCmSoftwarePlatform/rocFFT)

## A. Introduction: ##

This repository hosts the HCC based FFT Library, that targets GPU acceleration of FFT routines on AMD devices. To know what HCC compiler features, refer [here](https://github.com/RadeonOpenCompute/hcc).

The following are the sub-routines that are implemented

1. R2C : Transforms Real valued input in Time domain to Complex valued output in Frequency domain.
2. C2R : Transforms Complex valued input in Frequency domain to Real valued output in Real domain.
3. C2C : Transforms Complex valued input in Frequency domain to Complex valued output in Real domain or vice versa

To know more, go through the [Documentation](https://github.com/ROCmSoftwarePlatform/hcFFT/wiki)


## B. Key Features ##

* Support 1D, 2D and 3D Fast Fourier Transforms
* Supports R2C, C2R, C2C, D2Z, Z2D and Z2Z Transforms
* Support Out-Of-Place data storage
* Ability to Choose desired target accelerator
* Single and Double precision

## C. Prerequisites ##

* Refer Prerequisites section [here](https://github.com/ROCmSoftwarePlatform/hcFFT/wiki/Prerequisites)

## D. Tested Environment so far

* Refer Tested environments enumerated [here](https://github.com/ROCmSoftwarePlatform/hcFFT/wiki/Tested-Environments)


## E. Installation

* Follow installation steps as described [here](https://github.com/ROCmSoftwarePlatform/hcFFT/wiki/Installation)


## F. Unit testing

* Follow testing procedures as explained [here](https://github.com/ROCmSoftwarePlatform/hcFFT/wiki/Unit-testing)

## G. API reference

* The Specification of API's supported along with description  can be found [here](http://hcfft.readthedocs.org/en/latest/#hcfft-api-reference)

## H. Example Code

* Refer Examples section [here](https://github.com/ROCmSoftwarePlatform/hcRNG/wiki/Examples)

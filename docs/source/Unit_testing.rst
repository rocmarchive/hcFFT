******************
1.5. Unit testing
******************

1.5.1. Testing hcFFT against clFFT:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

a) Install clFFT library:

       ``git clone https://github.com/clMathLibraries/clFFT.git && cd clFFT``

       ``mkdir build && cd build``

       ``cmake ../src``

       ``make && sudo make install``

       ``export AMDAPPSDKROOT=/opt/AMDAPPSDK-x.y.z/``

b) Set Variables:

       ``export CLFFT_LIBRARY_PATH=/path/to/clFFT/build/library/``

       ``export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/hcFFT/library[libhcfft.so]``

c) Automated testing:

       ``cd ~/hcfft/``

       ``./build.sh --test=on``

d) Manual testing:

       ``cd ~/hcfft/build/test/src/bin/``

       choose the appropriate named binary

Here are some notes for performing manual testing:

|      N1 and N2 - Input sizes of 2D FFT

      * FFT

            ``./fft N1 N2``

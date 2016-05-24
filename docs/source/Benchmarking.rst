*****************
1.6. Benchmarking
*****************

1.6.1. Benchmarking hcFFT:
^^^^^^^^^^^^^^^^^^^^^^^^^^

a) Automated Benchmarking:

       ``cd ~/hcfft/``

       ``./build.sh --test=on --bench=on``

b) Manual Benchmarking:

       ``cd ~/hcfft/test/FFT_benchmark_Convolution_Networks``

       ``./runme_chronotimer.sh``

   It profiles hcFFT for input sizes in Input.txt and stores the output in Benchmark_fft.csv

1.6.2. Benchmarking hcFFT against clFFT:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

a) Manual Benchmarking:

       ``cd ~/hcfft/test/FFT_benchmark_Convolution_Networks/Comparison_tests``

       ``./run_clfftVShcfft.sh``

   It compares clFFT with hcFFT for input sizes in Input.txt and stores the output in Benchmark_clfftvshcfft.csv

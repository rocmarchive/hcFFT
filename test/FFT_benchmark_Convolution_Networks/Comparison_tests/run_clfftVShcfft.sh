#!/bin/bash -e

#This script is invoked to profile CLFFT & HCFFT

#CURRENT_WORK_DIRECTORY
CURRENTDIR=$PWD
hcfftPath=$CURRENTDIR/../../../
if [ ! -d "$hcfftPath" ]; then
  echo "Please run script[run_clfftVShcfft.sh] from hcfft/test/FFT_benchmark_Convolution_Networks/Comparison_tests/"
  exit
fi

export HCFFT_PATH=$hcfftPath
cd $CURRENTDIR

#Path to FFT executable
path2exe="$CURRENTDIR/../../../build/test/FFT_benchmark_Convolution_Networks/Comparison_tests/bin/clFFTvshcFFT-2D"
workingdir="$CURRENTDIR"

#Create Profile Data directory to store profile results
profDir="$workingdir/clfftvshcfftBenchData"
mkdir -p $profDir

echo -e "\n N1\t N2\t HCFFT R2C Avg Time(ms)\t CLFFT R2C Avg Time(ms)\t HCFFT C2R Avg Time(ms)\t CLFFT C2R Avg Time(ms)\t HCFFT C2C Avg Time(ms)\t CLFFT C2C Avg Time(ms)" >> $workingdir/Benchmark_clfftvshcfft.csv

#Start profiling fft
while read line; do
  N1value=$(echo $line | cut -f1 -d" " )
  N2value=$(echo $line | cut -f2 -d" " )
  datetime=$(date +%b-%d-%a_%H-%M-%S_)
  path2outdir="$profDir/$datetime$N1value$N2value"
  mkdir -p $path2outdir

#Check if executable exixts
 if [ -x $path2exe ]; then
   echo $path2exe $N1value $N2value
#Generate ATP file
   runcmd="$path2exe $N1value $N2value >> $path2outdir/output_$datetime.txt"
   echo $runcmd
   eval $runcmd
   filename="output_$datetime.txt"
   passarg=$path2outdir/$filename

#Store profile timings in CSV using python script
  if [ -f "$workingdir/extracttime_clfftvshcfft.py" ]; then
    python $workingdir/extracttime_clfftvshcfft.py $passarg $N1value $N2value 
  fi
  else
    echo $path2exe "doesnot exist" 
  fi

#Input file
done < $workingdir/Input.txt

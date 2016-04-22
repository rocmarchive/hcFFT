#!/bin/bash -e

#This script is invoked to profile FFT

#CURRENT_WORK_DIRECTORY
CURRENTDIR=$PWD
hcfftPath=$CURRENTDIR/../../
if [ ! -d "$hcfftPath" ]; then
  echo "Please run script[runme_chronotimer.sh] from hcfft/test/FFT_benchmark_Convolution_Networks"
  exit
fi

export HCFFT_PATH=$hcfftPath
cd $CURRENTDIR/../../build/test/ && cmake -DCMAKE_CXX_FLAGS=-fPIC $HCFFT_PATH/test/ 

set +e
mkdir -p $CURRENTDIR/../../build/test/src/bin/
mkdir -p $CURRENTDIR/../../build/test/unit/gtest/hcfft_1D_transform/bin/
mkdir -p $CURRENTDIR/../../build/test/unit/gtest/hcfft_2D_transform/bin/
mkdir -p $CURRENTDIR/../.../build/test/unit/gtest/hcfft_3D_transform/bin/
mkdir -p $CURRENTDIR/../../build/test/unit/gtest/hcfft_Create_Destroy_Plan/bin/
set -e

make

cd $CURRENTDIR

#Path to FFT executable
path2exe="$CURRENTDIR/../../build/test/src/bin/fft_timer"
workingdir="$CURRENTDIR"

#Create Profile Data directory to store profile results
profDir="$workingdir/fftbenchData"
mkdir -p $profDir

echo -e "\n N1\t N2\t R2C Avg Time(ms)\t C2R Avg Time(ms)" >> $workingdir/Benchmark_fft.csv

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
  if [ -f "$workingdir/extracttime_fft.py" ]; then
    python $workingdir/extracttime_fft.py $passarg $N1value $N2value 
  fi
  else
    echo $path2exe "doesnot exist" 
  fi

#Input file
done < $workingdir/Input.txt

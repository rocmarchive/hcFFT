# collecting data from the KernelSummary file and writing it to another file
import sys
import csv
import os

filename=str(sys.argv[1])
N1val=str(sys.argv[2])
N2val=str(sys.argv[3])
HCFFT_PATH=str(os.environ['HCFFT_PATH'])
inputfile=open(filename,"r")
out = csv.writer(open(HCFFT_PATH +"/test/FFT_benchmark_Convolution_Networks/Comparison_tests/Benchmark_clfftvshcfft.csv"  ,"a"), delimiter='\t',quoting=csv.QUOTE_NONE, quotechar='')
lines = inputfile.readlines()
hcr2cavgtime = lines[0].split(":")[1].split("\n")[0]
clr2cavgtime = lines[1].split(":")[1].split("\n")[0]
hcc2ravgtime = lines[2].split(":")[1].split("\n")[0]
clc2ravgtime = lines[3].split(":")[1].split("\n")[0]
hcc2cavgtime = lines[4].split(":")[1].split("\n")[0]
clc2cavgtime = lines[5].split(":")[1].split("\n")[0]
print hcr2cavgtime
print clr2cavgtime
print hcc2ravgtime
print clc2ravgtime
print hcc2cavgtime
print clc2cavgtime
vlist=[]
vlist = [N1val,N2val,hcr2cavgtime,clr2cavgtime,hcc2ravgtime,clc2ravgtime,hcc2cavgtime,clc2cavgtime]
out.writerow(vlist)
vlist = []

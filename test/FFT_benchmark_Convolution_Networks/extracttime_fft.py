# collecting data from the KernelSummary file and writing it to another file
from bs4 import BeautifulSoup
import sys
import csv
import os

filename=str(sys.argv[1])
N1val=str(sys.argv[2])
N2val=str(sys.argv[3])
HCFFT_PATH=str(os.environ['HCFFT_PATH'])
inputfile=open(filename,"r")
out = csv.writer(open(HCFFT_PATH +"/test/FFT_benchmark_Convolution_Networks/Benchmark_fft.csv","a"), delimiter='\t',quoting=csv.QUOTE_NONE, quotechar='')
lines = inputfile.readlines()
r2cavgtime = lines[0].split(":")[1].split("\n")[0]
c2ravgtime = lines[1].split(":")[1].split("\n")[0]
print r2cavgtime
print c2ravgtime
vlist=[]
vlist = [N1val,N2val,r2cavgtime,c2ravgtime]
out.writerow(vlist)
vlist = []

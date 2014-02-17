#!/bin/sh

echo "compiling " $1

if [ $# -gt 0 ]
then
	echo "using following options: " $2
fi

nvcc -o gpuknapsack $1 -arch=sm_11 src/gpuknapsack.cu
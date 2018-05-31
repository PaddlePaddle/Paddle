#!/bin/bash
set -e

# the number of process to run tests
NUM_PROC=6

# calculate and set the memory usage for each process
MEM_USAGE=$(printf "%.2f" `echo "scale=5; 1.0 / $NUM_PROC" | bc`)
export FLAGS_fraction_of_gpu_memory_to_use=$MEM_USAGE

# get the CUDA device count
CUDA_DEVICE_COUNT=$(nvidia-smi -L | wc -l)

for (( i = 0; i < $NUM_PROC; i++ )); do
    cuda_list=()
    for (( j = 0; j < $CUDA_DEVICE_COUNT; j++ )); do
        s=$[i+j]
        n=$[s%CUDA_DEVICE_COUNT]
        if [ $j -eq 0 ]; then
            cuda_list=("$n")
        else
            cuda_list="$cuda_list,$n"
        fi
    done
    echo $cuda_list
    # CUDA_VISIBLE_DEVICES http://acceleware.com/blog/cudavisibledevices-masking-gpus
    # ctest -I https://cmake.org/cmake/help/v3.0/manual/ctest.1.html?highlight=ctest
    env CUDA_VISIBLE_DEVICES=$cuda_list ctest -I $i,,$NUM_PROC --output-on-failure &
done
wait

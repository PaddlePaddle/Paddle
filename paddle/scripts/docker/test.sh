#!/bin/bash
set -xe

# the number of process to run tests
NUM_PROC=${CTEST_PARALLEL_LEVEL:-6}

# get the CUDA device count
CUDA_DEVICE_COUNT=$(nvidia-smi -L | wc -l)

# calculate and set the memory usage for each process
MEM_USAGE=$(printf "%.2f" `echo "scale=5; 0.9 * $CUDA_DEVICE_COUNT / $NUM_PROC" | bc`)
if (( `echo "$MEM_USAGE > 1.0"|bc -l` )); then
    MEM_USAGE=1.0
fi

export FLAGS_fraction_of_gpu_memory_to_use=$MEM_USAGE

echo "run ctest in parallel: $NUM_PROC"
pids=""
for (( i = 0; i < $NUM_PROC; i++ )); do
    cuda_list=()
    for (( j = 0; j < $CUDA_DEVICE_COUNT; j++ )); do
        s=$[i+j]
        n=$[s%CUDA_DEVICE_COUNT]
        if [ $j -eq 0 ]; then
            cuda_list=("$n")
        else
	    # only use up to two GPUs, the test speed decrease with
	    # more GPU because cuda init takes time. But we need at
	    # least two GPUs to test multiple GPU test cases.
	    if ! [[ $cuda_list = *,* ]]; then
		cuda_list="$cuda_list,$n"
	    fi
        fi
    done
    echo $cuda_list
    # CUDA_VISIBLE_DEVICES http://acceleware.com/blog/cudavisibledevices-masking-gpus
    # ctest -I https://cmake.org/cmake/help/v3.0/manual/ctest.1.html?highlight=ctest
    env CUDA_VISIBLE_DEVICES=$cuda_list ctest -I $i,,$NUM_PROC --output-on-failure &
    pids+=" $!"
done

failed_pids=""
for p in $pids; do
    if wait $p; then
        echo "ctest passed in process $p"
    else
        echo "ctest failed in process $p"
        failed_pids+=" $p"
    fi
done

if [ -z "$failed_pids" ]; then
    echo "all tests passed"
else
    echo "test(s) failed in process(es) $failed_pids"
    has_test_failed=true
fi

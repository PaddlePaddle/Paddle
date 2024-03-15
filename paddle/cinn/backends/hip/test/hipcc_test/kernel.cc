
#include <hip/hip_runtime.h>
#include "test_header.h"
#include "test_header1.h"
extern "C"
__global__ void saxpy_kernel(const real a, const realptr d_x, realptr d_y, const unsigned int size)
{
    const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(global_idx < size)
    {
        d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
    }
}
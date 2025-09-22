#pragma once

#include <vector>

#if defined(__NVCC__)
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "cutlass_matmul.cuh"
#include "math_function.h"
#include "profile.h"

using ap_bfloat16 = nv_bfloat16;
using ap_half = half;

using apStream_t = cudaStream_t;
#elif defined(__HIPCC__)
#include <hip/hip_runtime.h> 
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h> 

#include "ck_matmul.h"

using ap_bfloat16 = hip_bfloat16;
using ap_half = ck::half_t;

using apStream_t = hipStream_t;
#endif


template <typename ElementT,
          typename ElementComputeT,
          template <typename T>
          class VariadicFunctor,
          int AlignA,
          int AlignB,
          int ConfigId>
void MatmulAddVariadic(
    	const GemmEpilogueParams &params,
	const typename VariadicFunctor<ElementComputeT>::Arguments &variadic_args) {
#pragma once
#undef __HIP_PLATFORM_AMD__
#undef __HIP_PLATFORM_NVIDIA__
#define __HIP_PLATFORM_AMD__ 1
#undef __HIP_PLATFORM_HCC__
#undef __HIP_PLATFORM_NVCC__
#define __HIP_PLATFORM_HCC__ 1
#include_next <hip/hip_fp16.h>

// MIT License

// Copyright (c) 2025 DeepSeek

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include "paddle/fluid/distributed/collective/deep_ep/kernels/configs.cuh"

#ifndef SETUP_LAUNCH_CONFIG
#define SETUP_LAUNCH_CONFIG(num_sms, num_threads, stream) \
    cudaLaunchConfig_t cfg = {(num_sms), (num_threads), 0, stream, nullptr, 0}; \
    cudaLaunchAttribute attr[1]; \
    attr[0].id = cudaLaunchAttributeCooperative; \
    attr[0].val.cooperative = 1; \
    cfg.attrs = attr; \
    cfg.numAttrs = 1
#endif

#ifndef LAUNCH_KERNEL
#define LAUNCH_KERNEL(config, kernel, ...) CUDA_CHECK(cudaLaunchKernelEx(config, kernel, ##__VA_ARGS__))
#endif

#define SWITCH_RANKS(case_macro) \
    switch (num_ranks) { \
        case 2: case_macro(2); \
        case 4: case_macro(4); \
        case 8: case_macro(8); \
        default: EP_HOST_ASSERT(false and "Unsupported ranks"); \
    } while (false)

#define SWITCH_RDMA_RANKS(case_macro) \
    switch (num_ranks / NUM_MAX_NVL_PEERS) { \
        case 2: case_macro(2); \
        case 3: case_macro(3); \
        case 4: case_macro(4); \
        case 8: case_macro(8); \
        case 16: case_macro(16); \
        case 18: case_macro(18); \
        case 20: case_macro(20); \
        default: EP_HOST_ASSERT(false and "Unsupported RDMA ranks"); \
    } while (false)

#define SWITCH_RANKS_WITH_DTYPE(dtype, case_macro) \
    switch (num_ranks) { \
        case 2: case_macro(dtype, 2); \
        case 4: case_macro(dtype, 4); \
        case 8: case_macro(dtype, 8); \
        default: EP_HOST_ASSERT(false && "Unsupported ranks"); \
    } while (false)

#define SWITCH_TYPES(case_macro) \
    switch (type) { \
        case CUDA_R_16BF: case_macro(nv_bfloat16); \
        case CUDA_R_32F:  case_macro(float); \
        default: EP_HOST_ASSERT(false && "Unsupported type"); \
    } while (false)

#define SWITCH_HIDDEN(case_macro) \
    switch (hidden) { \
        case 2560: case_macro(2560); \
        case 5120: case_macro(5120); \
        case 7168: case_macro(7168); \
        default: EP_HOST_ASSERT(false && "Unsupported hidden"); \
    } while (false)

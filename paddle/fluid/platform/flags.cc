// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "gflags/gflags.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/cudnn_workspace_helper.h"
#endif

/**
 * NOTE(paddle-dev): This file is designed to define all public FLAGS.
 */

/* Paddle initialization related */
DEFINE_int32(paddle_num_threads, 1,
             "Number of threads for each paddle instance.");

/* Operator related */
DEFINE_bool(check_nan_inf, false,
            "Checking whether operator produce NAN/INF or not. It will be "
            "extremely slow so please use this flag wisely.");

/* CUDA related */
#ifdef PADDLE_WITH_CUDA
DEFINE_bool(
    enable_cublas_tensor_op_math, false,
    "The enable_cublas_tensor_op_math indicate whether to use Tensor Core, "
    "but it may loss precision. Currently, There are two CUDA libraries that"
    " use Tensor Cores, cuBLAS and cuDNN. cuBLAS uses Tensor Cores to speed up"
    " GEMM computations(the matrices must be either half precision or single "
    "precision); cuDNN uses Tensor Cores to speed up both convolutions(the "
    "input and output must be half precision) and recurrent neural networks "
    "(RNNs).");

DEFINE_string(selected_gpus, "",
              "A list of device ids separated by comma, like: 0,1,2,3. "
              "This option is useful when doing multi process training and "
              "each process have only one device (GPU). If you want to use "
              "all visible devices, set this to empty string. NOTE: the "
              "reason of doing this is that we want to use P2P communication"
              "between GPU devices, use CUDA_VISIBLE_DEVICES can only use"
              "share-memory only.");
#endif

/* CUDNN related */
#ifdef PADDLE_WITH_CUDA
DEFINE_bool(cudnn_deterministic, false,
            "Whether allow using an autotuning algorithm for convolution "
            "operator. The autotuning algorithm may be non-deterministic. If "
            "true, the algorithm is deterministic.");

DEFINE_uint64(conv_workspace_size_limit,
              paddle::platform::kDefaultConvWorkspaceSizeLimitMB,
              "cuDNN convolution workspace limit in MB unit.");

DEFINE_bool(cudnn_exhaustive_search, false,
            "Whether enable exhaustive search for cuDNN convolution or "
            "not, default is False.");

DEFINE_int64(cudnn_exhaustive_search_times, -1,
             "Exhaustive search times for cuDNN convolution, "
             "default is -1, not exhaustive search");

// CUDNN_BATCHNORM_SPATIAL_PERSISTENT in batchnorm. This mode can be faster in
// some tasks because an optimized path may be selected for CUDNN_DATA_FLOAT
// and CUDNN_DATA_HALF data types, compute capability 6.0 or higher. The
// reason we set it to false by default is that this mode may use scaled
// atomic integer reduction that may cause a numerical overflow for certain
// input data range.
DEFINE_bool(cudnn_batchnorm_spatial_persistent, false,
            "Whether enable CUDNN_BATCHNORM_SPATIAL_PERSISTENT mode for cudnn "
            "batch_norm, default is False.");
#endif

/* NCCL related */
#ifdef PADDLE_WITH_CUDA
// asynchronous nccl allreduce or synchronous issue:
// https://github.com/PaddlePaddle/Paddle/issues/15049
// If you want to change this default value, why?(gongwb)
DEFINE_bool(
    sync_nccl_allreduce, true,
    "If set true, will call `cudaStreamSynchronize(nccl_stream)`"
    "after allreduce, this mode can get better performance in some scenarios.");
#endif

/* Distributed related */
#ifdef PADDLE_WITH_DISTRIBUTE
DEFINE_int32(communicator_max_merge_var_num, 20,
             "max var num to merge and send");
DEFINE_int32(communicator_send_queue_size, 20,
             "queue size to recv gradient before send");
#endif

DEFINE_int32(dist_threadpool_size, 0,
             "number of threads used for distributed executed.");

/* Garbage collector related */
// Disable gc by default when inference library is built
#ifdef PADDLE_ON_INFERENCE
static const double kDefaultEagerDeleteTensorGB = -1;
#else
static const double kDefaultEagerDeleteTensorGB = 0;
#endif

DEFINE_double(
    eager_delete_tensor_gb, kDefaultEagerDeleteTensorGB,
    "Memory size threshold (GB) when the garbage collector clear tensors."
    "Disabled when this value is less than 0");

DEFINE_bool(fast_eager_deletion_mode, true,
            "Fast eager deletion mode. If enabled, memory would release "
            "immediately without waiting GPU kernel ends.");

DEFINE_double(memory_fraction_of_eager_deletion, 1.0,
              "Fraction of eager deletion. If less than 1.0, all variables in "
              "the program would be sorted according to its memory size, and "
              "only the FLAGS_memory_fraction_of_eager_deletion of the largest "
              "variables would be deleted.");

/* Allocator related */
DEFINE_string(allocator_strategy, "naive_best_fit",
              "The allocation strategy. naive_best_fit means the original best "
              "fit allocator of Fluid. "
              "auto_growth means the experimental auto-growth allocator. "
              "Enum in [naive_best_fit, auto_growth].");

DEFINE_double(fraction_of_cpu_memory_to_use, 1,
              "Default use 100% of CPU memory for PaddlePaddle,"
              "reserve the rest for page tables, etc");
DEFINE_uint64(initial_cpu_memory_in_mb, 500ul,
              "Initial CPU memory for PaddlePaddle, in MD unit.");

DEFINE_double(
    fraction_of_cuda_pinned_memory_to_use, 0.5,
    "Default use 50% of CPU memory as the pinned_memory for PaddlePaddle,"
    "reserve the rest for page tables, etc");

#ifdef PADDLE_WITH_CUDA
#ifndef _WIN32
constexpr static float fraction_of_gpu_memory_to_use = 0.92f;
#else
// fraction_of_gpu_memory_to_use cannot be too high on windows,
// since the win32 graphic sub-system can occupy some GPU memory
// which may lead to insufficient memory left for paddle
constexpr static float fraction_of_gpu_memory_to_use = 0.5f;
#endif

DEFINE_double(fraction_of_gpu_memory_to_use, fraction_of_gpu_memory_to_use,
              "Allocate a trunk of gpu memory that is this fraction of the "
              "total gpu memory size. Future memory usage will be allocated "
              "from the trunk. If the trunk doesn't have enough gpu memory, "
              "additional trunks of the same size will be requested from gpu "
              "until the gpu has no memory left for another trunk.");

DEFINE_uint64(
    initial_gpu_memory_in_mb, 0ul,
    "Allocate a trunk of gpu memory whose byte size is specified by "
    "the flag. Future memory usage will be allocated from the "
    "trunk. If the trunk doesn't have enough gpu memory, additional "
    "trunks of the gpu memory will be requested from gpu with size "
    "specified by FLAGS_reallocate_gpu_memory_in_mb until the gpu has "
    "no memory left for the additional trunk. Note: if you set this "
    "flag, the memory size set by "
    "FLAGS_fraction_of_gpu_memory_to_use will be overrided by this "
    "flag. If you don't set this flag, PaddlePaddle will use "
    "FLAGS_fraction_of_gpu_memory_to_use to allocate gpu memory");

DEFINE_uint64(reallocate_gpu_memory_in_mb, 0ul,
              "If this flag is set, Paddle will reallocate the gpu memory with "
              "size specified by this flag. Else Paddle will reallocate by "
              "FLAGS_fraction_of_gpu_memory_to_use");
#endif

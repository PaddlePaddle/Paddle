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

/**
 * Paddle initialization related FLAG
 * Name: FLAGS_paddle_num_threads
 * Since Version: 0.15.0
 * Value Range: int32, default=1
 * Example: FLAGS_paddle_num_threads=2, set the maximum thread number per
 * instance to 2
 * Note:
 */
DEFINE_int32(paddle_num_threads, 1,
             "Number of threads for each paddle instance.");

/**
 * Operator related FLAG
 * Name: FLAGS_check_nan_inf
 * Since Version: 0.13.0
 * Value Range: bool, default=false
 * Example:
 * Note: Used to debug. Checking whether operator produce NAN/INF or not.
 */
DEFINE_bool(check_nan_inf, false,
            "Checking whether operator produce NAN/INF or not. It will be "
            "extremely slow so please use this flag wisely.");

#ifdef PADDLE_WITH_CUDA

/**
 * CUDA related related FLAG
 * Name: FLAGS_enable_cublas_tensor_op_math
 * Since Version: 1.2.0
 * Value Range: bool, default=false
 * Example:
 * Note: whether to use Tensor Core, faster but it may loss precision.
 */
DEFINE_bool(
    enable_cublas_tensor_op_math, false,
    "The enable_cublas_tensor_op_math indicate whether to use Tensor Core, "
    "but it may loss precision. Currently, There are two CUDA libraries that"
    " use Tensor Cores, cuBLAS and cuDNN. cuBLAS uses Tensor Cores to speed up"
    " GEMM computations(the matrices must be either half precision or single "
    "precision); cuDNN uses Tensor Cores to speed up both convolutions(the "
    "input and output must be half precision) and recurrent neural networks "
    "(RNNs).");

/**
 * CUDA related FLAG
 * Name: FLAGS_selected_gpus
 * Since Version: 1.3.0
 * Value Range: integer list separated by comma, default empty list
 * Example: FLAGS_selected_gpus=0,1,2,3,4,5,6,7 to train or predict with 0~7 gpu
 * cards
 * Note: A list of device ids separated by comma, like: 0,1,2,3
 */
DEFINE_string(selected_gpus, "",
              "A list of device ids separated by comma, like: 0,1,2,3. "
              "This option is useful when doing multi process training and "
              "each process have only one device (GPU). If you want to use "
              "all visible devices, set this to empty string. NOTE: the "
              "reason of doing this is that we want to use P2P communication"
              "between GPU devices, use CUDA_VISIBLE_DEVICES can only use"
              "share-memory only.");
#endif

#ifdef PADDLE_WITH_CUDA

/**
 * CUDNN related FLAG
 * Name: FLAGS_cudnn_deterministic
 * Since Version: 0.13.0
 * Value Range: bool, default=false
 * Example:
 * Note: whether to use deterministic algorithm in cudnn.
 *       If true, it will slow down some operators such as conv and pooling.
 */
DEFINE_bool(cudnn_deterministic, false,
            "Whether allow using an autotuning algorithm for convolution "
            "operator. The autotuning algorithm may be non-deterministic. If "
            "true, the algorithm is deterministic.");

/**
 * CUDNN related FLAG
 * Name: FLAGS_conv_workspace_size_limit
 * Since Version: 0.13.0
 * Value Range: uint64, default=512 (MB)
 * Example:
 * Note: The internal function of cuDNN obtains the fastest matching algorithm
 *       within this memory limit. Usually, faster algorithms can be chosen in
 *       larger workspaces, but memory space can also be significantly
 * increased.
 *       Users need to balance memory and speed.
 */
DEFINE_uint64(conv_workspace_size_limit,
              paddle::platform::kDefaultConvWorkspaceSizeLimitMB,
              "cuDNN convolution workspace limit in MB unit.");

/**
 * CUDNN related FLAG
 * Name: FLAGS_cudnn_exhaustive_search
 * Since Version: 1.2.0
 * Value Range: bool, default=false
 * Example:
 * Note: Represents whether an exhaustive search method is used to
 *       select a convolution algorithm. There are two search methods in cuDNN,
 *       heuristic search and exhaustive search. Exhaustive search attempts
 *       all cuDNN algorithms to select the fastest. This method is very
 *       time-consuming, and the selected algorithm will be cached for a given
 *       layer specification. Once you change the layer specifications
 *       (such as batch size, feature map size), it will search again.
 */
DEFINE_bool(cudnn_exhaustive_search, false,
            "Whether enable exhaustive search for cuDNN convolution or "
            "not, default is False.");

/**
 * CUDNN related FLAG
 * Name: FLAGS_cudnn_exhaustive_search_times
 * Since Version:
 * Value Range:
 * Example:
 * Note: only used to predict for advanced developer
 */
DEFINE_int64(cudnn_exhaustive_search_times, -1,
             "Exhaustive search times for cuDNN convolution, "
             "default is -1, not exhaustive search");

/**
 * CUDNN related FLAG
 * Name: FLAGS_cudnn_batchnorm_spatial_persistent
 * Since Version: 1.4.0
 * Value Range: bool, default=false
 * Example:
 * Note: CUDNN_BATCHNORM_SPATIAL_PERSISTENT in batchnorm. This mode can be
 * faster in
 *       some tasks because an optimized path may be selected for
 * CUDNN_DATA_FLOAT
 *       and CUDNN_DATA_HALF data types, compute capability 6.0 or higher. The
 *       reason we set it to false by default is that this mode may use scaled
 *       atomic integer reduction that may cause a numerical overflow for
 * certain
 *       input data range.
 */
DEFINE_bool(cudnn_batchnorm_spatial_persistent, false,
            "Whether enable CUDNN_BATCHNORM_SPATIAL_PERSISTENT mode for cudnn "
            "batch_norm, default is False.");
#endif

#ifdef PADDLE_WITH_CUDA

/**
 * NCCL related FLAG
 * Name: FLAGS_enable_cublas_tensor_op_math
 * Since Version:
 * Value Range:
 * Example:
 * Note: asynchronous nccl allreduce or synchronous issue:
 *       https://github.com/PaddlePaddle/Paddle/issues/15049
 *       If you want to change this default value, why?(gongwb)
 */
DEFINE_bool(
    sync_nccl_allreduce, true,
    "If set true, will call `cudaStreamSynchronize(nccl_stream)`"
    "after allreduce, this mode can get better performance in some scenarios.");
#endif

#ifdef PADDLE_WITH_DISTRIBUTE
/**
 * Distributed related FLAG
 * Name: FLAGS_communicator_max_merge_var_num
 * Since Version: 1.5.0
 * Value Range: int32, default=20
 * Example:
 * Note: The maximum number of gradients to be merged into a gradient and
 *       sent through the communicator. The trainer puts all the gradients
 *       into the queue, and then the communicator takes the gradients out
 *       of the queue and sends them after merging.
 */
DEFINE_int32(communicator_max_merge_var_num, 20,
             "max var num to merge and send");

/**
 * Distributed related FLAG
 * Name: FLAGS_communicator_send_queue_size
 * Since Version: 1.5.0
 * Value Range: int32, default=20
 * Example:
 * Note: Size for each gradient queue. The trainer puts the gradient into
 *       the queue, and then the communicator takes it out of the queue and
 *       sends it out. When the communicator is slow, the queue may be full,
 *       and the trainer will be continuously blocked before the queue has
 *       space. It is used to avoid training much faster than communication,
 *       so that too many gradients are not sent out in time.
 */
DEFINE_int32(communicator_send_queue_size, 20,
             "queue size to recv gradient before send");
#endif

/**
 * Distributed related FLAG
 * Name: FLAGS_dist_threadpool_size
 * Since Version: 1.0.0
 * Value Range: int32, default=0
 * Example:
 * Note: Control the number of threads used for distributed modules.
 *       If it is not set, it is set to a hard thread.
 */
DEFINE_int32(dist_threadpool_size, 0,
             "number of threads used for distributed executed.");

/**
 * Garbage collector related FLAG
 * Name: FLAGS_eager_delete_tensor_gb
 * Since Version: 1.0.0
 * Value Range: double, default=kDefaultEagerDeleteTensorGB
 * Example: FLAGS_eager_delete_tensor_gb=0.0, Release memory garbage once it is
 * no longer used.
 *          FLAGS_eager_delete_tensor_gb=1.0, Release memory garbage when
 * garbage occupies 1.0GB of memory.
 *          FLAGS_eager_delete_tensor_gb=-1.0, Disable garbage collection
 * policy.
 * Note: Represents whether a garbage collection strategy is used to optimize
 * network memory usage.
 *       It is recommended that users set FLAGS_eager_delete_tensor_gb=0.0 to
 *       enable garbage collection strategy when training large networks.
 */
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

/**
 * Memory related FLAG
 * Name: FLAGS_fast_eager_deletion_mode
 * Since Version: 1.3.0
 * Value Range: bool, default=true
 * Example:
 * Note: Whether to use fast garbage collection strategy.
 *       If not set, the GPU memory is released at the end of the CUDA kernel.
 *       Otherwise, the GPU memory will be released before the CUDA kernel
 *       has finished, which will make the garbage collection strategy faster.
 *       Only works when garbage collection strategy is enabled.
 */
DEFINE_bool(fast_eager_deletion_mode, true,
            "Fast eager deletion mode. If enabled, memory would release "
            "immediately without waiting GPU kernel ends.");

/**
 * Memory related FLAG
 * Name: FLAGS_memory_fraction_of_eager_deletion
 * Since Version: 1.4
 * Value Range: double [0.0, 1.0], default=1.0
 * Example:
 * Note: The percentage of memory size of garbage collection policy
 *       to release variables.
 *       If FLAGS_memory_fraction_of_eager_deletion = 1.0,
 *       all temporary variables in the network will be released.
 *       If FLAGS_memory_fraction_of_eager_deletion = 0.0,
 *       no temporary variables in the network are released.
 *       If 0.0 < FLAGS_memory_fraction_of_eager_deletion < 1.0,
 *       all temporary variables will be sorted in descending order
 *       according to their memory size, and only variables with the
 *       largest FLAGS_memory_fraction_of_eager_deletion ratio will be released.
 *       The flag is only valid when running parallel data compilers.
 */
DEFINE_double(memory_fraction_of_eager_deletion, 1.0,
              "Fraction of eager deletion. If less than 1.0, all variables in "
              "the program would be sorted according to its memory size, and "
              "only the FLAGS_memory_fraction_of_eager_deletion of the largest "
              "variables would be deleted.");

/**
 * Allocator related FLAG
 * Name: FLAGS_allocator_strategy
 * Since Version: 1.2
 * Value Range: string, {naive_best_fit, auto_groth}, default=naive_best_fit
 * Example:
 * Note: Allocator policy for selecting Paddle Paddle.
 *       The allocator strategy is under development and the non-legacy
 *       allocator is not yet stable.
 */
DEFINE_string(allocator_strategy, "naive_best_fit",
              "The allocation strategy. naive_best_fit means the original best "
              "fit allocator of Fluid. "
              "auto_growth means the experimental auto-growth allocator. "
              "Enum in [naive_best_fit, auto_growth].");

/**
 * Memory related FLAG
 * Name: FLAGS_fraction_of_cpu_memory_to_use
 * Since Version: 0.12.0
 * Value Range: double, [0.0, 1.0], default=1
 * Example:
 * Note: Represents the proportion of allocated CPU memory blocks
 *       to the total memory size of the CPU. Future CPU memory usage
 *       will be allocated from this memory block. If the memory block does
 *       not have enough CUDA pinned memory, new memory blocks of the same
 *       size as the memory block will be allocated from the CUDA pinned
 *       request util the CPU does not have enough memory.
 */
DEFINE_double(fraction_of_cpu_memory_to_use, 1,
              "Default use 100% of CPU memory for PaddlePaddle,"
              "reserve the rest for page tables, etc");

/**
 * Memory related FLAG
 * Name: FLAGS_initial_cpu_memory_in_mb
 * Since Version: 0.14.0
 * Value Range: uint64, default=500 (MB)
 * Example:
 * Note: The CPU memory block size of the initial allocator in MB.
 *       The allocator takes the minimum values of
 *       FLAGS_initial_cpu_memory_in_mb and
 *       FLAGS_fraction_of_cpu_memory_to_use*(total physical memory)
 *       as memory block sizes.
 */
DEFINE_uint64(initial_cpu_memory_in_mb, 500ul,
              "Initial CPU memory for PaddlePaddle, in MD unit.");

/**
 * Memory related FLAG
 * Name: FLAGS_fraction_of_cuda_pinned_memory_to_use
 * Since Version: 0.12.0
 * Value Range: double, [0.0, 1.0], default=0.5
 * Example:
 * Note: Represents the proportion of allocated CUDA pinned memory blocks
 *       to the total memory size of the CPU. Future CUDA pinned memory usage
 *       will be allocated from this memory block. If the memory block does
 *       not have enough CPU memory, new memory blocks of the same
 *       size as the memory block will be allocated from the CPU
 *       request util the CPU does not have enough memory.
 */
DEFINE_double(
    fraction_of_cuda_pinned_memory_to_use, 0.5,
    "Default use 50% of CPU memory as the pinned_memory for PaddlePaddle,"
    "reserve the rest for page tables, etc");

#ifdef PADDLE_WITH_CUDA

/**
 * Memory related FLAG
 * Name: FLAGS_fraction_of_gpu_memory_to_use
 * Since Version: 1.2.0
 * Value Range: double, default=0.5 if win32, 0.92 else
 * Example:
 * Note: Represents the proportion of allocated memory blocks to the total
 * memory size
 *       of the GPU. Future memory usage will be allocated from this memory
 * block.
 *       If the memory block does not have enough GPU memory, new memory blocks
 * of
 *       the same size as the memory block will be allocated from the GPU
 * request
 *       until the GPU does not have enough memory.
 */

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

/**
 * Memory related FLAG
 * Name: FLAGS_initial_gpu_memory_in_mb
 * Since Version: 1.4.0
 * Value Range: uint64, default=0 (MB)
 * Example:
 * Note: Allocate a specified size of GPU memory block. Later memory usage
 *       will be allocated from that memory block. If the memory block does not
 *       have enough GPU memory, the memory block with the size
 *       FLAGS_reallocate_gpu_memory_in_mb will be requested from the GPU until
 *       the GPU has no remaining memory.
 */
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

/**
 * Memory related FLAG
 * Name: FLAGS_reallocate_gpu_memory_in_mb
 * Since Version: 1.4.0
 * Value Range: uint64, default=0 (MB)
 * Example:
 * Note: If the allocated GPU memory blocks are exhausted,
 *       additional GPU memory blocks are reallocated
 */
DEFINE_uint64(reallocate_gpu_memory_in_mb, 0ul,
              "If this flag is set, Paddle will reallocate the gpu memory with "
              "size specified by this flag. Else Paddle will reallocate by "
              "FLAGS_fraction_of_gpu_memory_to_use");

#endif

/**
 * Scope related FLAG
 * Name: local_exe_sub_scope_limit
 * Since Version: 1.6.0
 * Value Range: double, default=256 (MB)
 * Example:
 * Note:
 */
DEFINE_double(local_exe_sub_scope_limit, 256.0,  // MBytes
              "The memory up limit of sub-scopes of local execution scope for "
              "each CUDAPlace. If you don't need to limit the memory, "
              "you should set FLAGS_local_exe_sub_scope_limit=-1. "
              "The default value is 256 MBytes.");

/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/check_numerics_kernel.h"

#include "glog/logging.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/check_numerics_utils.h"
#include "paddle/phi/kernels/funcs/math_cuda_utils.h"

namespace phi {

static std::once_flag init_multi_gpu_op_var_map_flag;

// lazy init
static std::vector<
    std::unordered_map<std::string, phi::Allocator::AllocationPtr>>&
multi_op_var2gpu_str() {
  static std::vector<
      std::unordered_map<std::string, phi::Allocator::AllocationPtr>>
      _multi_op_var2gpu_str;
  return _multi_op_var2gpu_str;
}

static std::vector<std::mutex>& multi_op_var2gpu_str_mutex() {
  static std::vector<std::mutex> _multi_op_var2gpu_str_mutex;
  return _multi_op_var2gpu_str_mutex;
}

static void InitMultiGPUOpVarMap() {
  int dev_count = phi::backends::gpu::GetGPUDeviceCount();
  PADDLE_ENFORCE_GT(dev_count,
                    0,
                    phi::errors::NotFound(
                        "cuda device must > 0, now dev_count=%d", dev_count));

  // https://stackoverflow.com/questions/16465633/how-can-i-use-something-like-stdvectorstdmutex
  std::vector<std::unordered_map<std::string, phi::Allocator::AllocationPtr>>
      tmp_multi(dev_count);
  std::vector<std::mutex> tmp_multi_mutex(dev_count);

  multi_op_var2gpu_str().swap(tmp_multi);
  multi_op_var2gpu_str_mutex().swap(tmp_multi_mutex);
}

template <typename T>
__device__ __forceinline__ void PrintNanInfKernel(const T* value,
                                                  const size_t numel,
                                                  int print_num,
                                                  char* debug_info) {
  const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

  __shared__ unsigned int nan_count, inf_count, num_count;
  if (threadIdx.x == 0) nan_count = inf_count = num_count = 0;
  __syncthreads;

  for (size_t i = tid; i < numel; i += blockDim.x * gridDim.x) {
    unsigned int count = 0;
    if (isnan(value[i])) {
      count = atomicAdd(&nan_count, 1);
    } else if (isinf(value[i])) {
      count = atomicAdd(&inf_count, 1);
    } else {
      count = atomicAdd(&num_count, 1);
    }
    // for cuda, print in every block
    if (count < print_num) {
      printf("numel:%lu idx:%lu value:%f\n",
             static_cast<uint64_t>(numel),
             static_cast<uint64_t>(i),
             static_cast<float>(value[i]));
    }
  }
  __syncthreads;

#ifdef __HIPCC__
  if (true && hipThreadIdx_x == 0) {
    printf("In block %d, there has %u,%u,%u nan,inf,num\n",
           hipBlockIdx_x,
           nan_count,
           inf_count,
           num_count);
#else
  if (true && threadIdx.x == 0) {
    printf("In block %d, there has %u,%u,%u nan,inf,num\n",
           blockIdx.x,
           nan_count,
           inf_count,
           num_count);
#endif
    PADDLE_ENFORCE(false, "===ERROR: in %s find nan or inf===", debug_info);
  }
}

// Resnet 2gpus speed test, no check 270 images/s, this check 229 images/s
template <typename T>
__global__ void CheckNanInfKernel(const T* value,
                                  const size_t numel,
                                  int print_num,
                                  char* debug_info) {
  /// step 1, judge wheater has nan or inf
  __shared__ volatile int has_nan_inf;
  if (threadIdx.x == 0) has_nan_inf = false;
  __syncthreads();

  const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  T sum = static_cast<T>(0.0);
  // Todo(wangxi). simd speed up
  for (size_t i = tid; i < numel; i += blockDim.x * gridDim.x) {
    sum += (value[i] - value[i]);
  }

  if (isnan(sum) || isinf(sum)) has_nan_inf = true;
  __syncthreads();

  /// Note. different blocks may behave differently
  if (!has_nan_inf) return;

  PrintNanInfKernel(value, numel, print_num, debug_info);
}

template <typename T, int ReduceType>
__device__ T BlockReduce(T value) {
  __shared__ T shared_mem[1024];

  shared_mem[threadIdx.x] = value;
  __syncthreads();

  for (int stride = blockDim.x >> 1; stride > 0; stride = stride >> 1) {
    if (threadIdx.x < stride) {
      T value0 = shared_mem[threadIdx.x];
      T value1 = shared_mem[threadIdx.x + stride];
      T reduce_value;
      if (ReduceType == 0) {
        // max
        reduce_value = value0 > value1 ? value0 : value1;
      } else if (ReduceType == 1) {
        // min
        reduce_value = value0 < value1 ? value0 : value1;
      } else if (ReduceType == 2) {
        // sum
        reduce_value = value0 + value1;
      }
      shared_mem[threadIdx.x] = reduce_value;
    }

    if (stride > 16) {
      __syncthreads();
    }
  }

  __syncthreads();
  return shared_mem[0];
}

__device__ void BlockReduceNumNanInfAndWrite(const int64_t num_nan,
                                             const int64_t num_inf,
                                             const int64_t num_zero,
                                             int64_t offset,
                                             int64_t* num_nan_ptr,
                                             int64_t* num_inf_ptr,
                                             int64_t* num_zero_ptr) {
  int64_t block_num_nan = BlockReduce<int64_t, 2>(num_nan);
  int64_t block_num_inf = BlockReduce<int64_t, 2>(num_inf);
  int64_t block_num_zero = BlockReduce<int64_t, 2>(num_zero);

  if (threadIdx.x == 0) {
    num_nan_ptr[offset] = block_num_nan;
    num_inf_ptr[offset] = block_num_inf;
    num_zero_ptr[offset] = block_num_zero;
  }
}

template <
    typename T,
    std::enable_if_t<std::is_same<T, phi::dtype::complex<float>>::value ||
                         std::is_same<T, phi::dtype::complex<double>>::value,
                     bool> = true>
__device__ void BlockReduceMaxMinAndWrite(const T max_value,
                                          const T min_value,
                                          const T mean_value,
                                          int64_t offset,
                                          T* max_ptr,
                                          T* min_ptr,
                                          T* mean_ptr) {
  // TODO(Xreki): support complex
}

template <
    typename T,
    std::enable_if_t<!std::is_same<T, phi::dtype::complex<float>>::value &&
                         !std::is_same<T, phi::dtype::complex<double>>::value,
                     bool> = true>
__device__ void BlockReduceMaxMinAndWrite(const T max_value,
                                          const T min_value,
                                          const T mean_value,
                                          int64_t offset,
                                          T* max_ptr,
                                          T* min_ptr,
                                          T* mean_ptr) {
  if (max_ptr && min_ptr && mean_ptr) {
    __syncthreads();

    T block_max_value = phi::funcs::BlockReduceMax<T>(max_value, FINAL_MASK);
    T block_min_value = phi::funcs::BlockReduceMin<T>(min_value, FINAL_MASK);
    T block_mean_value = phi::funcs::BlockReduceSum<T>(mean_value, FINAL_MASK);

    if (threadIdx.x == 0) {
      max_ptr[offset] = block_max_value;
      min_ptr[offset] = block_min_value;
      mean_ptr[offset] = block_mean_value;
    }
  }
}

template <typename T, typename MT>
__global__ void FindNanInfAndBlockMaxMin(const T* value_ptr,
                                         const int64_t numel,
                                         int64_t* block_num_nan_ptr,
                                         int64_t* block_num_inf_ptr,
                                         int64_t* block_num_zero_ptr,
                                         MT* tensor_block_max_ptr,
                                         MT* tensor_block_min_ptr,
                                         MT* tensor_block_mean_ptr) {
  int64_t i = threadIdx.x + blockIdx.x * blockDim.x;

  int64_t num_nan = 0;
  int64_t num_inf = 0;
  int64_t num_zero = 0;

  MT max_value = static_cast<MT>(i < numel ? value_ptr[i] : value_ptr[0]);
  MT min_value = static_cast<MT>(i < numel ? value_ptr[i] : value_ptr[0]);
  MT mean_value = static_cast<MT>(0);
  for (; i < numel; i += blockDim.x * gridDim.x) {
    MT value = static_cast<MT>(value_ptr[i]);

    max_value = value > max_value ? value : max_value;
    min_value = value < min_value ? value : min_value;
    mean_value += value / static_cast<MT>(numel);

    if (isnan(value)) {
      num_nan += 1;
    } else if (isinf(value)) {
      num_inf += 1;
    }
    if (value == static_cast<MT>(0)) {
      num_zero += 1;
    }
  }

  BlockReduceNumNanInfAndWrite(num_nan,
                               num_inf,
                               num_zero,
                               blockIdx.x,
                               block_num_nan_ptr,
                               block_num_inf_ptr,
                               block_num_zero_ptr);

  BlockReduceMaxMinAndWrite<MT>(max_value,
                                min_value,
                                mean_value,
                                blockIdx.x,
                                tensor_block_max_ptr,
                                tensor_block_min_ptr,
                                tensor_block_mean_ptr);
}

template <typename T, typename MT>
__global__ void FindGlobalMaxMinAndPrint(const int64_t* block_num_nan_ptr,
                                         const int64_t* block_num_inf_ptr,
                                         const int64_t* block_num_zero_ptr,
                                         const MT* tensor_block_max_ptr,
                                         const MT* tensor_block_min_ptr,
                                         const MT* tensor_block_mean_ptr,
                                         const char* debug_info,
                                         int64_t numel,
                                         int64_t numel_max_min,
                                         int check_nan_inf_level,
                                         int64_t* stats_ptr,
                                         float* values_ptr) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    int64_t num_nan = 0;
    int64_t num_inf = 0;
    int64_t num_zero = 0;

    // numel_max_min <= 128
    for (int64_t i = 0; i < numel_max_min; ++i) {
      num_nan += block_num_nan_ptr[i];
      num_inf += block_num_inf_ptr[i];
      num_zero += block_num_zero_ptr[i];
    }

    MT max_value = static_cast<MT>(0);
    MT min_value = static_cast<MT>(0);
    MT mean_value = static_cast<MT>(0);
    if (tensor_block_max_ptr && tensor_block_min_ptr && tensor_block_mean_ptr) {
      max_value = tensor_block_max_ptr[0];
      min_value = tensor_block_min_ptr[0];
      mean_value = tensor_block_mean_ptr[0];

      // numel_max_min <= 128
      for (int64_t i = 1; i < numel_max_min; ++i) {
        MT tmp_max_value = tensor_block_max_ptr[i];
        MT tmp_min_value = tensor_block_min_ptr[i];
        MT tmp_mean_value = tensor_block_mean_ptr[i];

        max_value = tmp_max_value > max_value ? tmp_max_value : max_value;
        min_value = tmp_min_value < min_value ? tmp_min_value : min_value;
        mean_value += tmp_mean_value;
      }
      phi::funcs::SaveStatsAndValues<MT>(num_nan,
                                         num_inf,
                                         num_zero,
                                         max_value,
                                         min_value,
                                         mean_value,
                                         stats_ptr,
                                         values_ptr);
    }

    phi::funcs::PrintForDifferentLevel<T, MT>(debug_info,
                                              numel,
                                              num_nan,
                                              num_inf,
                                              num_zero,
                                              max_value,
                                              min_value,
                                              mean_value,
                                              check_nan_inf_level);
  }
}

template <typename T>
inline std::string GetHintString(const std::string& op_type,
                                 const std::string& var_name,
                                 const phi::Place& place,
                                 int dev_id = -1) {
  std::string op_var =
      phi::funcs::GetCpuHintString<T>(op_type, var_name, place, dev_id);
  PADDLE_ENFORCE_EQ(
      (dev_id >= 0 && dev_id < multi_op_var2gpu_str_mutex().size()),
      true,
      phi::errors::OutOfRange("GPU dev_id must >=0 and < dev_count=%d",
                              multi_op_var2gpu_str_mutex().size()));
  return op_var;
}

template <typename T>
static char* GetGpuHintStringPtr(const phi::GPUContext& ctx,
                                 const std::string& op_type,
                                 const std::string& var_name,
                                 int dev_id) {
  std::call_once(init_multi_gpu_op_var_map_flag, InitMultiGPUOpVarMap);

  std::string op_var =
      GetHintString<T>(op_type, var_name, ctx.GetPlace(), dev_id);
  char* gpu_str_ptr = nullptr;

  {
    auto& op_var2gpu_str_mutex = multi_op_var2gpu_str_mutex().at(dev_id);
    auto& op_var2gpu_str = multi_op_var2gpu_str().at(dev_id);

    std::lock_guard<std::mutex> guard(op_var2gpu_str_mutex);
    if (op_var2gpu_str.find(op_var) == op_var2gpu_str.end()) {  // insert
      auto gpu_str_tensor = phi::memory_utils::Alloc(
          ctx.GetPlace(),
          op_var.length() + 1,
          phi::Stream(reinterpret_cast<phi::StreamId>(ctx.stream())));
      gpu_str_ptr = reinterpret_cast<char*>(gpu_str_tensor->ptr());

      op_var2gpu_str.emplace(op_var, std::move(gpu_str_tensor));

      auto iter = op_var2gpu_str.find(op_var);
      PADDLE_ENFORCE_EQ(iter != op_var2gpu_str.end(),
                        true,
                        phi::errors::PreconditionNotMet(
                            "op_var=%s should successed insert into "
                            "op_var2gpu_str, but now failed",
                            op_var));

#ifdef __HIPCC__
      PADDLE_ENFORCE_GPU_SUCCESS(hipMemcpyAsync(gpu_str_ptr,
                                                iter->first.c_str(),
                                                op_var.length() + 1,
                                                hipMemcpyHostToDevice,
                                                ctx.stream()));
#else
      PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(gpu_str_ptr,
                                                 iter->first.c_str(),
                                                 op_var.length() + 1,
                                                 cudaMemcpyHostToDevice,
                                                 ctx.stream()));
#endif
    } else {  // get
      auto iter = op_var2gpu_str.find(op_var);
      PADDLE_ENFORCE_EQ(iter != op_var2gpu_str.end(),
                        true,
                        phi::errors::PreconditionNotMet(
                            "op_var=%s should be in the op_var2gpu_str, but "
                            "now can't find it",
                            op_var));
      gpu_str_ptr = reinterpret_cast<char*>(iter->second->ptr());
    }
  }
  return gpu_str_ptr;
}

template <typename T>
static void PrintStack(const phi::GPUContext& ctx,
                       const DenseTensor& stats,
                       const std::string& op_type,
                       const std::string& var_name,
                       int dev_id) {
  auto cpu_stats =
      phi::memory_utils::Alloc(phi::CPUPlace(), sizeof(int64_t) * 3);
  int64_t* cpu_stats_ptr = reinterpret_cast<int64_t*>(cpu_stats->ptr());
  phi::memory_utils::Copy(phi::CPUPlace(),
                          cpu_stats_ptr,
                          stats.place(),
                          stats.data(),
                          3 * sizeof(int64_t),
                          ctx.stream());
  ctx.Wait();
  if (cpu_stats_ptr[0] > 0 || cpu_stats_ptr[1] > 0) {
    const std::string debug_info =
        GetHintString<T>(op_type, var_name, stats.place(), dev_id);
    phi::funcs::PrintAndThrowError(debug_info.c_str(),
                                   cpu_stats_ptr[0],
                                   cpu_stats_ptr[1],
                                   cpu_stats_ptr[2]);
  }
}

template <typename T, typename MT>
static void WriteToOutputDir(const phi::GPUContext& ctx,
                             const DenseTensor& tensor,
                             const DenseTensor& stats,
                             const DenseTensor& values,
                             const std::string& op_type,
                             const std::string& var_name,
                             const std::string& output_dir,
                             const int check_nan_inf_level) {
  // Copy stats and values from GPU to CPU.
  phi::DenseTensor cpu_stats;
  cpu_stats.Resize({static_cast<int64_t>(3)});
  phi::Copy(ctx, stats, phi::CPUPlace(), false, &cpu_stats);

  phi::DenseTensor cpu_values;
  cpu_values.Resize({static_cast<int64_t>(3)});
  phi::Copy(ctx, values, phi::CPUPlace(), false, &cpu_values);
  ctx.Wait();

  int dev_id = tensor.place().device;
  const std::string debug_info =
      GetHintString<T>(op_type, var_name, tensor.place(), dev_id);
  std::string log_name = "gpu." + std::to_string(dev_id);
  int64_t* cpu_stats_ptr = cpu_stats.data<int64_t>();
  float* cpu_values_ptr = cpu_values.data<float>();
  phi::funcs::WriteToFileForDifferentLevel<T, MT>(debug_info.c_str(),
                                                  tensor.numel(),
                                                  cpu_stats_ptr[0],
                                                  cpu_stats_ptr[1],
                                                  cpu_stats_ptr[2],
                                                  cpu_values_ptr[0],
                                                  cpu_values_ptr[1],
                                                  cpu_values_ptr[2],
                                                  check_nan_inf_level,
                                                  log_name,
                                                  output_dir);
}

template <typename T, typename Context>
void CheckNumericsKernel(const Context& ctx,
                         const DenseTensor& tensor,
                         const std::string& op_type,
                         const std::string& var_name,
                         const int check_nan_inf_level,
                         const int stack_height_limit,
                         const std::string& output_dir,
                         DenseTensor* stats,
                         DenseTensor* values) {
  int dev_id = tensor.place().device;
  VLOG(6) << "op_type=" << op_type << ", var_name=" << var_name
          << ", dev_id=gpu:" << dev_id << ", numel=" << tensor.numel()
          << ", stack_height_limit=" << stack_height_limit
          << ", output_dir=" << output_dir;

  if (tensor.numel() <= 0) return;

  // Print to the standard output.
  char* gpu_str_ptr = GetGpuHintStringPtr<T>(ctx, op_type, var_name, dev_id);

  const size_t threads = 1024;
  size_t blocks =
      std::min(static_cast<size_t>(128),
               static_cast<size_t>((tensor.numel() + threads - 1) / threads));
#ifdef __HIPCC__
  int print_num = 3;

  hipLaunchKernelGGL(CheckNanInfKernel,
                     dim3(blocks),
                     dim3(threads),
                     0,
                     ctx.stream(),
                     tensor.data<T>(),
                     tensor.numel(),
                     print_num,
                     gpu_str_ptr);
#else
  using MT = typename phi::dtype::MPTypeTrait<T>::Type;

  int64_t numel_max_min = blocks;

  phi::DenseTensor block_num_nan_inf_zero;
  block_num_nan_inf_zero.Resize({static_cast<int64_t>(3 * numel_max_min)});
  int64_t* block_num_nan_ptr =
      ctx.template Alloc<int64_t>(&block_num_nan_inf_zero);
  int64_t* block_num_inf_ptr = block_num_nan_ptr + numel_max_min;
  int64_t* block_num_zero_ptr = block_num_inf_ptr + numel_max_min;

  phi::DenseTensor tensor_block_max_min;
  tensor_block_max_min.Resize({static_cast<int64_t>(3 * numel_max_min)});
  MT* tensor_block_max_ptr = ctx.template Alloc<MT>(&tensor_block_max_min);
  MT* tensor_block_min_ptr = tensor_block_max_ptr + numel_max_min;
  MT* tensor_block_mean_ptr = tensor_block_max_ptr + 2 * numel_max_min;

  FindNanInfAndBlockMaxMin<T, MT>
      <<<blocks, threads, 0, ctx.stream()>>>(tensor.data<T>(),
                                             tensor.numel(),
                                             block_num_nan_ptr,
                                             block_num_inf_ptr,
                                             block_num_zero_ptr,
                                             tensor_block_max_ptr,
                                             tensor_block_min_ptr,
                                             tensor_block_mean_ptr);

  // stats stores the checking result of num_nan, num_inf and num_zero.
  stats->Resize({static_cast<int64_t>(3)});
  int64_t* stats_ptr = ctx.template Alloc<int64_t>(stats);

  // values stores the max_value, min_value and mean_value.
  values->Resize({static_cast<int64_t>(3)});
  float* values_ptr = ctx.template Alloc<float>(values);

  FindGlobalMaxMinAndPrint<T, MT>
      <<<1, 1, 0, ctx.stream()>>>(block_num_nan_ptr,
                                  block_num_inf_ptr,
                                  block_num_zero_ptr,
                                  tensor_block_max_ptr,
                                  tensor_block_min_ptr,
                                  tensor_block_mean_ptr,
                                  gpu_str_ptr,
                                  tensor.numel(),
                                  numel_max_min,
                                  check_nan_inf_level,
                                  stats_ptr,
                                  values_ptr);

  if (output_dir.size() > 0) {
    // Write log to output_dir.
    WriteToOutputDir<T, MT>(ctx,
                            tensor,
                            *stats,
                            *values,
                            op_type,
                            var_name,
                            output_dir,
                            check_nan_inf_level);
  }

  if (check_nan_inf_level == 0 && stack_height_limit > 0) {
    PrintStack<T>(ctx, *stats, op_type, var_name, dev_id);
  }
#endif
}

}  // namespace phi

PD_REGISTER_KERNEL(check_numerics,
                   GPU,
                   ALL_LAYOUT,
                   phi::CheckNumericsKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

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

#include "paddle/fluid/framework/details/nan_inf_utils_detail.h"
#include "paddle/fluid/framework/details/nan_inf_utils.h"

#include <algorithm>
#include <unordered_map>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/kernels/funcs/math_cuda_utils.h"

DECLARE_bool(check_tensor_max_min);

namespace paddle {
namespace framework {
namespace details {

static std::once_flag init_multi_gpu_op_var_map_flag;

// lazy init
static std::vector<std::unordered_map<std::string, memory::AllocationPtr>>&
multi_op_var2gpu_str() {
  static std::vector<std::unordered_map<std::string, memory::AllocationPtr>>
      _multi_op_var2gpu_str;
  return _multi_op_var2gpu_str;
}

static std::vector<std::mutex>& multi_op_var2gpu_str_mutex() {
  static std::vector<std::mutex> _multi_op_var2gpu_str_mutex;
  return _multi_op_var2gpu_str_mutex;
}

static void InitMultiGPUOpVarMap() {
  int dev_count = platform::GetGPUDeviceCount();
  PADDLE_ENFORCE_GT(dev_count,
                    0,
                    platform::errors::NotFound(
                        "cuda device must > 0, now dev_count=%d", dev_count));

  // https://stackoverflow.com/questions/16465633/how-can-i-use-something-like-stdvectorstdmutex
  std::vector<std::unordered_map<std::string, memory::AllocationPtr>> tmp_multi(
      dev_count);
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

template <typename T, int ReduceType>
__device__ T BlockReduce(T value) {
  __shared__ T shared_mem[1024];

  shared_mem[threadIdx.x] = value;
  __syncthreads();

  if (threadIdx.x == 0) {
    T reduce_value = shared_mem[0];
    for (int i = 1; i < blockDim.x; ++i) {
      if (ReduceType == 0) {
        // max
        reduce_value =
            reduce_value > shared_mem[i] ? reduce_value : shared_mem[i];
      } else if (ReduceType == 1) {
        // min
        reduce_value =
            reduce_value < shared_mem[i] ? reduce_value : shared_mem[i];
      } else if (ReduceType == 2) {
        // sum
        reduce_value += shared_mem[i];
      }
    }
    return reduce_value;
  } else {
    return static_cast<T>(0);
  }
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

    // T block_max_value = phi::funcs::blockReduceMax<T>(max_value, FINAL_MASK);
    // T block_min_value = phi::funcs::blockReduceMin<T>(min_value, FINAL_MASK);
    // T block_mean_value = phi::funcs::blockReduceSum<T>(mean_value,
    // FINAL_MASK);
    T block_max_value = BlockReduce<T, 0>(max_value);
    T block_min_value = BlockReduce<T, 1>(min_value);
    T block_mean_value = BlockReduce<T, 2>(mean_value);

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
                                         int* found_nan_inf_ptr,
                                         MT* tensor_block_max_ptr,
                                         MT* tensor_block_min_ptr,
                                         MT* tensor_block_mean_ptr) {
  bool has_nan = false;
  bool has_inf = false;

  int64_t i = threadIdx.x + blockIdx.x * blockDim.x;

  MT max_value = static_cast<MT>(i < numel ? value_ptr[i] : value_ptr[0]);
  MT min_value = static_cast<MT>(i < numel ? value_ptr[i] : value_ptr[0]);
  MT mean_value = static_cast<MT>(0);
  for (; i < numel; i += blockDim.x * gridDim.x) {
    MT value = static_cast<MT>(value_ptr[i]);

    max_value = value > max_value ? value : max_value;
    min_value = value < min_value ? value : min_value;
    mean_value += value;

    if (isnan(value)) {
      has_nan = true;
    }
    if (isinf(value)) {
      has_inf = true;
    }

    if (has_nan || has_inf) {
      if (!tensor_block_max_ptr && !tensor_block_min_ptr &&
          !tensor_block_mean_ptr) {
        break;
      }
    }
  }
  if (has_nan) {
    found_nan_inf_ptr[0] = 1;
  }
  if (has_inf) {
    found_nan_inf_ptr[1] = 1;
  }

  BlockReduceMaxMinAndWrite<MT>(max_value,
                                min_value,
                                mean_value,
                                blockIdx.x,
                                tensor_block_max_ptr,
                                tensor_block_min_ptr,
                                tensor_block_mean_ptr);
}

template <typename T>
__global__ void FindGlobalMaxMinAndPrint(const int* found_nan_inf_ptr,
                                         const T* tensor_block_max_ptr,
                                         const T* tensor_block_min_ptr,
                                         const T* tensor_block_mean_ptr,
                                         const char* debug_info,
                                         int64_t numel,
                                         int64_t numel_max_min,
                                         bool check_tensor_max_min) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    int has_nan = found_nan_inf_ptr[0];
    int has_inf = found_nan_inf_ptr[1];

    T max_value = static_cast<T>(0);
    T min_value = static_cast<T>(0);
    T mean_value = static_cast<T>(0);
    if (tensor_block_max_ptr && tensor_block_min_ptr && tensor_block_mean_ptr) {
      max_value = tensor_block_max_ptr[0];
      min_value = tensor_block_min_ptr[0];
      mean_value = tensor_block_mean_ptr[0];
      // numel_max_min <= 128
      for (int64_t i = 1; i < numel_max_min; ++i) {
        T tmp_max_value = tensor_block_max_ptr[i];
        T tmp_min_value = tensor_block_min_ptr[i];
        T tmp_mean_value = tensor_block_mean_ptr[i];

        max_value = tmp_max_value > max_value ? tmp_max_value : max_value;
        min_value = tmp_min_value < min_value ? tmp_min_value : min_value;
        mean_value += tmp_mean_value;
      }

      mean_value = mean_value / static_cast<T>(numel);
    }

    // PADDLE_ENFORCE(found_nan_inf == 0, "===ERROR: in %s find nan or inf===",
    // debug_info);
    if (has_nan || has_inf) {
      printf(
          "===[PRECISION] [ERROR] in %s, numel=%ld, find_nan=%d, find_inf=%d, "
          "max=%e, "
          "min=%e, mean=%e===\n",
          debug_info,
          numel,
          has_nan,
          has_inf,
          static_cast<float>(max_value),
          static_cast<float>(min_value),
          static_cast<float>(mean_value));
    } else if (check_tensor_max_min) {
      printf("[PRECISION] in %s, numel=%ld, max=%e, min=%e, mean=%e\n",
             debug_info,
             numel,
             static_cast<float>(max_value),
             static_cast<float>(min_value),
             static_cast<float>(mean_value));
    }
  }
}

template <>
template <typename T>
void TensorCheckerVisitor<phi::GPUContext>::apply(
    typename std::enable_if<
        std::is_floating_point<T>::value ||
        std::is_same<T, ::paddle::platform::complex<float>>::value ||
        std::is_same<T, ::paddle::platform::complex<double>>::value>::type*)
    const {
  //  int print_num = 3;

  auto* dev_ctx = reinterpret_cast<phi::GPUContext*>(
      platform::DeviceContextPool::Instance().Get(tensor_.place()));
  int dev_id = tensor_.place().device;
  PADDLE_ENFORCE_EQ(
      (dev_id >= 0 && dev_id < multi_op_var2gpu_str_mutex().size()),
      true,
      platform::errors::OutOfRange("GPU dev_id must >=0 and < dev_count=%d",
                                   multi_op_var2gpu_str_mutex().size()));

  std::string dtype_str = DataTypeToString(DataTypeTrait<T>::DataType());
  if (dtype_str == "::paddle::platform::float16") {
    dtype_str = "float16";
  }
  std::string op_var = "[op=" + op_type_ + "] [tensor=" + var_name_ +
                       "] [dtype=" + dtype_str + "]";
  char* gpu_str_ptr = NULL;

  {
    auto& op_var2gpu_str_mutex = multi_op_var2gpu_str_mutex().at(dev_id);
    auto& op_var2gpu_str = multi_op_var2gpu_str().at(dev_id);

    std::lock_guard<std::mutex> guard(op_var2gpu_str_mutex);
    if (op_var2gpu_str.find(op_var) == op_var2gpu_str.end()) {  // insert
      auto gpu_str_tensor = paddle::memory::Alloc(
          dev_ctx->GetPlace(),
          op_var.length() + 1,
          phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx->stream())));
      gpu_str_ptr = reinterpret_cast<char*>(gpu_str_tensor->ptr());

      op_var2gpu_str.emplace(op_var, std::move(gpu_str_tensor));

      auto iter = op_var2gpu_str.find(op_var);
      PADDLE_ENFORCE_EQ(iter != op_var2gpu_str.end(),
                        true,
                        platform::errors::PreconditionNotMet(
                            "op_var=%s should successed insert into "
                            "op_var2gpu_str, but now failed",
                            op_var));

#ifdef __HIPCC__
      PADDLE_ENFORCE_GPU_SUCCESS(hipMemcpyAsync(gpu_str_ptr,
                                                iter->first.c_str(),
                                                op_var.length() + 1,
                                                hipMemcpyHostToDevice,
                                                dev_ctx->stream()));
#else
      PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(gpu_str_ptr,
                                                 iter->first.c_str(),
                                                 op_var.length() + 1,
                                                 cudaMemcpyHostToDevice,
                                                 dev_ctx->stream()));
#endif
    } else {  // get
      auto iter = op_var2gpu_str.find(op_var);
      PADDLE_ENFORCE_EQ(iter != op_var2gpu_str.end(),
                        true,
                        platform::errors::PreconditionNotMet(
                            "op_var=%s should be in the op_var2gpu_str, but "
                            "now can't find it",
                            op_var));
      gpu_str_ptr = reinterpret_cast<char*>(iter->second->ptr());
    }
  }

#ifdef __HIPCC__
  // HIP will throw GPU memory access fault if threads > 256
  const size_t threads = 256;
#else
  const size_t threads = 1024;
#endif
  size_t blocks =
      std::min(static_cast<size_t>(128),
               static_cast<size_t>((tensor_.numel() + threads - 1) / threads));
#ifdef __HIPCC__
  hipLaunchKernelGGL(CheckNanInfKernel,
                     dim3(blocks),
                     dim3(threads),
                     0,
                     dev_ctx->stream(),
                     tensor_.data<T>(),
                     tensor_.numel(),
                     print_num,
                     gpu_str_ptr);
#else
  // CheckNanInfKernel<<<blocks, threads, 0, dev_ctx->stream()>>>(
  //     tensor_.data<T>(), tensor_.numel(), print_num, gpu_str_ptr);

  using MT = typename phi::dtype::MPTypeTrait<T>::Type;

  phi::DenseTensor found_nan_inf;
  found_nan_inf.Resize({2});
  int* found_nan_inf_ptr = found_nan_inf.mutable_data<int>(tensor_.place());
  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemsetAsync(
      found_nan_inf_ptr, 0, 2 * sizeof(int), dev_ctx->stream()));

  int64_t numel_max_min = blocks;

  phi::DenseTensor tensor_block_max;
  tensor_block_max.Resize({static_cast<int64_t>(numel_max_min)});
  MT* tensor_block_max_ptr = tensor_block_max.mutable_data<MT>(tensor_.place());

  phi::DenseTensor tensor_block_min;
  tensor_block_min.Resize({static_cast<int64_t>(numel_max_min)});
  MT* tensor_block_min_ptr = tensor_block_min.mutable_data<MT>(tensor_.place());

  phi::DenseTensor tensor_block_mean;
  tensor_block_mean.Resize({static_cast<int64_t>(numel_max_min)});
  MT* tensor_block_mean_ptr =
      tensor_block_mean.mutable_data<MT>(tensor_.place());

  FindNanInfAndBlockMaxMin<T, MT>
      <<<blocks, threads, 0, dev_ctx->stream()>>>(tensor_.data<T>(),
                                                  tensor_.numel(),
                                                  found_nan_inf_ptr,
                                                  tensor_block_max_ptr,
                                                  tensor_block_min_ptr,
                                                  tensor_block_mean_ptr);

  bool check_tensor_max_min = FLAGS_check_tensor_max_min;
  FindGlobalMaxMinAndPrint<MT>
      <<<1, 1, 0, dev_ctx->stream()>>>(found_nan_inf_ptr,
                                       tensor_block_max_ptr,
                                       tensor_block_min_ptr,
                                       tensor_block_mean_ptr,
                                       gpu_str_ptr,
                                       tensor_.numel(),
                                       numel_max_min,
                                       check_tensor_max_min);
#endif
}

template <>
void tensor_check<phi::GPUContext>(const std::string& op_type,
                                   const std::string& var_name,
                                   const phi::DenseTensor& tensor,
                                   const platform::Place& place) {
  std::call_once(init_multi_gpu_op_var_map_flag, InitMultiGPUOpVarMap);

  TensorCheckerVisitor<phi::GPUContext> vistor(
      op_type, var_name, tensor, place);
  VisitDataType(framework::TransToProtoVarType(tensor.dtype()), vistor);
}

}  // namespace details
}  // namespace framework
}  // namespace paddle

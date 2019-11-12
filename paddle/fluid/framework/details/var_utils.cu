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

#include "paddle/fluid/framework/details/var_utils.h"
#include "paddle/fluid/framework/details/var_utils_detail.h"

#include <algorithm>

namespace paddle {
namespace framework {
namespace details {

// Resnet 2gpus speed test, no check 270 images/s, this check 223 images/s
template <typename T>
__global__ void CheckNanInfKernel(const T* value, const size_t numel,
                                  int print_num, char* debug_info) {
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

  /// notify, different blocks may behave differently
  if (!has_nan_inf) return;

  /// step 2, has nan or inf, print part of value
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
      printf("numel:%lu idx:%lu value:%f\n", static_cast<uint64_t>(numel),
             static_cast<uint64_t>(i), static_cast<float>(value[i]));
    }
  }

  if (true && threadIdx.x == 0) {
    printf("In block %d, there has %u,%u,%u nan,inf,num\n", blockIdx.x,
           nan_count, inf_count, num_count);
    PADDLE_ENFORCE(!has_nan_inf, "===ERROR: in %s find nan or inf===",
                   debug_info);
  }
}

template <>
template <typename T>
void TensorCheckerVisitor<platform::CUDADeviceContext>::apply(
    typename std::enable_if<std::is_floating_point<T>::value>::type*) const {
  int print_num = 3;

  auto* dev_ctx = reinterpret_cast<platform::CUDADeviceContext*>(
      platform::DeviceContextPool::Instance().Get(tensor_.place()));

  std::string debug_str = "[op=" + op_type_ + "] [tensor=" + var_name_ + "]";
  auto debug_tensor = paddle::memory::Alloc(*dev_ctx, debug_str.length() + 1);
  char* debug_ptr = reinterpret_cast<char*>(debug_tensor->ptr());

  PADDLE_ENFORCE_CUDA_SUCCESS(
      cudaMemcpyAsync(debug_ptr, debug_str.c_str(), debug_str.length() + 1,
                      cudaMemcpyHostToDevice, dev_ctx->stream()));

  const size_t threads = 1024;
  size_t blocks = std::min(128ul, (tensor_.numel() + threads - 1) / threads);
  CheckNanInfKernel<<<blocks, threads, 0, dev_ctx->stream()>>>(
      tensor_.data<T>(), tensor_.numel(), print_num, debug_ptr);
}

template <>
void tensor_check<platform::CUDADeviceContext>(const std::string& op_type,
                                               const std::string& var_name,
                                               const framework::Tensor& tensor,
                                               const platform::Place& place) {
  TensorCheckerVisitor<platform::CUDADeviceContext> vistor(op_type, var_name,
                                                           tensor, place);
  VisitDataType(tensor.type(), vistor);
}

}  // namespace details
}  // namespace framework
}  // namespace paddle

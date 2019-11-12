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

#include <algorithm>

#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/platform/cuda_device_function.h"
#include "paddle/fluid/platform/for_range.h"

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

template <typename T>
void CheckNanInf(const T* value, const size_t numel, int print_num,
                 const std::string& op_type, const std::string& var_name) {
  T sum = static_cast<T>(0.0);
#pragma omp parallel for reduction(+ : sum)
  for (int i = 0; i < numel; ++i) {
    sum += (value[i] - value[i]);
  }

  if (std::isnan(sum) || std::isinf(sum)) {
    // cpu print all value
    for (int i = 0; i < numel; ++i) {
      printf("idx:%u value:%f\n", i, value[i]);
    }
    PADDLE_ENFORCE_EQ(1, 0,
                      "===ERROR: in [op=%s] [tensor=%s] find nan or inf===",
                      op_type, var_name);
  }
}

template <typename DeviceContext>
struct CheckNanInfTool {
  template <typename T>
  void run(const std::string& op_type, const std::string& var_name,
           const framework::Tensor& tensor, const platform::Place& place,
           int print_num,
           typename std::enable_if<std::is_integral<T>::value>::type* = 0);

  template <typename T>
  void run(
      const std::string& op_type, const std::string& var_name,
      const framework::Tensor& tensor, const platform::Place& place,
      int print_num,
      typename std::enable_if<std::is_floating_point<T>::value>::type* = 0);
};

template <typename DeviceContext>
template <typename T>
void CheckNanInfTool<DeviceContext>::run(
    const std::string& op_type, const std::string& var_name,
    const framework::Tensor& tensor, const platform::Place& place,
    int print_num, typename std::enable_if<std::is_integral<T>::value>::type*) {
  VLOG(10) << var_name << " need not to check, it's type is not float point";
}

template <>
template <typename T>
void CheckNanInfTool<platform::CUDADeviceContext>::run(
    const std::string& op_type, const std::string& var_name,
    const framework::Tensor& tensor, const platform::Place& place,
    int print_num,
    typename std::enable_if<std::is_floating_point<T>::value>::type*) {
  auto* dev_ctx = reinterpret_cast<platform::CUDADeviceContext*>(
      platform::DeviceContextPool::Instance().Get(tensor.place()));

  std::string debug_str = "[op=" + op_type + "] [tensor=" + var_name + "]";
  auto debug_tensor = paddle::memory::Alloc(*dev_ctx, debug_str.length() + 1);
  char* debug_ptr = reinterpret_cast<char*>(debug_tensor->ptr());

  PADDLE_ENFORCE_CUDA_SUCCESS(
      cudaMemcpyAsync(debug_ptr, debug_str.c_str(), debug_str.length() + 1,
                      cudaMemcpyHostToDevice, dev_ctx->stream()));

  const size_t threads = 1024;
  size_t blocks = std::min(128ul, (tensor.numel() + threads - 1) / threads);
  CheckNanInfKernel<<<blocks, threads, 0, dev_ctx->stream()>>>(
      tensor.data<T>(), tensor.numel(), print_num, debug_ptr);
}

template <>
template <typename T>
void CheckNanInfTool<platform::CPUDeviceContext>::run(
    const std::string& op_type, const std::string& var_name,
    const framework::Tensor& tensor, const platform::Place& place,
    int print_num,
    typename std::enable_if<std::is_floating_point<T>::value>::type*) {
  platform::DeviceContextPool::Instance().Get(tensor.place());

  CheckNanInf(tensor.data<T>(), tensor.numel(), print_num, op_type, var_name);
}

struct TensorCheckerVisitor {
  TensorCheckerVisitor(const std::string& op_type, const std::string& var_name,
                       const framework::Tensor& tensor,
                       const platform::Place& place)
      : op_type_(op_type),
        var_name_(var_name),
        tensor_(tensor),
        place_(place) {}

  template <typename T>
  void apply() const {
    int print_num = 3;

    if (platform::is_gpu_place(tensor_.place())) {
#ifdef PADDLE_WITH_CUDA
      CheckNanInfTool<platform::CUDADeviceContext> tools;
      tools.run<T>(op_type_, var_name_, tensor_, place_, print_num);
#else
      PADDLE_THROW("PaddlePaddle should compile with GPU.");
#endif
      return;
    }

    CheckNanInfTool<platform::CPUDeviceContext> tools;
    tools.run<T>(op_type_, var_name_, tensor_, place_, print_num);
  }

  std::string op_type_;
  std::string var_name_;
  const framework::Tensor& tensor_;
  const platform::Place& place_;
};

void EnforceNoNanOrInf(const std::string& op_type,
                       const framework::Scope& scope,
                       const std::string& var_name,
                       const platform::Place& place) {
  auto* var = scope.FindVar(var_name);
  PADDLE_ENFORCE_NOT_NULL(var, "can't find var:%s", var_name);

  const Tensor* tensor{nullptr};
  if (var->IsType<framework::LoDTensor>()) {
    tensor = &var->Get<framework::LoDTensor>();
  } else if (var->IsType<framework::SelectedRows>()) {
    tensor = &var->Get<framework::SelectedRows>().value();
  } else {
    VLOG(10) << var_name << " var_name need not to check";
    return;
  }

  if (tensor->memory_size() == 0) {
    VLOG(10) << var_name << " var_name need not to check, size == 0";
    return;
  }

  VLOG(10) << "begin check " << op_type << " var_name:" << var_name
           << ", place:" << tensor->place() << ", numel:" << tensor->numel();

  TensorCheckerVisitor vistor(op_type, var_name, *tensor, place);
  VisitDataType(tensor->type(), vistor);
}

}  // namespace details
}  // namespace framework
}  // namespace paddle

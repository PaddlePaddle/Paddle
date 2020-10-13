// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <cuda_runtime.h>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/allclose_op.h"

namespace paddle {
namespace operators {

template <typename T>
struct GetTensorValue<platform::CUDADeviceContext, T> {
  T operator()(const framework::Tensor& tensor) const {
    const T* data = tensor.data<T>();
    T value;
    cudaMemcpy(&value, data, sizeof(T), cudaMemcpyDeviceToHost);
    return value;
  }
};

template <typename T>
__global__ void AllcloseCUDAKernel(const T* in_a, const T* in_b,
                                   const double rtol, const double atol,
                                   bool equal_nan, bool* out_data) {
  int tid = threadIdx.x;
  const T a = in_a[tid], b = in_b[tid];
  bool val;
  __shared__ bool val_;
  if (tid == 0) {
    val_ = true;
  }
  __syncthreads();
  if (isnan(a) || isnan(b)) {
    val = equal_nan && isnan(a) == isnan(b);
  } else {
    T left = (a > b ? a - b : b - a);
    T right = atol + (b > 0 ? rtol * b : (-rtol) * b);
    T dif = (left > right ? left - right : right - left);
    val = a == b || left <= right || dif <= 1e-15;
  }
  atomicAnd(reinterpret_cast<int*>(&val_), static_cast<int>(val));
  __syncthreads();
  if (tid == 0) {
    *out_data = static_cast<bool>(val_);
  }
}

template <typename T>
struct AllcloseFunctor<platform::CUDADeviceContext, T> {
  void operator()(const platform::CUDADeviceContext& ctx,
                  const framework::Tensor& in, const framework::Tensor& other,
                  const double rtol, const double atol, bool equal_nan,
                  framework::Tensor* output) {
    auto in_dims = in.numel();
    auto other_dims = other.numel();

    PADDLE_ENFORCE_EQ(in_dims == other_dims, true,
                      platform::errors::InvalidArgument(
                          "Dims of input(a) and dims of other(b) should"
                          "be equal, but received the dims of input is : %d ,"
                          "received the dims of other is :%d. ",
                          in_dims, other_dims));
    const T* in_data = in.data<T>();
    const T* other_data = other.data<T>();
    bool* out_data = output->mutable_data<bool>(ctx.GetPlace());
    int grid = 1;
    int block = in_dims;

    AllcloseCUDAKernel<T><<<1, block, 0, ctx.stream()>>>(
        in_data, other_data, rtol, atol, equal_nan, out_data);
  }
};

// template struct AllcloseFunctor<platform::CUDADeviceContext, float>;
template struct AllcloseFunctor<platform::CUDADeviceContext, double>;

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CUDA = paddle::platform::CUDADeviceContext;
REGISTER_OP_CUDA_KERNEL(allclose, ops::AllcloseKernel<CUDA, float>,
                        ops::AllcloseKernel<CUDA, double>);

// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/isclose_op.h"

namespace paddle {
namespace operators {

template <typename T>
struct GetTensorValue<platform::CUDADeviceContext, T> {
  T operator()(const platform::CUDADeviceContext& dev_ctx,
               const framework::Tensor& tensor) const {
    const T* data = tensor.data<T>();
    T value;
    const auto gpu_place = dev_ctx.GetPlace();
    memory::Copy(platform::CPUPlace(), &value, gpu_place, data, sizeof(T),
                 dev_ctx.stream());
    return value;
  }
};

template <typename T>
__global__ void IscloseCUDAKernel(const T* in_data, const T* other_data,
                                  const double rtol, const double atol,
                                  bool equal_nan, int num, bool* out_data) {
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
  bool val;
  for (int i = idx; i < num; i += blockDim.x * gridDim.x) {
    const T a = in_data[i], b = other_data[i];
    if (isnan(a) || isnan(b)) {
      val = equal_nan && isnan(a) == isnan(b);
    } else {
      T left = (a > b ? a - b : b - a);
      T right = atol + (b > 0 ? rtol * b : (-rtol) * b);
      T diff = (left > right ? left - right : right - left);
      val = a == b || left <= right || diff <= 1e-15;
    }
    out_data[i] = val;
    // if (!val) *out_data = false;
  }
}

template <typename T>
struct IscloseFunctor<platform::CUDADeviceContext, T> {
  void operator()(const platform::CUDADeviceContext& dev_ctx,
                  const framework::Tensor& in, const framework::Tensor& other,
                  const double rtol, const double atol, bool equal_nan,
                  framework::Tensor* output) {
    int num = in.numel();
    const T* in_data = in.data<T>();
    const T* other_data = other.data<T>();
    bool* out_data = output->mutable_data<bool>(dev_ctx.GetPlace());
    int block = 1024;
    int grid = (block - 1 + num) / block;
    grid = (grid > block) ? block : grid;
#ifdef PADDLE_WITH_HIP
    hipMemset(out_data, true, num * sizeof(bool));
#else
    cudaMemset(out_data, true, num * sizeof(bool));
#endif
    IscloseCUDAKernel<T><<<grid, block, 0, dev_ctx.stream()>>>(
        in_data, other_data, rtol, atol, equal_nan, num, out_data);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CUDA = paddle::platform::CUDADeviceContext;
REGISTER_OP_CUDA_KERNEL(isclose, ops::IscloseKernel<CUDA, float>,
                        ops::IscloseKernel<CUDA, double>);

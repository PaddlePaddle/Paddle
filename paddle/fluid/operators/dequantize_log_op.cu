/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/dequantize_log_op.h"
#include "paddle/fluid/operators/math.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/pten/core/hostdevice.h"

namespace paddle {
namespace operators {

template <typename T>
__global__ void KeDequantize(const T* in, const float* dict, int num,
                             float* out) {
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < num) {
    if (in[idx] < 0) {
      out[idx] = -dict[in[idx] + 128];
    } else {
      out[idx] = dict[in[idx]];
    }
  }
}

template <typename T>
struct DequantizeFunctor<platform::CUDADeviceContext, T> {
  void operator()(const platform::CUDADeviceContext& dev_ctx,
                  const framework::Tensor* in, const framework::Tensor* dict,
                  framework::Tensor* out) {
    const T* in_data = in->data<T>();
    const float* dict_data = dict->data<float>();
    float* out_data = out->mutable_data<float>(dev_ctx.GetPlace());

    int num = in->numel();
    int block = 512;
    int grid = (num + block - 1) / block;

    KeDequantize<T><<<grid, block, 0, dev_ctx.stream()>>>(in_data, dict_data,
                                                          num, out_data);
  }
};

template struct DequantizeFunctor<platform::CUDADeviceContext, int8_t>;

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CUDA = paddle::platform::CUDADeviceContext;
REGISTER_OP_CUDA_KERNEL(dequantize_log, ops::DequantizeLogKernel<CUDA, int8_t>);

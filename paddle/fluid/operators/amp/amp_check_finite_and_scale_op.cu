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

#include <cuda.h>
#include "paddle/fluid/operators/amp/amp_check_finite_and_scale_op.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {

template <typename T>
__global__ void AmpCheckFiniteAndScale(const T* in, const T* scale, int num,
                                       int* found_inf, T* out) {
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx < num) {
    if (!isfinite(in[idx])) {
      *found_inf = 1;
    }
    out[idx] = *found_inf ? in[idx] : in[idx] * scale[0];
  }
}

template <typename T>
class AmpCheckFiniteAndScaleKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const {
    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    const auto xs = ctx.MultiInput<framework::Tensor>("X");
    const auto* scale = ctx.Input<framework::Tensor>("Scale");
    auto outs = ctx.MultiOutput<framework::Tensor>("Out");
    auto* found_inf = ctx.Output<framework::Tensor>("FoundInfinite");

    const T* scale_data = scale->data<T>();
    int* found_inf_data = found_inf->mutable_data<int>(dev_ctx.GetPlace());
    cudaMemset(found_inf_data, false, found_inf->numel() * sizeof(bool));

    for (size_t i = 0; i < xs.size(); ++i) {
      const auto* x = xs[i];
      auto* out = outs[i];
      const T* x_data = x->data<T>();
      T* out_data = out->mutable_data<T>(dev_ctx.GetPlace());

      int num = x->numel();
      int block = 512;
      int grid = (num + block - 1) / block;
      VLOG(3) << "launch kernel";
      AmpCheckFiniteAndScale<T><<<grid, block, 0, dev_ctx.stream()>>>(
          x_data, scale_data, num, found_inf_data, out_data);
      VLOG(3) << "finish kernel";
    }
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    amp_check_finite_and_scale,
    ops::AmpCheckFiniteAndScaleKernel<paddle::platform::CUDADeviceContext,
                                      float>,
    ops::AmpCheckFiniteAndScaleKernel<paddle::platform::CUDADeviceContext,
                                      double>);

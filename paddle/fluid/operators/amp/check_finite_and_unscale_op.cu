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

#include "paddle/fluid/operators/amp/check_finite_and_unscale_op.h"

namespace paddle {
namespace operators {

template <typename T>
__global__ void InverseAndMemset(const T* s, T* o, bool* found_inf) {
  *o = Inverse<T>(*s);
  *found_inf = false;
}

template <typename T>
__global__ void CheckFiniteAndUnscale(const T* in, const T* scale, int num,
                                      bool* found_inf, T* out) {
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx < num) {
    T val = in[idx] * (*scale);
    out[idx] = val;
    if (!isfinite(val)) {
      *found_inf = true;
    }
  }
}

template <typename T>
class CheckFiniteAndUnscaleGpuKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const {
    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    const auto xs = ctx.MultiInput<framework::Tensor>("X");
    const auto* scale = ctx.Input<framework::Tensor>("Scale");
    auto outs = ctx.MultiOutput<framework::Tensor>("Out");
    auto* found_inf = ctx.Output<framework::Tensor>("FoundInfinite");

    const T* scale_data = scale->data<T>();
    bool* found_inf_data = found_inf->mutable_data<bool>(dev_ctx.GetPlace());

    framework::Tensor inverse_scale =
        ctx.AllocateTmpTensor<T, platform::CUDADeviceContext>({1}, dev_ctx);
    T* inverse_scale_v = inverse_scale.template data<T>();

    InverseAndMemset<T><<<1, 1, 0, dev_ctx.stream()>>>(
        scale_data, inverse_scale_v, found_inf_data);

    for (size_t i = 0; i < xs.size(); ++i) {
      const auto* x = xs[i];
      auto* out = outs[i];
      const T* x_data = x->data<T>();
      T* out_data = out->mutable_data<T>(dev_ctx.GetPlace());

      int num = x->numel();
      int block = 1024;
      int grid = (num + block - 1) / block;
      VLOG(3) << "launch kernel";
      CheckFiniteAndUnscale<T><<<grid, block, 0, dev_ctx.stream()>>>(
          x_data, inverse_scale_v, num, found_inf_data, out_data);
      VLOG(3) << "finish kernel";
    }
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(check_finite_and_unscale,
                        ops::CheckFiniteAndUnscaleGpuKernel<float>,
                        ops::CheckFiniteAndUnscaleGpuKernel<double>);

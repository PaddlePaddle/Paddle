/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/cast_op.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {
template <typename InT, typename OutT>
__global__ void DoCastKernel(const InT* in, const int n, OutT* out) {
  int bid = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = bid; i < n; i += blockDim.x * gridDim.x) {
    out[i] = static_cast<OutT>(in[i]);
  }
}

template <typename InT>
struct CastOpFunctor<platform::CUDADeviceContext, InT> {
  const framework::Tensor* in_;
  framework::Tensor* out_;
  const platform::CUDADeviceContext& ctx_;
  CastOpFunctor(const framework::Tensor* in, framework::Tensor* out,
                const platform::CUDADeviceContext& ctx)
      : in_(in), out_(out), ctx_(ctx) {}

  template <typename OutT>
  void apply() const {
    auto* in = in_->data<InT>();
    auto num = in_->numel();
    auto* out = out_->mutable_data<OutT>(ctx_.GetPlace());
    int block = 1024;
    int grid = (block - 1 + num) / block;
    DoCastKernel<InT, OutT><<<grid, block, 0, ctx_.stream()>>>(in, num, out);
  }
};

}  // namespace operators
}  // namespace paddle

template <typename T>
using CastOpKernel =
    paddle::operators::CastOpKernel<paddle::platform::CUDADeviceContext, T>;

REGISTER_OP_CUDA_KERNEL(cast, CastOpKernel<float>, CastOpKernel<double>,
                        CastOpKernel<int>, CastOpKernel<int64_t>,
                        CastOpKernel<bool>, CastOpKernel<uint8_t>,
                        CastOpKernel<paddle::platform::float16>);

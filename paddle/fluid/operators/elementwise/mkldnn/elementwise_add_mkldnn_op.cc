// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/elementwise/mkldnn/elementwise_mkldnn_op.h"

namespace paddle {
namespace framework {
class ExecutionContext;
}  // namespace framework
namespace platform {
class CPUDeviceContext;
struct CPUPlace;
}  // namespace platform
}  // namespace paddle

namespace paddle {
namespace operators {
template <typename T>
class EltwiseAddMKLDNNGradKernel : public ElemwiseGradKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    ElemwiseGradKernel<T>::Compute(ctx);
    using Tensor = framework::Tensor;

    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<Tensor>(framework::GradVarName("Y"));

    auto set_mkldnn_format = [](Tensor* in, const Tensor* out) {
      in->set_layout(DataLayout::kMKLDNN);
      in->set_format(out->format());
    };

    // TODO(jczaja): Double check if vcopy works for blocked data
    auto blas = math::GetBlas<paddle::platform::CPUDeviceContext, T>(ctx);
    if (dx) {
      blas.VCOPY(dout->numel(), dout->data<T>(),
                 dx->mutable_data<T>(ctx.GetPlace()));
      set_mkldnn_format(dx, dout);
    }

    if (dy) {
      blas.VCOPY(dout->numel(), dout->data<T>(),
                 dy->mutable_data<T>(ctx.GetPlace()));
      set_mkldnn_format(dy, dout);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(
    elementwise_add, MKLDNN, ::paddle::platform::CPUPlace,
    ops::EltwiseMKLDNNKernel<float, dnnl::algorithm::binary_add>,
    ops::EltwiseMKLDNNKernel<paddle::platform::bfloat16,
                             dnnl::algorithm::binary_add>,
    ops::EltwiseMKLDNNKernel<int8_t, dnnl::algorithm::binary_add>,
    ops::EltwiseMKLDNNKernel<uint8_t, dnnl::algorithm::binary_add>)

REGISTER_OP_KERNEL(elementwise_add_grad, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::EltwiseAddMKLDNNGradKernel<float>)

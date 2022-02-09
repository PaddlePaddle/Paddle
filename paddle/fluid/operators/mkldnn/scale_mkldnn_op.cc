/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/platform/mkldnn_reuse.h"
#include "paddle/pten/kernels/scale_kernel.h"
namespace paddle {
namespace operators {

using paddle::framework::Tensor;

template <typename T>
class ScaleMKLDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    this->RunKernel(ctx);
  }

  void RunKernel(const framework::ExecutionContext& ctx) const {
    const auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();
    auto* x = ctx.Input<Tensor>("X");
    auto* out = ctx.Output<Tensor>("Out");

    bool bias_after_scale = ctx.Attr<bool>("bias_after_scale");
    float scale;
    if (ctx.HasInput("ScaleTensor")) {
      auto* scale_tensor = ctx.Input<Tensor>("ScaleTensor");
      scale = static_cast<float>(*(scale_tensor->data<T>()));
    } else {
      scale = ctx.Attr<float>("scale");
    }

    auto bias = ctx.Attr<float>("bias");

    pten::ScaleKernel<T>(dev_ctx, *x, scale, bias, bias_after_scale, out);
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_KERNEL(scale, MKLDNN, paddle::platform::CPUPlace,
                   ops::ScaleMKLDNNKernel<float>,
                   ops::ScaleMKLDNNKernel<paddle::platform::bfloat16>);

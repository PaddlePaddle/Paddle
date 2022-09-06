/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class InstanceNormNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const auto epsilon = ctx.Attr<float>("epsilon");
    const auto* x = ctx.Input<Tensor>("X");
    const auto* scale = ctx.Input<Tensor>("Scale");
    const auto* bias = ctx.Input<Tensor>("Bias");
    auto* y = ctx.Output<Tensor>("Y");
    auto* mean = ctx.Output<Tensor>("SavedMean");
    auto* variance = ctx.Output<Tensor>("SavedVariance");
    auto& dev_ctx = ctx.template device_context<DeviceContext>();

    dev_ctx.template Alloc<T>(y);
    dev_ctx.template Alloc<T>(mean);
    dev_ctx.template Alloc<T>(variance);

    auto x_dims = x->dims();
    auto y_dims = y->dims();

    PADDLE_ENFORCE(x_dims.size() <= 5 && x_dims.size() >= 3,
                   platform::errors::InvalidArgument(
                       "InstanceNorm only supports the dimension of input "
                       " less equal to 5 and greater equal to 3. the dimension "
                       "of input is %d.",
                       x_dims.size()));

    auto tmp_x_dims = phi::vectorize<int>(x_dims);
    auto tmp_y_dims = phi::vectorize<int>(y_dims);
    if (x_dims.size() < 5) {
      for (size_t i = x_dims.size(); i < 5; ++i) {
        tmp_x_dims.insert(tmp_x_dims.begin() + 2, 1);
        tmp_y_dims.insert(tmp_y_dims.begin() + 2, 1);
      }
    }

    Tensor tmp_x, tmp_y;
    tmp_x.ShareDataWith(*x);

    tmp_x.Resize(phi::make_ddim(tmp_x_dims));
    tmp_x.set_layout(paddle::framework::DataLayout::NCDHW);
    tmp_y.ShareDataWith(*y);
    tmp_y.Resize(phi::make_ddim(tmp_y_dims));
    tmp_y.set_layout(paddle::framework::DataLayout::NCDHW);

    NpuOpRunner runner;

    runner.SetType("InstanceNorm")
        .AddInput(tmp_x)
        .AddInput(*scale)
        .AddInput(*bias)
        .AddAttr("data_format", std::string("NCDHW"))
        .AddAttr("epsilon", epsilon)
        .AddOutput(tmp_y)
        .AddOutput(*mean)
        .AddOutput(*variance);
    runner.Run(dev_ctx.stream());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_NPU_KERNEL(
    instance_norm,
    ops::InstanceNormNPUKernel<paddle::platform::NPUDeviceContext,
                               plat::float16>,
    ops::InstanceNormNPUKernel<paddle::platform::NPUDeviceContext, float>);

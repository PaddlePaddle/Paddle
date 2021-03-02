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

#ifdef PADDLE_WITH_ASCEND_CL

#include "paddle/fluid/operators/layer_norm_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using DDim = framework::DDim;

template <typename T>
class LayerNormNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const auto begin_norm_axis = ctx.Attr<int>("begin_norm_axis");
    const auto epsilon = ctx.Attr<float>("epsilon");
    const auto* x = ctx.Input<Tensor>("X");
    const auto* scale = ctx.Input<Tensor>("Scale");
    const auto* bias = ctx.Input<Tensor>("Bias");
    auto* y = ctx.Output<Tensor>("Y");
    auto* mean = ctx.Output<Tensor>("Mean");
    auto* variance = ctx.Output<Tensor>("Variance");
    const auto& x_dims = x->dims();
    auto matrix_dim = framework::flatten_to_2d(x_dims, begin_norm_axis);
    int right = static_cast<int>(matrix_dim[1]);

    auto place = ctx.GetPlace();
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    if (!scale) {
      Tensor default_scale(x->type());
      default_scale.mutable_data<T>({right}, place);
      Tensor value(x->type());
      value.mutable_data<T>({1}, place);
      TensorFromVector(std::vector<T>{static_cast<T>(1.0)},
                       ctx.device_context(), &value);
      auto runner =
          NpuOpRunner("FillD", {value}, {default_scale}, {{"dims", {right}}});
      runner.Run(stream);
      scale = &default_scale;
    }
    if (!bias) {
      Tensor default_bias(x->type());
      default_bias.mutable_data<T>({right}, place);
      Tensor value(x->type());
      value.mutable_data<T>({1}, place);
      TensorFromVector(std::vector<T>{static_cast<T>(0)}, ctx.device_context(),
                       &value);
      auto runner =
          NpuOpRunner("FillD", {value}, {default_bias}, {{"dims", {right}}});
      runner.Run(stream);
      bias = &default_bias;
    }
    y->mutable_data<T>(ctx.GetPlace());
    mean->mutable_data<T>(ctx.GetPlace());
    variance->mutable_data<T>(ctx.GetPlace());

    auto runner =
        NpuOpRunner("LayerNorm", {*x, *scale, *bias}, {*y, *mean, *variance},
                    {{"begin_norm_axis", begin_norm_axis},
                     {"begin_params_axis", begin_norm_axis},
                     {"epsilon", epsilon}});
    runner.Run(stream);
  }
};

template <typename T>
class LayerNormGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const auto begin_norm_axis = ctx.Attr<int>("begin_norm_axis");
    const auto* x = ctx.Input<Tensor>("X");
    const auto& x_dims = x->dims();
    const auto* mean = ctx.Input<Tensor>("Mean");
    const auto* variance = ctx.Input<Tensor>("Variance");
    const auto* scale = ctx.Input<Tensor>("Scale");
    const auto* dy = ctx.Input<Tensor>(framework::GradVarName("Y"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dscale = ctx.Output<Tensor>(framework::GradVarName("Scale"));
    auto* dbias = ctx.Output<Tensor>(framework::GradVarName("Bias"));

    auto matrix_dim = framework::flatten_to_2d(x_dims, begin_norm_axis);
    int right = static_cast<int>(matrix_dim[1]);

    auto place = ctx.GetPlace();
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    // No need to compute any gradient, jusr return
    if (!dx && !dscale && !dbias) {
      return;
    }

    Tensor default_scale(x->type());
    if (!scale) {
      default_scale.mutable_data<T>({right}, place);
      Tensor value(x->type());
      value.mutable_data<T>({1}, place);
      TensorFromVector(std::vector<T>{static_cast<T>(1.0)},
                       ctx.device_context(), &value);
      auto runner =
          NpuOpRunner("FillD", {value}, {default_scale}, {{"dims", {right}}});
      runner.Run(stream);
      scale = &default_scale;
    }

    Tensor dx_(dy->type()), dscale_(dy->type()), dbias_(dy->type());
    dx = (dx == nullptr) ? &dx_ : dx;
    dscale = (dscale == nullptr) ? &dscale_ : dscale;
    dbias = (dbias == nullptr) ? &dbias_ : dbias;

    dscale->mutable_data<T>(ctx.GetPlace());
    dbias->mutable_data<T>(ctx.GetPlace());
    dx->mutable_data<T>(ctx.GetPlace());

    auto runner =
        NpuOpRunner("LayerNormGrad", {*dy, *x, *variance, *mean, *scale},
                    {*dx, *dscale, *dbias}, {});
    runner.Run(stream);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_NPU_KERNEL(layer_norm, ops::LayerNormNPUKernel<float>,
                       ops::LayerNormNPUKernel<plat::float16>);
REGISTER_OP_NPU_KERNEL(layer_norm_grad, ops::LayerNormGradNPUKernel<float>,
                       ops::LayerNormGradNPUKernel<plat::float16>);

#endif  // PADDLE_WITH_ASCEND_CL

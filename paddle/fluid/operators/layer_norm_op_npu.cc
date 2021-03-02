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
    std::vector<int> axes;
    auto matrix_dim = framework::flatten_to_2d(x_dims, begin_norm_axis);
    int right = static_cast<int>(matrix_dim[1]);

    // The shape of scale and bias should be equal to x.shape[begin_norm_axis:],
    // required by Ascend.
    for (auto i = begin_norm_axis; i < x_dims.size(); ++i) {
      axes.push_back(x_dims[i]);
    }
    auto place = ctx.GetPlace();
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    Tensor default_scale(x->type());
    if (!scale) {
      default_scale.mutable_data<T>(framework::make_ddim(axes), place);
      Tensor value(x->type());
      value.mutable_data<T>({1}, place);
      TensorFromVector(std::vector<T>{static_cast<T>(1.0)},
                       ctx.device_context(), &value);
      auto runner =
          NpuOpRunner("FillD", {value}, {default_scale}, {{"dims", axes}});
      runner.Run(stream);
      scale = &default_scale;
    } else {
      const_cast<Tensor*>(scale)->Resize(framework::make_ddim(axes));
    }

    Tensor default_bias(x->type());
    if (!bias) {
      default_bias.mutable_data<T>(framework::make_ddim(axes), place);
      Tensor value(x->type());
      value.mutable_data<T>({1}, place);
      TensorFromVector(std::vector<T>{static_cast<T>(0)}, ctx.device_context(),
                       &value);
      auto runner =
          NpuOpRunner("FillD", {value}, {default_bias}, {{"dims", axes}});
      runner.Run(stream);
      bias = &default_bias;
    } else {
      const_cast<Tensor*>(bias)->Resize(framework::make_ddim(axes));
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
    // revert shape of scale and bias
    // TODO(zhiqiu): better implementation, use tmp tensor to avoid write input
    // tensor.
    const_cast<Tensor*>(scale)->Resize(framework::make_ddim({right}));
    const_cast<Tensor*>(bias)->Resize(framework::make_ddim({right}));
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

    std::vector<int> axes;
    for (auto i = begin_norm_axis; i < x_dims.size(); ++i) {
      axes.push_back(x_dims[i]);
    }

    auto place = ctx.GetPlace();
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    // No need to compute any gradient, jusr return
    if (!dx && !dscale && !dbias) {
      return;
    }

    // The rank of mean should be equal to x, required by Ascend.
    std::vector<int> new_shape;
    for (auto i = 0; i < begin_norm_axis; ++i) {
      new_shape.push_back(x_dims[i]);
    }
    for (auto i = begin_norm_axis; i < x_dims.size(); ++i) {
      new_shape.push_back(1);
    }

    auto mean_dims = mean->dims();
    const_cast<Tensor*>(mean)->Resize(framework::make_ddim({new_shape}));
    const_cast<Tensor*>(variance)->Resize(framework::make_ddim({new_shape}));

    Tensor default_scale(x->type());
    if (!scale) {
      default_scale.mutable_data<T>(framework::make_ddim(axes), place);
      Tensor value(x->type());
      value.mutable_data<T>({1}, place);
      TensorFromVector(std::vector<T>{static_cast<T>(1.0)},
                       ctx.device_context(), &value);
      auto runner =
          NpuOpRunner("FillD", {value}, {default_scale}, {{"dims", axes}});
      runner.Run(stream);
      scale = &default_scale;
    } else {
      const_cast<Tensor*>(scale)->Resize(framework::make_ddim(axes));
    }

    Tensor dx_(dy->type()), dscale_(dy->type()), dbias_(dy->type());
    dx = (dx == nullptr) ? &dx_ : dx;
    dscale = (dscale == nullptr) ? &dscale_ : dscale;
    dbias = (dbias == nullptr) ? &dbias_ : dbias;

    dscale->Resize(framework::make_ddim(axes));
    dscale->mutable_data<T>(ctx.GetPlace());

    dbias->Resize(framework::make_ddim(axes));
    dbias->mutable_data<T>(ctx.GetPlace());

    dx->Resize(x->dims());
    dx->mutable_data<T>(ctx.GetPlace());

    auto runner =
        NpuOpRunner("LayerNormGrad", {*dy, *x, *variance, *mean, *scale},
                    {*dx, *dscale, *dbias}, {});
    runner.Run(stream);

    const_cast<Tensor*>(mean)->Resize(mean_dims);
    const_cast<Tensor*>(variance)->Resize(mean_dims);
    const_cast<Tensor*>(scale)->Resize(framework::make_ddim({right}));
    dscale->Resize(framework::make_ddim({right}));
    dbias->Resize(framework::make_ddim({right}));
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

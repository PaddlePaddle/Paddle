//   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/inplace_abn_op.h"
#include <string>
#include <vector>
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/operators/batch_norm_op.h"

namespace paddle {
namespace operators {

class InplaceABNOp : public paddle::operators::BatchNormOp {
 public:
  using paddle::operators::BatchNormOp::BatchNormOp;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto input_data_type = ctx.Input<Tensor>("X")->type();
    // By default, the type of the scale, bias, mean,
    // and var tensors should both be float. (For float or float16 input tensor)
    // or double (For double input tensor).
    auto bn_param_type = framework::proto::VarType::FP32;
    if (input_data_type == framework::proto::VarType::FP64) {
      bn_param_type = framework::proto::VarType::FP64;
    }
    PADDLE_ENFORCE_EQ(bn_param_type, ctx.Input<Tensor>("Scale")->type(),
                      "Scale input should be of float type");
    PADDLE_ENFORCE_EQ(bn_param_type, ctx.Input<Tensor>("Bias")->type(),
                      "Bias input should be of float type");
    PADDLE_ENFORCE_EQ(bn_param_type, ctx.Input<Tensor>("Mean")->type(),
                      "Mean input should be of float type");
    PADDLE_ENFORCE_EQ(bn_param_type, ctx.Input<Tensor>("Variance")->type(),
                      "Variance input should be of float type");

    framework::LibraryType library = framework::LibraryType::kPlain;
    framework::DataLayout layout = framework::DataLayout::kAnyLayout;

    return framework::OpKernelType(input_data_type, ctx.GetPlace(), layout,
                                   library);
  }
};

class InplaceABNGradOp : public paddle::operators::BatchNormGradOp {
 public:
  using paddle::operators::BatchNormGradOp::BatchNormGradOp;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    const auto* var = ctx.InputVar(framework::GradVarName("Y"));
    auto input_data_type = ctx.Input<Tensor>("X")->type();
    if (var == nullptr) {
      PADDLE_THROW("can't find Y@GRAD");
    }
    const Tensor* t = nullptr;
    if (var->IsType<Tensor>()) {
      t = &var->Get<Tensor>();
    } else if (var->IsType<LoDTensor>()) {
      t = &var->Get<LoDTensor>();
    }
    if (t == nullptr) {
      PADDLE_THROW("can't find Y@GRAD");
    }
    framework::LibraryType library = framework::LibraryType::kPlain;
    framework::DataLayout layout = framework::DataLayout::kAnyLayout;

    return framework::OpKernelType(input_data_type, ctx.GetPlace(), layout,
                                   library);
  }
};

class InplaceABNOpMaker : public paddle::operators::BatchNormOpMaker {
 public:
  void Make() override {
    BatchNormOpMaker::Make();
    AddAttr<std::string>(
        "activation",
        "(enum string, default identity, can be identity|elu|leakyrelu) "
        "The activation type used for output candidate {h}_t.")
        .SetDefault("");
    AddAttr<float>("alpha",
                   "(float, default 1.0) Only used in inplace-abn kernel,"
                   "the activation type(identity|elu|leakyrelu) would be fused "
                   "with batch_norm, "
                   "this is the alpha value for elu|leakyrelu.")
        .SetDefault(0.1f);
    AddAttr<bool>("use_sync_bn",
                  "(bool, default false) Whether use synchronize batch "
                  "normalization.")
        .SetDefault(false);
  }
};

class InplaceABNOpGradMaker : public paddle::operators::BatchNormGradMaker {
 public:
  using paddle::operators::BatchNormGradMaker::BatchNormGradMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    std::unique_ptr<framework::OpDesc> op = BatchNormGradMaker::Apply();
    op->SetInput("Y", Output("Y"));
    return op;
  }
};

template <typename DeviceContext, typename T>
class InplaceABNKernel
    : public paddle::operators::BatchNormKernel<DeviceContext, T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* y = ctx.Output<Tensor>("Y");
    auto activation =
        GetInplaceABNActivationType(ctx.Attr<std::string>("activation"));
    auto& place = *ctx.template device_context<DeviceContext>().eigen_device();
    BatchNormKernel<DeviceContext, T>::Compute(ctx);

    auto cur_y = EigenVector<T>::Flatten(*y);
    InplaceABNActivation<DeviceContext, T> functor;
    functor.Compute(ctx, activation, place, cur_y, cur_y);
  }
};

template <typename DeviceContext, typename T>
class InplaceABNGradKernel
    : public paddle::operators::BatchNormGradKernel<DeviceContext, T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* y = ctx.Input<Tensor>("Y");
    auto* d_y = ctx.Input<Tensor>(framework::GradVarName("Y"));
    auto& place = *ctx.template device_context<DeviceContext>().eigen_device();
    auto activation =
        GetInplaceABNActivationType(ctx.Attr<std::string>("activation"));

    auto py = *y;
    auto pd_y = *d_y;
    auto cur_y = EigenVector<T>::Flatten(py);
    auto cur_dy = EigenVector<T>::Flatten(pd_y);

    InplaceABNActivation<DeviceContext, T> functor;
    functor.GradCompute(ctx, activation, place, cur_y, cur_y, cur_dy, cur_dy);

    BatchNormGradKernel<DeviceContext, T>::Compute(ctx);
  }
};

DECLARE_INPLACE_OP_INFERER(InplaceABNInplaceInferer, {"X", "Y"});
DECLARE_INPLACE_OP_INFERER(InplaceABNGradInplaceInferer,
                           {framework::GradVarName("Y"),
                            framework::GradVarName("X")});

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(inplace_abn, ops::InplaceABNOp, ops::InplaceABNOpMaker,
                  ops::InplaceABNOpGradMaker, ops::InplaceABNInplaceInferer,
                  ops::BatchNormOpInferVarType)
REGISTER_OPERATOR(inplace_abn_grad, ops::InplaceABNGradOp,
                  ops::InplaceABNGradInplaceInferer)

REGISTER_OP_CPU_KERNEL(
    inplace_abn,
    ops::InplaceABNKernel<paddle::platform::CPUDeviceContext, float>,
    ops::InplaceABNKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    inplace_abn_grad,
    ops::InplaceABNGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::InplaceABNGradKernel<paddle::platform::CPUDeviceContext, double>);

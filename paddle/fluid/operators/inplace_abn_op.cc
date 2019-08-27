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

class InplaceABNOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {}

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(ctx.Input<Tensor>("X")->type(),
                                   ctx.device_context());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name, const Tensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const override {
    return framework::OpKernelType(expected_kernel_type.data_type_,
                                   tensor.place(), tensor.layout());
  }
};

class InplaceABNGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {}

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(ctx.Input<Tensor>("X")->type(),
                                   ctx.device_context());
  }
};

class InplaceABNOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(LoDTensor, LoDTensor<int>) Input variable with rank at least 2. "
             "The last dimension of X should be 1. Each value of X is an index "
             "to indicate the position.");
    AddAttr<int>("activation",
                 "(enum int, default leakyrelu) "
                 "The activation type used for output candidate {h}_t.")
        .SetDefault(identity)
        .InEnum({identity, leakyrelu, elu});
    AddComment(R"DOC(
)DOC");
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

    // apply in-place activation calculate
    auto cur_x = EigenMatrix<T>::From(*y);
    auto cur_y = EigenMatrix<T>::From(*y);
    InplaceABNActivation<DeviceContext, T> functor;
    functor.Compute(activation, place, cur_x, cur_y);
  }
};

template <typename DeviceContext, typename T>
class InplaceABNGradKernel
    : public paddle::operators::BatchNormGradKernel<DeviceContext, T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Output<Tensor>("Y");
    auto* d_y = ctx.Input<Tensor>(framework::GradVarName("Y"));
    auto* d_x = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto& place = *ctx.template device_context<DeviceContext>().eigen_device();
    auto activation =
        GetInplaceABNActivationType(ctx.Attr<std::string>("activation"));

    // apply in-place activation calculate
    auto cur_x = EigenMatrix<T>::From(*x);
    auto cur_y = EigenMatrix<T>::From(*y);
    auto cur_dx = EigenMatrix<T>::From(*d_x);
    auto cur_dy = EigenMatrix<T>::From(*d_y);
    InplaceABNActivation<DeviceContext, T> functor;
    functor.GradCompute(activation, place, cur_x, cur_y, cur_dx, cur_dy);

    auto inp_cur_dy = EigenMatrix<T>::From(const_cast<Tensor&>(*d_y));
    inp_cur_dy.device(place) = cur_dx;
    BatchNormGradKernel<DeviceContext, T>::Compute(ctx);
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(inplace_abn, ops::InplaceABNOp, ops::InplaceABNOpMaker);
REGISTER_OPERATOR(inplace_abn_grad, ops::InplaceABNGradOp)

REGISTER_OP_CPU_KERNEL(
    inplace_abn,
    ops::InplaceABNKernel<paddle::platform::CPUDeviceContext, float>,
    ops::InplaceABNKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    inplace_abn_grad,
    ops::InplaceABNGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::InplaceABNGradKernel<paddle::platform::CPUDeviceContext, double>);

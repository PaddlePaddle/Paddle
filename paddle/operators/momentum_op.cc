/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/operators/momentum_op.h"

namespace paddle {
namespace operators {

class MomentumOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Param"),
                   "Input(param) of Momentum should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Grad"),
                   "Input(grad) of Momentum should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Velocity"),
                   "Input(velocity) of Momentum should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("LearningRate"),
                   "Input(LearningRate) of Momentum should not be null.");

    PADDLE_ENFORCE(ctx->HasOutput("ParamOut"),
                   "Output(ParamOut) of Momentum should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("VelocityOut"),
                   "Output(VelocityOut) of Momentum should not be null.");

    auto param_dim = ctx->GetInputDim("Param");
    PADDLE_ENFORCE_EQ(
        param_dim, ctx->GetInputDim("Grad"),
        "Param and Grad input of MomentumOp should have the same dimension.");
    PADDLE_ENFORCE_EQ(
        param_dim, ctx->GetInputDim("Velocity"),
        "Param and Velocity of MomentumOp should have the same dimension.");
    PADDLE_ENFORCE_EQ(framework::product(ctx->GetInputDim("LearningRate")), 1,
                      "Learning_rate should be a scalar");

    ctx->SetOutputDim("ParamOut", param_dim);
    ctx->SetOutputDim("VelocityOut", param_dim);
  }
};

class MomentumOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  MomentumOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("Param",
             "(Tensor, default Tensor<float>) "
             "Input parameter that has to be updated");
    AddInput("Grad",
             "(Tensor, default Tensor<float>) "
             "Input gradient of the parameter");
    AddInput("Velocity",
             "(Tensor, default Tensor<float>) "
             "Input velocity (corresponding to the parameter) "
             "that has to be updated");
    AddInput("LearningRate",
             "(Tensor, default Tensor<float>) "
             "Input learning rate");

    AddOutput("ParamOut",
              "(Tensor) This output is updated parameter. "
              "It shared memory with Input(Param).");
    AddOutput("VelocityOut",
              "(Tensor) This output is updated velocity. "
              "It shared memory with Input(Velocity).");

    AddAttr<float>("mu", "(float) Momentum coefficient");
    AddAttr<bool>("use_nesterov",
                  "(bool, default false) "
                  "Use Nesterov Momentum")
        .SetDefault(false);
    AddAttr<bool>("use_local_lr",
                  "(bool, default false) "
                  "Use LARS")
        .SetDefault(false);
    AddAttr<float>("local_gw_ratio", "(float) LARS coefficient")
        .SetDefault(0.001);
    AddAttr<float>("weight_decay", "(float) LARS weight decay")
        .SetDefault(0.0005);

    AddComment(R"DOC(
Momentum Optimizer.

This optimizer has a flag for Nestrov Momentum.
Thie optimizer has attributes for LARS to adjust local LR for large batch training of CNN.
paper : https://arxiv.org/abs/1708.03888.
The update equations are as follows:

$$
velocity = mu * velocity + gradient \\
if (use\_nesterov):   \\
  param = param - gradient * learning\_rate + mu * velocity * learning\_rate \\
else if (use\_lcoal\_lr): \\
  learning\_rate *= local\_gw\_ratio * sqrt(sumsq(param)) 
                  / (sqrt(sumsq(gradient))+ weight\_decay * sqrt(sumsq(param))) \\
  param = param - learning\_rate * velocity. \\
else: \\
  param = param - learning\_rate * velocity. \\
$$

)DOC");
  }
};

template <typename T>
class MomentumOpCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto param_out = ctx.Output<framework::Tensor>("ParamOut");
    auto velocity_out = ctx.Output<framework::Tensor>("VelocityOut");
    auto param = ctx.Input<framework::Tensor>("Param");
    auto velocity = ctx.Input<framework::Tensor>("Velocity");
    auto grad = ctx.Input<framework::Tensor>("Grad");
    auto learning_rate = ctx.Input<framework::Tensor>("LearningRate");

    param_out->mutable_data<T>(ctx.GetPlace());
    velocity_out->mutable_data<T>(ctx.GetPlace());

    T mu = static_cast<T>(ctx.Attr<float>("mu"));
    bool use_nesterov = ctx.Attr<bool>("use_nesterov");
    bool use_local_lr = ctx.Attr<bool>("use_local_lr");
    T local_gw_ratio = static_cast<T>(ctx.Attr<float>("local_gw_ratio"));
    T weight_decay = static_cast<T>(ctx.Attr<float>("weight_decay"));

    auto p_out = framework::EigenVector<T>::Flatten(*param_out);
    auto v_out = framework::EigenVector<T>::Flatten(*velocity_out);

    auto p = framework::EigenVector<T>::Flatten(*param);
    auto v = framework::EigenVector<T>::Flatten(*velocity);
    auto g = framework::EigenVector<T>::Flatten(*grad);
    auto *lr = learning_rate->data<T>();

    T local_lr = lr[0];
    if (use_local_lr) {
      Eigen::TensorFixedSize<T, Eigen::Sizes<>, Eigen::RowMajor,
                             Eigen::DenseIndex>
          p_norm = p.square().sum().sqrt();
      Eigen::TensorFixedSize<T, Eigen::Sizes<>, Eigen::RowMajor,
                             Eigen::DenseIndex>
          g_norm = g.square().sum().sqrt();
      if ((p_norm(0) > static_cast<T>(0)) && (g_norm(0) > static_cast<T>(0)))
        local_lr = lr[0] * local_gw_ratio * p_norm(0) /
                   (g_norm(0) + weight_decay * p_norm(0));
    }

    v_out = v * mu + g;
    if (use_nesterov) {
      p_out = p - (g - v_out * mu) * lr[0];
    } else {
      p_out = p - local_lr * v_out;
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(momentum, ops::MomentumOp, ops::MomentumOpMaker);
REGISTER_OP_CPU_KERNEL(momentum, ops::MomentumOpCPUKernel<float>,
                       ops::MomentumOpCPUKernel<double>);

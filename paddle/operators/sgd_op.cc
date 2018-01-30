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

#include "paddle/operators/sgd_op.h"

namespace paddle {
namespace operators {

class SGDOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Param"),
                   "Input(Param) of SGDOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Grad"),
                   "Input(Grad) of SGDOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("LearningRate"),
                   "Input(LearningRate) of SGDOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("ParamOut"),
                   "Output(ParamOut) of SGDOp should not be null.");

    auto lr_dims = ctx->GetInputDim("LearningRate");
    PADDLE_ENFORCE_EQ(framework::product(lr_dims), 1,
                      "Learning rate should have 1 element");
    auto param_dim = ctx->GetInputDim("Param");
    // TODO(qijun): check dimensions of Param and Grad at complie
    // and run time.
    ctx->SetOutputDim("ParamOut", param_dim);
  }
};

class SGDOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SGDOpMaker(OpProto* proto, OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("Param", "(Tensor) Input parameter");
    AddInput("LearningRate", "(Tensor) Learning rate of SGD");
    AddInput("Grad", "(Tensor) Input gradient");
    AddOutput("ParamOut", "(Tensor) Output parameter");
    AddAttr<bool>("use_local_lr",
                  "(bool, default false) "
                  "Use LARS")
        .SetDefault(false);
    AddAttr<float>("local_gw_ratio", "(float) LARS coefficient")
        .SetDefault(0.001);
    AddAttr<float>("weight_decay", "(float) LARS weight decay")
        .SetDefault(0.0005);

    AddComment(R"DOC(

SGD operator

This operator implements one step of the stochastic gradient descent algorithm.
This optimizer has attributes for LARS to adjust local LR for large batch training of CNN.
paper : https://arxiv.org/abs/1708.03888.
$$
if (use\_local\_lr):  \\
  learning\_rate *= local\_gw\_ratio * sqrt(sumsq(param)) 
                  / (sqrt(sumsq(grad))+ weight\_decay * sqrt(sumsq(param))) \\
param\_out = param - learning\_rate * grad
$$
)DOC");
  }
};

template <typename T>
class SGDOpCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* param = ctx.Input<framework::Tensor>("Param");
    auto* param_out = ctx.Output<framework::Tensor>("ParamOut");
    auto* learning_rate = ctx.Input<framework::Tensor>("LearningRate");

    auto* grad_var = ctx.InputVar("Grad");

    bool use_local_lr = ctx.Attr<bool>("use_local_lr");
    T local_gw_ratio = static_cast<T>(ctx.Attr<float>("local_gw_ratio"));
    T weight_decay = static_cast<T>(ctx.Attr<float>("weight_decay"));

    // Actually, all tensors are LoDTensor except SelectedRows.
    if (grad_var->IsType<framework::LoDTensor>()) {
      param_out->mutable_data<T>(ctx.GetPlace());
      auto* grad = ctx.Input<framework::Tensor>("Grad");

      auto p = framework::EigenVector<T>::Flatten(*param);
      auto g = framework::EigenVector<T>::Flatten(*grad);
      auto o = framework::EigenVector<T>::Flatten(*param_out);
      auto* lr = learning_rate->data<T>();

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
      o = p - local_lr * g;
    } else if (grad_var->IsType<framework::SelectedRows>()) {
      // TODO(qijun): In Sparse SGD operator, in-place update is enforced.
      // This manual optimization brings difficulty to track data dependency.
      // It's better to find a more elegant solution.
      PADDLE_ENFORCE_EQ(param, param_out);
      auto* grad = ctx.Input<framework::SelectedRows>("Grad");

      auto in_height = grad->height();
      auto out_dims = param_out->dims();
      PADDLE_ENFORCE_EQ(in_height, out_dims[0]);

      auto& in_value = grad->value();
      auto& in_rows = grad->rows();

      int64_t in_row_numel = in_value.numel() / in_rows.size();
      PADDLE_ENFORCE_EQ(in_row_numel, param_out->numel() / in_height);

      auto* in_data = in_value.data<T>();
      auto* out_data = param_out->data<T>();
      auto* lr = learning_rate->data<T>();

      for (size_t i = 0; i < in_rows.size(); i++) {
        for (int64_t j = 0; j < in_row_numel; j++) {
          out_data[in_rows[i] * in_row_numel + j] -=
              lr[0] * in_data[i * in_row_numel + j];
        }
      }
    } else {
      PADDLE_THROW("Unsupported Variable Type of Grad");
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(sgd, ops::SGDOp, ops::SGDOpMaker);
REGISTER_OP_CPU_KERNEL(sgd, ops::SGDOpCPUKernel<float>,
                       ops::SGDOpCPUKernel<double>);

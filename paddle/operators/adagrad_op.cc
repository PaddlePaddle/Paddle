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

#include "paddle/operators/adagrad_op.h"

#include <cmath>

#include "paddle/operators/math/math_function.h"
#include "paddle/operators/math/selected_rows_functor.h"

namespace paddle {
namespace operators {

class AdagradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Param"),
                   "Input(Param) of AdagradOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Grad"),
                   "Input(Grad) of AdagradOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Moment"),
                   "Input(Moment) of AdagradOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("LearningRate"),
                   "Input(LearningRate) of AdagradOp should not be null.");

    PADDLE_ENFORCE(ctx->HasOutput("ParamOut"),
                   "Output(ParamOut) of AdagradOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("MomentOut"),
                   "Output(MomentOut) of AdagradOp should not be null.");

    auto lr_dims = ctx->GetInputDim("LearningRate");
    PADDLE_ENFORCE_EQ(framework::product(lr_dims), 1,
                      "LearningRate should have one element");
    auto param_dims = ctx->GetInputDim("Param");
    PADDLE_ENFORCE_EQ(
        param_dims, ctx->GetInputDim("Grad"),
        "Param and Grad input of AdagradOp should have the same dimension.");
    PADDLE_ENFORCE_EQ(
        param_dims, ctx->GetInputDim("Moment"),
        "Param and Moment input of AdagradOp should have the same dimension.");

    ctx->SetOutputDim("ParamOut", param_dims);
    ctx->SetOutputDim("MomentOut", param_dims);
  }
};

class AdagradOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  AdagradOpMaker(framework::OpProto* proto,
                 framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("Param", "(Tensor) Input parameter");
    AddInput("Grad", "(Tensor) Input gradient");
    AddInput("Moment", "(Tensor) Second moment");
    AddInput("LearningRate", "(Tensor) Learning rate");

    AddOutput("ParamOut", "(Tensor) Output parameter");
    AddOutput("MomentOut", "(Tensor) Output second moment");

    AddAttr<float>("epsilon",
                   "(float, default 1.0e-6) "
                   "Constant for numerical stability")
        .SetDefault(1.0e-6f);
    AddComment(R"DOC(

Adaptive Gradient Algorithm (Adagrad).

The update is done as follows:

$$momentOut = moment + grad * grad \break
paramOut = param - learningRate * grad / ($\sqrt{momentOut}$ + \epsilon) \break
$$

The original paper(http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
does not have the epsilon attribute. It is added here in our implementation
as also proposed here: http://cs231n.github.io/neural-networks-3/#ada
for numerical stability to avoid the division by zero error.

)DOC");
  }
};

namespace {
size_t FindPos(const std::vector<int64_t>& rows, int64_t value) {
  return std::find(rows.begin(), rows.end(), value) - rows.begin();
}
}  // namespace

template <typename T>
struct SparseAdagradFunctor<platform::CPUPlace, T> {
  void operator()(const platform::DeviceContext& context,
                  const framework::SelectedRows& grad,
                  const framework::Tensor& learning_rate, T epsilon,
                  framework::Tensor* moment, framework::Tensor* param) {
    // 1. g_m.rows = set(g.rows)
    auto grad_rows = grad.rows();
    std::set<int64_t> row_set(grad_rows.begin(), grad_rows.end());
    std::vector<int64_t> merge_rows(row_set.begin(), row_set.end());

    auto grad_width = grad.value().dims()[1];
    std::unique_ptr<framework::SelectedRows> grad_merge{
        new framework::SelectedRows()};
    grad_merge->set_rows(merge_rows);
    grad_merge->set_height(grad.height());
    grad_merge->mutable_value()->mutable_data<T>(
        framework::make_ddim(
            {static_cast<int64_t>(merge_rows.size()), grad_width}),
        context.GetPlace());

    math::SetConstant<platform::CPUPlace, T> constant_functor;
    constant_functor(context, grad_merge->mutable_value(), 0.0);

    auto* grad_merge_data = grad_merge->mutable_value()->data<T>();
    auto* grad_data = grad.value().data<T>();

    for (size_t i = 0; i < grad_rows.size(); i++) {
      size_t grad_merge_i = FindPos(merge_rows, grad_rows[i]);
      for (int64_t j = 0; j < grad_width; j++) {
        grad_merge_data[grad_merge_i * grad_width + j] +=
            grad_data[i * grad_width + j];
      }
    }

    // 2. m += g_m * g_m
    std::unique_ptr<framework::SelectedRows> grad_square{
        new framework::SelectedRows()};
    grad_square->set_rows(grad_merge->rows());
    grad_square->set_height(grad_merge->height());
    grad_square->mutable_value()->mutable_data<T>(grad_merge->value().dims(),
                                                  context.GetPlace());
    auto gs =
        framework::EigenVector<T>::Flatten(*(grad_square->mutable_value()));
    auto gm = framework::EigenVector<T>::Flatten(grad_merge->value());
    gs.device(*context.GetEigenDevice<platform::CPUPlace>()) = gm * gm;

    math::SelectedRowsAddToTensor<platform::CPUPlace, T> functor;
    functor(context, *grad_square, moment);

    // 3. update parameter
    auto* lr = learning_rate.data<T>();
    auto* param_data = param->data<T>();
    auto* moment_data = moment->data<T>();

    for (size_t i = 0; i < merge_rows.size(); i++) {
      for (int64_t j = 0; j < grad_width; j++) {
        param_data[merge_rows[i] * grad_width + j] -=
            lr[0] * grad_merge_data[i * grad_width + j] /
            (std::sqrt(moment_data[merge_rows[i] * grad_width + j]) + epsilon);
      }
    }
  }
};

template struct SparseAdagradFunctor<platform::CPUPlace, float>;
template struct SparseAdagradFunctor<platform::CPUPlace, double>;
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(adagrad, ops::AdagradOp, ops::AdagradOpMaker);
REGISTER_OP_CPU_KERNEL(
    adagrad, ops::AdagradOpKernel<paddle::platform::CPUPlace, float>,
    ops::AdagradOpKernel<paddle::platform::CPUPlace, double>);

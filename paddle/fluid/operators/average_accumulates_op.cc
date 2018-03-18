/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/average_accumulates_op.h"

namespace paddle {
namespace operators {

template <>
void getAccumulators<paddle::platform::CPUDeviceContext>(
    const framework::ExecutionContext& ctx, int64_t& num_updates_,
    int64_t& num_accumulates_, int64_t& old_num_accumulates_) {
  auto* in_old_num_accumulates = ctx.Input<Tensor>("in_old_num_accumulates");
  auto* in_num_accumulates = ctx.Input<Tensor>("in_num_accumulates");
  auto* in_num_updates = ctx.Input<Tensor>("in_num_updates");

  old_num_accumulates_ = in_old_num_accumulates->data<int64_t>()[0];
  num_accumulates_ = in_num_accumulates->data<int64_t>()[0];
  num_updates_ = in_num_updates->data<int64_t>()[0];
}

template <>
void setAccumulators<paddle::platform::CPUDeviceContext>(
    const framework::ExecutionContext& ctx, int64_t num_updates_,
    int64_t num_accumulates_, int64_t old_num_accumulates_) {
  auto* out_old_num_accumulates = ctx.Output<Tensor>("out_old_num_accumulates");
  auto* out_num_accumulates = ctx.Output<Tensor>("out_num_accumulates");
  auto* out_num_updates = ctx.Output<Tensor>("out_num_updates");

  out_old_num_accumulates->data<int64_t>()[0] = old_num_accumulates_;
  out_num_accumulates->data<int64_t>()[0] = num_accumulates_;
  out_num_updates->data<int64_t>()[0] = num_updates_;
}

class AverageAccumulatesOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(
        ctx->HasInput("param"),
        "Input (param) of average_accumulates op should not be null.");
    PADDLE_ENFORCE(
        ctx->HasInput("in_sum_1"),
        "Input (sum_1) of average_accumulates op should not be null.");
    PADDLE_ENFORCE(
        ctx->HasInput("in_sum_2"),
        "Input (sum_2) of average_accumulates op should not be null.");
    PADDLE_ENFORCE(
        ctx->HasInput("in_sum_3"),
        "Input (sum_3) of average_accumulates op should not be null.");
    PADDLE_ENFORCE(
        ctx->HasInput("in_num_accumulates"),
        "Input (in_num_accumulates) of average_accumulates op should "
        "not be null.");
    PADDLE_ENFORCE(ctx->HasInput("in_old_num_accumulates"),
                   "Input (old_num_accumulates) of average_accumulates op "
                   "should not be null.");
    PADDLE_ENFORCE(
        ctx->HasInput("in_num_updates"),
        "Input (num_updates) of average_accumulates op should not be null.");

    PADDLE_ENFORCE(
        ctx->HasOutput("out_sum_1"),
        "Output (sum_1) of average_accumulates op should not be null.");
    PADDLE_ENFORCE(
        ctx->HasOutput("out_sum_2"),
        "Output (sum_2) of average_accumulates op should not be null.");
    PADDLE_ENFORCE(
        ctx->HasOutput("out_sum_3"),
        "Output (sum_3) of average_accumulates op should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("out_num_accumulates"),
                   "Output (num_accumulates) of average_accumulates op should "
                   "not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("out_old_num_accumulates"),
                   "Output (old_num_accumulates) of average_accumulates op "
                   "should not be null.");
    PADDLE_ENFORCE(
        ctx->HasOutput("out_num_updates"),
        "Output (num_updates) of average_accumulates op should not be null.");

    auto in_dim = ctx->GetInputDim("param");

    ctx->SetOutputDim("out_sum_1", in_dim);
    ctx->SetOutputDim("out_sum_2", in_dim);
    ctx->SetOutputDim("out_sum_3", in_dim);
    ctx->SetOutputDim("out_num_accumulates", {1});
    ctx->SetOutputDim("out_old_num_accumulates", {1});
    ctx->SetOutputDim("out_num_updates", {1});
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(ctx.Input<Tensor>("param")->type()),
        ctx.GetPlace());
  }
};

class AverageAccumulatesOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  AverageAccumulatesOpMaker(OpProto* proto, OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("param",
             "Input(Tensor or LoDTensor): The parameter to be accumulated.");
    AddInput("in_sum_1",
             "Input(Tensor or LoDTensor): A tensor used to store the parameter "
             "sums with the same shape as input(param).");
    AddInput("in_sum_2",
             "Input(Tensor or LoDTensor): A auxiliary tensor to help "
             "accumulating sums of parameter values with the same shape as "
             "input(param). It is used to avoid loss of precision due to too "
             "many sums.");
    AddInput("in_sum_3",
             "Input(Tensor or LoDTensor): A auxiliary tensor to help "
             "accumulating sums of parameter values with the same shape as "
             "input(param).");
    AddInput("in_num_accumulates",
             "Input(Tensor): The accumulating times of current window with "
             "shape [1].");
    AddInput("in_old_num_accumulates",
             "Input(Tensor): The accumulating times of previous window with "
             "shape [1].");
    AddInput("in_num_updates",
             "Input(Tensor): The total number of batches used by trainning "
             "before this batch with shape [1].");

    AddOutput("out_sum_1",
              "Output(Tensor or LoDTensor): A tensor used to store the "
              "parameter sums with the same shape as input(param).");
    AddOutput("out_sum_2",
              "Output(Tensor or LoDTensor): A auxiliary tensor to help "
              "accumulating sums of parameter values with the same shape as "
              "input(param). It is used to avoid loss of precision due to too "
              "many sums.");
    AddOutput("out_sum_3",
              "Output(Tensor or LoDTensor): A auxiliary tensor to help "
              "accumulating sums of parameter values with the same shape as "
              "input(param).");
    AddOutput("out_num_accumulates",
              "Output(Tensor): The accumulating times of current window with "
              "shape [1].");
    AddOutput("out_old_num_accumulates",
              "Output(Tensor): The accumulating times of previous window with "
              "shape [1].");
    AddOutput("out_num_updates",
              "Output(Tensor): The total number of batches used by trainning "
              "before this batch with shape [1].");

    AddAttr<float>("average_window",
                   "The rate of average window size relative to num_updates.");
    AddAttr<int64_t>("max_average_window", "Maximum size of average window.");
    AddAttr<int64_t>("min_average_window", "Minimu size of average window.");

    AddComment(R"DOC(
AverageAccumulates Operator.
Accumulate the sum of parameter whtin sliding window. The size of sliding window is determined by 'average_window', 'max_average_window' and 'min_average_window'.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(average_accumulates, ops::AverageAccumulatesOp,
                  ops::AverageAccumulatesOpMaker,
                  paddle::framework::EmptyGradOpMaker);
REGISTER_OP_CPU_KERNEL(
    average_accumulates,
    ops::AverageAccumulatesKernel<paddle::platform::CPUDeviceContext, float>,
    ops::AverageAccumulatesKernel<paddle::platform::CPUDeviceContext, double>);

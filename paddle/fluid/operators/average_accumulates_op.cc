/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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
  auto* in_old_num_accumulates = ctx.Input<Tensor>("old_num_accumulates");
  auto* in_num_accumulates = ctx.Input<Tensor>("num_accumulates");
  auto* in_num_updates = ctx.Input<Tensor>("num_updates");

  old_num_accumulates_ = in_old_num_accumulates->data<int64_t>()[0];
  num_accumulates_ = in_num_accumulates->data<int64_t>()[0];
  num_updates_ = in_num_updates->data<int64_t>()[0];
}

template <>
void setAccumulators<paddle::platform::CPUDeviceContext>(
    const framework::ExecutionContext& ctx, int64_t num_updates_,
    int64_t num_accumulates_, int64_t old_num_accumulates_) {
  auto* out_old_num_accumulates = ctx.Output<Tensor>("old_num_accumulates");
  auto* out_num_accumulates = ctx.Output<Tensor>("num_accumulates");
  auto* out_num_updates = ctx.Output<Tensor>("num_updates");

  out_old_num_accumulates->data<int64_t>()[0] = old_num_accumulates_;
  out_num_accumulates->data<int64_t>()[0] = num_accumulates_;
  out_num_updates->data<int64_t>()[0] = num_updates_;
}

class AverageAccumulatesOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(
        ctx->HasInput("Param"),
        "Input (Param) of average_accumulates op should not be null.");
    PADDLE_ENFORCE(
        ctx->HasInput("Grad"),
        "Input (Grad) of average_accumulates op should not be null.");
    PADDLE_ENFORCE(
        ctx->HasInput("sum_1"),
        "Input (sum_1) of average_accumulates op should not be null.");
    PADDLE_ENFORCE(
        ctx->HasInput("sum_2"),
        "Input (sum_2) of average_accumulates op should not be null.");
    PADDLE_ENFORCE(
        ctx->HasInput("sum_3"),
        "Input (sum_3) of average_accumulates op should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("num_accumulates"),
                   "Input (num_accumulates) of average_accumulates op should "
                   "not be null.");
    PADDLE_ENFORCE(ctx->HasInput("old_num_accumulates"),
                   "Input (old_num_accumulates) of average_accumulates op "
                   "should not be null.");
    PADDLE_ENFORCE(
        ctx->HasInput("num_updates"),
        "Input (num_updates) of average_accumulates op should not be null.");

    PADDLE_ENFORCE(
        ctx->HasOutput("sum_1"),
        "Output (sum_1) of average_accumulates op should not be null.");
    PADDLE_ENFORCE(
        ctx->HasOutput("sum_2"),
        "Output (sum_2) of average_accumulates op should not be null.");
    PADDLE_ENFORCE(
        ctx->HasOutput("sum_3"),
        "Output (sum_3) of average_accumulates op should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("num_accumulates"),
                   "Output (num_accumulates) of average_accumulates op should "
                   "not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("old_num_accumulates"),
                   "Output (old_num_accumulates) of average_accumulates op "
                   "should not be null.");
    PADDLE_ENFORCE(
        ctx->HasOutput("num_updates"),
        "Output (num_updates) of average_accumulates op should not be null.");

    auto in_dim = ctx->GetInputDim("Param");

    ctx->SetOutputDim("sum_1", in_dim);
    ctx->SetOutputDim("sum_2", in_dim);
    ctx->SetOutputDim("sum_3", in_dim);
    ctx->SetOutputDim("num_accumulates", {1});
    ctx->SetOutputDim("old_num_accumulates", {1});
    ctx->SetOutputDim("num_updates", {1});
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(ctx.Input<Tensor>("Param")->type()),
        ctx.GetPlace());
  }
};

class AverageAccumulatesOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  AverageAccumulatesOpMaker(OpProto* proto, OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("sum_1", "");
    AddInput("sum_2", "");
    AddInput("sum_3", "");
    AddInput("num_accumulates", "");
    AddInput("old_num_accumulates", "");
    AddInput("num_updates", "");

    AddOutput("sum_1", "");
    AddOutput("sum_2", "");
    AddOutput("sum_3", "");
    AddOutput("num_accumulates", "");
    AddOutput("old_num_accumulates", "");
    AddOutput("num_updates", "");

    AddAttr<float>("", "average_window");
    AddAttr<float>("", "max_average_window");
    AddAttr<float>("", "min_average_window");

    AddComment(R"DOC(
AverageAccumulates Operator.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(average_accumulate, ops::AverageAccumulatesOp,
                  ops::AverageAccumulatesOpMaker,
                  paddle::framework::EmptyGradOpMaker);
REGISTER_OP_CPU_KERNEL(
    average_accumulate,
    ops::AverageAccumulatesKernel<paddle::platform::CPUDeviceContext, float>,
    ops::AverageAccumulatesKernel<paddle::platform::CPUDeviceContext, double>);

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

#include "paddle/fluid/operators/range_op.h"

namespace paddle {
namespace operators {

class RangeOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    if (ctx->HasInput("Start")) {
      auto s_dims = ctx->GetInputDim("Start");
      PADDLE_ENFORCE((s_dims.size() == 1) && (s_dims[0] == 1),
                     "The shape of Input(Start) should be [1].");
    }
    if (ctx->HasInput("End")) {
      auto e_dims = ctx->GetInputDim("End");
      PADDLE_ENFORCE((e_dims.size() == 1) && (e_dims[0] == 1),
                     "The shape of Input(End) should be [1].");
    }
    if (ctx->HasInput("Step")) {
      auto step_dims = ctx->GetInputDim("Step");
      PADDLE_ENFORCE((step_dims.size() == 1) && (step_dims[0] == 1),
                     "The shape of Input(Step) should be [1].");
    }
    ctx->SetOutputDim("Out", {-1});
  }
};

class RangeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Start",
             "Start of interval. The interval includes this value. It is a "
             "tensor with shape=[1].");
    AddInput("End",
             "End of interval. The interval does not include this value, "
             "except in some cases where step is not an integer and floating "
             "point round-off affects the length of out. It is a tensor with "
             "shape=[1].");
    AddInput("Step", "Spacing between values. It is a tensor with shape=[1].");
    AddOutput("Out", "A sequence of numbers.");
    AddComment(R"DOC(
    Return evenly spaced values within a given interval. Values are generated within the half-open interval [start, stop) (in other words, the interval including start but excluding stop). Like arange function of numpy.
)DOC");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(range, ops::RangeOp, ops::RangeOpMaker);
REGISTER_OP_CPU_KERNEL(range, ops::CPURangeKernel<int>,
                       ops::CPURangeKernel<float>, ops::CPURangeKernel<double>,
                       ops::CPURangeKernel<int64_t>);

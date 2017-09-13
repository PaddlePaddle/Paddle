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

#include "paddle/operators/expand_op.h"

namespace paddle {
namespace operators {

using framework::Tensor;

class ExpandOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext& ctx) const override {
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("X"), "X must be initialized.");
    std::vector<int> expand_times = Attr<std::vector<int>>("expandTimes");
    auto* x = ctx.Input<Tensor>("X");
    auto x_dims = x->dims();

    PADDLE_ENFORCE_EQ(static_cast<size_t>(framework::arity(x_dims)),
                      expand_times.size(),
                      "Number of attribute (expandTimes) value must be equal "
                      "to rank of X.");
    PADDLE_ENFORCE_LE(framework::arity(x_dims), 6,
                      "Rank of X must not be greater than 6.");

    std::vector<int64_t> out_shape(x_dims.size());
    for (size_t i = 0; i < expand_times.size(); ++i) {
      PADDLE_ENFORCE_GE(expand_times[i], 1,
                        "Each value of expand times should not be "
                        "less than 1.");
      out_shape[i] = x_dims[i] * expand_times[i];
    }
    auto* out = ctx.Output<Tensor>("Out");
    out->Resize(framework::make_ddim(out_shape));
  }
};

class ExpandOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  ExpandOpMaker(framework::OpProto* proto, framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input tensor.");
    AddOutput("Out", "Expanded result by tiling input X.");
    AddAttr<std::vector<int>>("expandTimes",
                              "Expand times for each dimension.");
    AddComment(R"DOC(
Expand operator tiles the input by given times. You should set times for each
dimension by providing attribute 'expandTimes'. Rank of input tensor should be
in [1, 6]. Please draw an inttention that size of 'expandTimes' must be same
with rank of input tensor.
)DOC");
  }
};

class ExpandGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext& ctx) const override {
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("X"), "X must be initialized.");
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar(framework::GradVarName("Out")),
                            "Input(Out@GRAD) should not be null.");
    auto x_dims = ctx.Input<Tensor>("X")->dims();
    std::vector<int> expand_times = Attr<std::vector<int>>("expandTimes");
    auto out_dims = ctx.Input<Tensor>(framework::GradVarName("Out"))->dims();
    auto* x_grad = ctx.Output<Tensor>(framework::GradVarName("X"));

    for (size_t i = 0; i < expand_times.size(); ++i) {
      PADDLE_ENFORCE_EQ(x_dims[i] * expand_times[i], out_dims[i],
                        "Size of each dimension of Input(Out@GRAD) should be "
                        "equal to multiplication of crroresponding sizes of "
                        "Input(X) and expandTimes.");
    }

    if (x_grad) x_grad->Resize(x_dims);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(expand, ops::ExpandOp, ops::ExpandOpMaker, expand_grad,
            ops::ExpandGradOp);
REGISTER_OP_CPU_KERNEL(expand,
                       ops::ExpandKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(
    expand_grad, ops::ExpandGradKernel<paddle::platform::CPUPlace, float>);

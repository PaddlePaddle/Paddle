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

#include "paddle/operators/reduce_op.h"

namespace paddle {
namespace operators {

using framework::Tensor;
using framework::DDim;

class ReduceOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("X"), "Input(X) should not be null");
    auto x_dims = ctx.Input<Tensor>("X")->dims();
    auto x_rank = x_dims.size();
    PADDLE_ENFORCE_LE(x_rank, 6, "Tensors with rank at most 6 are supported");
    int dim = ctx.Attr<int>("dim");
    if (dim < 0) dim = x_rank + dim;
    PADDLE_ENFORCE_LT(
        dim, x_rank,
        "The dim should be in the range [-rank(input), rank(input))");
    PADDLE_ENFORCE_GE(ctx.Attr<int>("keep_dim"), 0, "keep_dim must be 0 or 1");
    PADDLE_ENFORCE_LE(ctx.Attr<int>("keep_dim"), 1, "keep_dim must be 0 or 1");
    bool keep_dim = ctx.Attr<int>("keep_dim") == 1;
    auto dims_vector = vectorize(x_dims);
    if (keep_dim || x_rank == 1) {
      dims_vector[dim] = 1;
    } else {
      dims_vector.erase(dims_vector.begin() + dim);
    }
    auto out_dims = framework::make_ddim(dims_vector);
    ctx.Output<Tensor>("Out")->Resize(out_dims);
  }
};

class ReduceGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("X"), "Input(X) should not be null");
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar(framework::GradVarName("Out")),
                            "Input(Out@GRAD) should not be null");
    auto x_dims = ctx.Input<Tensor>("X")->dims();
    auto x_rank = x_dims.size();
    PADDLE_ENFORCE_LE(x_rank, 6, "Tensors with rank at most 6 are supported");
    int dim = ctx.Attr<int>("dim");
    if (dim < 0) dim = x_rank + dim;
    PADDLE_ENFORCE_LT(
        dim, x_rank,
        "The dim should be in the range [-rank(input), rank(input))");
    auto *x_grad = ctx.Output<Tensor>(framework::GradVarName("X"));
    if (x_grad) x_grad->Resize(x_dims);
  }
};

class ReduceSumOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  ReduceSumOpMaker(framework::OpProto *proto,
                   framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput(
        "X",
        "(Tensor) The input tensor. Tensors with rank at most 6 are supported");
    AddOutput("Out", "(Tensor) The result tensor.");
    AddComment(R"DOC(
ReduceMean operator computes the sum of input tensor along the given dimension. 
The result tensor has 1 fewer dimension than the input unless `keep_dim` is true.
)DOC");
    AddAttr<int>("dim",
                 "(int, default 0) The dimension to reduce. "
                 "Must be in the range [-rank(input), rank(input))")
        .SetDefault(0);
    AddAttr<int>(
        "keep_dim",
        "(int, default 0) "
        "Must be 0 or 1. If 1, retain the reduced dimension with length 1.")
        .SetDefault(0);
  }
};

class ReduceMeanOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  ReduceMeanOpMaker(framework::OpProto *proto,
                    framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput(
        "X",
        "(Tensor) The input tensor. Tensors with rank at most 6 are supported");
    AddOutput("Out", "(Tensor) The result tensor.");
    AddComment(R"DOC(
ReduceMean operator computes the mean of input tensor along the given dimension. 
The result tensor has 1 fewer dimension than the input unless `keep_dim` is true.
)DOC");
    AddAttr<int>("dim",
                 "(int, default 0) The dimension to reduce. "
                 "Must be in the range [-rank(input), rank(input))")
        .SetDefault(0);
    AddAttr<int>(
        "keep_dim",
        "(int, default 0) "
        "Must be 0 or 1. If 1, retain the reduced dimension with length 1.")
        .SetDefault(0);
  }
};

class ReduceMaxOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  ReduceMaxOpMaker(framework::OpProto *proto,
                   framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput(
        "X",
        "(Tensor) The input tensor. Tensors with rank at most 6 are supported");
    AddOutput("Out", "(Tensor) The result tensor.");
    AddComment(R"DOC(
ReduceMax operator computes the maximum of input tensor along the given dimension. 
The result tensor has 1 fewer dimension than the input unless `keep_dim` is true.
)DOC");
    AddAttr<int>("dim",
                 "(int, default 0) The dimension to reduce. "
                 "Must be in the range [-rank(input), rank(input))")
        .SetDefault(0);
    AddAttr<int>(
        "keep_dim",
        "(int, default 0) "
        "Must be 0 or 1. If 1, retain the reduced dimension with length 1.")
        .SetDefault(0);
  }
};

class ReduceMinOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  ReduceMinOpMaker(framework::OpProto *proto,
                   framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput(
        "X",
        "(Tensor) The input tensor. Tensors with rank at most 6 are supported");
    AddOutput("Out", "(Tensor) The result tensor.");
    AddComment(R"DOC(
ReduceMin operator computes the minimum of input tensor along the given dimension. 
The result tensor has 1 fewer dimension than the input unless `keep_dim` is true.
)DOC");
    AddAttr<int>("dim",
                 "(int, default 0) The dimension to reduce. "
                 "Must be in the range [-rank(input), rank(input))")
        .SetDefault(0);
    AddAttr<int>(
        "keep_dim",
        "(int, default 0) "
        "Must be 0 or 1. If 1, retain the reduced dimension with length 1.")
        .SetDefault(0);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP(reduce_sum, ops::ReduceOp, ops::ReduceSumOpMaker, reduce_sum_grad,
            ops::ReduceGradOp);
REGISTER_OP_CPU_KERNEL(
    reduce_sum,
    ops::ReduceKernel<paddle::platform::CPUPlace, float, ops::SumFunctor>);
REGISTER_OP_CPU_KERNEL(reduce_sum_grad,
                       ops::ReduceGradKernel<paddle::platform::CPUPlace, float,
                                             ops::SumGradFunctor>);

REGISTER_OP(reduce_mean, ops::ReduceOp, ops::ReduceMeanOpMaker,
            reduce_mean_grad, ops::ReduceGradOp);
REGISTER_OP_CPU_KERNEL(
    reduce_mean,
    ops::ReduceKernel<paddle::platform::CPUPlace, float, ops::MeanFunctor>);
REGISTER_OP_CPU_KERNEL(reduce_mean_grad,
                       ops::ReduceGradKernel<paddle::platform::CPUPlace, float,
                                             ops::MeanGradFunctor>);

REGISTER_OP(reduce_max, ops::ReduceOp, ops::ReduceMaxOpMaker, reduce_max_grad,
            ops::ReduceGradOp);
REGISTER_OP_CPU_KERNEL(
    reduce_max,
    ops::ReduceKernel<paddle::platform::CPUPlace, float, ops::MaxFunctor>);
REGISTER_OP_CPU_KERNEL(reduce_max_grad,
                       ops::ReduceGradKernel<paddle::platform::CPUPlace, float,
                                             ops::MaxOrMinGradFunctor>);

REGISTER_OP(reduce_min, ops::ReduceOp, ops::ReduceMaxOpMaker, reduce_min_grad,
            ops::ReduceGradOp);
REGISTER_OP_CPU_KERNEL(
    reduce_min,
    ops::ReduceKernel<paddle::platform::CPUPlace, float, ops::MinFunctor>);
REGISTER_OP_CPU_KERNEL(reduce_min_grad,
                       ops::ReduceGradKernel<paddle::platform::CPUPlace, float,
                                             ops::MaxOrMinGradFunctor>);

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

#include "paddle/fluid/operators/reduce_op.h"

#include <string>
#include <vector>

namespace paddle {
namespace operators {

using framework::Tensor;

class ReduceOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of ReduceOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of ReduceOp should not be null.");
    auto x_dims = ctx->GetInputDim("X");
    auto x_rank = x_dims.size();
    PADDLE_ENFORCE_LE(x_rank, 6, "Tensors with rank at most 6 are supported.");
    int dim = ctx->Attrs().Get<int>("dim");
    if (dim < 0) dim = x_rank + dim;
    PADDLE_ENFORCE_LT(
        dim, x_rank,
        "The dim should be in the range [-rank(input), rank(input)).");
    bool reduce_all = ctx->Attrs().Get<bool>("reduce_all");
    bool keep_dim = ctx->Attrs().Get<bool>("keep_dim");
    if (reduce_all) {
      if (keep_dim)
        ctx->SetOutputDim(
            "Out", framework::make_ddim(std::vector<int64_t>(x_rank, 1)));
      else
        ctx->SetOutputDim("Out", {1});
    } else {
      auto dims_vector = vectorize(x_dims);
      if (keep_dim || x_rank == 1) {
        dims_vector[dim] = 1;
      } else {
        dims_vector.erase(dims_vector.begin() + dim);
      }
      auto out_dims = framework::make_ddim(dims_vector);
      ctx->SetOutputDim("Out", out_dims);
      if (dim != 0) {
        // Only pass LoD when not reducing on the first dim.
        ctx->ShareLoD("X", /*->*/ "Out");
      }
    }
  }
};

class ReduceGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should not be null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) should not be null.");
    auto x_dims = ctx->GetInputDim("X");
    auto x_rank = x_dims.size();
    PADDLE_ENFORCE_LE(x_rank, 6, "Tensors with rank at most 6 are supported.");
    int dim = ctx->Attrs().Get<int>("dim");
    if (dim < 0) dim = x_rank + dim;
    PADDLE_ENFORCE_LT(
        dim, x_rank,
        "The dim should be in the range [-rank(input), rank(input)).");
    auto x_grad_name = framework::GradVarName("X");
    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, x_dims);
      ctx->ShareLoD("X", /*->*/ x_grad_name);
    }
  }
};

class ReduceOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  ReduceOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X",
             "(Tensor) The input tensor. Tensors with rank at most 6 are "
             "supported.");
    AddOutput("Out", "(Tensor) The result tensor.");
    AddAttr<int>(
        "dim",
        "(int, default 0) The dimension to reduce. "
        "Must be in the range [-rank(input), rank(input)). "
        "If `dim < 0`, the dim to reduce is `rank + dim`. "
        "Note that reducing on the first dim will make the LoD info lost.")
        .SetDefault(0);
    AddAttr<bool>("keep_dim",
                  "(bool, default false) "
                  "If true, retain the reduced dimension with length 1.")
        .SetDefault(false);
    AddAttr<bool>("reduce_all",
                  "(bool, default false) "
                  "If true, output a scalar reduced along all dimensions.")
        .SetDefault(false);
    comment_ = R"DOC(
{ReduceOp} Operator.

This operator computes the {reduce} of input tensor along the given dimension. 
The result tensor has 1 fewer dimension than the input unless keep_dim is true.
If reduce_all is true, just reduce along all dimensions and output a scalar.

)DOC";
    AddComment(comment_);
  }

 protected:
  std::string comment_;

  void Replace(std::string *src, std::string from, std::string to) {
    std::size_t len_from = std::strlen(from.c_str());
    std::size_t len_to = std::strlen(to.c_str());
    for (std::size_t pos = src->find(from); pos != std::string::npos;
         pos = src->find(from, pos + len_to)) {
      src->replace(pos, len_from, to);
    }
  }

  void SetComment(std::string name, std::string op) {
    Replace(&comment_, "{ReduceOp}", name);
    Replace(&comment_, "{reduce}", op);
  }
};

class ReduceSumOpMaker : public ReduceOpMaker {
 public:
  ReduceSumOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : ReduceOpMaker(proto, op_checker) {
    SetComment("ReduceSum", "sum");
    AddComment(comment_);
  }
};

class ReduceMeanOpMaker : public ReduceOpMaker {
 public:
  ReduceMeanOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : ReduceOpMaker(proto, op_checker) {
    SetComment("ReduceMean", "mean");
    AddComment(comment_);
  }
};

class ReduceMaxOpMaker : public ReduceOpMaker {
 public:
  ReduceMaxOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : ReduceOpMaker(proto, op_checker) {
    SetComment("ReduceMax", "max");
    AddComment(comment_);
  }
};

class ReduceMinOpMaker : public ReduceOpMaker {
 public:
  ReduceMinOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : ReduceOpMaker(proto, op_checker) {
    SetComment("ReduceMin", "min");
    AddComment(comment_);
  }
};

class ReduceProdOpMaker : public ReduceOpMaker {
 public:
  ReduceProdOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : ReduceOpMaker(proto, op_checker) {
    SetComment("ReduceProd", "production");
    AddComment(comment_);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(reduce_sum, ops::ReduceOp, ops::ReduceSumOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(reduce_sum_grad, ops::ReduceGradOp);

REGISTER_OPERATOR(reduce_mean, ops::ReduceOp, ops::ReduceMeanOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(reduce_mean_grad, ops::ReduceGradOp);

REGISTER_OPERATOR(reduce_max, ops::ReduceOp, ops::ReduceMaxOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(reduce_max_grad, ops::ReduceGradOp);

REGISTER_OPERATOR(reduce_min, ops::ReduceOp, ops::ReduceMinOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(reduce_min_grad, ops::ReduceGradOp);

REGISTER_OPERATOR(reduce_prod, ops::ReduceOp, ops::ReduceProdOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(reduce_prod_grad, ops::ReduceGradOp);

#define REGISTER_REDUCE_CPU_KERNEL(reduce_type, functor, grad_functor)         \
  REGISTER_OP_CPU_KERNEL(reduce_type,                                          \
                         ops::ReduceKernel<paddle::platform::CPUDeviceContext, \
                                           float, ops::functor>,               \
                         ops::ReduceKernel<paddle::platform::CPUDeviceContext, \
                                           double, ops::functor>,              \
                         ops::ReduceKernel<paddle::platform::CPUDeviceContext, \
                                           int, ops::functor>,                 \
                         ops::ReduceKernel<paddle::platform::CPUDeviceContext, \
                                           int64_t, ops::functor>);            \
  REGISTER_OP_CPU_KERNEL(                                                      \
      reduce_type##_grad,                                                      \
      ops::ReduceGradKernel<paddle::platform::CPUDeviceContext, float,         \
                            ops::grad_functor>,                                \
      ops::ReduceGradKernel<paddle::platform::CPUDeviceContext, double,        \
                            ops::grad_functor>,                                \
      ops::ReduceGradKernel<paddle::platform::CPUDeviceContext, int,           \
                            ops::grad_functor>,                                \
      ops::ReduceGradKernel<paddle::platform::CPUDeviceContext, int64_t,       \
                            ops::grad_functor>);

FOR_EACH_KERNEL_FUNCTOR(REGISTER_REDUCE_CPU_KERNEL);

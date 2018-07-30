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

#include "paddle/fluid/operators/argmax_op.h"

namespace paddle {
namespace operators {

using framework::Tensor;

class ArgExtremeOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of ArgMaxOp/ArgMinOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of ArgMaxOp/ArgMinOp should not be null.");
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
        ctx->ShareLoD("X", /*->*/ "Out");
      }
    }
  }
};

class ArgExtremeGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should not be null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@Grad) should be null");
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

class ArgExtremeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  ArgExtremeOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "(Tensor) Input of Argmax operator");
    AddOutput("Out", "(Tensor) Output of Argmax operator");
    AddAttr<int>(
        "dim",
        "(int, default 0) The axis in which to compute the arg indices. "
        "Must be in the range [-rank(input), rank(input)). "
        "if `dim < 0`, the dim to calculate is `rank + dim`. ")
        .SetDefault(0);
    AddAttr<bool>("keep_dim",
                  "(bool, default true) "
                  "Keep the reduced dimension or not.")
        .SetDefault(false);
    AddAttr<bool>("reduce_all",
                  "(bool, default false) "
                  "If true, output a scalar reduced along all dimensions.")
        .SetDefault(false);
    comment_ = R"DOC(
{ArgEx} Operator.

{ArgEx} computes the indices of the {extreme} elements of the input tensor's element along the provided axis. 
The resulted tensor has the same rank as the input if keepdims equal 1. 
If keepdims equal 0, then the resulted tensor have the reduced dimension pruned. 
The type of the output tensor is integer.

)DOC";
    AddComment(comment_);
  }

 protected:
  std::string comment_;

  void Replace(std::string &src, std::string from, std::string to) {
    std::size_t len_from = std::strlen(from.c_str());
    std::size_t len_to = std::strlen(to.c_str());
    for (std::size_t pos = src.find(from); pos != std::string::npos;
         pos = src.find(from, pos + len_to)) {
      src.replace(pos, len_from, to);
    }
  }

  void SetComment(std::string name, std::string op) {
    Replace(comment_, "{ArgEx}", name);
    Replace(comment_, "{extreme}", op);
  }
};

class ArgMaxOpMaker : public ArgExtremeOpMaker {
 public:
  ArgMaxOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : ArgExtremeOpMaker(proto, op_checker) {
    SetComment("ArgMax", "max");
    AddComment(comment_);
  }
};

class ArgMinOpMaker : public ArgExtremeOpMaker {
 public:
  ArgMinOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : ArgExtremeOpMaker(proto, op_checker) {
    SetComment("ArgMin", "min");
    AddComment(comment_);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP(argmax, ops::ArgExtremeOp, ops::ArgMaxOpMaker, argmax_grad,
            ops::ArgExtremeGradOp);

REGISTER_OP(argmin, ops::ArgExtremeOp, ops::ArgMinOpMaker, argmin_grad,
            ops::ArgExtremeGradOp);

#define REGISTER_ARGEXTREME_CPU_KERNEL(arg_type, functor, grad_functor)      \
  REGISTER_OP_CPU_KERNEL(                                                    \
      arg_type, ops::ArgExtremeKernel<paddle::platform::CPUDeviceContext,    \
                                      float, ops::functor>,                  \
      ops::ArgExtremeKernel<paddle::platform::CPUDeviceContext, double,      \
                            ops::functor>,                                   \
      ops::ArgExtremeKernel<paddle::platform::CPUDeviceContext, int,         \
                            ops::functor>,                                   \
      ops::ArgExtremeKernel<paddle::platform::CPUDeviceContext, int64_t,     \
                            ops::functor>);                                  \
  REGISTER_OP_CPU_KERNEL(                                                    \
      arg_type##_grad,                                                       \
      ops::ArgExtremeGradKernel<paddle::platform::CPUDeviceContext, float,   \
                                ops::grad_functor>,                          \
      ops::ArgExtremeGradKernel<paddle::platform::CPUDeviceContext, double,  \
                                ops::grad_functor>,                          \
      ops::ArgExtremeGradKernel<paddle::platform::CPUDeviceContext, int,     \
                                ops::grad_functor>,                          \
      ops::ArgExtremeGradKernel<paddle::platform::CPUDeviceContext, int64_t, \
                                ops::grad_functor>);

FOR_EACH_KERNEL_FUNCTOR(REGISTER_ARGEXTREME_CPU_KERNEL);

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

#include <algorithm>
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
    auto dims = ctx->Attrs().Get<std::vector<int>>("dim");
    for (size_t i = 0; i < dims.size(); ++i) {
      if (dims[i] < 0) dims[i] = x_rank + dims[i];
      PADDLE_ENFORCE_LT(
          dims[i], x_rank,
          "The dim should be in the range [-rank(input), rank(input)).");
    }
    sort(dims.begin(), dims.end());
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
      if (keep_dim) {
        for (size_t i = 0; i < dims.size(); ++i) {
          dims_vector[dims[i]] = 1;
        }
      } else {
        const int kDelFlag = -2;
        for (size_t i = 0; i < dims.size(); ++i) {
          dims_vector[dims[i]] = kDelFlag;
        }
        dims_vector.erase(
            remove(dims_vector.begin(), dims_vector.end(), kDelFlag),
            dims_vector.end());
      }
      auto out_dims = framework::make_ddim(dims_vector);
      ctx->SetOutputDim("Out", out_dims);
      if (dims[0] != 0) {
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
    auto dims = ctx->Attrs().Get<std::vector<int>>("dim");
    for (size_t i = 0; i < dims.size(); ++i) {
      if (dims[i] < 0) dims[i] = x_rank + dims[i];
      PADDLE_ENFORCE_LT(
          dims[i], x_rank,
          "The dim should be in the range [-rank(input), rank(input)).");
    }
    sort(dims.begin(), dims.end());
    auto x_grad_name = framework::GradVarName("X");
    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, x_dims);
      ctx->ShareLoD("X", /*->*/ x_grad_name);
    }
  }
};

class ReduceOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() final {
    AddInput("X",
             "(Tensor) The input tensor. Tensors with rank at most 6 are "
             "supported.");
    AddOutput("Out", "(Tensor) The result tensor.");
    AddAttr<std::vector<int>>(
        "dim",
        "(list<int>, default {0}) The dimensions to reduce. "
        "Must be in the range [-rank(input), rank(input)). "
        "If `dim[i] < 0`, the dims[i] to reduce is `rank + dims[i]`. "
        "Note that reducing on the first dim will make the LoD info lost.")
        .SetDefault({0});
    AddAttr<bool>("keep_dim",
                  "(bool, default false) "
                  "If true, retain the reduced dimension with length 1.")
        .SetDefault(false);
    AddAttr<bool>("reduce_all",
                  "(bool, default false) "
                  "If true, output a scalar reduced along all dimensions.")
        .SetDefault(false);
    AddComment(string::Sprintf(R"DOC(
%s Operator.

This operator computes the %s of input tensor along the given dimension.
The result tensor has 1 fewer dimension than the input unless keep_dim is true.
If reduce_all is true, just reduce along all dimensions and output a scalar.

)DOC",
                               GetOpType(), GetName()));
  }

 protected:
  virtual std::string GetName() const = 0;
  virtual std::string GetOpType() const = 0;
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

#define REGISTER_REDUCE_OP(op_name)                                        \
  class __##op_name##Maker__ : public ops::ReduceOpMaker {                 \
   protected:                                                              \
    virtual std::string GetName() const { return #op_name; }               \
    virtual std::string GetOpType() const { return "Reduce " #op_name; }   \
  };                                                                       \
  REGISTER_OPERATOR(reduce_##op_name, ops::ReduceOp, __##op_name##Maker__, \
                    paddle::framework::DefaultGradOpDescMaker<true>);      \
  REGISTER_OPERATOR(reduce_##op_name##_grad, ops::ReduceGradOp)

REGISTER_REDUCE_OP(sum);
REGISTER_REDUCE_OP(mean);
REGISTER_REDUCE_OP(max);
REGISTER_REDUCE_OP(min);
REGISTER_REDUCE_OP(prod);

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

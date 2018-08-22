// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

inline void GetPrePostForStackOp(const framework::DDim &dim, int axis, int *pre,
                                 int *post) {
  *pre = 1;
  for (auto i = 0; i < axis; ++i) (*pre) *= dim[i];
  *post = 1;
  for (auto i = axis; i < dim.size(); ++i) (*post) *= dim[i];
}

class StackOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_GT(ctx->Inputs("X").size(), 0,
                      "Number of Inputs(X) must be larger than 0");
    PADDLE_ENFORCE(ctx->HasOutput("Y"), "Output(Y) must exist.");

    auto input_dims = ctx->GetInputsDim("X");
    for (size_t i = 1; i < input_dims.size(); ++i) {
      PADDLE_ENFORCE_EQ(input_dims[i], input_dims[0],
                        "Dims of all Inputs(X) must be the same");
    }

    // Only lod of X[0] would be shared with Y
    ctx->ShareLoD("X", /*->*/ "Y");

    int axis = ctx->Attrs().Get<int>("axis");
    int rank = input_dims[0].size();
    PADDLE_ENFORCE(
        axis >= -(rank + 1) && axis < rank + 1,
        "Attr(axis) must be inside [-(rank+1), rank+1), where rank = %d", rank);
    if (axis < 0) axis += (rank + 1);

    auto vec = framework::vectorize2int(input_dims[0]);
    vec.insert(vec.begin() + axis, input_dims.size());
    ctx->SetOutputDim("Y", framework::make_ddim(vec));
  }
};

class StackOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input of stack op.").AsDuplicable();
    AddOutput("Y", "The output of stack op.");
    AddAttr<int>("axis",
                 "The axis along which all of the Inputs(X) should be stacked.")
        .SetDefault(0);
    AddComment(R"DOC(
      Stack Operator.

      Stack all of the Inputs(X) into one tensor along Attr(axis). The dims of all Inputs(X) must be the same.
    )DOC");
  }
};

template <typename DeviceContext, typename T, typename Functor>
class StackKernel : public framework::OpKernel<T> {
  using Tensor = framework::LoDTensor;

 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto x = ctx.MultiInput<Tensor>("X");
    auto *y = ctx.Output<Tensor>("Y");

    int axis = ctx.Attr<int>("axis");
    if (axis < 0) axis += (x[0]->dims().size() + 1);

    int n = static_cast<int>(x.size());
    auto *y_data = y->mutable_data<T>(ctx.GetPlace());
    std::vector<const T *> x_datas(n);
    for (int i = 0; i < n; i++) x_datas[i] = x[i]->data<T>();

    int pre = 1, post = 1;
    auto &dim = x[0]->dims();
    for (auto i = 0; i < axis; ++i) pre *= dim[i];
    for (auto i = axis; i < dim.size(); ++i) post *= dim[i];

    Functor functor;
    functor(ctx.template device_context<DeviceContext>(), x_datas, y_data, pre,
            n, post);
  }
};

class StackOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Y")),
                   "Input(Y@Grad) must exist.");

    int axis = ctx->Attrs().Get<int>("axis");
    auto dy_dim = ctx->GetInputDim(framework::GradVarName("Y"));
    int rank = dy_dim.size();
    PADDLE_ENFORCE(axis >= -rank && axis < rank,
                   "Attr(axis) must be inside [-rank, rank), where rank = %d",
                   rank);
    if (axis < 0) axis += rank;

    PADDLE_ENFORCE_EQ(ctx->Outputs(framework::GradVarName("X")).size(),
                      static_cast<size_t>(dy_dim[axis]),
                      "Number of Outputs(X@Grad) is wrong");
    auto vec = framework::vectorize2int(dy_dim);
    vec.erase(vec.begin() + axis);
    ctx->SetOutputsDim(
        framework::GradVarName("X"),
        std::vector<framework::DDim>(dy_dim[axis], framework::make_ddim(vec)));
  }
};

class StackGradOpDescMaker
    : public framework::
          SingleGradOpDescMaker /*framework::GradOpDescMakerBase*/ {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;
  /*
  using framework::GradOpDescMakerBase::GradOpDescMakerBase;

  std::vector<std::unique_ptr<framework::OpDesc>> operator ()() const override {
                auto x_grads = InputGrad("X", false);
    std::vector<std::unique_ptr<framework::OpDesc>> grad_ops;
    grad_ops.reserve(x_grads.size());
    auto og = OutputGrad("Y");
    std::transform(x_grads.begin(), x_grads.end(), std::back_inserter(grad_ops),
                   [&og](const std::string& x_grad) {
                     auto* grad_op = new framework::OpDesc();
                     grad_op->SetInput("X", og);
                     grad_op->SetOutput("Y", {x_grad});
                     grad_op->SetAttrMap(Attrs());
                     return std::unique_ptr<framework::OpDesc>(grad_op);
                   });
    return grad_ops;
  }
  */

  std::unique_ptr<framework::OpDesc> Apply() const override {
    std::unique_ptr<framework::OpDesc> op(new framework::OpDesc());
    op->SetType("stack_grad");
    op->SetInput(framework::GradVarName("Y"), OutputGrad("Y"));
    op->SetOutput(framework::GradVarName("X"), InputGrad("X", false));
    op->SetAttrMap(Attrs());
    return op;
  }
};

template <typename DeviceContext, typename T, typename GradFunctor>
class StackGradKernel : public framework::OpKernel<T> {
  using Tensor = framework::LoDTensor;

 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *dy = ctx.Input<Tensor>(framework::GradVarName("Y"));
    auto dx = ctx.MultiOutput<Tensor>(framework::GradVarName("X"));
    int axis = ctx.Attr<int>("axis");
    if (axis < 0) axis += dy->dims().size();

    int n = dy->dims()[axis];
    std::vector<T *> dx_datas(n);  // NOLINT
    for (int i = 0; i < n; i++)
      dx_datas[i] = dx[i]->mutable_data<T>(ctx.GetPlace());
    auto dy_data = dy->data<T>();

    int pre = 1;
    for (int i = 0; i < axis; ++i) pre *= dy->dims()[i];
    int post = dy->numel() / (n * pre);
    GradFunctor functor;
    functor(ctx.template device_context<DeviceContext>(), dx_datas, dy_data,
            pre, n, post);
  }
};

}  // namespace operators
}  // namespace paddle

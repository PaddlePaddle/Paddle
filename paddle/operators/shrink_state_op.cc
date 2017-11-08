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
#include "paddle/framework/lod_rank_table.h"
#include "paddle/operators/array_operator.h"
#include "paddle/operators/math/math_function.h"

namespace paddle {
namespace operators {

class ShrinkStateOp : public ArrayOp {
 public:
  ShrinkStateOp(const std::string &type,
                const framework::VariableNameMap &inputs,
                const framework::VariableNameMap &outputs,
                const framework::AttributeMap &attrs)
      : ArrayOp(type, inputs, outputs, attrs) {}

  void Run(const framework::Scope &scope,
           const platform::DeviceContext &dev_ctx) const override {
    auto *x_var = scope.FindVar(Input("X"));
    PADDLE_ENFORCE(x_var != nullptr, "Input X must be set");
    auto &x_tensor = x_var->Get<framework::LoDTensor>();
    size_t offset = this->GetOffset(scope, dev_ctx);
    auto *rank_table_var = scope.FindVar(Input("RankTable"));
    PADDLE_ENFORCE(rank_table_var != nullptr, "RankTable must be set");
    auto &rank_table = rank_table_var->Get<framework::LoDRankTable>();

    int dst_num_rows = 0;

    {
      auto &rank_items = rank_table.items();
      for (auto &rank_item : rank_items) {
        if (offset < rank_item.length) {
          ++dst_num_rows;
        } else {
          break;
        }
      }
    }

    auto *out_var = scope.FindVar(Output("Out"));
    PADDLE_ENFORCE(out_var != nullptr, "Output Out must be set");
    auto &out_tensor = *out_var->GetMutable<framework::LoDTensor>();
    if (dst_num_rows != 0) {
      out_tensor.ShareDataWith(x_tensor.Slice(0, dst_num_rows));
    }
  }
};

class ShrinkStateOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  ShrinkStateOpProtoMaker(framework::OpProto *proto,
                          framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "");
    AddInput("RankTable", "");
    AddInput("I", "");
    AddOutput("Out", "");
    AddComment("");
  }
};

class ShrinkStateOpInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *context) const override {
    PADDLE_ENFORCE(context->HasInput("X"));
    PADDLE_ENFORCE(context->HasInput("I"));
    PADDLE_ENFORCE(context->HasInput("RankTable"));
    context->SetOutputDim("Out", context->GetInputDim("X"));
  }
};

class ShrinkStateGradOp : public ArrayOp {
 public:
  ShrinkStateGradOp(const std::string &type,
                    const framework::VariableNameMap &inputs,
                    const framework::VariableNameMap &outputs,
                    const framework::AttributeMap &attrs)
      : ArrayOp(type, inputs, outputs, attrs) {}

  void Run(const framework::Scope &scope,
           const platform::DeviceContext &dev_ctx) const override {
    auto *dout_var = scope.FindVar(Input(framework::GradVarName("Out")));
    auto dx_name = Output(framework::GradVarName("X"));
    auto *dx_var = scope.FindVar(dx_name);
    PADDLE_ENFORCE(dx_var != nullptr, "Input Gradient should not be nullptr");
    auto *x_var = scope.FindVar(Input("X"));
    PADDLE_ENFORCE(x_var != nullptr);

    auto &x_tensor = x_var->Get<framework::LoDTensor>();
    auto &dx_tensor = *dx_var->GetMutable<framework::LoDTensor>();
    dx_tensor.Resize(x_tensor.dims());
    dx_tensor.mutable_data(x_tensor.place(), x_tensor.type());

    if (dout_var == nullptr) {  // dx_tensor fill zero
      math::set_constant(dev_ctx, &dx_tensor, 0.0f);
    } else {
      auto &dout_tensor = dout_var->Get<framework::LoDTensor>();
      auto height = dout_tensor.dims()[0];
      dx_tensor.Slice(0, static_cast<int>(height))
          .CopyFrom(dout_tensor, dout_tensor.place(), dev_ctx);
      if (height < dout_tensor.dims()[0]) {
        auto rest_tensor = dx_tensor.Slice(
            static_cast<int>(height), static_cast<int>(dout_tensor.dims()[0]));
        math::set_constant(dev_ctx, &rest_tensor, 0.0f);
      }
    }
  }
};

class ShrikStateGradInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *context) const override {
    PADDLE_ENFORCE(context->HasInput("X"));
    PADDLE_ENFORCE(context->HasOutput(framework::GradVarName("X")));
    context->SetOutputDim(framework::GradVarName("X"),
                          context->GetInputDim("X"));
  }
};

class ShrinkStateGradOpMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDescBind> Apply() const override {
    auto *op = new framework::OpDescBind();
    op->SetType("shrink_state_grad");
    op->SetInput("X", Input("X"));
    op->SetInput(framework::GradVarName("Out"), OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), InputGrad("X"));
    op->SetAttrMap(Attrs());
    return std::unique_ptr<framework::OpDescBind>(op);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(shrink_state, ops::ShrinkStateOp,
                  ops::ShrinkStateOpInferShape, ops::ShrinkStateOpProtoMaker,
                  ops::ShrinkStateGradOpMaker);
REGISTER_OPERATOR(shrink_state_grad, ops::ShrinkStateGradOp,
                  ops::ShrikStateGradInferShape);

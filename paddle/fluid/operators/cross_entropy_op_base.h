// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>
#include <unordered_map>
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

class CrossEntropyOpBase : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput("Label"), "Input(Label) should be not null.");

    PADDLE_ENFORCE(ctx->HasOutput("Y"), "Output(Y) should be not null.");

    auto x_dims = ctx->GetInputDim("X");
    auto label_dims = ctx->GetInputDim("Label");
    int rank = x_dims.size();
    PADDLE_ENFORCE_EQ(rank, label_dims.size(),
                      "Input(X) and Input(Label) shall have the same rank.");
    bool check = true;
    if ((!ctx->IsRuntime()) && (framework::product(x_dims) <= 0 ||
                                framework::product(label_dims) <= 0)) {
      check = false;
    }
    if (check) {
      PADDLE_ENFORCE_EQ(framework::slice_ddim(x_dims, 0, rank - 1),
                        framework::slice_ddim(label_dims, 0, rank - 1),
                        "Input(X) and Input(Label) shall have the same shape "
                        "except the last dimension.");
    }

    if (IsSoftLabel(ctx)) {
      if (check) {
        PADDLE_ENFORCE_EQ(x_dims[rank - 1], label_dims[rank - 1],
                          "If Attr(soft_label) == true, the last dimension of "
                          "Input(X) and Input(Label) should be equal.");
      }
    } else {
      PADDLE_ENFORCE_EQ(label_dims[rank - 1], 1UL,
                        "If Attr(softLabel) == false, the last dimension of "
                        "Input(Label) should be 1.");
    }

    auto y_dims = x_dims;
    y_dims[rank - 1] = 1;
    ctx->SetOutputDim("Y", y_dims);
    ctx->ShareLoD("X", /*->*/ "Y");
  }

 protected:
  // Explicitly set that the data type of computation kernel of cross_entropy
  // is determined by its input "X".
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(ctx.Input<Tensor>("X")->type(),
                                   ctx.device_context());
  }

  virtual bool IsSoftLabel(framework::InferShapeContext* ctx) const {
    return ctx->Attrs().Get<bool>("soft_label");
  }
};

class CrossEntropyOpInferVarType
    : public framework::PassInDtypeAndVarTypeToOutput {
 protected:
  std::unordered_map<std::string, std::string> GetInputOutputWithSameType()
      const override {
    return std::unordered_map<std::string, std::string>{{"X", /*->*/ "Y"}};
  }
};

class CrossEntropyGradientOpBase : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const {
    PADDLE_ENFORCE(ctx->HasInput("Label"), "Input(Label) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Y")),
                   "Input(Y@GRAD) shoudl be not null.");
    PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("X")),
                   "Output(X@GRAD) should be not null.");

    auto x_dims = GetXDim(ctx);
    auto label_dims = ctx->GetInputDim("Label");
    auto dy_dims = ctx->GetInputDim(framework::GradVarName("Y"));
    int rank = x_dims.size();
    PADDLE_ENFORCE_EQ(dy_dims.size(), rank,
                      "Input(Y@Grad) and Input(X) should have the same rank.");
    PADDLE_ENFORCE_EQ(label_dims.size(), rank,
                      "Input(Label) and Input(X) should have the same rank.");

    bool check = true;
    if ((!ctx->IsRuntime()) && (framework::product(x_dims) <= 0 ||
                                framework::product(label_dims) <= 0)) {
      check = false;
    }

    if (check) {
      PADDLE_ENFORCE_EQ(framework::slice_ddim(x_dims, 0, rank - 1),
                        framework::slice_ddim(label_dims, 0, rank - 1),
                        "The Input(X) and Input(Label) should have the same "
                        "shape except the last dimension.");
      PADDLE_ENFORCE_EQ(framework::slice_ddim(x_dims, 0, rank - 1),
                        framework::slice_ddim(dy_dims, 0, rank - 1),
                        "The Input(X) and Input(Y@Grad) should have the same "
                        "shape except the last dimension.");
    }
    if (IsSoftLabel(ctx)) {
      if (check) {
        PADDLE_ENFORCE_EQ(
            x_dims[rank - 1], label_dims[rank - 1],
            "When Attr(soft_label) == true, the last dimension of "
            "Input(X) and Input(Label) should be equal.");
      }
    } else {
      PADDLE_ENFORCE_EQ(label_dims[rank - 1], 1,
                        "When Attr(soft_label) == false, the last dimension of "
                        "Input(Label) should be 1.");
    }
    ctx->SetOutputDim(framework::GradVarName("X"), x_dims);
    PADDLE_ENFORCE_EQ(dy_dims[rank - 1], 1,
                      "The last dimension of Input(Y@Grad) should be 1.");
    ctx->SetOutputDim(framework::GradVarName("X"), x_dims);
    ctx->ShareLoD(VarNameWithXLoD(), framework::GradVarName("X"));
  }

 protected:
  // Explicitly set that the data type of computation kernel of cross_entropy
  // is determined by its input "X".
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        ctx.Input<Tensor>(framework::GradVarName("Y"))->type(),
        ctx.device_context());
  }

  virtual framework::DDim GetXDim(framework::InferShapeContext* ctx) const {
    return ctx->GetInputDim("X");
  }

  virtual const char* VarNameWithXLoD() const { return "X"; }

  virtual bool IsSoftLabel(framework::InferShapeContext* ctx) const {
    return ctx->Attrs().Get<bool>("soft_label");
  }
};

}  // namespace operators
}  // namespace paddle

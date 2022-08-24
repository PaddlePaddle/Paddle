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

#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace operators {

using framework::Tensor;

class TileOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name,
      const Tensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const override {
    if (var_name == "repeat_times_tensor" || var_name == "RepeatTimes") {
      return expected_kernel_type;
    }
    return framework::OpKernelType(
        expected_kernel_type.data_type_, tensor.place(), tensor.layout());
  }
};

class TileOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensor, default Tensor<float>). X is the input to be titled.");
    AddInput(
        "RepeatTimes",
        "(Tensor<int>, optional). If provided, it is the number of repeat times"
        " along specific axis. It has a higher priority than "
        "repeat_times_tensor and the repeat_times attribute.")
        .AsDispensable();
    AddInput("repeat_times_tensor",
             "(Tensor Tensor<int>), repeat times for X."
             "It has a higher priority than repeat_times, but a lower priority "
             "than RepeatTimes")
        .AsDuplicable()
        .AsDispensable();
    AddOutput("Out",
              "(Tensor, default Tensor<float>). A tensor with rank in [1, 6]."
              "After tiling, size of each dimension of Output(Out) is equal "
              "to size of the corresponding dimension of Input(X) multiplying "
              "the corresponding value given by Attr(repeat_times).");
    AddAttr<std::vector<int>>("repeat_times",
                              "The number of repeat times for each dimension.")
        .SetDefault({})
        .SupportTensor();
    AddComment(R"DOC(
Tile operator repeats the input by given times number. You should set times
number for each dimension by providing attribute 'repeat_times'. The rank of X
should be in [1, 6]. Please note that size of 'repeat_times' must be the same
with X's rank. Following is a using case:

Input(X) is a 3-D tensor with shape [2, 3, 1]:

        [
           [[1], [2], [3]],
           [[4], [5], [6]]
        ]

Attr(repeat_times):  [1, 2, 2]

Output(Out) is a 3-D tensor with shape [2, 6, 2]:

        [
            [[1, 1], [2, 2], [3, 3], [1, 1], [2, 2], [3, 3]],
            [[4, 4], [5, 5], [6, 6], [4, 4], [5, 5], [6, 6]]
        ]

)DOC");
  }
};

class TileGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "TileGrad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")),
                   "Input",
                   framework::GradVarName("Out"),
                   "TileGrad");

    auto x_dims = ctx->GetInputDim("X");
    auto x_grad_name = framework::GradVarName("X");

    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, x_dims);
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.device_context());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name,
      const Tensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const override {
    if (var_name == "repeat_times_tensor" || var_name == "RepeatTimes") {
      return expected_kernel_type;
    }
    return framework::OpKernelType(
        expected_kernel_type.data_type_, tensor.place(), tensor.layout());
  }
};

template <typename T>
class TileGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("tile_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetInput("repeat_times_tensor", this->Input("repeat_times_tensor"));
    op->SetInput("RepeatTimes", this->Input("RepeatTimes"));
    op->SetAttrMap(this->Attrs());
  }
};

template <typename T>
class TileDoubleGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("tile");
    op->SetInput("X", this->OutputGrad(framework::GradVarName("X")));
    op->SetOutput("Out", this->InputGrad(framework::GradVarName("Out")));
    if (this->HasInput("repeat_times_tensor")) {
      op->SetInput("repeat_times_tensor", this->Input("repeat_times_tensor"));
    }
    if (this->HasInput("RepeatTimes")) {
      op->SetInput("RepeatTimes", this->Input("RepeatTimes"));
    }
    op->SetAttrMap(this->Attrs());
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(TileGradNoNeedBufVarsInferer, "X");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

DECLARE_INFER_SHAPE_FUNCTOR(tile,
                            TileInferMetaFunctor,
                            PD_INFER_META(phi::TileInferMeta));

REGISTER_OPERATOR(tile,
                  ops::TileOp,
                  ops::TileOpMaker,
                  ops::TileGradOpMaker<paddle::framework::OpDesc>,
                  ops::TileGradOpMaker<paddle::imperative::OpBase>,
                  TileInferMetaFunctor);
REGISTER_OPERATOR(tile_grad,
                  ops::TileGradOp,
                  ops::TileDoubleGradOpMaker<paddle::framework::OpDesc>,
                  ops::TileDoubleGradOpMaker<paddle::imperative::OpBase>,
                  ops::TileGradNoNeedBufVarsInferer);

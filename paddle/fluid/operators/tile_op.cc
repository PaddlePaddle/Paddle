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

#include "paddle/fluid/operators/tile_op.h"
#include <memory>
#include <string>
#include <vector>

namespace paddle {
namespace operators {

using framework::Tensor;

class TileOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "Tile");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "Tile");
    auto x_dims = ctx->GetInputDim("X");
    auto repeat_times = ctx->Attrs().Get<std::vector<int>>("repeat_times");

    if (repeat_times.size() == 0) {
      repeat_times = std::vector<int>(x_dims.size(), -1);
    }

    PADDLE_ENFORCE_EQ(
        static_cast<size_t>(x_dims.size()), repeat_times.size(),
        platform::errors::InvalidArgument(
            "The number of elements (%d) of 'repeat_times' for "
            "Op(tile) must be equal to the number of dimensions "
            "(%d) of the input.",
            repeat_times.size(), static_cast<size_t>(x_dims.size())));
    PADDLE_ENFORCE_LE(
        x_dims.size(), 6,
        platform::errors::InvalidArgument(
            "The number of dimensions of the input for Op(tile) "
            "must not be greater than 6, but the value received is %d.",
            x_dims.size()));

    std::vector<int64_t> out_shape(x_dims.size());
    for (size_t i = 0; i < repeat_times.size(); ++i) {
      if (x_dims[i] == -1 || repeat_times[i] == -1) {
        out_shape[i] = -1;
      } else {
        PADDLE_ENFORCE_GT(
            repeat_times[i], 0,
            platform::errors::InvalidArgument(
                "The %uth element of 'repeat_times' for Op(tile) must be "
                "greater than 0, but the value given is %d.",
                i, repeat_times[i]));
        out_shape[i] = x_dims[i] * repeat_times[i];
      }
    }

    ctx->SetOutputDim("Out", framework::make_ddim(out_shape));
    if (out_shape[0] == x_dims[0]) {
      ctx->ShareLoD("X", "Out");
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name, const Tensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const override {
    if (var_name == "repeat_times_tensor" || var_name == "RepeatTimes") {
      return expected_kernel_type;
    }
    return framework::OpKernelType(expected_kernel_type.data_type_,
                                   tensor.place(), tensor.layout());
  }
};

class TileOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensor, default Tensor<float>). A tensor with rank in [1, 6]."
             "X is the input to be expanded.");
    AddInput(
        "TileTimes",
        "(Tensor<int>), optional). If provided, repeat along specific "
        "axis according to this given value. It has a higher priority than "
        "repeat_times_tensor and repeat_times.")
        .AsDispensable();
    AddInput("repeat_times_tensor",
             "(Tensor Tensor<int>), repeat times for X."
             "It has a higher priority than repeat_times, but a lower priority "
             "than RepeatTimes")
        .AsDuplicable()
        .AsDispensable();
    AddOutput("Out",
              "(Tensor, default Tensor<float>). A tensor with rank in [1, 6]."
              "The rank of Output(Out) have the same with Input(X). "
              "After tiling, size of each dimension of Output(Out) is equal "
              "to size of the corresponding dimension of Input(X) multiplying "
              "the corresponding value given by Attr(repeat_times).");
    AddAttr<std::vector<int>>("repeat_times",
                              "Repeat times number for each dimension.")
        .SetDefault({});
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
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   framework::GradVarName("Out"), "TileGrad");

    auto x_dims = ctx->GetInputDim("X");
    std::vector<int> repeat_times =
        ctx->Attrs().Get<std::vector<int>>("repeat_times");

    auto out_dims = ctx->GetInputDim(framework::GradVarName("Out"));

    size_t start_pos = 0u;
    if (!ctx->IsRuntime() && x_dims[0] < 0) {
      PADDLE_ENFORCE_EQ(
          x_dims[0], out_dims[0],
          platform::errors::InvalidArgument(
              "The first dimension size (%d) of Input(Out@GRAD) should be "
              "equal to the crroresponding dimension size (%d) of Input(X)",
              out_dims[0], x_dims[0]));
      start_pos = 1u;
    }

    for (size_t i = start_pos; i < repeat_times.size(); ++i) {
      if (repeat_times[i] == -1) {
        continue;
      } else {
        if (ctx->IsRuntime()) {
          PADDLE_ENFORCE_EQ(
              x_dims[i] * repeat_times[i], out_dims[i],
              platform::errors::InvalidArgument(
                  "The %uth dimension size (%d) of Input(Out@GRAD) should be "
                  "equal to the multiplication of the crroresponding dimension "
                  "sizes of Input(X) (%d) and repeat_times (%d).",
                  i, out_dims[i], x_dims[i], repeat_times[i]));
        }
      }
    }
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
      const std::string& var_name, const Tensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const override {
    if (var_name == "repeat_times_tensor") {
      return expected_kernel_type;
    }
    return framework::OpKernelType(expected_kernel_type.data_type_,
                                   tensor.place(), tensor.layout());
  }
};

template <typename T>
class TileGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("repeat_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetInput("repeat_times_tensor", this->Input("repeat_times_tensor"));
    op->SetInput("RepeatTimes", this->Input("RepeatTimes"));
    op->SetAttrMap(this->Attrs());
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(TileGradNoNeedBufVarsInferer, "X");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(tile, ops::TileOp, ops::TileOpMaker,
                  ops::TileGradOpMaker<paddle::framework::OpDesc>,
                  ops::TileGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(tile_grad, ops::TileGradOp,
                  ops::TileGradNoNeedBufVarsInferer);
REGISTER_OP_CPU_KERNEL(
    tile, ops::TileKernel<paddle::platform::CPUDeviceContext, float>,
    ops::TileKernel<paddle::platform::CPUDeviceContext, double>,
    ops::TileKernel<paddle::platform::CPUDeviceContext, int>,
    ops::TileKernel<paddle::platform::CPUDeviceContext, int64_t>,
    ops::TileKernel<paddle::platform::CPUDeviceContext, bool>);
REGISTER_OP_CPU_KERNEL(
    tile_grad, ops::TileGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::TileGradKernel<paddle::platform::CPUDeviceContext, double>,
    ops::TileGradKernel<paddle::platform::CPUDeviceContext, int>,
    ops::TileGradKernel<paddle::platform::CPUDeviceContext, int64_t>);

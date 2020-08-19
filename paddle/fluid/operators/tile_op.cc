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

    PADDLE_ENFORCE_LE(
        x_dims.size(), MAX_RANK_SUPPORTED,
        platform::errors::InvalidArgument(
            "The rank of the input 'x' for tile op "
            "must not be greater than %d, but the value received is %d.",
            MAX_RANK_SUPPORTED, x_dims.size()));
    PADDLE_ENFORCE_LE(
        repeat_times.size(), MAX_RANK_SUPPORTED,
        platform::errors::InvalidArgument(
            "The size of the shape of input 'repeat_times' for tile op "
            "must not be greater than %d, but the value received is %d.",
            MAX_RANK_SUPPORTED, repeat_times.size()));
    PADDLE_ENFORCE_GE(
        repeat_times.size(), 1,
        platform::errors::InvalidArgument(
            "The size of the shape of input 'repeat_times' for tile op "
            "must be positive integers, but the value received is %d.",
            repeat_times.size()));

    auto out_rank =
        std::max(static_cast<size_t>(x_dims.size()), repeat_times.size());
    std::vector<int64_t> out_shape(out_rank);
    auto x_dim_vec = framework::vectorize<int>(x_dims);
    if (x_dim_vec.size() > repeat_times.size()) {
      auto diff = x_dim_vec.size() - repeat_times.size();
      repeat_times.insert(repeat_times.begin(), diff, -1);
    } else {
      auto diff = repeat_times.size() - x_dim_vec.size();
      x_dim_vec.insert(x_dim_vec.begin(), diff, -1);
    }
    for (size_t i = 0; i < repeat_times.size(); ++i) {
      if (x_dim_vec[i] == -1 || repeat_times[i] == -1) {
        out_shape[i] = -1;
      } else {
        PADDLE_ENFORCE_GT(
            repeat_times[i], 0,
            platform::errors::InvalidArgument(
                "Every element of the input 'repeat_times' for tile op must be "
                "greater than 0, but the value given is %d.",
                repeat_times[i]));
        out_shape[i] = x_dim_vec[i] * repeat_times[i];
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
    if (repeat_times.size() == 0) {
      repeat_times = std::vector<int>(x_dims.size(), -1);
    }

    auto out_dims = ctx->GetInputDim(framework::GradVarName("Out"));
    auto x_dim_vec = framework::vectorize<int>(x_dims);
    if (x_dim_vec.size() > repeat_times.size()) {
      auto diff = x_dim_vec.size() - repeat_times.size();
      repeat_times.insert(repeat_times.begin(), diff, -1);
    } else {
      auto diff = repeat_times.size() - x_dim_vec.size();
      x_dim_vec.insert(x_dim_vec.begin(), diff, -1);
    }

    for (size_t i = 0; i < repeat_times.size(); ++i) {
      if (repeat_times[i] == -1 || x_dim_vec[i] == -1) {
        continue;
      } else {
        if (ctx->IsRuntime()) {
          PADDLE_ENFORCE_EQ(
              x_dim_vec[i] * repeat_times[i], out_dims[i],
              platform::errors::InvalidArgument(
                  "The size (%d) of the dimension %d of Input(Out@GRAD) should "
                  "be equal to the multiplication of the crroresponding "
                  "dimension size of Input(X) (%d) and repeat_times (%d).",
                  out_dims[i], i, x_dim_vec[i], repeat_times[i]));
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
    if (var_name == "repeat_times_tensor" || var_name == "RepeatTimes") {
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
    op->SetType("tile_grad");
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

/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/strided_slice_op.h"
#include <algorithm>
#include <memory>
#include <vector>

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

class StridedSliceOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("Input"), true,
                      "Input (Input) of slice op should not be null.");
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Out"), true,
                      "Output (Out) of slice op should not be null.");

    auto in_dims = ctx->GetInputDim("Input");
    PADDLE_ENFORCE_LT(in_dims.size(), 7,
                      "The rank of input should be less than 7.");
    auto starts = ctx->Attrs().Get<std::vector<int>>("starts");
    auto ends = ctx->Attrs().Get<std::vector<int>>("ends");
    auto strides = ctx->Attrs().Get<std::vector<int>>("strides");
    auto axes = ctx->Attrs().Get<std::vector<int>>("axes");

    PADDLE_ENFORCE_EQ(starts.size(), ends.size(),
                      "starts and ends dim size must to be same");
    PADDLE_ENFORCE_EQ(ends.size(), strides.size(),
                      "ends and strides dim size must to be same");
    PADDLE_ENFORCE_EQ(ends.size(), axes.size(),
                      "axes, end and start dim size must to be same");

    // we need to analysis strided slice op is valid for
    // the parameter that we get from python front
    int stride_index, start_index, end_index;
    std::vector<int> out_dims_vector(in_dims.size());
    for (int i = 0; i < in_dims.size(); i++) {
      out_dims_vector[i] = in_dims[i];
    }
    for (size_t i = 0; i < starts.size(); i++) {
      PADDLE_ENFORCE_NE(strides[i], 0, "stride must not to be zero");
      int axes_index = axes[i];
      start_index = starts[i];
      end_index = ends[i];
      stride_index = strides[i];
      int axis_size = in_dims[axes_index];
      if (axis_size < 0) {
        continue;
      }

      if (start_index < 0) {
        start_index = start_index + axis_size;
      }
      if (end_index < 0) {
        end_index = end_index + axis_size;
      }

      if (stride_index < 0) {
        start_index = start_index + 1;
        end_index = end_index + 1;
      }

      bool zero_dim_condition =
          ((stride_index < 0 && (start_index <= end_index)) ||
           (stride_index > 0 && (start_index >= end_index)));
      PADDLE_ENFORCE_EQ(zero_dim_condition, false,
                        "starts and end must meet requirement in different "
                        "stride conditiont");
      int left = std::max(0, std::min(start_index, end_index));
      int right = std::min(axis_size, std::max(start_index, end_index));
      int step = std::abs(stride_index);
      auto out_dims_index = (std::abs(right - left) + step - 1) / step;

      out_dims_vector[axes_index] = out_dims_index;
    }
    framework::DDim out_dims(framework::make_ddim(out_dims_vector));

    ctx->SetOutputDim("Out", out_dims);
    ctx->ShareLoD("Input", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(ctx.Input<Tensor>("Input")->type(),
                                   ctx.Input<Tensor>("Input")->place());
  }
};

class StridedSliceOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Input", "Tensor of data to extract slices from.");
    AddOutput("Out", "Sliced data tensor.");

    AddAttr<std::vector<int>>(
        "axes", "(list<int> Axes stride from the start to the end)");
    AddAttr<std::vector<int>>(
        "starts", "(list<int>)  start  that the tensor slice start.");
    AddAttr<std::vector<int>>("ends",
                              "(list<int>) end that the tensor slice end");
    AddAttr<std::vector<int>>(
        "strides", "(list<int> stride stride from the start to the end)");
    AddComment(R"DOC(
Strided Slice Operator.
Instead of calling this op directly most users will want to use the
NumPy-style slicing syntax.
For Example:
data = fluid.layers.fill_constant(shape=[3, 3], value=0, dtype='int64')
y = fluid.layers.strided_slice(data, [0, 1], [1,0], [2, 3], [1, 1])
)DOC");
  }
};

class StridedSliceOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("Input"), true, "Input should not be null");
    PADDLE_ENFORCE_EQ(ctx->HasInput(framework::GradVarName("Out")), true,
                      "Input(Out@GRAD) should not be null");
    auto x_dims = ctx->GetInputDim("Input");
    auto x_grad_name = framework::GradVarName("Input");
    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, x_dims);
    }
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        ctx.Input<framework::Tensor>(framework::GradVarName("Out"))->type(),
        ctx.GetPlace());
  }
};

class StridedSliceOpGradMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    auto* bind = new framework::OpDesc();
    bind->SetInput(framework::GradVarName("Out"), OutputGrad("Out"));
    bind->SetInput("Input", Input("Input"));
    bind->SetOutput(framework::GradVarName("Input"), InputGrad("Input"));
    bind->SetAttrMap(Attrs());
    bind->SetType("strided_slice_grad");
    return std::unique_ptr<framework::OpDesc>(bind);
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERENCE(
    StridedSliceOpGradNoNeedBufferVarsInference, "Input");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(strided_slice, ops::StridedSliceOp, ops::StridedSliceOpMaker,
                  ops::StridedSliceOpGradMaker);
REGISTER_OPERATOR(strided_slice_grad, ops::StridedSliceOpGrad,
                  ops::StridedSliceOpGradNoNeedBufferVarsInference);

REGISTER_OP_CPU_KERNEL(
    strided_slice,
    ops::StridedSliceKernel<paddle::platform::CPUDeviceContext, int>,
    ops::StridedSliceKernel<paddle::platform::CPUDeviceContext, int64_t>,
    ops::StridedSliceKernel<paddle::platform::CPUDeviceContext, float>,
    ops::StridedSliceKernel<paddle::platform::CPUDeviceContext, double>);

REGISTER_OP_CPU_KERNEL(
    strided_slice_grad,
    ops::StridedSliceGradKernel<paddle::platform::CPUDeviceContext, int>,
    ops::StridedSliceGradKernel<paddle::platform::CPUDeviceContext, int64_t>,
    ops::StridedSliceGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::StridedSliceGradKernel<paddle::platform::CPUDeviceContext, double>);

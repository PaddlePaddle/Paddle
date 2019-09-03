/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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
    PADDLE_ENFORCE(ctx->HasInput("Input"),
                   "Input (Input) of slice op should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output (Out) of slice op should not be null.");

    auto in_dims = ctx->GetInputDim("Input");
    PADDLE_ENFORCE(in_dims.size() < 7,
                   "The rank of input should be less than 7.");
    auto begin = ctx->Attrs().Get<std::vector<int>>("begin");
    auto end = ctx->Attrs().Get<std::vector<int>>("end");
    auto stride = ctx->Attrs().Get<std::vector<int>>("stride");

    PADDLE_ENFORCE_EQ(begin.size(), end.size());
    PADDLE_ENFORCE_EQ(begin.size(), stride.size());

    // we need to analysis strided slice op is valid for
    // the parameter that we get from python front
    int stride_index, start_index, end_index;
    std::vector<int> out_dims_vector;
    for (size_t i = 0; i < begin.size(); i++) {
      PADDLE_ENFORCE(stride[i] != 0, "stride must not tp be zero");
      start_index = begin[i];
      end_index = end[i];
      stride_index = stride[i];
      int axis_size = in_dims[i];
      if (start_index < 0) {
        start_index = (start_index + axis_size) % axis_size;
      }
      if (end_index < 0) {
        end_index = (end_index + axis_size) % axis_size;
      }

      bool zero_dim_condition =
          ((stride_index < 0 && (start_index < end_index)) ||
           (stride_index > 0 && (start_index > end_index)));
      auto out_dims_index =
          std::abs(end_index - start_index) / std::abs(stride_index);
      if (zero_dim_condition) {
        out_dims_index = 0;
      }

      out_dims_vector.push_back(out_dims_index);
    }
    framework::DDim out_dims(framework::make_ddim(out_dims_vector));

    ctx->SetOutputDim("Out", out_dims);
    if (out_dims.size() == in_dims.size()) {
      ctx->ShareLoD("Input", /*->*/ "Out");
    }
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

    AddAttr<std::vector<int>>("begin",
                              "(list<int>) Axes that the tensor slice start.");
    AddAttr<std::vector<int>>("end",
                              "(list<int>) Axes that the tensor slice end");
    AddAttr<std::vector<int>>(
        "stride", "(list<int> Axes stride from the start to the end)");
    AddComment(R"DOC(
Strided Slice Operator.

)DOC");
  }
};

class StridedSliceOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Input"), "Input should not be null");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
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

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

#include "paddle/fluid/operators/slice_op.h"
#include <algorithm>
#include <memory>
#include <vector>

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

class SliceOp : public framework::OperatorWithKernel {
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
    framework::DDim out_dims(in_dims);
    auto axes = ctx->Attrs().Get<std::vector<int>>("axes");
    auto starts = ctx->Attrs().Get<std::vector<int>>("starts");
    auto ends = ctx->Attrs().Get<std::vector<int>>("ends");
    auto decrease_axis = ctx->Attrs().Get<std::vector<int>>("decrease_axis");

    PADDLE_ENFORCE_EQ(starts.size(), ends.size());
    PADDLE_ENFORCE_EQ(starts.size(), axes.size());
    int dim_value, start, end;
    for (size_t i = 0; i < axes.size(); ++i) {
      dim_value = out_dims[axes[i]];
      if (dim_value > 0) {
        start = starts[i] < 0 ? (starts[i] + dim_value) : starts[i];
        end = ends[i] < 0 ? (ends[i] + dim_value) : ends[i];
        start = std::max(start, 0);
        end = std::max(end, 0);
        // start = std::min(start, dim_value);
        end = std::min(end, dim_value);
        // start = std::min(start, end);
        PADDLE_ENFORCE_GT(end, start, "end should greater than start");
        out_dims[axes[i]] = end - start;
      }
    }

    // generate new shape
    if (decrease_axis.size() > 0) {
      std::vector<int> new_out_shape;
      for (size_t i = 0; i < decrease_axis.size(); ++i) {
        if (ctx->IsRuntime()) {
          PADDLE_ENFORCE_EQ(out_dims[decrease_axis[i]], 1,
                            "decrease dim should be 1");
        }
        out_dims[decrease_axis[i]] = 0;
      }

      for (int i = 0; i < out_dims.size(); ++i) {
        if (out_dims[i] != 0) {
          new_out_shape.push_back(out_dims[i]);
        }
      }
      if (new_out_shape.size() == 0) {
        new_out_shape.push_back(1);
      }

      out_dims = framework::make_ddim(new_out_shape);
    }

    ctx->SetOutputDim("Out", out_dims);
    if (axes[0] != 0) {
      ctx->ShareLoD("Input", /*->*/ "Out");
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(ctx.Input<Tensor>("Input")->type(),
                                   ctx.GetPlace());
  }
};

class SliceOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Input", "Tensor of data to extract slices from.");
    AddOutput("Out", "Sliced data tensor.");

    AddAttr<std::vector<int>>(
        "axes",
        "(list<int>) Axes that `starts` and `ends` apply to. It's optional."
        "If not present, will be treated as [0, 1, ..., len(`starts`) - 1].");
    AddAttr<std::vector<int>>(
        "starts",
        "(list<int>) Starting indices of corresponding axis in `axes`");
    AddAttr<std::vector<int>>(
        "ends",
        "(list<int>) Starting indices of corresponding axis in `axes`.");
    AddAttr<std::vector<int>>("decrease_axis", "(list<int>) decrease_axis")
        .SetDefault({});
    AddComment(R"DOC(
Slice Operator.

Produces a slice of the input tensor along multiple axes. Similar to numpy:
https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
Slice uses `axes`, `starts` and `ends` attributes to specify the start and
end dimension for each axis in the list of axes, it uses this information
to slice the input data tensor. If a negative value is passed for any of
the start or end indices, it represents number of elements before the end
of that dimension. If the value passed to start or end is larger than
the n (the number of elements in this dimension), it represents n.
For slicing to the end of a dimension with unknown size, it is recommended
to pass in INT_MAX. The size of axes must be equal to starts\' and ends\'.
Following examples will explain how slice works:

.. code-block:: text

    Case1:
        Given:
            data = [ [1, 2, 3, 4], [5, 6, 7, 8], ]
            axes = [0, 1]
            starts = [1, 0]
            ends = [2, 3]
        Then:
            result = [ [5, 6, 7], ]

    Case2:
        Given:
            data = [ [1, 2, 3, 4], [5, 6, 7, 8], ]
            starts = [0, 1]
            ends = [-1, 1000]
        Then:
            result = [ [2, 3, 4], ]
)DOC");
  }
};

class SliceOpGrad : public framework::OperatorWithKernel {
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

class SliceOpGradMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    auto* bind = new framework::OpDesc();
    bind->SetInput("Input", Input("Input"));
    bind->SetInput(framework::GradVarName("Out"), OutputGrad("Out"));
    bind->SetOutput(framework::GradVarName("Input"), InputGrad("Input"));
    bind->SetAttrMap(Attrs());
    bind->SetType("slice_grad");
    return std::unique_ptr<framework::OpDesc>(bind);
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERENCE(SliceOpGradNoNeedBufferVarsInference,
                                      "Input");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(slice, ops::SliceOp, ops::SliceOpMaker,
                  ops::SliceOpGradMaker);
REGISTER_OPERATOR(slice_grad, ops::SliceOpGrad,
                  ops::SliceOpGradNoNeedBufferVarsInference);

REGISTER_OP_CPU_KERNEL(
    slice, ops::SliceKernel<paddle::platform::CPUDeviceContext, int>,
    ops::SliceKernel<paddle::platform::CPUDeviceContext, int64_t>,
    ops::SliceKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SliceKernel<paddle::platform::CPUDeviceContext, double>);

REGISTER_OP_CPU_KERNEL(
    slice_grad, ops::SliceGradKernel<paddle::platform::CPUDeviceContext, int>,
    ops::SliceGradKernel<paddle::platform::CPUDeviceContext, int64_t>,
    ops::SliceGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SliceGradKernel<paddle::platform::CPUDeviceContext, double>);

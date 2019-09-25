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
#include <string>
#include <vector>
#include "paddle/fluid/operators/slice_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

class StridedSliceOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
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
    auto infer_flags = ctx->Attrs().Get<std::vector<int>>("infer_flags");

    auto starts_size = starts.size();
    auto ends_size = ends.size();
    auto strides_size = strides.size();

    if (ctx->HasInputs("StartsTensorList")) {
      auto StartsTensorList = ctx->Inputs("StartsTensorList");
      PADDLE_ENFORCE_GT(StartsTensorList.size(), 0,
                        "StartsTensorList size can't be zero");
      starts_size = StartsTensorList.size();
    }
    if (ctx->HasInputs("EndsTensorList")) {
      auto EndsTensorList = ctx->Inputs("EndsTensorList");
      PADDLE_ENFORCE_GT(EndsTensorList.size(), 0,
                        "EndsTensorList size can't be zero");
      ends_size = EndsTensorList.size();
    }
    if (ctx->HasInputs("StridesTensorList")) {
      auto StridesTensorList = ctx->Inputs("StridesTensorList");
      PADDLE_ENFORCE_GT(StridesTensorList.size(), 0,
                        "StridesTensorList size can't be zero");
      strides_size = StridesTensorList.size();
    }

    auto tensor_input = false;
    if (ctx->HasInput("EndsTensor") || ctx->HasInput("StartsTensor") ||
        ctx->HasInput("StridesTensor")) {
      tensor_input = true;
    }
    if (ctx->HasInput("EndsTensor") == false) {
      PADDLE_ENFORCE_EQ(ends_size, axes.size(),
                        "The size of ends must be equal to the size of axes.");
    }
    if (ctx->HasInput("StartsTensor") == false) {
      PADDLE_ENFORCE_EQ(
          starts_size, axes.size(),
          "The size of starts must be equal to the size of axes.");
    }
    if (ctx->HasInput("StridesTensor") == false) {
      PADDLE_ENFORCE_EQ(
          strides_size, axes.size(),
          "The size of strides must be equal to the size of axes.");
    }
    // we need to analysis strided slice op is valid for
    // the parameter that we get from python front
    std::vector<int> out_dims_vector(in_dims.size(), -1);
    if (!tensor_input) {
      StridedSliceOutDims(starts, ends, strides, axes, infer_flags, in_dims,
                          out_dims_vector.data(), axes.size(), true);
    }
    framework::DDim out_dims(framework::make_ddim(out_dims_vector));

    ctx->SetOutputDim("Out", out_dims);
    ctx->ShareLoD("Input", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(ctx.Input<Tensor>("Input")->type(),
                                   ctx.Input<Tensor>("Input")->place());
  }
  framework::OpKernelType GetKernelTypeForVar(
      const std::string &var_name, const Tensor &tensor,
      const framework::OpKernelType &expected_kernel_type) const override {
    if (var_name == "StartsTensor" || var_name == "EndsTensor" ||
        var_name == "StridesTensor") {
      return expected_kernel_type;
    }
    if (var_name == "StartsTensorList" || var_name == "EndsTensorList" ||
        var_name == "StridesTensorList") {
      return expected_kernel_type;
    }
    return framework::OpKernelType(expected_kernel_type.data_type_,
                                   tensor.place(), tensor.layout());
  }
};

class StridedSliceOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Input", "Tensor of data to extract slices from.");
    AddOutput("Out", "Strided Sliced data tensor.");

    AddInput("StartsTensor",
             "(Tensor<int32>, optional) If provided, slice will use this."
             "It has the highest priority of StartsTensor, StartsTensorList "
             "and attr(starts).")
        .AsDispensable();
    AddInput("EndsTensor",
             "(Tensor<int32>, optional) If provided, slice will use this."
             "It has the highest priority of EndsTensor, EndsTensorList and "
             "attr(ends).")
        .AsDispensable();
    AddInput(
        "StridesTensor",
        "(Tensor<int32>, optional) If provided, slice will use this."
        "It has the highest priority of StridesTensor, StridesTensorList and "
        "attr(ends).")
        .AsDispensable();
    AddInput(
        "StartsTensorList",
        "(vector<Tensor<int32>>, optional) If provided, slice will use this."
        "The shape of the tensor in vector MUST BE [1]."
        "It has higher priority compare with attr(starts).")
        .AsDuplicable()
        .AsDispensable();
    AddInput(
        "EndsTensorList",
        "(vector<Tensor<int32>>, optional) If provided, slice will use this."
        "The shape of the tensor in vector MUST BE [1]."
        "It has higher priority compare with attr(ends).")
        .AsDuplicable()
        .AsDispensable();
    AddInput(
        "StridesTensorList",
        "(vector<Tensor<int32>>, optional) If provided, slice will use this."
        "The shape of the tensor in vector MUST BE [1]."
        "It has higher priority compare with attr(strides).")
        .AsDuplicable()
        .AsDispensable();
    AddAttr<std::vector<int>>(
        "axes", "(list<int>) Axes that `starts` and `ends` apply to.");
    AddAttr<std::vector<int>>(
        "starts", "(list<int>) Start indices for the strided slice start.")
        .SetDefault({});
    AddAttr<std::vector<int>>("ends",
                              "(list<int>) End indices the tensor slice end")
        .SetDefault({});
    AddAttr<std::vector<int>>(
        "strides", "(list<int> Stride step from the start to the end)")
        .SetDefault({});
    AddAttr<std::vector<int>>(
        "infer_flags", "(list<int>) Flags of inferring dims in attributes.")
        .SetDefault({});
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

  void InferShape(framework::InferShapeContext *ctx) const override {
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
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        ctx.Input<framework::Tensor>(framework::GradVarName("Out"))->type(),
        ctx.GetPlace());
  }
  framework::OpKernelType GetKernelTypeForVar(
      const std::string &var_name, const Tensor &tensor,
      const framework::OpKernelType &expected_kernel_type) const override {
    if (var_name == "StartsTensor" || var_name == "EndsTensor") {
      return expected_kernel_type;
    }
    if (var_name == "StartsTensorList" || var_name == "EndsTensorList") {
      return expected_kernel_type;
    }
    return framework::OpKernelType(expected_kernel_type.data_type_,
                                   tensor.place(), tensor.layout());
  }
};

class StridedSliceOpGradMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    auto *bind = new framework::OpDesc();
    bind->SetInput(framework::GradVarName("Out"), OutputGrad("Out"));
    bind->SetInput("Input", Input("Input"));
    bind->SetInput("StartsTensor", Input("StartsTensor"));
    bind->SetInput("EndsTensor", Input("EndsTensor"));
    bind->SetInput("StridesTensor", Input("StridesTensor"));
    bind->SetInput("StartsTensorList", Input("StartsTensorList"));
    bind->SetInput("EndsTensorList", Input("EndsTensorList"));
    bind->SetInput("StridesTensorList", Input("StridesTensorList"));
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

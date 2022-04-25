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

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/slice_op.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/backward.h"
#include "paddle/phi/kernels/funcs/strided_slice.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

class StridedSliceOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto *in_var = ctx.InputVar("Input");
    auto is_in_var_array = in_var->IsType<framework::LoDTensorArray>();
    if (is_in_var_array) {
      auto &tensor_array = in_var->Get<framework::LoDTensorArray>();
      for (auto &tensor : tensor_array) {
        if (!platform::is_cuda_pinned_place(tensor.place())) {
          PADDLE_ENFORCE_EQ(
              platform::is_same_place(tensor.place(),
                                      ctx.device_context().GetPlace()),
              true,
              platform::errors::InvalidArgument(
                  "Place of context is %s. Place of input tensor is %s. They "
                  "are should be same, but reveived different place.",
                  string::to_string(ctx.device_context().GetPlace()),
                  string::to_string(tensor.place())));
        }
      }
      return framework::OpKernelType(
          OperatorWithKernel::IndicateVarDataType(ctx, "Input"),
          ctx.device_context());
    }
    // NOTE: cuda pinned tensor need to copy its data to target place
    auto in_tensor = ctx.Input<Tensor>("Input");
    if (platform::is_cuda_pinned_place(in_tensor->place())) {
      return framework::OpKernelType(
          framework::TransToProtoVarType(in_tensor->dtype()),
          ctx.device_context());
    }
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Input"),
        in_tensor->place());
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

class StridedSliceOpVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    ctx->SetOutputType("Out", ctx->GetInputType("Input"));
    ctx->SetOutputDataType("Out", ctx->GetInputDataType("Input"));
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
    AddAttr<std::vector<int>>("decrease_axis", "(list<int>) decrease_axis")
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

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.GetPlace());
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

template <typename T>
class StridedSliceOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> bind) const override {
    bind->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    bind->SetInput("Input", this->Input("Input"));
    bind->SetInput("StartsTensor", this->Input("StartsTensor"));
    bind->SetInput("EndsTensor", this->Input("EndsTensor"));
    bind->SetInput("StridesTensor", this->Input("StridesTensor"));
    bind->SetInput("StartsTensorList", this->Input("StartsTensorList"));
    bind->SetInput("EndsTensorList", this->Input("EndsTensorList"));
    bind->SetInput("StridesTensorList", this->Input("StridesTensorList"));
    bind->SetOutput(framework::GradVarName("Input"), this->InputGrad("Input"));
    bind->SetAttrMap(this->Attrs());
    bind->SetType("strided_slice_grad");
  }
};
class StridedSliceGradOpVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    ctx->SetOutputType(framework::GradVarName("Input"),
                       ctx->GetInputType(framework::GradVarName("Out")));
    ctx->SetOutputDataType(
        framework::GradVarName("Input"),
        ctx->GetInputDataType(framework::GradVarName("Out")));
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(StridedSliceOpGradNoNeedBufferVarsInferer,
                                    "Input");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

DECLARE_INFER_SHAPE_FUNCTOR(strided_slice, StridedSliceInferShape,
                            PD_INFER_META(phi::StridedSliceRawInferMeta));

REGISTER_OPERATOR(strided_slice, ops::StridedSliceOp, ops::StridedSliceOpMaker,
                  ops::StridedSliceOpGradMaker<paddle::framework::OpDesc>,
                  ops::StridedSliceOpGradMaker<paddle::imperative::OpBase>,
                  ops::StridedSliceOpVarTypeInference, StridedSliceInferShape);

DECLARE_INFER_SHAPE_FUNCTOR(strided_slice_grad, StridedSliceGradInferShape,
                            PD_INFER_META(phi::GeneralUnaryGradInferMeta));

REGISTER_OPERATOR(strided_slice_grad, ops::StridedSliceOpGrad,
                  ops::StridedSliceOpGradNoNeedBufferVarsInferer,
                  ops::StridedSliceGradOpVarTypeInference,
                  StridedSliceGradInferShape);

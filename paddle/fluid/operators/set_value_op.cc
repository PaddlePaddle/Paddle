//   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/set_value_op.h"

#include <string>

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace framework {
class InferShapeContext;
class OpDesc;
template <typename T>
class EmptyGradOpMaker;
}  // namespace framework
namespace imperative {
class OpBase;
}  // namespace imperative
}  // namespace paddle

namespace paddle {
namespace operators {

using Tensor = phi::DenseTensor;

class SetValue : public framework::OperatorWithKernel {
 public:
  SetValue(const std::string &type,
           const framework::VariableNameMap &inputs,
           const framework::VariableNameMap &outputs,
           const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Input"), ctx.GetPlace());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string &var_name,
      const Tensor &tensor,
      const framework::OpKernelType &expected_kernel_type) const override {
    if (var_name == "StartsTensorList" || var_name == "EndsTensorList" ||
        var_name == "StepsTensorList") {
      return expected_kernel_type;
    }
    return framework::OpKernelType(
        expected_kernel_type.data_type_, tensor.place(), tensor.layout());
  }
};

class SetValueMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    // Input
    AddInput("Input", "(Tensor) Input tensor of set_value operator.");
    AddInput("ValueTensor", "(Tensor) Value tensor of set_value operator.")
        .AsDispensable();
    AddInput("StartsTensorList",
             "(vector<Tensor<int32>>, optional) If provided, set_value will "
             "use this. The shape of the tensor in vector must be [1]."
             "It has higher priority compare with attr(starts).")
        .AsDuplicable()
        .AsDispensable();
    AddInput("EndsTensorList",
             "(vector<Tensor<int32>>, optional) If provided, set_value will "
             "use this. The shape of the tensor in vector must BE [1]."
             "It has higher priority compare with attr(ends).")
        .AsDuplicable()
        .AsDispensable();

    AddInput("StepsTensorList",
             "(vector<Tensor<int32>>, optional) If provided, set_value will "
             "use this. The shape of the tensor in vector must BE [1]."
             "It has higher priority compare with attr(steps).")
        .AsDuplicable()
        .AsDispensable();

    // Output
    AddOutput("Out",
              "(Tensor) Output tensor of set_value operator. The output is the "
              "same Tensor as input");

    // Attr
    AddAttr<int>("dtype", "data type of input.")
        .InEnum({framework::proto::VarType::BOOL,
                 framework::proto::VarType::INT32,
                 framework::proto::VarType::INT64,
                 framework::proto::VarType::FP32,
                 framework::proto::VarType::FP64})
        .SetDefault(framework::proto::VarType::FP32);
    AddAttr<std::vector<int64_t>>(
        "axes", "(list<int64_t>) Axes that `starts` and `ends` apply to.");
    AddAttr<std::vector<int64_t>>(
        "starts",
        "(list<int64_t>) Starting indices of corresponding axis in `axes`.")
        .SetDefault({});
    AddAttr<std::vector<int64_t>>(
        "ends",
        "(list<int64_t>) Ending indices of corresponding axis in `axes`.")
        .SetDefault({});
    AddAttr<std::vector<int64_t>>(
        "steps", "(list<int64_t>) Stride step from the start to the end.")
        .SetDefault({});
    AddAttr<std::vector<int64_t>>("decrease_axes",
                                  "(list<int>) The axes to decrease.")
        .SetDefault({});
    AddAttr<std::vector<int64_t>>("none_axes", "(list<int>) The axes to none.")
        .SetDefault({});

    AddAttr<std::vector<int>>("bool_values", "Store the bool values.")
        .SetDefault({});
    AddAttr<std::vector<float>>("fp32_values", "Store the float32 values.")
        .SetDefault({});
    AddAttr<std::vector<int>>("int32_values", "Store the int32 values.")
        .SetDefault({});
    AddAttr<std::vector<int64_t>>("int64_values", "Store the int64 values.")
        .SetDefault({});
    AddAttr<std::vector<double>>("fp64_values", "Store the float64 values.")
        .SetDefault({});

    AddAttr<std::vector<int64_t>>("shape", "(vector<int64_t>) Shape of values.")
        .SetDefault({});
    AddComment(R"DOC(SetValue operator.
Assignment to a Tensor in static mode.
)DOC");
  }
};

template <typename T>
class SetValueGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    if (this->HasInput("ValueTensor")) {
      op->SetType("set_value_grad");

      op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
      op->SetInput("ValueTensor", this->Input("ValueTensor"));
      if (this->HasInput("StartsTensorList")) {
        op->SetInput("StartsTensorList", this->Input("StartsTensorList"));
      }
      if (this->HasInput("EndsTensorList")) {
        op->SetInput("EndsTensorList", this->Input("EndsTensorList"));
      }
      if (this->HasInput("StepsTensorList")) {
        op->SetInput("StepsTensorList", this->Input("StepsTensorList"));
      }

      op->SetAttrMap(this->Attrs());

      op->SetOutput(framework::GradVarName("ValueTensor"),
                    this->InputGrad("ValueTensor"));
      op->SetOutput(framework::GradVarName("Input"), this->InputGrad("Input"));

    } else {
      op->SetType("assign");
      op->SetInput("X", this->OutputGrad("Out"));
      op->SetOutput("Out", this->InputGrad("Input"));
    }
  }
};

class SetValueGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")),
                   "Input",
                   framework::GradVarName("Out"),
                   "set_value_grad");

    auto in_dims = ctx->GetInputDim(framework::GradVarName("Out"));
    PADDLE_ENFORCE_LT(
        in_dims.size(),
        7,
        platform::errors::InvalidArgument(
            "The dimension of set_value_grad operator's input should be less "
            "than 7, but received dimension is %d.",
            in_dims.size()));

    if (ctx->HasOutput(framework::GradVarName("ValueTensor"))) {
      ctx->ShareDim("ValueTensor",
                    /*->*/ framework::GradVarName("ValueTensor"));
      ctx->ShareLoD("ValueTensor",
                    /*->*/ framework::GradVarName("ValueTensor"));
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto in_tensor = ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   in_tensor->place());
  }
  framework::OpKernelType GetKernelTypeForVar(
      const std::string &var_name,
      const Tensor &tensor,
      const framework::OpKernelType &expected_kernel_type) const override {
    if (var_name == "StartsTensorList" || var_name == "EndsTensorList" ||
        var_name == "StepsTensorList") {
      return expected_kernel_type;
    }
    return framework::OpKernelType(
        expected_kernel_type.data_type_, tensor.place(), tensor.layout());
  }
};

DECLARE_INPLACE_OP_INFERER(SetValueOpInplaceInferer, {"Input", "Out"});

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

DECLARE_INFER_SHAPE_FUNCTOR(set_value,
                            SetValueInferShapeFunctor,
                            PD_INFER_META(phi::SetValueInferMeta));

REGISTER_OPERATOR(set_value,
                  ops::SetValue,
                  ops::SetValueMaker,
                  ops::SetValueGradMaker<paddle::framework::OpDesc>,
                  ops::SetValueGradMaker<paddle::imperative::OpBase>,
                  ops::SetValueOpInplaceInferer,
                  SetValueInferShapeFunctor);

REGISTER_OPERATOR(set_value_grad, ops::SetValueGrad);

REGISTER_OP_VERSION(set_value)
    .AddCheckpoint(
        R"ROC(
Upgrade set_value, add 3 inputs [StartsTensorList, EndsTensorList, StepsTensorList] and 1 attribute [steps].
              )ROC",
        paddle::framework::compatible::OpVersionDesc()
            .NewInput("StartsTensorList",
                      "If provided, set_value will use this.The shape of the "
                      "tensor in vector must be [1]. It has higher priority "
                      "compare with attr(starts).")
            .NewInput("EndsTensorList",
                      "If provided, set_value will use this.The shape of the "
                      "tensor in vector must be [1]. It has higher priority "
                      "compare with attr(ends).")
            .NewInput("StepsTensorList",
                      "If provided, set_value will use this.The shape of the "
                      "tensor in vector must be [1]. It has higher priority "
                      "compare with attr(steps).")
            .ModifyAttr("starts",
                        "Starting indices of corresponding axis in `axes`.",
                        std::vector<int64_t>{})
            .ModifyAttr("ends",
                        "Ending indices of corresponding axis in `axes`.",
                        std::vector<int64_t>{})
            .NewAttr("steps",
                     "Stride step from the start to the end.",
                     std::vector<int64_t>{}))
    .AddCheckpoint(
        R"ROC(
Upgrade set_value, add 1 attribute [decrease_axes].
              )ROC",
        paddle::framework::compatible::OpVersionDesc().NewAttr(
            "decrease_axes", "The axes to decrease.", std::vector<int64_t>{}))
    .AddCheckpoint(
        R"ROC(
Upgrade set_value, add 1 attribute [none_axes].
              )ROC",
        paddle::framework::compatible::OpVersionDesc().NewAttr(
            "none_axes", "The axes with none index.", std::vector<int64_t>{}));

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

#include "paddle/fluid/operators/fill_constant_op.h"
#include <string>
#include "paddle/fluid/framework/op_version_registry.h"
namespace paddle {
namespace operators {

class FillConstantOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "FillConstant");

    auto& shape = ctx->Attrs().Get<std::vector<int64_t>>("shape");
    if (!ctx->HasInput("ShapeTensor") && !ctx->HasInputs("ShapeTensorList")) {
      for (size_t i = 0; i < shape.size(); ++i) {
        PADDLE_ENFORCE_GE(
            shape[i], 0,
            platform::errors::InvalidArgument(
                "Each value of attribute 'shape' is expected to be no less "
                "than 0. But recieved: shape[%u] = %d; shape = [%s].",
                i, shape[i], framework::make_ddim(shape)));
      }
    }
    if (shape.empty() && ctx->HasInput("ShapeTensor")) {
      auto shape_dims = ctx->GetInputDim("ShapeTensor");
      int num_ele = 1;
      for (int i = 0; i < shape_dims.size(); ++i) {
        num_ele *= shape_dims[i];
      }
      auto vec_dims = std::vector<int>(num_ele, -1);
      ctx->SetOutputDim("Out", framework::make_ddim(vec_dims));

      return;
    }
    ctx->SetOutputDim("Out", framework::make_ddim(shape));
  }

 protected:
  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name, const framework::Tensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const override {
    if (var_name == "ShapeTensor" || var_name == "ShapeTensorList") {
      return expected_kernel_type;
    } else {
      return framework::OpKernelType(expected_kernel_type.data_type_,
                                     tensor.place(), tensor.layout());
    }
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    framework::OpKernelType kt = framework::OpKernelType(
        framework::proto::VarType::Type(ctx.Attr<int>("dtype")),
        ctx.GetPlace());
    // TODO(zyfncg) The force_cpu and place_type are conflicted, it's a issue
    // lefted before, and we may merge them in the future.
    // In order to invoke new fill_constant kernel, the place of OpKernelType
    // will be setted by force_cpu and place_type here.
    if (ctx.Attr<bool>("force_cpu")) {
      kt.place_ = platform::CPUPlace();
    }
    auto place_type = ctx.Attr<int>("place_type");
    if (place_type != -1) {
      switch (place_type) {
        case 0:
          kt.place_ = platform::CPUPlace();
          break;
        case 1:
        case 2:
          kt.place_ = platform::CUDAPlace();
          break;
        case 3:
          kt.place_ = platform::XPUPlace();
          break;
        case 4:
          kt.place_ = platform::NPUPlace();
          break;
        default:
          PADDLE_THROW(platform::errors::Unimplemented(
              "Could NOT determine the place of variable, place_type = %d .",
              place_type));
      }
    }

    return kt;
  }

  framework::KernelSignature GetExpectedPtenKernelArgs(
      const framework::ExecutionContext& ctx) const override {
    std::string shape;
    if (ctx.HasInput("ShapeTensor")) {
      shape = "ShapeTensor";
    } else if (ctx.MultiInput<framework::Tensor>("ShapeTensorList").size()) {
      shape = "ShapeTensorList";
    } else {
      shape = "shape";
    }
    std::string value;
    if (ctx.HasInput("ValueTensor")) {
      value = "ValueTensor";
    } else {
      const auto& str_value = ctx.Attr<std::string>("str_value");
      value = str_value.empty() ? "value" : "str_value";
    }
    if (!ctx.OutputVar("Out")->IsType<framework::SelectedRows>()) {
      return framework::KernelSignature("full", {}, {shape, value}, {"Out"});
    }
    return framework::KernelSignature("fill_constant.unregistered", {}, {}, {});
  }
};

class FillConstantOpVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext* ctx) const override {
    auto data_type = static_cast<framework::proto::VarType::Type>(
        BOOST_GET_CONST(int, ctx->GetAttr("dtype")));
    ctx->SetOutputDataType("Out", data_type);
  }
};

class FillConstantOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddAttr<int>("dtype",
                 "(int, default 5 (FP32)) "
                 "Output data type")
        .SetDefault(framework::proto::VarType::FP32);
    AddAttr<std::vector<int64_t>>("shape",
                                  "(vector<int64_t>) The shape of the output")
        .SetDefault({});
    AddInput("ValueTensor",
             "(Tensor, optional) If provided, fill_constant Op will use this "
             "as value to set the output Tensor, this has a higher priority "
             "than attr(str_value), the shape of this tensor MUST BE [1].")
        .AsDispensable();
    AddInput("ShapeTensor",
             "(Tensor<int>), optional). The shape of the output."
             "It has a higher priority than Attr(shape).")
        .AsDispensable();
    AddInput("ShapeTensorList",
             "(vector<Tensor<int>>, optional). The shape of the output. "
             "It has a higher priority than Attr(shape)."
             "The shape of the element in vector must be [1].")
        .AsDuplicable()
        .AsDispensable();
    AddAttr<float>("value", "(float, default 0.0f) The value to be filled")
        .SetDefault(0.0f);
    AddAttr<std::string>(
        "str_value",
        "(string, default empty) The str convert to value to be filled")
        .SetDefault("");
    AddAttr<bool>("force_cpu",
                  "(bool, default false) Force fill output variable to cpu "
                  "memory. Otherwise, fill output variable to the running "
                  "device")
        .SetDefault(false);
    AddAttr<int>("place_type",
                 "(int, default -1) allow mamually setting place where the "
                 "variable should be hold. "
                 "-1: not set manually, determine the place by executor. "
                 "0: CPUPlace. "
                 "1: CUDAPlace. "
                 "2: CUDAPinnedPlace. "
                 "3: XPUPlace. "
                 "4: NPUPlace. ")
        .SetDefault(-1);
    AddOutput("Out",
              "(Tensor) Tensor of specified shape will be filled "
              "with the specified value");
    AddComment(R"DOC(
FillConstantBatchSizeLike Operator.

Fill up a variable with specified constant value.

)DOC");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(
    fill_constant, ops::FillConstantOp, ops::FillConstantOpMaker,
    ops::FillConstantOpVarTypeInference,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OP_CPU_KERNEL(
    fill_constant, ops::FillConstantKernel<float>,
    ops::FillConstantKernel<double>, ops::FillConstantKernel<uint8_t>,
    ops::FillConstantKernel<int16_t>, ops::FillConstantKernel<int>,
    ops::FillConstantKernel<int64_t>, ops::FillConstantKernel<bool>,
    ops::FillConstantKernel<paddle::platform::float16>,
    ops::FillConstantKernel<paddle::platform::bfloat16>,
    ops::FillConstantKernel<paddle::platform::complex<float>>,
    ops::FillConstantKernel<paddle::platform::complex<double>>);

REGISTER_OP_VERSION(fill_constant)
    .AddCheckpoint(
        R"ROC(
      Upgrade fill_constant, add a new input [ValueTensor].
    )ROC",
        paddle::framework::compatible::OpVersionDesc().NewInput(
            "ValueTensor",
            "In order to support new feature tensor support of Value"))
    .AddCheckpoint(
        R"ROC(
      Upgrade fill_constant to add a new attribute [place_type].
    )ROC",
        paddle::framework::compatible::OpVersionDesc().NewAttr(
            "place_type",
            "In order to support tensor in CUDAPinnedPlace and XPUPlace", -1));

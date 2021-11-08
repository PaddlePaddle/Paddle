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

#include "paddle/fluid/operators/eye_op.h"

namespace paddle {
namespace operators {

class EyeOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Out"), true,
                      platform::errors::InvalidArgument(
                          "Output(Out) of EyeOP should not be null."));
    auto num_rows = ctx->Attrs().Get<int64_t>("num_rows");
    auto num_columns = ctx->Attrs().Get<int64_t>("num_columns");
    if (num_columns == -1) num_columns = num_rows;
    ctx->SetOutputDim("Out", {num_rows, num_columns});
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        framework::proto::VarType::Type(ctx.Attr<int>("dtype")),
        ctx.GetPlace());
  }
  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name, const Tensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const override {
    if (var_name == "NumRows" || var_name == "NumColumns") {
      return expected_kernel_type;
    }
    return framework::OpKernelType(expected_kernel_type.data_type_,
                                   tensor.place(), tensor.layout());
  }
};

class EyeOpVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext* ctx) const override {
    auto data_type = static_cast<framework::proto::VarType::Type>(
        BOOST_GET_CONST(int, ctx->GetAttr("dtype")));
    ctx->SetOutputDataType("Out", data_type);
  }
};

class EyeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddAttr<int>("dtype",
                 "(int, default 5 (FP32)) "
                 "Output data type")
        .SetDefault(framework::proto::VarType::FP32);
    AddInput("NumRows",
             "(Tensor<int64_t>, optional) If provided, eye will use this."
             "It has the highest priority of NumRows "
             "and attr(num_rows).")
        .AsDispensable();
    AddInput("NumColumns",
             "(Tensor<int64_t>, optional) If provided, eye will use this."
             "It has the highest priority of NumColumns and "
             "attr(num_columns).")
        .AsDispensable();
    AddAttr<int64_t>("num_rows",
                     "(int64_t) the number of rows in output tensor")
        .SetDefault(-1);
    AddAttr<int64_t>("num_columns",
                     "(int64_t) the number of columns in output tensor."
                     "Default -1 means that num_columns=num_rows")
        .SetDefault(-1);
    AddOutput("Out",
              "(Tensor) Construct an identity tensor with "
              "specified shape [num_rows, num_columns]");
    AddComment(R"DOC(
Return an identity tensor whose shape is [num_rows, num_columns].
)DOC");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CPU = paddle::platform::CPUDeviceContext;

REGISTER_OPERATOR(
    eye, ops::EyeOp, ops::EyeOpMaker, ops::EyeOpVarTypeInference,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OP_CPU_KERNEL(eye, ops::EyeKernel<CPU, float>,
                       ops::EyeKernel<CPU, double>,
                       ops::EyeKernel<CPU, int64_t>, ops::EyeKernel<CPU, int>,
                       ops::EyeKernel<CPU, paddle::platform::float16>);

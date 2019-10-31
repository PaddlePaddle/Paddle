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
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of EyeOP should not be null.");
    auto num_rows = ctx->Attrs().Get<int64_t>("num_rows");
    PADDLE_ENFORCE(num_rows >= 0,
                   "The value of Input(num_rows) should be non-negative int.");
    auto num_columns = ctx->Attrs().Get<int64_t>("num_columns");
    if (num_columns == -1) num_columns = num_rows;
    PADDLE_ENFORCE(
        num_columns >= 0,
        "The value of Input(num_columns) should be non-negative int.");
    ctx->SetOutputDim("Out", {num_rows, num_columns});
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        framework::proto::VarType::Type(ctx.Attr<int>("dtype")),
        ctx.GetPlace());
  }
};

class EyeOpVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext* ctx) const override {
    auto data_type = static_cast<framework::proto::VarType::Type>(
        boost::get<int>(ctx->GetAttr("dtype")));
    auto& out_var_name = ctx->Output("Out").front();
    ctx->SetDataType(out_var_name, data_type);
  }
};

class EyeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddAttr<int>("dtype",
                 "(int, default 5 (FP32)) "
                 "Output data type")
        .SetDefault(framework::proto::VarType::FP32);
    AddAttr<int64_t>("num_rows",
                     "(int64_t) the number of rows in output tensor");
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
using float16 = paddle::platform::float16;

REGISTER_OPERATOR(
    eye, ops::EyeOp, ops::EyeOpMaker, ops::EyeOpVarTypeInference,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OP_CPU_KERNEL(eye, ops::EyeKernel<CPU, float>,
                       ops::EyeKernel<CPU, double>,
                       ops::EyeKernel<CPU, int64_t>, ops::EyeKernel<CPU, int>,
                       ops::EyeKernel<CPU, float16>);

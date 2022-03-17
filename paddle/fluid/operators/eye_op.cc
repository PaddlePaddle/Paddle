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

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/nullary.h"

namespace paddle {
namespace operators {

class EyeOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

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
DECLARE_INFER_SHAPE_FUNCTOR(eye, EyeInferShapeFunctor,
                            PD_INFER_META(phi::EyeInferMeta));

REGISTER_OPERATOR(
    eye, ops::EyeOp, ops::EyeOpMaker, ops::EyeOpVarTypeInference,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    EyeInferShapeFunctor);

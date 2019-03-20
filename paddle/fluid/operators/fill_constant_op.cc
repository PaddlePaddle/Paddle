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

namespace paddle {
namespace operators {

class FillConstantOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of FillConstantOp should not be null.");
    auto& shape = ctx->Attrs().Get<std::vector<int64_t>>("shape");
    ctx->SetOutputDim("Out", framework::make_ddim(shape));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        framework::proto::VarType::Type(ctx.Attr<int>("dtype")),
        ctx.GetPlace());
  }
};

class FillConstantOpVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext* ctx) const override {
    auto data_type = static_cast<framework::proto::VarType::Type>(
        boost::get<int>(ctx->GetAttr("dtype")));
    auto& out_var_name = ctx->Output("Out").front();
    ctx->SetDataType(out_var_name, data_type);
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
                                  "(vector<int64_t>) The shape of the output");
    AddAttr<float>("value", "(float, default 0) The value to be filled")
        .SetDefault(0.0f);
    AddAttr<bool>("force_cpu",
                  "(bool, default false) Force fill output variable to cpu "
                  "memory. Otherwise, fill output variable to the running "
                  "device")
        .SetDefault(false);
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

REGISTER_OPERATOR(fill_constant, ops::FillConstantOp, ops::FillConstantOpMaker,
                  ops::FillConstantOpVarTypeInference,
                  paddle::framework::EmptyGradOpMaker);

REGISTER_OP_CPU_KERNEL(fill_constant, ops::FillConstantKernel<float>,
                       ops::FillConstantKernel<double>,
                       ops::FillConstantKernel<int64_t>,
                       ops::FillConstantKernel<int>,
                       ops::FillConstantKernel<paddle::platform::float16>);

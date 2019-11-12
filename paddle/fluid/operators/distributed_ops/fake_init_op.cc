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

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

template <typename T>
class FakeInitKernel : public framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext &ctx) const override {
    framework::Tensor *tensor = nullptr;
    auto &out_var = *ctx.OutputVar("Out");

    auto data_type =
        static_cast<framework::proto::VarType::Type>(ctx.Attr<int>("dtype"));
    auto &shape = ctx.Attr<std::vector<int64_t>>("shape");

    if (out_var.IsType<framework::LoDTensor>()) {
      tensor = out_var.GetMutable<framework::LoDTensor>();
      tensor->Resize(framework::make_ddim(shape));
    } else if (out_var.IsType<framework::SelectedRows>()) {
      tensor = out_var->GetMutable<framework::SelectedRows>()->mutable_value();
      tensor->Resize(shape);
    } else {
      PADDLE_THROW(
          "fake init op's output only"
          "supports SelectedRows and LoDTensor");
    }
  }
};

class FillConstantOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of FakeInitOp should not be null.");
    auto &shape = ctx->Attrs().Get<std::vector<int64_t>>("shape");
    ctx->SetOutputDim("Out", framework::make_ddim(shape));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        framework::proto::VarType::Type(ctx.Attr<int>("dtype")),
        ctx.GetPlace());
  }
};

class FakeInitOpVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    auto data_type = static_cast<framework::proto::VarType::Type>(
        boost::get<int>(ctx->GetAttr("dtype")));
    auto &out_var_name = ctx->Output("Out").front();
    ctx->SetDataType(out_var_name, data_type);
  }
};

class FakeInitOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddAttr<int>("dtype",
                 "(int, default 5 (FP32)) "
                 "Output data type")
        .SetDefault(framework::proto::VarType::FP32);

    AddAttr<std::vector<int64_t>>("shape",
                                  "(vector<int64_t>) The shape of the output");
    AddOutput("Out",
              "(Tensor) Tensor of specified shape will be filled "
              "with the specified value");
    AddComment(R"DOC(
FakeInit Operator.

Init an variable but not alloc memory for it, it is used for init the
table parameter at trainer side in distributed lookup table.

)DOC");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(fake_init, ops::FillConstantOp, ops::FakeInitOpMaker,
                  ops::FakeInitOpVarTypeInference,
                  paddle::framework::EmptyGradOpMaker);

REGISTER_OP_CPU_KERNEL(fake_init, ops::FakeInitKernel<float>,
                       ops::FakeInitKernel<double>,
                       ops::FakeInitKernel<int64_t>, ops::FakeInitKernel<int>,
                       ops::FakeInitKernel<bool>,
                       ops::FakeInitKernel<paddle::platform::float16>);

/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/framework/data_type.h"
#include "paddle/framework/op_registry.h"
#include "paddle/operators/math/math_function.h"

namespace paddle {
namespace operators {

class FillConstantInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of FillConstantOp should not be null.");
    auto &shape = ctx->Attrs().Get<std::vector<int>>("shape");
    ctx->SetOutputDim("Out", framework::make_ddim(shape));
  }
};

class FillConstantOp : public framework::OperatorBase {
 public:
  using framework::OperatorBase::OperatorBase;
  void Run(const framework::Scope &scope,
           const platform::DeviceContext &dev_ctx) const override {
    auto data_type = static_cast<framework::DataType>(Attr<int>("data_type"));
    auto value = Attr<float>("value");
    auto force_cpu = Attr<bool>("force_cpu");
    auto &out =
        *scope.FindVar(Output("Out"))->GetMutable<framework::LoDTensor>();
    out.Resize(framework::make_ddim(Attr<std::vector<int>>("shape")));
    if (force_cpu) {
      auto cpu = platform::CPUPlace();
      out.mutable_data(cpu, framework::ToTypeIndex(data_type));
    } else {
      out.mutable_data(dev_ctx.GetPlace(), framework::ToTypeIndex(data_type));
    }
    math::set_constant(dev_ctx, &out, value);
  }
};

class FillConstantOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  FillConstantOpMaker(framework::OpProto *proto,
                      framework::OpAttrChecker *op_checker)
      : framework::OpProtoAndCheckerMaker(proto, op_checker) {
    AddAttr<int>("data_type",
                 "(int, default 5 (FP32)) "
                 "Output data type")
        .SetDefault(framework::DataType::FP32);
    AddAttr<std::vector<int>>("shape", "(vector<int>) The shape of the output");
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
REGISTER_OPERATOR(fill_constant, ops::FillConstantOp,
                  ops::FillConstantInferShape, ops::FillConstantOpMaker,
                  paddle::framework::EmptyGradOpMaker);

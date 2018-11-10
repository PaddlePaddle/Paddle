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

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

class FillConstantInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of FillConstantOp should not be null.");
    auto &shape = ctx->Attrs().Get<std::vector<int64_t>>("shape");
    ctx->SetOutputDim("Out", framework::make_ddim(shape));
  }
};

class FillConstantOp : public framework::OperatorBase {
 public:
  using framework::OperatorBase::OperatorBase;

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
    auto data_type =
        static_cast<framework::proto::VarType::Type>(Attr<int>("dtype"));
    auto value = Attr<float>("value");
    auto force_cpu = Attr<bool>("force_cpu");

    framework::Tensor *tensor = nullptr;

    auto &out_var = *scope.FindVar(Output("Out"));

    if (out_var.IsType<framework::LoDTensor>()) {
      tensor = out_var.GetMutable<framework::LoDTensor>();
      tensor->Resize(framework::make_ddim(Attr<std::vector<int64_t>>("shape")));
    } else if (out_var.IsType<framework::SelectedRows>()) {
      tensor = out_var.GetMutable<framework::SelectedRows>()->mutable_value();
      tensor->Resize(framework::make_ddim(Attr<std::vector<int64_t>>("shape")));
    } else {
      PADDLE_THROW(
          "fill constant op's output only"
          "supports SelectedRows and LoDTensor");
    }

    if (force_cpu) {
      auto cpu = platform::CPUPlace();
      tensor->mutable_data(cpu, framework::ToTypeIndex(data_type));
    } else {
      tensor->mutable_data(dev_place, framework::ToTypeIndex(data_type));
    }

    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    auto &dev_ctx = *pool.Get(dev_place);
    math::set_constant(dev_ctx, tensor, value);
  }
};

class FillConstantOpVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(const framework::OpDesc &op_desc,
                  framework::BlockDesc *block) const override {}
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
REGISTER_OPERATOR(fill_constant, ops::FillConstantOp,
                  ops::FillConstantInferShape, ops::FillConstantOpMaker,
                  paddle::framework::EmptyGradOpMaker,
                  ops::FillConstantOpVarTypeInference);

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

class FakeInitInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of FakeInitOp should not be null.");
    auto &shape = ctx->Attrs().Get<std::vector<int64_t>>("shape");
    ctx->SetOutputDim("Out", framework::make_ddim(shape));
  }
};

class FakeInitOp : public framework::OperatorBase {
 public:
  using framework::OperatorBase::OperatorBase;

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
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
          "fake init op's output only"
          "supports SelectedRows and LoDTensor");
    }
  }
};

class FakeInitOpVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {}
};

class FakeInitOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
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
REGISTER_OPERATOR(fake_init, ops::FakeInitOp, ops::FakeInitInferShape,
                  ops::FakeInitOpMaker, paddle::framework::EmptyGradOpMaker,
                  ops::FakeInitOpVarTypeInference);

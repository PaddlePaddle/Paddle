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

#include "paddle/framework/data_type.h"
#include "paddle/framework/op_registry.h"
#include "paddle/framework/var_type.h"

namespace paddle {
namespace operators {
class AssignFunctor {
 public:
  AssignFunctor(framework::Variable *out,
                const platform::DeviceContext &dev_ctx)
      : out_(out), dev_ctx_(dev_ctx) {}

  void operator()(const framework::LoDTensor &lod_tensor) const {
    auto &out_tensor = *out_->GetMutable<framework::LoDTensor>();
    copy_tensor(lod_tensor, &out_tensor);
  }

  void operator()(const framework::LoDTensorArray &array) const {
    auto &out_array = *out_->GetMutable<framework::LoDTensorArray>();
    out_array.resize(array.size());
    for (size_t i = 0; i < array.size(); ++i) {
      copy_tensor(array[i], &out_array[i]);
    }
  }

  void operator()(const framework::SelectedRows &rows) const {
    framework::SelectedRows &out_rows =
        *out_->GetMutable<framework::SelectedRows>();
    out_rows.set_rows(rows.rows());
    out_rows.set_height(rows.height());
    auto &t = rows.value();
    out_rows.mutable_value()->CopyFrom(t, t.place(), dev_ctx_);
  }

  template <typename T>
  void operator()(const T &v) const {
    PADDLE_THROW("Not support type for assign op %s", typeid(T).name());
  }

 private:
  void copy_tensor(const framework::LoDTensor &lod_tensor,
                   framework::LoDTensor *out) const {
    auto &out_tensor = *out;
    out_tensor.CopyFrom(lod_tensor, lod_tensor.place(), dev_ctx_);
    out_tensor.set_lod(lod_tensor.lod());
  }

  framework::Variable *out_;
  const platform::DeviceContext &dev_ctx_;
};

class AssignOp : public framework::OperatorBase {
 public:
  AssignOp(const std::string &type, const framework::VariableNameMap &inputs,
           const framework::VariableNameMap &outputs,
           const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}
  void Run(const framework::Scope &scope,
           const platform::DeviceContext &dev_ctx) const override {
    auto *x = scope.FindVar(Input("X"));
    if (x == nullptr) {
      return;
    }
    auto *out = scope.FindVar(Output("Out"));
    PADDLE_ENFORCE(
        out != nullptr,
        "The Output(Out) should not be null if the Input(X) is set.");
    framework::VisitVarType(*x, AssignFunctor(out, dev_ctx));
  }
};

class AssignOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  AssignOpProtoMaker(framework::OpProto *proto,
                     framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X",
             "(LoDTensor, SelectedRows or LoDTensorArray) The input variable "
             "could be LoDTensor, SelectedRows or LoDTensorArray.")
        .AsDispensable();
    AddOutput("Out",
              "(LoDTensor, SelectedRows or LoDTensorArray) The type of output "
              "is the same as input X.");
    AddComment(R"DOC(Assign Operator

Out = X,  when type in [LoDTensor/SelectedRows/LoDTensorArray]
raise error if the type is not listed above.
)DOC");
  }
};

class AssignInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *context) const override {
    if (context->HasInput("X")) {
      auto type = context->GetInputsVarType("X")[0];
      if (type == framework::VarDesc_VarType_SELECTED_ROWS ||
          type == framework::VarDesc_VarType_LOD_TENSOR) {
        context->SetOutputDim("Out", context->GetInputDim("X"));
      }
    }
  }
};

class AssignGradMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDescBind> Apply() const override {
    auto *op = new framework::OpDescBind();
    op->SetType("assign");
    op->SetInput("X", OutputGrad("Out"));
    op->SetOutput("Out", InputGrad("X"));
    return std::unique_ptr<framework::OpDescBind>(op);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(assign, ops::AssignOp, ops::AssignGradMaker,
                  ops::AssignInferShape, ops::AssignOpProtoMaker);

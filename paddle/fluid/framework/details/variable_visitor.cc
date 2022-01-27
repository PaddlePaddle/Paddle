//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/framework/details/variable_visitor.h"

#include "paddle/fluid/framework/selected_rows_utils.h"

namespace pten {
class DenseTensor;
}  // namespace pten

namespace paddle {
namespace framework {
class Variable;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace details {
template <typename Func>
static void VisitVariable(Variable* var, Func* func) {
  if (var->IsType<LoDTensor>()) {
    (*func)(var->GetMutable<LoDTensor>());
  } else if (var->IsType<SelectedRows>()) {
    (*func)(var->GetMutable<SelectedRows>());
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "VisitVariable is not supported for type %s.",
        ToTypeName(var->Type())));
  }
}

template <typename Func>
static void VisitVariable(const Variable& var, Func* func) {
  if (var.IsType<LoDTensor>()) {
    (*func)(var.Get<LoDTensor>());
  } else if (var.IsType<SelectedRows>()) {
    (*func)(var.Get<SelectedRows>());
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "VisitVariable is not supported for type %s.", ToTypeName(var.Type())));
  }
}

struct TensorVisitor {
  Tensor* result_{nullptr};

  void operator()(LoDTensor* tensor) { result_ = tensor; }

  void operator()(SelectedRows* selected_rows) {
    result_ = selected_rows->mutable_value();
  }

  template <typename T>
  void operator()() {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Getting tensor from type %s is not supported.", typeid(T).name()));
  }
};

Tensor& VariableVisitor::GetMutableTensor(Variable* var) {
  TensorVisitor vistor;
  VisitVariable(var, &vistor);
  return *vistor.result_;
}

struct ShareDimsAndLoDVisitor {
  Variable* trg_;
  void operator()(const LoDTensor& val) {
    auto* tensor = trg_->GetMutable<LoDTensor>();
    tensor->set_layout(val.layout());
    tensor->set_lod(val.lod());
    tensor->Resize(val.dims());
  }

  void operator()(const SelectedRows& val) {
    auto* selected_rows = trg_->GetMutable<SelectedRows>();
    selected_rows->set_rows(val.rows());
    selected_rows->set_height(val.height());
    selected_rows->mutable_value()->Resize(val.value().dims());
  }

  template <typename T>
  void operator()(const T&) {
    PADDLE_THROW(platform::errors::Unimplemented(
        "ShareDimsAndLoD is not supported for type %s.", typeid(T).name()));
  }
};

void VariableVisitor::ShareDimsAndLoD(const Variable& src, Variable* trg) {
  ShareDimsAndLoDVisitor visitor{trg};
  VisitVariable(src, &visitor);
}

struct EnforceShapeAndDTypeEQVisitor {
  const Variable* dst_;

  void operator()(const LoDTensor& src) {
    auto& tensor = dst_->Get<LoDTensor>();
    PADDLE_ENFORCE_EQ(
        src.place().GetType(), tensor.place().GetType(),
        platform::errors::PreconditionNotMet(
            "The place type of the two variables is not equal. The src place "
            "is %s, but the dst place is %s",
            src.place().DebugString(), tensor.place().DebugString()));
    PADDLE_ENFORCE_EQ(src.type(), tensor.type(),
                      platform::errors::PreconditionNotMet(
                          "The dtype of the two variables is not equal."));
    PADDLE_ENFORCE_EQ(
        src.dims(), tensor.dims(),
        platform::errors::PreconditionNotMet(
            "The layout of the two variables' tensors is not equal."));
    PADDLE_ENFORCE_EQ(src.lod(), tensor.lod(),
                      platform::errors::PreconditionNotMet(
                          "The lod of the two variable is not equal."));
    PADDLE_ENFORCE_EQ(
        src.layout(), tensor.layout(),
        platform::errors::PreconditionNotMet(
            "The layout of the two variables' tensors tensor is not equal."));
  }

  void operator()(const SelectedRows& src) {
    auto& selected_rows = dst_->Get<SelectedRows>();
    PADDLE_ENFORCE_EQ(
        src.place().GetType(), selected_rows.place().GetType(),
        platform::errors::PreconditionNotMet(
            "The place type of the two variables is not equal. The src place "
            "is %s, but the dst place is %s",
            src.place().DebugString(), selected_rows.place().DebugString()));
    PADDLE_ENFORCE_EQ(src.value().type(), selected_rows.value().type(),
                      platform::errors::PreconditionNotMet(
                          "The dtype of the two variables is not equal."));
    PADDLE_ENFORCE_EQ(
        src.value().layout(), selected_rows.value().layout(),
        platform::errors::PreconditionNotMet(
            "The layout of the two variables' tensors is not equal."));
    PADDLE_ENFORCE_EQ(src.height(), selected_rows.height(),
                      platform::errors::PreconditionNotMet(
                          "The height of the two variables is not equal."));
    PADDLE_ENFORCE_EQ(src.GetCompleteDims(), selected_rows.GetCompleteDims(),
                      platform::errors::PreconditionNotMet(
                          "The dims of the two variables is not equal."));
  }

  template <typename T>
  void operator()(const T&) {
    PADDLE_THROW(platform::errors::Unimplemented(
        "EnforceShapeAndDTypeEQ is not supported for type %s.",
        typeid(T).name()));
  }
};

void VariableVisitor::EnforceShapeAndDTypeEQ(const Variable& var1,
                                             const Variable& var2) {
  EnforceShapeAndDTypeEQVisitor visitor{&var1};
  VisitVariable(var2, &visitor);
}

}  // namespace details
}  // namespace framework
}  // namespace paddle

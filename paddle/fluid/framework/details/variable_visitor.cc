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
#include "paddle/fluid/framework/selected_rows.h"
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
    PADDLE_THROW("Not supported type %s", var->Type().name());
  }
}

template <typename Func>
static void VisitVariable(const Variable& var, Func* func) {
  if (var.IsType<LoDTensor>()) {
    (*func)(var.Get<LoDTensor>());
  } else if (var.IsType<SelectedRows>()) {
    (*func)(var.Get<SelectedRows>());
  } else {
    PADDLE_THROW("Not supported type %s", var.Type().name());
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
    PADDLE_THROW("Not Support to get LoDTensor from %s", typeid(T).name());
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
    PADDLE_ENFORCE("ShareDimsAndLoD is not supported by type %s",
                   typeid(T).name());
  }
};

void VariableVisitor::ShareDimsAndLoD(const Variable& src, Variable* trg) {
  ShareDimsAndLoDVisitor visitor{trg};
  VisitVariable(src, &visitor);
}

}  // namespace details
}  // namespace framework
}  // namespace paddle

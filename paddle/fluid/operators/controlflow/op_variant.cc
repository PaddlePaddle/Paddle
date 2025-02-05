// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/controlflow/op_variant.h"

namespace paddle::operators {

struct InputsVisitor {
  template <typename OpType>
  const framework::VariableNameMap *operator()(const OpType *op) const {
    return &(op->Inputs());
  }
};

struct OutputsVisitor {
  template <typename OpType>
  const framework::VariableNameMap *operator()(const OpType *op) const {
    return &(op->Outputs());
  }
};

struct AttributeMapVisitor {
  const framework::AttributeMap *operator()(const framework::OpDesc *op) const {
    return &(op->GetAttrMap());
  }

  const framework::AttributeMap *operator()(
      const framework::OperatorBase *op) const {
    return &(op->Attrs());
  }
};

struct RawPointerVisitor {
  template <typename OpType>
  const void *operator()(const OpType *op) const {
    return op;
  }
};

const framework::VariableNameMap &OpVariant::Inputs() const {
  return *paddle::visit(InputsVisitor(), op_);
}

const framework::VariableNameMap &OpVariant::Outputs() const {
  return *paddle::visit(OutputsVisitor(), op_);
}

const framework::AttributeMap &OpVariant::Attrs() const {
  return *paddle::visit(AttributeMapVisitor(), op_);
}

const void *OpVariant::RawPointer() const {
  return paddle::visit(RawPointerVisitor(), op_);
}

void AppendOpVariantByOpName(const std::vector<framework::OpDesc *> &op_descs,
                             const std::string &candidate_op_name,
                             std::vector<OpVariant> *result_ops) {
  PADDLE_ENFORCE_NOT_NULL(
      result_ops,
      common::errors::Unavailable("result_ops should not be a null_ptr."));
  for (auto *op_desc : op_descs) {
    PADDLE_ENFORCE_NOT_NULL(
        op_desc,
        common::errors::Unavailable("op_desc should not be a null_ptr."));
    if (op_desc->Type() == candidate_op_name) {
      result_ops->emplace_back(op_desc);
    }
  }
}

void AppendOpVariantByOpName(
    const std::vector<framework::OpDesc *> &op_descs,
    const std::string &candidate_op_name,
    std::unordered_set<OpVariant, OpVariant::Hasher> *result_ops) {
  PADDLE_ENFORCE_NOT_NULL(
      result_ops,
      common::errors::Unavailable("result_ops should not be a null_ptr."));
  for (auto *op_desc : op_descs) {
    PADDLE_ENFORCE_NOT_NULL(
        op_desc,
        common::errors::Unavailable("op_desc should not be a null_ptr."));
    if (op_desc->Type() == candidate_op_name) {
      result_ops->emplace(op_desc);
    }
  }
}

}  // namespace paddle::operators

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

namespace paddle {
namespace operators {

struct InputsVisitor
    : public boost::static_visitor<const framework::VariableNameMap *> {
  template <typename OpType>
  const framework::VariableNameMap *operator()(const OpType *op) const {
    return &(op->Inputs());
  }
};

struct OutputsVisitor
    : public boost::static_visitor<const framework::VariableNameMap *> {
  template <typename OpType>
  const framework::VariableNameMap *operator()(const OpType *op) const {
    return &(op->Outputs());
  }
};

struct AttributeMapVisitor
    : public boost::static_visitor<const framework::AttributeMap *> {
  const framework::AttributeMap *operator()(const framework::OpDesc *op) const {
    return &(op->GetAttrMap());
  }

  const framework::AttributeMap *operator()(
      const framework::OperatorBase *op) const {
    return &(op->Attrs());
  }
};

struct RawPointerVisitor : public boost::static_visitor<const void *> {
  template <typename OpType>
  const void *operator()(const OpType *op) const {
    return op;
  }
};

const framework::VariableNameMap &OpVariant::Inputs() const {
  return *boost::apply_visitor(InputsVisitor(), op_);
}

const framework::VariableNameMap &OpVariant::Outputs() const {
  return *boost::apply_visitor(OutputsVisitor(), op_);
}

const framework::AttributeMap &OpVariant::Attrs() const {
  return *boost::apply_visitor(AttributeMapVisitor(), op_);
}

const void *OpVariant::RawPointer() const {
  return boost::apply_visitor(RawPointerVisitor(), op_);
}

void AppendOpVariantByOpName(const std::vector<framework::OpDesc *> &op_descs,
                             const std::string &candidate_op_name,
                             std::vector<OpVariant> *result_ops) {
  PADDLE_ENFORCE_NOT_NULL(
      result_ops,
      platform::errors::Unavailable("result_ops should not be a null_ptr."));
  for (auto *op_desc : op_descs) {
    PADDLE_ENFORCE_NOT_NULL(op_desc, platform::errors::Unavailable(
                                         "op_desc should not be a null_ptr."));
    if (op_desc->Type() == candidate_op_name) {
      result_ops->emplace_back(op_desc);
    }
  }
}

}  // namespace operators
}  // namespace paddle

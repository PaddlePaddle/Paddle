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

}  // namespace operators
}  // namespace paddle

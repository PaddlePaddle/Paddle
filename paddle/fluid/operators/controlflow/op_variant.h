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

#pragma once

#include <string>

#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/platform/variant.h"

namespace paddle {
namespace operators {

// OpVariant is a wrapper class of OpDesc and OperatorBase pointer
// So that API would be the same.
class OpVariant {
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
    const framework::AttributeMap *operator()(
        const framework::OpDesc *op) const {
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

 public:
  OpVariant(const framework::OperatorBase *op) : op_(op) {}  // NOLINT

  OpVariant(const framework::OpDesc *op) : op_(op) {}  // NOLINT

  const framework::VariableNameMap &Inputs() const;

  const framework::VariableNameMap &Outputs() const;

  const framework::AttributeMap &Attrs() const;

  template <typename AttrType>
  const AttrType &Attr(const std::string &name) const {
    auto &attrs = Attrs();
    auto it = attrs.find(name);
    PADDLE_ENFORCE(it != attrs.end(), "Cannot find attribute %s", name);
    return boost::get<AttrType>(it->second);
  }

  bool operator==(const OpVariant &other) const {
    return RawPointer() == other.RawPointer();
  }

  const void *RawPointer() const {
    return boost::apply_visitor(RawPointerVisitor(), op_);
  }

  int which() const { return static_cast<int>(op_.which()); }

  struct Hasher {
    size_t operator()(const OpVariant &op) const {
      return reinterpret_cast<size_t>(op.RawPointer());
    }
  };

 private:
  const boost::variant<const framework::OperatorBase *,
                       const framework::OpDesc *>
      op_;
};

}  // namespace operators
}  // namespace paddle

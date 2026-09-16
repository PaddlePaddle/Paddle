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

namespace paddle {
namespace framework {
class OpDesc;
class OperatorBase;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace operators {

// OpVariant is a wrapper class of OpDesc and OperatorBase pointer
// So that API would be the same.
class OpVariant {
 public:
  OpVariant(const framework::OperatorBase *op) : op_(op) {}  // NOLINT

  OpVariant(const framework::OpDesc *op) : op_(op) {}  // NOLINT

  const framework::VariableNameMap &Inputs() const;

  const framework::VariableNameMap &Outputs() const;

  const framework::AttributeMap &Attrs() const;

  const void *RawPointer() const;

  template <typename AttrType>
  const AttrType &Attr(const std::string &name) const {
    auto &attrs = Attrs();
    auto it = attrs.find(name);
    PADDLE_ENFORCE_NE(
        it,
        attrs.end(),
        common::errors::NotFound("Cannot find attribute %s.", name));
    return PADDLE_GET_CONST(AttrType, it->second);
  }

  bool operator==(const OpVariant &other) const {
    return RawPointer() == other.RawPointer();
  }

  int index() const { return static_cast<int>(op_.index()); }

  struct Hasher {
    size_t operator()(const OpVariant &op) const {
      return reinterpret_cast<size_t>(op.RawPointer());
    }
  };

 private:
  const paddle::variant<const framework::OperatorBase *,
                        const framework::OpDesc *>
      op_;
};

void AppendOpVariantByOpName(const std::vector<framework::OpDesc *> &op_descs,
                             const std::string &candidate_op_name,
                             std::vector<OpVariant> *result_ops);

void AppendOpVariantByOpName(
    const std::vector<framework::OpDesc *> &op_descs,
    const std::string &candidate_op_name,
    std::unordered_set<OpVariant, OpVariant::Hasher> *result_ops);

}  // namespace operators
}  // namespace paddle

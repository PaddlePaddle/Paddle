// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/adt/adt.h"
#include "paddle/cinn/adt/m_expr.h"

namespace cinn::hlir::framework {
class Node;
}

namespace cinn {
namespace adt {
namespace m_ir {

// SSAShadowTensor = (tSSAShadow Name, const Graph::NodeData*)
class SSAShadowTensor final : public Tuple<tSSAShadow<Name>, m_expr::Tensor> {
 public:
  using Tuple<tSSAShadow<Name>, m_expr::Tensor>::Tuple;
};
OVERLOAD_OPERATOR_EQ_NE(tSSAShadow<Name>, TagEqual);
OVERLOAD_OPERATOR_EQ_NE(SSAShadowTensor, TupleEqual);

OVERRIDE_TAG_GET_HASH_VALUE(tSSAShadow<Name>);

inline std::size_t GetHashValue(const SSAShadowTensor& shadow_tensor) {
  const auto& [shadow_name, tensor] = shadow_tensor.tuple();
  return hash_combine(GetHashValue(shadow_name), tensor);
}

// Tensor = const Graph::NodeData* | SSAShadowTensor
DEFINE_ADT_UNION(Tensor, m_expr::Tensor, SSAShadowTensor);
OVERRIDE_UNION_GET_HASH_VALUE(Tensor);
OVERLOAD_OPERATOR_EQ_NE(Tensor, UnionEqual);

// Arg = Tensor
using Arg = Tensor;

// Op = const Graph::Node*
using Op = const cinn::hlir::framework::Node*;

// OpStmtNode = (Op, In [Arg], Out [Arg])
class OpStmtNode final : public Tuple<Op, In<List<Arg>>, Out<List<Arg>>> {
 public:
  using Tuple<Op, In<List<Arg>>, Out<List<Arg>>>::Tuple;

  bool operator==(const OpstmtNode& other) const {
    return &this->tuple() == &other.tuple();
  }
};

inline std::size_t GetHashValue(const OpStmtNode& op_stmt_node) {
  return &op_stmt_node.tuple();
}

// MapIR = [OpStmtNode]
using MapIR = List<OpStmtNode>;

}  // namespace m_ir
}  // namespace adt
}  // namespace cinn

namespace std {

template <>
struct hash<cinn::adt::m_ir::Tensor> {
  std::size_t operator()(const cinn::adt::m_ir::Tensor& tensor) const {
    return cinn::adt::m_ir::GetHashValue(tensor);
  }
};

template <>
struct hash<cinn::adt::m_ir::OpStmtNode> {
  std::size_t operator()(
      const cinn::adt::m_ir::OpStmtNode& op_stmt_node) const {
    return cinn::adt::m_ir::GetHashValue(op_stmt_node);
  }
};

}  // namespace std

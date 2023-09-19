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

#include <functional>

#include "paddle/cinn/adt/adapter.h"
#include "paddle/cinn/adt/adt.h"
#include "paddle/cinn/adt/equation_value.h"
#include "paddle/cinn/adt/schedule_descriptor.h"
#include "paddle/cinn/adt/tags.h"

namespace cinn {
namespace hlir {
namespace framework {
class Node;
class NodeData;
}  // namespace framework
}  // namespace hlir
}  // namespace cinn

namespace cinn {
namespace adt {

// Offset = Int64
using Offset = std::int64_t;

class GlobalMemoryType final {
 public:
  bool operator==(const GlobalMemoryType& global_memory_type) const {
    return this == &global_memory_type;
  }
};

inline std::size_t GetHashValueImpl(const GlobalMemoryType&) {
  static GlobalMemoryType global_memory_type;
  return reinterpret_cast<std::size_t>(&global_memory_type);
}

class SharedMemoryType final {
 public:
  bool operator==(const SharedMemoryType& shared_memory_type) const {
    return this == &shared_memory_type;
  }
};

inline std::size_t GetHashValueImpl(const SharedMemoryType&) {
  static SharedMemoryType shared_memory_type;
  return reinterpret_cast<std::size_t>(&shared_memory_type);
}

// MemoryType = GlobalMemoryType | SharedMemoryType
DEFINE_ADT_UNION(MemoryType, GlobalMemoryType, SharedMemoryType);
OVERLOAD_OPERATOR_EQ_NE(MemoryType, UnionEqual);
OVERRIDE_UNION_GET_HASH_VALUE(MemoryType);

// TempStorage = (Name, Offset, MemoryType)
class TempStorage final : public Tuple<Name, Offset, MemoryType> {
 public:
  using Tuple<Name, Offset, MemoryType>::Tuple;
};
OVERLOAD_OPERATOR_EQ_NE(TempStorage, TupleEqual);
inline std::size_t GetHashValueImpl(const TempStorage& temp_storage) {
  const auto& [var_name, offset, memory_type] = temp_storage.tuple();
  std::size_t hash_value = std::hash<std::string>()(var_name);
  hash_value = hash_combine(hash_value, offset);
  hash_value = hash_combine(hash_value, GetHashValue(memory_type));
  return hash_value;
}

// SSAShadowTensor = (tSSAShadow Name, const Graph::NodeData*)
class SSAShadowTensor final : public Tuple<tSSAShadow<Name>, adapter::Tensor> {
 public:
  using Tuple<tSSAShadow<Name>, adapter::Tensor>::Tuple;
};

OVERLOAD_OPERATOR_EQ_NE(SSAShadowTensor, TupleEqual);

OVERRIDE_TAG_GET_HASH_VALUE(tSSAShadow<Name>);

inline std::size_t GetHashValueImpl(const SSAShadowTensor& shadow_tensor) {
  const auto& [shadow_name, tensor] = shadow_tensor.tuple();
  return hash_combine(GetHashValue(shadow_name), GetHashValueImpl(tensor));
}

// Tensor = adapter::Tensor | SSAShadowTensor | TempStorage
DEFINE_ADT_UNION(Tensor, adapter::Tensor, SSAShadowTensor, TempStorage);
OVERRIDE_UNION_GET_HASH_VALUE(Tensor);
OVERLOAD_OPERATOR_EQ_NE(Tensor, UnionEqual);

// Op = const Node*
DEFINE_ADT_UNION(Op, const hlir::framework::Node*);

using Arg = Tensor;

// OpStmt = (Op, In [Arg], Out [Arg])
class OpStmt final : public Tuple<Op, tIn<List<Arg>>, tOut<List<Arg>>> {
 public:
  using Tuple<Op, tIn<List<Arg>>, tOut<List<Arg>>>::Tuple;

  bool operator==(const OpStmt& other) const {
    return &this->tuple() == &other.tuple();
  }
};

inline std::size_t GetHashValue(const OpStmt& op_stmt_node) {
  return reinterpret_cast<std::size_t>(&op_stmt_node.tuple());
}

// MapStmt T = (ScheduleDescriptor, [T])
template <typename T>
class MapStmt final : public Tuple<ScheduleDescriptor, List<T>> {
 public:
  using Tuple<ScheduleDescriptor, List<T>>::Tuple;
};

// Stmt = OpStmt | MapStmt Stmt
DEFINE_ADT_UNION(Stmt, OpStmt, MapStmt<Stmt>);

using TensorIndexExpr = Value;
using TensorIndexExpr4TensorT =
    std::function<const TensorIndexExpr*(const Tensor&)>;

// AnchoredMapStmt = (MapStmt Stmt, tAnchor Tensor, TensorIndexExpr4TensorT)
class AnchoredMapStmt final
    : public Tuple<MapStmt<Stmt>, tAnchor<Tensor>, TensorIndexExpr4TensorT> {
 public:
  using Tuple<MapStmt<Stmt>, tAnchor<Tensor>, TensorIndexExpr4TensorT>::Tuple;
};

// Kernel = ([AnchoredMapStmt], In [Tensor], Out [Tensor])
class Kernel final : public Tuple<List<AnchoredMapStmt>,
                                  tIn<List<Tensor>>,
                                  tOut<List<Tensor>>> {
 public:
  using Tuple<List<AnchoredMapStmt>, tIn<List<Tensor>>, tOut<List<Tensor>>>::
      Tuple;
};

// MapExpr = Kernel;
using MapExpr = Kernel;

}  // namespace adt
}  // namespace cinn

namespace std {

template <>
struct hash<cinn::adt::Tensor> {
  std::size_t operator()(const cinn::adt::Tensor& tensor) const {
    return cinn::adt::GetHashValue(tensor);
  }
};

template <>
struct hash<cinn::adt::OpStmt> {
  std::size_t operator()(const cinn::adt::OpStmt& op_stmt_node) const {
    return cinn::adt::GetHashValue(op_stmt_node);
  }
};

}  // namespace std

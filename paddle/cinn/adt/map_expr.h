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

#include "paddle/cinn/adt/adapter_dynamic_tensor.h"
#include "paddle/cinn/adt/adapter_tensor.h"
#include "paddle/cinn/adt/adt.h"
#include "paddle/cinn/adt/arithmetic.h"
#include "paddle/cinn/adt/equation_value.h"
#include "paddle/cinn/adt/logical.h"
#include "paddle/cinn/adt/schedule_descriptor.h"
#include "paddle/cinn/adt/schedule_mesh.h"
#include "paddle/cinn/adt/tags.h"
#include "paddle/cinn/adt/tree.h"

namespace pir {
class Operation;
}

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

// Tensor = adapter::Tensor | adapter::DynamicTensor | TempStorage
DEFINE_ADT_UNION(Tensor, adapter::Tensor, adapter::DynamicTensor, TempStorage);
OVERRIDE_UNION_GET_HASH_VALUE(Tensor);
OVERLOAD_OPERATOR_EQ_NE(Tensor, UnionEqual);

// Op = const pir::Operation*
//    | tReduceInit<const pir::Operation*>
//    | tReduceAcc<const pir::Operation*>
DEFINE_ADT_UNION(Op,
                 const ::pir::Operation*,
                 tReduceInit<const ::pir::Operation*>,
                 tReduceAcc<const ::pir::Operation*>);

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

using LoopIterators = List<Iterator>;
// MapStmt T = ([Iterator], [T])
template <typename T>
class MapStmt final : public Tuple<LoopIterators, List<T>> {
 public:
  using value_type = LoopIterators;
  using Tuple<LoopIterators, List<T>>::Tuple;
};

// Stmt = OpStmt | MapStmt Stmt
using Stmt = Tree<MapStmt, OpStmt>;

template <typename OutT, typename InT>
class Store final : public Tuple<OutT, InT> {
 public:
  using Tuple<OutT, InT>::Tuple;
};

template <typename T>
class Load final : public Tuple<T> {
 public:
  using Tuple<T>::Tuple;
};

// OpCall T = (Op, [T])
template <typename T>
class OpCall final : public Tuple<Op, List<T>> {
 public:
  using Tuple<Op, List<T>>::Tuple;
};

// OpExpr = Tree OpCall (Load Tensor)
using OpExpr = Tree<OpCall, Load<Tensor>>;
// OpExprStmt = Store Tensor OpExpr
using OpExprStmt = Store<Tensor, OpExpr>;

using InlineStmt = Tree<MapStmt, OpExprStmt>;

using TensorIndexExpr = Value;
using TensorIndexExpr4TensorT = std::function<TensorIndexExpr(const Tensor&)>;
using TensorIteratorExpr = Value;
using TensorIteratorExpr4TensorT =
    std::function<List<TensorIteratorExpr>(const Tensor&)>;
using LoopDescriptor4LoopIteratorT =
    std::function<LoopDescriptor(const Iterator&)>;

// AnchoredMapStmt = (MapStmt Stmt, ScheduleMesh, tAnchor Tensor,
// TensorIndexExpr4TensorT, TensorIteratorExpr4TensorT,
// LoopDescriptor4LoopIteratorT)
class AnchoredMapStmt final : public Tuple<MapStmt<Stmt>,
                                           ScheduleMesh,
                                           tAnchor<Tensor>,
                                           TensorIndexExpr4TensorT,
                                           TensorIteratorExpr4TensorT,
                                           LoopDescriptor4LoopIteratorT> {
 public:
  using Tuple<MapStmt<Stmt>,
              ScheduleMesh,
              tAnchor<Tensor>,
              TensorIndexExpr4TensorT,
              TensorIteratorExpr4TensorT,
              LoopDescriptor4LoopIteratorT>::Tuple;

  TensorIndexExpr GetTensorIndexExpr(const Tensor& tensor) const {
    const auto& TensorIndexExpr4Tensor = std::get<3>(tuple());
    return TensorIndexExpr4Tensor(tensor);
  }
};

DEFINE_ADT_UNION(GenericDim, SymbolicDim, std::int64_t);
using KernelCondition = Logical<Tree<Arithmetic, GenericDim>>;

template <typename T>
class ConditionalAnchoredMapStmt
    : public Tuple<List<KernelCondition>, tTrue<T>, tFalse<T>> {
 public:
  using Tuple<List<KernelCondition>, tTrue<T>, tFalse<T>>::Tuple;
};

using KernelBody = Tree<ConditionalAnchoredMapStmt, AnchoredMapStmt>;

// Kernel = (KernelBody, In [Tensor], Out [Tensor])
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

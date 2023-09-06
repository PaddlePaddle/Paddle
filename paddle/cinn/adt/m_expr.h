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

#include <function>

#include "paddle/cinn/adt/adt.h"
#include "paddle/cinn/adt/equation_value.h"
#include "paddle/cinn/adt/schedule_descriptor.h"

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
namespace m_expr {

// Offset = Int64
using Offset = std::int64_t;

class GlobalMemoryType final {};
class SharedMemoryType final {};
// MemoryType = GlobalMemoryType | SharedMemoryType
DEFINE_ADT_UNION(MemoryType, GlobalMemoryType, SharedMemoryType);

// TempStorage = (tVar Name, Offset, MemoryType)
class TempStorage final : public Tuple<tVar<Name>, Offset, MemoryType> {
 public:
  using Tuple<tVar<Name>, Offset, MemoryType>::Tuple;
};

// SSAShadowTensor = (tSSAShadow Name, const Graph::NodeData*)
using SSAShadowTensor = m_ir::SSAShadowTensor;

// Tensor = const Graph::NodeData* | SSAShadowTensor | TempStorage
DEFINE_ADT_UNION(Tensor,
                 const hlir::framework::NodeData*,
                 SSAShadowTensor,
                 TempStorage);

// Arg = (Tensor, TensorIndexExpr)
using TensorIndexExpr = equation::Value;
using Arg = Tuple<Tensor, TensorIndexExpr>;

// MemoryBarrier = {}    // (Sync Thread)
class MemoryBarrier final {};

// BuiltinReduceRelatedOp = Zeros | InplaceAdd
class Zeros final {};
class InplaceAdd final {};
DEFINE_ADT_UNION(BuiltinReduceRelatedOp, Zeros, InplaceAdd);

// Op = const Node* | BuiltinReduceRelatedOp | MemoryBarrier
DEFINE_ADT_UNION(Op,
                 const hlir::framework::Node*,
                 BuiltinReduceRelatedOp,
                 MemoryBarrier);

// OpStmtNode = (Op, In [Arg], Out [Arg])
class OpStmtNode final : public Tuple<Op, In<List<Arg>>, Out<List<Arg>>> {
 public:
  using Tuple<Op, In<List<Arg>>, Out<List<Arg>>>::Tuple;

  bool operator==(const OpstmtNode& other) const {
    return &this->tuple() == &other.tuple();
  }
};

// MapNode T = (ScheduleDescriptor, [T])
template <typename T>
class MapNode final : public Tuple<ScheduleDescriptor, List<T>> {
 public:
  using Tuple<ScheduleDescriptor, List<T>>::Tuple;
};

// Stmt = OpStmtNode | MapNode Stmt
DEFINE_ADT_UNION(Stmt, OpStmtNode, MapNode<Stmt>);

// IGroup = (MapStmtNode, tAnchor Tensor)
class IGroup final : public Tuple<MapNode<Stmt>, tAnchor<Tensor>> {
 public:
  using Tuple<MapNode<Stmt>, tAnchor<Tensor>>::Tuple;
};

// Kernel = ([IGroup], In [Tensor], Out [Tensor])
class Kernel final
    : public Tuple<List<IGroup>, In<List<Tensor>>, Out<List<Tensor>>> {
 public:
  using Tuple<List<IGroup>, In<List<Tensor>>, Out<List<Tensor>>>::Tuple;
};

// MapExpr = Kernel
using MapExpr = Kernel;

}  // namespace m_expr
}  // namespace adt
}  // namespace cinn

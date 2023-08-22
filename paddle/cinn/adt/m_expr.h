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
#include "paddle/cinn/adt/tensor_flatten_index_lambda.h"

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

class AutoSize final {};

// ScheduleSize = Int64 | AutoSize
using ScheduleSize = Union<std::int64_t, AutoSize>;

// S(Spatial): S0 = BlockIdx; S1 = ThreadIdx
// ScheduleType = S0x | S0y | S0z | S1x | S1y | S1z | Temporal | Vectorize |
// Unroll
class S0x final {};
class S0y final {};
class S0z final {};
class S1x final {};
class S1y final {};
class S1z final {};
class Temporal final {};
class Vectorize final {};
class Unroll final {};
using ScheduleType =
    Union<S0x, S0y, S0z, S1x, S1y, S1z, Temporal, Vectorize, Unroll>;

// ScheduleDescriptor = [(ScheduleType, ScheduleSize)]
using ScheduleDescriptor = List<Tuple<ScheduleType, ScheduleSize>>;

// Offset = Int64
using Offset = std::int64_t;

class GlobalMemoryType final {};
class SharedMemoryType final {};
// MemoryType = GlobalMemoryType | SharedMemoryType
using MemoryType = Union<GlobalMemoryType, SharedMemoryType>;

// TempStorage = (tVar Name, Offset, MemoryType)
using TempStorage = Tuple<tVar<Name>, Offset, MemoryType>;

// Tensor = const Graph::NodeData*
using Tensor = const hlir::framework::NodeData*;

// TensorOrBuf = Tensor | TempStorage
using TensorOrBuf = Union<Tensor, TempStorage>;

// TensorArg = (TensorOrBuf, TensorFlattenIndexLambda <- [tScheduleIterVar Name,
// ScheduleSize])
using TensorArg =
    Tuple<TensorOrBuf,
          std::function<TensorFlattenIndexLambda(
              List<Tuple<tScheduleIterVar<Name>, ScheduleSize>>)>>;

class OpExprNode;
// OpExpr = Box OpExprNode
using OpExpr = std::shared_ptr<OpExprNode>;

// Arg = TensorArg | OpExpr
using Arg = Union<TensorArg, OpExpr>;

// MemoryBarrier = {}    // (Sync Thread)
class MemoryBarrier final {};

// BuiltinReduceRelatedOp = Zeros | InplaceAdd
class Zeros final {};
class InplaceAdd final {};
using BuiltinReduceRelatedOp = Union<Zeros, InplaceAdd>;

// Op = const Node* | BuiltinReduceRelatedOp | MemoryBarrier
using Op =
    Union<const hlir::framework::Node*, BuiltinReduceRelatedOp, MemoryBarrier>;

// OpExprNode = (Op, In [Arg])
class OpExprNode final : public Tuple<Op, In<Arg>> {
  using Tuple<Op, In<Arg>>::Tuple;
};

class StmtNode;
// Stmt = Box StmtNode
using Stmt = Box<StmtNode>;

// OpStmtNode = (Op, In [Arg], Out [TensorArg])
using OpStmtNode = Tuple<Op, In<List<Arg>>, Out<List<TensorArg>>>;

// MapStmtNode = (ScheduleDescriptor, tAnchor Tensor, [Stmt])
using MapStmtNode = Tuple<ScheduleDescriptor, tAnchor<Tensor>, List<Stmt>>;

// StmtNode = OpStmtNode | MapStmtNode
class StmtNode final : public Union<OpStmtNode, MapStmtNode> {
  using Union<OpStmtNode, MapStmtNode>::Union;
};

// Kernel = (MapStmtNode, In [Tensor], Out [Tensor])
using Kernel = Tuple<MapStmtNode, In<List<Tensor>>, Out<List<Tensor>>>;

// MapExpr = Kernel
using MapExpr = Kernel;

}  // namespace m_expr
}  // namespace adt
}  // namespace cinn

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

#include "paddle/cinn/adt/adt.h"

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
namespace m_ir {

// ScheduleSize = Int64
using ScheduleSize = int64_t;

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

// RankedSchedulePolicy = [(ScheduleType, ScheduleSize)]
using RankedSchedulePolicy = List<Tuple<ScheduleType, ScheduleSize>>;

// SchedulePolicy = RankedSchedulePolicy | [RankedSchedulePolicy]
using SchedulePolicy = Union<RankedSchedulePolicy, List<RankedSchedulePolicy>>;

// DimPatternKind = ElementwiseKind | InjectiveKind | BroadcastKind | ReduceKind
// | OpaqueKind
class ElementwiseKind final {};
class InjectiveKind final {};
class BroadcastKind final {};
class ReduceKind final {};
class OpaqueKind final {};
using DimPatternKind = Union<ElementwiseKind,
                             InjectiveKind,
                             BroadcastKind,
                             ReduceKind,
                             OpaqueKind>;

// ScheduleDescriptor = (DimPatternKind, SchedulePolicy)
using ScheduleDescriptor = Tuple<DimPatternKind, SchedulePolicy>;

// DimSize = Int64
using DimSize = int64_t;

// Stride = Int64
using Stride = int64_t;

// BroadcastedDimSize = Int64
using BroadcastedDimSize = int64_t;

// Tensor = const Graph::NodeData*
using Tensor = const hlir::framework::NodeData*;

// Arg = (Tensor, [(DimSize, Stride, BroadcastedDimSize)])
using Arg = Tuple<Tensor, List<Tuple<DimSize, Stride, BroadcastedDimSize>>>;

// InArgs = [Arg]
using InArgs = List<Arg>;

// OutArgs = [Arg]
using OutArgs = List<Arg>;

// MemoryBarrier = {}    // (Sync Thread)
class MemoryBarrier final {};

// BuiltinReduceRelatedOp = Zeros | InplaceAdd
class Zeros final {};
class InplaceAdd final {};
using BuiltinReduceRelatedOp = Union<Zeros, InplaceAdd>;

// Op = const Node* | BuiltinReduceRelatedOp | MemoryBarrier
using Op =
    Union<const hlir::framework::Node*, BuiltinReduceRelatedOp, MemoryBarrier>;

class StmtNode;
// Stmt = Box StmtNode
using Stmt = Box<StmtNode>;

// OpStmtNode = (Op, InArgs, OutArgs)
using OpStmtNode = Tuple<Op, InArgs, OutArgs>;

// MapStmtNode = ([ScheduleDescriptor], [Stmt])
using MapStmtNode = Tuple<List<ScheduleDescriptor>, List<Stmt>>;

// StmtNode = OpStmtNode | MapStmtNode
class StmtNode final : public Union<OpStmtNode, MapStmtNode> {
  using Union<OpStmtNode, MapStmtNode>::Union;
};

// Kernel = (MapStmtNode, InArgs, OutArgs)
using Kernel = Tuple<MapStmtNode, InArgs, OutArgs>;

// InterfaceADT = Kernel
using InterfaceADT = Kernel;

}  // namespace m_ir
}  // namespace adt
}  // namespace cinn

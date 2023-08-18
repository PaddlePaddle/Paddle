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

namespace cinn {
namespace adt {
namespace m_ir {

// SymbolicScheduleType = Var Name
using SymbolicScheduleType = Var<Name>;

// SymbolicScheduleSize = Var Name
using SymbolicScheduleSize = Var<Name>;

// SymbolicSchedulePolicy = [(SymbolicScheduleType, SymbolicScheduleSize)]
using SymbolicSchedulePolicy =
    List<Tuple<SymbolicScheduleType, SymbolicScheduleSize>>;

// DimPatternKind = ElementwiseKind | InjectiveKind | BroadcastKind | ReduceKind
// | OpaqueKind
using DimPatternKind = m_expr::DimPatternKind;

// SymbolicScheduleDescriptor = (DimPatternKind, SymbolicSchedulePolicy)
using SymbolicScheduleDescriptor =
    Tuple<DimPatternKind, SymbolicSchedulePolicy>;

// SymbolicDimSize = Var Name
using SymbolicDimSize = Var<Name>;

// SymbolicStride = Var Name
using SymbolicStride = Var<Name>;

// Tensor = (const Graph::NodeData*, [[(SymbolicDimSize, SymbolicStride)]])
using Tensor =
    Tuple<m_expr::Tensor, List<List<Tuple<SymbolicDimSize, SymbolicStride>>>>;

// Arg = (Tensor, [[tag.Broadcasted SymbolicDimSize]])
using Arg = Tuple<Tensor, List<List<tag::Broadcasted<SymbolicDimSize>>>>;

// Op = const Graph::Node* | BuiltinReduceRelatedOp | MemoryBarrier
// BuiltinReduceRelatedOp = Zeros | InplaceAdd
// MemoryBarrier = {}    // (Sync Thread)
using Op = m_expr::Op;

// OpStmtNode = (Op, In [Arg], Out [Arg])
using OpStmtNode = Tuple<Op, In<List<Arg>>, Out<List<Arg>>>;

// MapStmtNode = ([SymbolicScheduleDescriptor], OpStmtNode])
using MapStmtNode = Tuple<List<SymbolicScheduleDescriptor>, OpStmtNode>;

// InternalADT = [MapStmtNode]
using InternalADT = List<MapStmtNode>;

}  // namespace m_ir
}  // namespace adt
}  // namespace cinn

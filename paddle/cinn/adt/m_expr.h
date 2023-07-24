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
#include "paddle/cinn/adt/m_ir.h"

namespace cinn {
namespace adt {
namespace m_expr {

// UniqueName = std::string
using UniqueName = std::string;

// SymbolicScheduleType = (UniqueName,)
using SymbolicScheduleType = Tuple<UniqueName>;

// SymbolicScheduleSize = (UniqueName,)
using SymbolicScheduleSize = Tuple<UniqueName>;

// SymbolicSchedulePolicy = [(SymbolicScheduleType, SymbolicScheduleSize)]
using SymbolicSchedulePolicy =
    List<Tuple<SymbolicScheduleType, SymbolicScheduleSize>>;

// DimPatternKind = ElementwiseKind | InjectiveKind | BroadcastKind | ReduceKind
// | OpaqueKind
using DimPatternKind = m_ir::DimPatternKind;

// SymbolicScheduleDescriptor = (DimPatternKind, SymbolicSchedulePolicy)
using SymbolicScheduleDescriptor =
    Tuple<DimPatternKind, SymbolicSchedulePolicy>;

// SymbolicDimSize = (UniqueName,)
using SymbolicDimSize = Tuple<UniqueName>;

// SymbolicStride = (UniqueName,)
using SymbolicStride = Tuple<UniqueName>;

// SymbolicBroadcastedDimSize = (UniqueName,)
using SymbolicBroadcastedDimSize = Tuple<UniqueName>;

// Tensor = const Graph::NodeData*
using Tensor = m_ir::Tensor;

// Arg = (Tensor, [(SymbolicDimSize, SymbolicStride,
// SymbolicBroadcastedDimSize)])
using Arg = Tuple<
    Tensor,
    List<Tuple<SymbolicDimSize, SymbolicStride, SymbolicBroadcastedDimSize>>>;

// InArgs = [Arg]
using InArgs = List<Arg>;

// OutArgs = [Arg]
using OutArgs = List<Arg>;

// Op = const Graph::Node* | BuiltinReduceRelatedOp | MemoryBarrier
// BuiltinReduceRelatedOp = Zeros | InplaceAdd
// MemoryBarrier = {}    // (Sync Thread)
using Op = m_ir::Op;

// OpStmtNode = (Op, InArgs, OutArgs)
using OpStmtNode = Tuple<Op, InArgs, OutArgs>;

// MapStmtNode = ([SymbolicScheduleDescriptor], OpStmtNode])
using MapStmtNode = Tuple<List<SymbolicScheduleDescriptor>, OpStmtNode>;

// InternalADT = [MapStmtNode]
using InternalADT = List<MapStmtNode>;

}  // namespace m_expr
}  // namespace adt
}  // namespace cinn

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

// AutoSize = {}
class AutoSize final {};

// ScheduleSize = Int64 | AutoSize
using ScheduleSize = Tuple<std::size_t, AutoSize>;

// ScheduleType = S0x | S0y | S0z | S1x | S1y | S1z | Temporal | Vectorize |
// Unroll
using ScheduleType = m_expr::ScheduleType;

// ScheduleDescriptor = [(ScheduleType, ScheduleSize)]
using ScheduleDescriptor = List<Tuple<ScheduleType, ScheduleSize>>;

// SymbolicDim = tVar Name
using SymbolicDim = tVar<Name>;

// SSAShadowTensor = (tSSAShadow Name, const Graph::NodeData*)
using SSAShadowTensor = Tuple<tSSAShadow<Name>, m_expr::Tensor>;

// Tensor = const Graph::NodeData* | SSAShadowTensor
using Tensor = Union<m_expr::Tensor, SSAShadowTensor>;

// Arg = (Tensor, [SymbolicDim])
using Arg = Tuple<Tensor, List<SymbolicDim>>;

// Op = const Graph::Node* | BuiltinReduceRelatedOp | MemoryBarrier
// BuiltinReduceRelatedOp = Zeros | InplaceAdd
// MemoryBarrier = {}    // (Sync Thread)
using Op = m_expr::Op;

// OpStmtNode = (Op, In [Arg], Out [Arg])
using OpStmtNode = Tuple<Op, In<List<Arg>>, Out<List<Arg>>>;

// MapStmtNode = (ScheduleDescriptor, OpStmtNode)
using MapStmtNode = Tuple<ScheduleDescriptor, OpStmtNode>;

// MapIR = [MapStmtNode]
using MapIR = List<MapStmtNode>;

}  // namespace m_ir
}  // namespace adt
}  // namespace cinn

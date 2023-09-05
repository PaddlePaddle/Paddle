// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/cinn/adt/equation_value.h"
#include "paddle/cinn/adt/m_ir.h"

namespace cinn::adt {

using ScheduleIterators = List<equation::IterVar>;
// [(ScheduleDescriptor, [OpNode])]
using Schedule4MergedOps = List<Tuple<ScheduleIterators, List<m_ir::Op>>>;

Schedule4MergedOps GenerateClusterOpsForLoopFuse(
    const m_ir::MapIR& map_ir,
    const ScheduleIterators& sd_iters,
    const std::function<const m_expr::ScheduleDescriptor&(
        const equation::IterVar&)>& GetScheduleType,
    const std::function<TensorIndexExpr(
        const cinn::hlir::framework::NodeData*)>& GetTensorIndexes);

}  // namespace cinn::adt

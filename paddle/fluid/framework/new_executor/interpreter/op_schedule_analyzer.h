// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include <cstdint>
#include <memory>
#include <vector>

#include "paddle/fluid/framework/new_executor/instruction/instruction_base.h"
#include "paddle/fluid/framework/new_executor/instruction/instruction_defs.h"
#include "paddle/phi/common/place.h"
#include "paddle/pir/include/core/operation.h"

namespace paddle {
namespace framework {
namespace interpreter {

// Single-op per-tensor element threshold for classifying a static-shape CPU
// PhiKernelOp as HostSyncCost::kInlineMetadata.
constexpr int64_t kInlineMetadataMaxOpNumel = 64;

// Cumulative host element budget across all kInlineMetadata instructions in a
// graph before PirInterpreter falls back from TraceRun to MultiThreadRun.
constexpr int64_t kGraphInlineMetadataNumelBudget = 1024;

// Determines the low-level execution thread class (kCpuSync / kGpuSync /
// kGpuAsync) for a PIR operation on `place`.
OpFuncType AnalyzeOpFuncType(pir::Operation* op, const Place& place);

// Unified static scheduling classification for a PIR operation on `place`.
// Combines OpFuncType analysis with a 3-tier host sync cost model executed
// once at graph-build time (zero per-step runtime overhead):
//   Tier 1 (Blocklist): blocking D2H / I/O / Python / custom / control-flow /
//                       superlinear CPU ops -> kBlockingDeviceIO or kCompute.
//   Tier 2 (Intrinsic): O(1) metadata ops (shape / shape64 / full_int_array /
//                       numel / builtin.combine of CPU metadata) ->
//                       kInlineMetadata.
//   Tier 3 (Cost model): standard pd_kernel PhiKernelOp CPU ops whose static
//                        input & output dense-tensor numels are all known
//                        (with all inputs on CPUPlace) and total <=
//                        kInlineMetadataMaxOpNumel -> kInlineMetadata.
OpScheduleInfo AnalyzeOpScheduleInfo(pir::Operation* op, const Place& place);

// Overload that preserves an already-determined OpFuncType on an instruction
// (e.g., control-flow instructions that override KernelType()) while filling
// in HostSyncCost and estimated_host_numel.
OpScheduleInfo AnalyzeOpScheduleInfo(const InstructionBase* ins,
                                     const Place& place);

// Counts instructions that block TraceRun, applying the per-op HostSyncCost
// classification and the graph-level cumulative host element budget
// (kGraphInlineMetadataNumelBudget).
int64_t CountTraceBlockingSyncOps(
    const std::vector<std::unique_ptr<InstructionBase>>& instructions);

}  // namespace interpreter
}  // namespace framework
}  // namespace paddle

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

#include "paddle/cinn/optim/pass_registry.h"

#include <unordered_map>

#include "paddle/cinn/ir/schedule/ir_schedule_util.h"
#include "paddle/cinn/optim/eliminate_invariant_loop.h"
#include "paddle/cinn/optim/entail_loop_condition_pass.h"
#include "paddle/cinn/optim/extern_call_process_pass.h"
#include "paddle/cinn/optim/if_fold_pass.h"
#include "paddle/cinn/optim/if_fusion_pass.h"
#include "paddle/cinn/optim/ir_simplify.h"
#include "paddle/cinn/optim/lower_intrin.h"
#include "paddle/cinn/optim/map_extern_call.h"
#include "paddle/cinn/optim/realize_composite_reduce_pass.h"
#include "paddle/cinn/optim/rearrange_load_instruction_pass.h"
#include "paddle/cinn/optim/reindex_transpose_buffer_pass.h"
#include "paddle/cinn/optim/remove_schedule_block_pass.h"
#include "paddle/cinn/optim/replace_cross_block_reduction.h"
#include "paddle/cinn/optim/replace_cross_thread_reduction.h"
#include "paddle/cinn/optim/trans_buffer_with_dynamic_shape.h"
#include "paddle/cinn/optim/transform_gpu_forloop.h"
#include "paddle/cinn/optim/vectorize_for_trans.h"
#include "paddle/cinn/pass/pass_manager.h"

namespace cinn {
namespace optim {

static const std::unordered_map<std::string, PassAction>& GetBuiltinPassMap() {
  // clang-format off
  static const std::unordered_map<std::string, PassAction> kMap = {
    {"Simplify", [](ir::LoweredFunc f, const Target&) {
      Simplify(&f->body);
    }},
    {"EliminateInvariantLoop", [](ir::LoweredFunc f, const Target&) {
      EliminateInvariantLoop(&f->body);
    }},
    {"RealizeCompositeReduce", [](ir::LoweredFunc f, const Target& t) {
      FuncPassManager mgr;
      mgr.AddPass(CreateRealizeCompositeReducePass(t));
      mgr.Run(f);
    }},
    {"ReindexTransposeBuffer", [](ir::LoweredFunc f, const Target&) {
      FuncPassManager mgr;
      mgr.AddPass(CreateReindexTransposeBufferPass());
      mgr.Run(f);
    }},
    {"ReplaceCrossThreadReduction", [](ir::LoweredFunc f, const Target&) {
      ReplaceCrossThreadReduction(f);
    }},
    {"ReplaceCrossBlockReduction", [](ir::LoweredFunc f, const Target&) {
      ReplaceCrossBlockReduction(f);
    }},
    {"SetCudaAxisInfo", [](ir::LoweredFunc f, const Target&) {
      ir::SetCudaAxisInfo(f);
    }},
    {"RemoveGpuForLoops", [](ir::LoweredFunc f, const Target&) {
      FuncPassManager mgr;
      mgr.AddPass(CreateRemoveGpuForLoopsPass());
      mgr.Run(f);
    }},
    {"CudaSyncThreadsDropIfThenElse", [](ir::LoweredFunc f, const Target&) {
      BlockPassManager mgr;
      mgr.AddPass(CreateCudaSyncThreadsDropIfThenElsePass());
      mgr.Run(f->body_block);
    }},
    {"TransBufferWithDynamicShape", [](ir::LoweredFunc f, const Target&) {
      FuncPassManager mgr;
      mgr.AddPass(CreateTransBufferWithDynamicShapePass());
      mgr.Run(f);
    }},
    {"SimplifyUnitBlock", [](ir::LoweredFunc f, const Target&) {
      SimplifyUnitBlock(&f->body);
    }},
    {"MapExternCall", [](ir::LoweredFunc f, const Target& t) {
      MapExternCall(&f->body, t);
    }},
    {"ExternCallMultiOutputShallowStore", [](ir::LoweredFunc f, const Target&) {
      BlockPassManager mgr;
      mgr.AddPass(CreateExternCallMultiOutputShallowStorePass());
      mgr.Run(f);
    }},
    {"IfFusion", [](ir::LoweredFunc f, const Target&) {
      BlockPassManager mgr;
      mgr.AddPass(CreateIfFusionPass());
      mgr.Run(f);
    }},
    {"EntailLoopCondition", [](ir::LoweredFunc f, const Target&) {
      BlockPassManager mgr;
      mgr.AddPass(CreateEntailLoopConditionPass());
      mgr.Run(f);
    }},
    {"RearrangeLoadInstruction", [](ir::LoweredFunc f, const Target&) {
      FuncPassManager mgr;
      mgr.AddPass(CreateRearrangeLoadInstructionPass());
      mgr.Run(f);
    }},
    {"VectorizeForTrans", [](ir::LoweredFunc f, const Target&) {
      VectorizeForTrans(&f->body);
    }},
    {"RemoveScheduleBlock", [](ir::LoweredFunc f, const Target&) {
      BlockPassManager mgr;
      mgr.AddPass(CreateRemoveScheduleBlockPass());
      mgr.Run(f);
    }},
    {"IfFold", [](ir::LoweredFunc f, const Target&) {
      StmtPassManager mgr;
      mgr.AddPass(CreateIfFoldPass());
      mgr.Run(f);
    }},
    {"LowerIntrin", [](ir::LoweredFunc f, const Target& t) {
      LowerIntrin(&f->body, t);
    }},
    {"PrepareBufferCastExprs", [](ir::LoweredFunc f, const Target&) {
      f->PrepareBufferCastExprs(false);
    }},
  };
  // clang-format on
  return kMap;
}

std::vector<std::string> GetDefaultGpuPassPipeline() {
  return {
      "Simplify",
      "EliminateInvariantLoop",
      "RealizeCompositeReduce",
      "ReindexTransposeBuffer",
      "ReplaceCrossThreadReduction",
      "ReplaceCrossBlockReduction",
      "SetCudaAxisInfo",
      "RemoveGpuForLoops",
      "CudaSyncThreadsDropIfThenElse",
      "TransBufferWithDynamicShape",
      "SimplifyUnitBlock",
      "MapExternCall",
      "ExternCallMultiOutputShallowStore",
      "Simplify",
      "IfFusion",
      "EntailLoopCondition",
      "RearrangeLoadInstruction",
      "VectorizeForTrans",
      "Simplify",
      "RemoveScheduleBlock",
      "IfFold",
      "LowerIntrin",
      "PrepareBufferCastExprs",
  };
}

const PassAction* LookupBuiltinPass(const std::string& name) {
  auto& map = GetBuiltinPassMap();
  auto it = map.find(name);
  return it != map.end() ? &it->second : nullptr;
}

}  // namespace optim
}  // namespace cinn

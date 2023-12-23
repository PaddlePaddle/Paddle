// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/optim/optimize.h"

#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/schedule/ir_schedule_util.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/cinn/optim/call_arg_list_to_pod_value.h"
#include "paddle/cinn/optim/cast_bool_to_int8.h"
#include "paddle/cinn/optim/eliminate_broadcast_in_forloop.h"
#include "paddle/cinn/optim/extern_call_process.h"
#include "paddle/cinn/optim/fold_cinn_call_arguments.h"
#include "paddle/cinn/optim/insert_debug_log_callee.h"
#include "paddle/cinn/optim/ir_simplify.h"
#include "paddle/cinn/optim/lower_function_call_bind_vars.h"
#include "paddle/cinn/optim/lower_intrin.h"
#include "paddle/cinn/optim/map_extern_call.h"
#include "paddle/cinn/optim/remove_schedule_block.h"
#include "paddle/cinn/optim/replace_const_param_to_integer.h"
#include "paddle/cinn/optim/replace_cross_thread_reduction.h"
#include "paddle/cinn/optim/transform_gpu_forloop.h"
#include "paddle/cinn/optim/transform_polyfor_to_for.h"
#include "paddle/cinn/optim/unroll_loops.h"
#include "paddle/cinn/optim/vectorize_loops.h"

namespace cinn {
namespace optim {

Expr Optimize(Expr e,
              Target target,
              bool runtime_debug_info,
              bool remove_gpu_for_loops) {
  CHECK(e.defined());
  auto copied = ir::ir_utils::IRCopy(e);

  FoldCINNCallArguments(&copied);
  TransformPolyForToFor(&copied);
  ReplaceConstParamToInteger(&copied);
  // Simplify already contains CastSimplify
  Simplify(&copied);
  ReplaceCrossThreadReduction(&copied);
  UnrollLoop(&copied);
  VLOG(4) << "After Optimize UnrollLoop:" << copied;

  VectorizeLoops(&copied, target);
  VLOG(4) << "After Optimize VectorizeLoops:" << copied;
#ifdef CINN_WITH_CUDA
  if (copied.as_lowered_func()) {
    ir::SetCudaAxisInfo(&copied);
  }
  if (remove_gpu_for_loops) {
    RemoveGpuForloopsAxis(&copied);
  }
  CudaSyncThreadsDropIfThenElse(&copied);
#endif

  SimplifyBlocks(&copied);
  VLOG(4) << "After SimplifyBlocks:" << copied;

  MapExternCall(&copied, target);
  VLOG(10) << "After Optimize MapExternCall:" << copied;

  ExternCallMultiOutputShallowStore(&copied);
  VLOG(10) << "After Optimize ExternCallMultiOutputShallowStore:" << copied;
  // Simplify already contains CastSimplify
  Simplify(&copied);
  VLOG(10) << "After Optimize Simplify:" << copied;

  if (runtime_debug_info) {
    LOG(WARNING) << "Turn on runtime debug information output";
    InsertDebugLogCallee(&copied);
  }
  return copied;
}

ir::Module Optimize(const ir::Module& module, const Target& target) {
  auto copied = ir::ir_utils::IRCopy(Expr(module));
  ReplaceCrossThreadReduction(&copied);
  UnrollLoop(&copied);
  VectorizeLoops(&copied, Target());
  VLOG(10) << "After VectorizeLoops:" << copied.as_module_ref();
  RemoveScheduleBlock(&copied);
  VLOG(10) << "After RemoveScheduleBlock:" << copied.as_module_ref();
  LowerFunctionCallBindVars(&copied);
  VLOG(10) << "After LowerFunctionCallBindVars:" << copied.as_module_ref();
  CallArgListToPodValue(&copied);
  VLOG(10) << "After CallArgListToPodValue:" << copied.as_module_ref();
  LowerIntrin(&copied, target);
  VLOG(10) << "After LowerIntrin:" << copied.as_module_ref();

  return copied.as_module_ref();
}

}  // namespace optim
}  // namespace cinn

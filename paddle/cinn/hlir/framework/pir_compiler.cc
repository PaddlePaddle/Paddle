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

#include "paddle/cinn/hlir/framework/pir_compiler.h"

#include <absl/types/variant.h>
#include "paddle/cinn/hlir/framework/pir/compilation_task.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/cinn/utils/multi_threading.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"

PD_DECLARE_bool(cinn_bucket_compile);
PD_DECLARE_int32(cinn_parallel_compile_thread);

namespace cinn {
namespace hlir {
namespace framework {

PirCompiler::CompileResult PirCompiler::Build(
    const std::vector<pir::GroupPtr>& groups) {
  if (FLAGS_cinn_bucket_compile) {
    return BucketBuild(groups);
  } else {
    return StaticBuild(groups);
  }
}

PirCompiler::CompileResult PirCompiler::BucketBuild(
    const std::vector<pir::GroupPtr>& groups) {
  std::vector<pir::CINNKernelInfo> cinn_kernel_info_vecs(groups.size());
  for (int i = 0; i < groups.size(); ++i) {
    group_compilation_contexts_.emplace_back(target_, groups[i], scope_);
  }
  auto worker_fn = [&](int index) {
    CompilationTask task(&group_compilation_contexts_[index]);
    task();
    cinn_kernel_info_vecs[index] = task.BuildPirCINNKernelInfo();
  };
  utils::parallel_run(
      worker_fn, utils::SequenceDispatcher(0, groups.size()), -1);
  return cinn_kernel_info_vecs;
}

PirCompiler::CompileResult PirCompiler::StaticBuild(
    const std::vector<pir::GroupPtr>& groups) {
  std::vector<pir::CINNKernelInfo> cinn_kernel_info_vecs(groups.size());
  auto op_lowerer = CreateOpLowerer<pir::GroupPtr>(target_);
  std::vector<std::vector<ir::LoweredFunc>> lowered_funcs;
  for (int i = 0; i < groups.size(); ++i) {
    lowered_funcs.emplace_back(op_lowerer.Lower(groups[i]));
  }

  for (auto&& lowered_func : lowered_funcs) {
    ProcessFunction(lowered_func);
  }
  compiler_ = backends::Compiler::Create(target_);
  auto build_module = m_builder_.Build();
  compiler_->Build(build_module, "");

  auto fn_ptrs = compiler_->GetFnPtr();

  for (int idx = 0; idx < groups.size(); ++idx) {
    pir::CINNKernelInfo cinn_kernel_info;
    auto fn_name = groups[idx]->FuncName();
    auto fn_ptr = compiler_->Lookup(fn_name);
    cinn_kernel_info.fn_ptr = fn_ptr;
    cinn_kernel_info.int_args_map = groups[idx]->int_args_map;

    cinn_kernel_info_vecs[idx] = cinn_kernel_info;
  }
  return cinn_kernel_info_vecs;
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn

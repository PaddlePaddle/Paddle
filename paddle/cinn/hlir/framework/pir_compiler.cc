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

#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/cinn/utils/multi_threading.h"

namespace cinn::hlir::framework {

std::vector<pir::CINNKernelInfo> PirCompiler::Build(
    const std::vector<pir::OpLoweringGroupPtr>& groups) {
  std::vector<pir::CINNKernelInfo> kernel_infos(groups.size());
  for (int i = 0; i < groups.size(); ++i) {
    group_compilation_contexts_.emplace_back(target_, groups[i]);
  }
  auto worker_fn = [&](int index) {
    CompilationTask task(&group_compilation_contexts_[index]);
    task();
    kernel_infos[index] = task.GetCINNKernelInfo();
  };
  utils::parallel_run(
      worker_fn, utils::SequenceDispatcher(0, groups.size()), -1);
  return kernel_infos;
}

}  // namespace cinn::hlir::framework

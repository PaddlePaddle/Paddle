// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/lite/core/mir/static_kernel_pick_pass.h"
#include <algorithm>
#include <utility>
#include <vector>
#include "paddle/fluid/lite/core/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

bool KernelScoreCmp(const std::pair<size_t, std::unique_ptr<KernelBase>>& a,
                    const std::pair<size_t, std::unique_ptr<KernelBase>>& b) {
  return a.first > b.first;
}

void StaticKernelPickPass::Apply(std::unique_ptr<mir::SSAGraph>& graph) {
  CHECK(kernel_pick_factors_.AnyFactorConsidered())
      << "kernel_pick_factors should be specified first";
  CHECK(graph) << "graph not valid";
  // sort kernels by the factors.
  for (auto& node : graph->mutable_nodes()) {
    if (!node.IsInstruct()) continue;
    auto& instruct = node.AsInstruct();
    std::vector<std::pair<size_t, std::unique_ptr<KernelBase>>> scored;
    for (auto&& kernel : instruct.valid_kernels) {
      size_t score = KernelGrade(*kernel);
      LOG(INFO) << "kernel " << kernel->summary() << " " << score;
      scored.emplace_back(score, std::move(kernel));
    }

    std::sort(scored.begin(), scored.end(), KernelScoreCmp);

    // Move kernel back
    // Just keep a single best kernel.
    // TODO(Superjomn) reconsider this.
    instruct.valid_kernels.clear();
    instruct.valid_kernels.emplace_back(std::move(scored.front().second));
    LOG(INFO) << "pick " << instruct.valid_kernels.front()->name();
  }
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(static_kernel_pick_pass,
                  paddle::lite::mir::StaticKernelPickPass);

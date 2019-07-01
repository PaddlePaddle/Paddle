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
#include <memory>
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

void StaticKernelPickPass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  CHECK(kernel_pick_factors_.any_factor_considered())
      << "kernel_pick_factors should be specified first";
  CHECK(graph) << "graph not valid";
  // sort kernels by the factors.

  for (auto& node : graph->mutable_nodes()) {
    if (!node.IsStmt()) continue;
    auto& instruct = node.AsStmt();

    // Get candidate kernels
    std::vector<std::pair<size_t, std::unique_ptr<KernelBase>>> scored;
    CHECK(!instruct.kernels().empty()) << "No kernels found for "
                                       << instruct.op_type();
    for (auto&& kernel : instruct.kernels()) {
      size_t score = KernelGrade(*kernel);
      scored.emplace_back(score, std::move(kernel));
    }
    std::sort(scored.begin(), scored.end(), KernelScoreCmp);
    instruct.kernels().clear();

    if (!instruct.op_info()->HasAttr("enable_int8")) {
      // Move kernel back
      // Just keep a single best kernel.
      // TODO(Superjomn) reconsider this.
      instruct.kernels().emplace_back(std::move(scored.front().second));
      VLOG(2) << "pick " << instruct.kernels().front()->name();

    } else {
      bool out_type_int8 = true;
      // Only if all ops linked to this op output has enable_int8 attr,
      // then the op output type is int8, or fp32.
      for (auto* out_n : node.outlinks) {
        CHECK(out_n->IsArg());
        for (auto* tmp_op : out_n->outlinks) {
          CHECK(tmp_op->IsStmt());
          if (!tmp_op->AsStmt().op_info()->HasAttr("enable_int8")) {
            out_type_int8 = false;
            break;
          }
        }
        if (!out_type_int8) break;
      }

      // According to the out type, we pick the kernel.
      auto output_arguments = instruct.op_info()->OutputArgumentNames();
      for (auto& candidate : scored) {
        bool all_output_type_match = true;
        auto expect_output_type =
            out_type_int8 ? PRECISION(kInt8) : PRECISION(kFloat);

        for (auto& arg_name : output_arguments) {
          const Type* out_arg_ty =
              candidate.second->GetOutputDeclType(arg_name);
          if (out_arg_ty->precision() != expect_output_type) {
            all_output_type_match = false;
          }
        }

        if (all_output_type_match) {
          instruct.kernels().emplace_back(std::move(candidate.second));
          VLOG(2) << "pick " << instruct.kernels().front()->name();
          break;
        }
      }
      CHECK(!instruct.kernels().empty()) << "No kernels found for "
                                         << instruct.op_type();
    }
  }
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(static_kernel_pick_pass,
                  paddle::lite::mir::StaticKernelPickPass);

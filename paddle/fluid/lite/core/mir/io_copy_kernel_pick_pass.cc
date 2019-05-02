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

#include "paddle/fluid/lite/core/mir/pass.h"
#include "paddle/fluid/lite/core/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

class IoCopyKernelPickPass : public StmtPass {
 public:
  void Apply(std::unique_ptr<mir::SSAGraph>& graph) override {
    for (auto& node : graph->mutable_nodes()) {
      if (!node.IsStmt()) continue;
      auto& inst = node.AsStmt();
      if (inst.op_type != "io_copy") continue;

      LOG(INFO) << "....> picking a IO COPY kernel";

      auto& kernels = node.AsStmt().valid_kernels;
      CHECK(!kernels.empty()) << "No valid kernels found for IoCopy Op";
      const auto* inty = node.inlinks.front()->AsArg().type;
      const auto* outy = node.outlinks.front()->AsArg().type;
      LOG(INFO) << "input type " << *inty;
      LOG(INFO) << "output type " << *outy;

      bool is_found = false;
      LOG(INFO) << "kernels size " << kernels.size();
      for (auto& kernel : kernels) {
        CHECK_EQ(node.inlinks.size(), 1UL);
        CHECK_EQ(node.outlinks.size(), 1UL);

        const Type* in_arg_ty = kernel->GetInputDeclType("Input");
        const Type* out_arg_ty = kernel->GetOutputDeclType("Out");
        LOG(INFO) << "checking kernel candidate " << *in_arg_ty << "->"
                  << *out_arg_ty;
        if (inty->target() == in_arg_ty->target()) {
          // Both the input and output type matches, remove other kernels
          // directly.
          if (out_arg_ty->target() == outy->target()) {
            LOG(INFO) << "get a IOCopy kernel";
            auto x = std::move(kernel);
            kernels.clear();
            kernels.emplace_back(std::move(x));
            is_found = true;
            break;
          }
        }
      }

      CHECK(is_found) << "Can't find a IoCopy kernel for IO: " << *inty << "->"
                      << *outy;
    }
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(io_copy_kernel_pick_pass,
                  paddle::lite::mir::IoCopyKernelPickPass);

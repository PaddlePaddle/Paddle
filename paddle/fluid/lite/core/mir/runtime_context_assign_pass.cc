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

class RuntimeContextAssignPass : public InstructionPass {
 public:
  RuntimeContextAssignPass() {
#ifdef LITE_WITH_CUDA
    InitCudaBlas();
#endif
  }

  void Apply(std::unique_ptr<mir::SSAGraph>& graph) override {
    for (auto& node : graph->mutable_nodes()) {
      if (!node.IsInstruct()) continue;

      auto& inst = node.AsInstruct();

      switch (inst.picked_kernel().target()) {
        case TARGET(kHost):
        case TARGET(kX86):
          inst.picked_kernel().SetContext(NewHostContext());
          break;
        case TARGET(kCUDA):
          inst.picked_kernel().SetContext(NewCudaContext());
          break;
        default:
          LOG(FATAL) << "unsupported target "
                     << TargetToStr(inst.picked_kernel().target());
      }
    }
  }

  std::unique_ptr<KernelContext> NewHostContext() {
    std::unique_ptr<KernelContext> ctx(new KernelContext);
    ctx->AsX86Context();
    // Some initialization here.
    return ctx;
  }

  std::unique_ptr<KernelContext> NewCudaContext() {
    std::unique_ptr<KernelContext> ctx(new KernelContext);
    auto& cuda = ctx->AsCudaContext();
    // Some initialization here.
    CHECK(cublas_fp32_) << "cublas_fp32 should be set first";
    cuda.blas_fp32 = cublas_fp32_;
    return ctx;
  }

  void InitCudaBlas() {
    cublas_fp32_ = std::make_shared<lite::cuda::Blas<float>>();
  }

 private:
  std::shared_ptr<lite::cuda::Blas<float>> cublas_fp32_;
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(runtime_context_assign_pass,
                  paddle::lite::mir::RuntimeContextAssignPass);

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

#pragma once

#include <limits>
#include "paddle/fluid/lite/core/mir/pass.h"
#include "paddle/fluid/lite/core/types.h"

namespace paddle {
namespace lite {
namespace mir {

/*
 * StaticKernelPickPass is a simple strategy for picking the kernel for each
 * Operator using operator developer defined rule, there are many other tactics
 * such as considering IO or kernel execution latency and we will implement them
 * latter.
 *
 * There are two argument for this pass:
 * - place, the target place.
 * - kernel_pick_factors, the factors to consider in picking kernels.
 * Set them first before execute the pass.
 */
class StaticKernelPickPass : public mir::InstructionPass {
 public:
  void Apply(std::unique_ptr<mir::SSAGraph>& graph) override;

  void SetPreferPlace(const Place& place) { place_ = place; }
  const Place& place() const { return place_; }
  const core::KernelPickFactor& kernel_pick_factors() const {
    return kernel_pick_factors_;
  }
  core::KernelPickFactor* mutable_kernel_pick_factors() {
    return &kernel_pick_factors_;
  }

 private:
  // Score the kernel.
  size_t KernelGrade(const lite::KernelBase& kernel) {
    size_t score{};
    const int kMax =
        std::numeric_limits<core::KernelPickFactor::value_type>::max();

    // The more important factor comes first
    if (kernel_pick_factors_.IsTargetConsidered() &&
        (place().target == kernel.target() || kernel.target() == TARGET(kAny) ||
         place().target == TARGET(kAny))) {
      score +=
          kMax / static_cast<int>(core::KernelPickFactor::Factor::TargetFirst);
    }
    if (kernel_pick_factors_.IsPrecisionConsidered() &&
        (place().precision == kernel.precision() ||
         kernel.precision() == PRECISION(kAny) ||
         place().precision == PRECISION(kAny))) {
      score += kMax /
               static_cast<int>(core::KernelPickFactor::Factor::PrecisionFirst);
    }
    if (kernel_pick_factors_.IsDataLayoutConsidered() &&
        (place().layout == kernel.layout() ||
         kernel.layout() == DATALAYOUT(kAny) ||
         place().layout == DATALAYOUT(kAny))) {
      score += kMax / static_cast<int>(
                          core::KernelPickFactor::Factor::DataLayoutFirst);
    }
    LOG(INFO) << "picker tactic " << kernel_pick_factors_;
    LOG(INFO) << "kernel place " << kernel.place();
    LOG(INFO) << "picker place " << place();
    LOG(INFO) << "score " << score;

    // The data layout is not considered, for the input and output arguments
    // might have different data layout.
    // TODO(Superjomn) reconsider the idea of taking the data layout as a kernel
    // specification.
    return score;
  }

 private:
  core::KernelPickFactor kernel_pick_factors_;
  Place place_;
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

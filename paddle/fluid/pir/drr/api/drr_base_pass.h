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

#include "paddle/pir/pass/pass.h"
#include "paddle/pir/pattern_rewrite/pattern_rewrite_driver.h"

#pragma once

namespace pir {
namespace drr {

class DrrBasePass : public pir::Pass {
 public:
  DrrBasePass(const std::string &name = "DrrBasePass",
              uint8_t opt_level = 1,
              const std::vector<std::string> &dependents = {})
      : pir::Pass(name, opt_level, dependents) {}

  virtual void DrrPassInitialize(pir::RewritePatternSet *ps,
                                 pir::IrContext *context) {
    return;
  }

  bool Initialize(pir::IrContext *context) final {
    pir::RewritePatternSet ps(context);
    DrrPassInitialize(&ps, context);

    patterns_ = pir::FrozenRewritePatternSet(std::move(ps));
    return true;
  }

  void Run(pir::Operation *op) final {
    pir::GreedyRewriteConfig cfg;
    cfg.use_top_down_traversal = true;
    cfg.max_iterations = 10;
    pir::ApplyPatternsGreedily(op->region(0), patterns_, cfg);
  }

  bool CanApplyOn(pir::Operation *op) const final {
    return op->name() == "builtin.module" && op->num_regions() > 0;
  }

 private:
  pir::FrozenRewritePatternSet patterns_;
};

}  // namespace drr
}  // namespace pir

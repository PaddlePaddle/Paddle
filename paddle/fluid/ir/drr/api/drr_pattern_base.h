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

#pragma once

#include "paddle/fluid/ir/drr/api/drr_pattern_context.h"
#include "paddle/fluid/ir/drr/drr_rewrite_pattern.h"

namespace ir {
namespace drr {

class DrrPatternBase {
 public:
  // Define the Drr Pattern.
  virtual void operator()(ir::drr::DrrPatternContext* ctx) const = 0;

  std::unique_ptr<DrrRewritePattern> Build(
      ir::IrContext* ir_context, ir::PatternBenefit benefit = 1) const {
    DrrPatternContext drr_context;
    this->operator()(&drr_context);
    return std::make_unique<DrrRewritePattern>(
        drr_context, ir_context, benefit);
  }
};

}  // namespace drr
}  // namespace ir

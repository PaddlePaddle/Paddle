// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/ap/include/axpr/atomic_builder.h"
#include "paddle/ap/include/axpr/core_expr.h"

namespace ap::axpr {

class CoreExprBuilder : public AtomicExprBuilder<CoreExpr> {
 public:
  CoreExprBuilder() {}
  CoreExprBuilder(const CoreExprBuilder&) = delete;
  CoreExprBuilder(CoreExprBuilder&&) = delete;

  ap::axpr::ComposedCallAtomic<CoreExpr> ComposedCallAtomic(
      const Atomic<CoreExpr>& outer_func,
      const Atomic<CoreExpr>& inner_func,
      const std::vector<Atomic<CoreExpr>>& args) {
    return ap::axpr::ComposedCallAtomic<CoreExpr>{outer_func, inner_func, args};
  }

 private:
};

}  // namespace ap::axpr

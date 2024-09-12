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

#include "paddle/cinn/optim/merge_block_utils.h"

#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/common/enforce.h"

namespace cinn {
namespace optim {

struct ForInfoAnalyzer {
 public:
  bool IsBlockForAllEqual(const std::vector<ir::For*>& first,
                          const std::vector<ir::For*>& second) {
    auto ForVarExtentEqual = [&](const std::vector<ir::For*>& first,
                                 const std::vector<ir::For*>& second) -> bool {
      for (size_t i = 0; i < first.size(); ++i) {
        const ir::Expr lhs = first[i]->extent;
        const ir::Expr rhs = second[i]->extent;
        if (cinn::common::AutoSimplify(ir::Sub::Make(lhs, rhs)) !=
            ir::Expr(0)) {
          return false;
        }
      }
      return true;
    };

    if (first.size() != second.size()) return false;
    return ForVarExtentEqual(first, second);
  }
};

bool CanMergeBlocks(const std::vector<ir::For*>& first,
                    const std::vector<ir::For*>& second) {
  ForInfoAnalyzer for_info_analyzer;
  return for_info_analyzer.IsBlockForAllEqual(first, second);
}

}  // namespace optim
}  // namespace cinn

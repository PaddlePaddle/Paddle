// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/common/macros.h"
#include "paddle/cinn/ir/dy_schedule/ir_schedule.h"

namespace cinn {
namespace ir {

std::vector<Expr> DyScheduleImpl::Split(const Expr& loop,
                                        const std::vector<int>& factors) {
  CINN_NOT_IMPLEMENTED;
}

Expr DyScheduleImpl::Fuse(const std::vector<Expr>& loops) {
  CINN_NOT_IMPLEMENTED;
}

Expr DyScheduleImpl::Fuse(const std::string& block_name,
                          const std::vector<int>& loops_index) {
  CINN_NOT_IMPLEMENTED;
}

Expr DyScheduleImpl::Fuse(const Expr& block,
                          const std::vector<int>& loops_index) {
  CINN_NOT_IMPLEMENTED;
}

Expr DyScheduleImpl::Reorder(const std::vector<Expr>& loops) {
  CINN_NOT_IMPLEMENTED;
}

Expr DyScheduleImpl::Reorder(const std::string& block_name,
                             const std::vector<int>& loops_index) {
  CINN_NOT_IMPLEMENTED;
}

Expr DyScheduleImpl::Reorder(const Expr& block,
                             const std::vector<int>& loops_index) {
  CINN_NOT_IMPLEMENTED;
}

Expr DyScheduleImpl::AddUnitLoop(const Expr& block) const {
  CINN_NOT_IMPLEMENTED;
}

void DyScheduleImpl::FlattenLoops(const std::vector<Expr>& loops,
                                  const bool force_flat) {
  CINN_NOT_IMPLEMENTED;
}

}  // namespace ir
}  // namespace cinn

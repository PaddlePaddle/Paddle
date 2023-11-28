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

void DyScheduleImpl::MutateForType(const Expr& loop,
                                   ForType for_type,
                                   int factor) {
  CINN_NOT_IMPLEMENTED;
}

void DyScheduleImpl::Parallel(const Expr& loop) { CINN_NOT_IMPLEMENTED; }

void DyScheduleImpl::Vectorize(const Expr& loop, int factor) {
  CINN_NOT_IMPLEMENTED;
}

void DyScheduleImpl::Unroll(const Expr& loop) { CINN_NOT_IMPLEMENTED; }

void DyScheduleImpl::Bind(const Expr& loop, const std::string& thread_axis) {
  CINN_NOT_IMPLEMENTED;
}

}  // namespace ir
}  // namespace cinn

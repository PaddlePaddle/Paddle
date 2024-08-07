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
#include <optional>
#include "paddle/cinn/hlir/framework/pir/op_lowering_group.h"

using OpLoweringGroup = cinn::hlir::framework::pir::OpLoweringGroup;
using OpLoweringGroupPtr = std::shared_ptr<OpLoweringGroup>;

namespace cinn::hlir::framework::pir {
// The optimized information for a broadcast group consists of a list of groups,
// where each group maintains the same structure as the original group but
// differs in shape information. In these groups, broadcast shapes have been
// eliminated, and we utilize broadcast shape conditions to determine which
// group to execute.
std::optional<std::vector<OpLoweringGroupPtr>> GetBroadcastGroupListForOptimize(
    const OpLoweringGroupPtr& group);
}  // namespace cinn::hlir::framework::pir

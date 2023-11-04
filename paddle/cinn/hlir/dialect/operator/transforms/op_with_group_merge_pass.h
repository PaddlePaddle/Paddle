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

#include "paddle/cinn/hlir/dialect/operator/transforms/op_with_group_merge_util.h"
#include "paddle/pir/core/program.h"

namespace cinn {
namespace dialect {
namespace ir {

using GroupPtr = std::shared_ptr<Group>;
using GroupList = std::vector<GroupPtr>;

GroupList OpFusionPassInternal(const std::vector<pir::Operation*>& op_list);

GroupList GeneralFusionMergePassInternal(const GroupList& group_list);

}  // namespace ir
}  // namespace dialect
}  // namespace cinn

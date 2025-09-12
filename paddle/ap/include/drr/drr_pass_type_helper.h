// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/ap/include/adt/adt.h"
#include "paddle/ap/include/drr/drr_pass_type.h"

namespace ap::drr {

struct DrrPassTypeHelper {
  bool SupportReifying(const std::optional<DrrPassType>& type) const {
    if (!type.has_value()) return false;
    return type.value().Match(
        [&](const AbstractDrrPassType&) { return true; },
        [&](const ReifiedDrrPassType&) { return false; },
        [&](const AccessTopoDrrPassType&) { return false; });
  }

  bool SupportOptionalPackedOp(const std::optional<DrrPassType>& type) const {
    if (!type.has_value()) return false;
    return type.value().Match(
        [&](const AbstractDrrPassType&) { return true; },
        [&](const ReifiedDrrPassType&) { return false; },
        [&](const AccessTopoDrrPassType&) { return false; });
  }
};

}  // namespace ap::drr

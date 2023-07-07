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

#include "paddle/cinn/auto_schedule/search_strategy/mutate_rule/mutate_rule.h"

#include "paddle/cinn/auto_schedule/search_strategy/mutate_rule/mutate_tile_size.h"

namespace cinn {
namespace auto_schedule {

std::unique_ptr<MutateRule> MutateRule::Make(const std::string& name) {
  if (name == "mutate_tile_size") {
    return std::make_unique<MutateTileSize>();
  } else {
    LOG(FATAL) << "MutateRule " << name << " is not supported.";
  }
  return nullptr;
}

}  // namespace auto_schedule
}  // namespace cinn

// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/common/graph_utils.h"
#include "paddle/cinn/hlir/framework/graph.h"
#include "paddle/cinn/hlir/framework/pass.h"

namespace cinn {
namespace hlir {
namespace pass {

void InferShape(
    framework::Node* node,
    absl::flat_hash_map<std::string, common::Type>& dtype_dict,  // NOLINT
    absl::flat_hash_map<std::string, framework::shape_t>&
        shape_dict);  // NOLINT

}  // namespace pass
}  // namespace hlir
}  // namespace cinn

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
#include <vector>
#include "paddle/ap/include/adt/adt.h"
#include "paddle/pir/include/core/attribute.h"
#include "paddle/pir/include/core/builder.h"
#include "paddle/pir/include/core/type.h"

namespace pir {

class Operation;

}

namespace ap::paddle {

// Returns nullopt if op_name not supported.
adt::Result<std::optional<pir::Operation*>> CreateOperation(
    pir::Builder* builder,
    const std::string& op_name,
    const std::vector<pir::Value>& inputs,
    const pir::AttributeMap& attributes);

}  // namespace ap::paddle

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

#include <memory>
#include <set>
#include <string>
#include "paddle/pir/include/core/dll_decl.h"

namespace pir {

class Pass;

// ops_in_NHWC: the op that should be in NHWC layout.
IR_API std::unique_ptr<Pass> CreateAutoLayoutInsertPass(
    const std::set<std::string>& ops_in_NHWC);

}  // namespace pir

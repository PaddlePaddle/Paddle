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

#include <memory>
#include <optional>
#include "paddle/pir/include/pass/pass.h"

namespace ap::memory {

class CirclableRefListBase;

}

namespace ap::axpr {

struct Value;

}

namespace ap {
namespace paddle {

std::unique_ptr<::pir::Pass> CreateFallbackFusionOpToPhiPass();

std::unique_ptr<::pir::Pass> CreateFallbackNestedFusionOpToPhiPass();

}  // namespace paddle
}  // namespace ap

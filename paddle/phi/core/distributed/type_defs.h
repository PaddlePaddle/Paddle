//   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <functional>
#include <string>
#include <vector>

#include "paddle/utils/variant.h"

namespace phi {
namespace distributed {
class TensorDistAttr;
class DistMetaTensor;

using ArgDistAttr =
    paddle::variant<TensorDistAttr, std::vector<TensorDistAttr>>;

using SpmdInfo = std::pair<std::vector<ArgDistAttr>, std::vector<ArgDistAttr>>;

}  // namespace distributed
}  // namespace phi

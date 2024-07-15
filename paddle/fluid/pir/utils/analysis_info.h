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

#include <unordered_map>
#include <vector>

#include "paddle/pir/include/core/value.h"

namespace pir {
namespace pass {
// QuantAnalysis is used to transfer quantification scale information between
// PIR Passes
struct QuantAnalysis {
  std::unordered_map<pir::Value, std::vector<float>> scale_map;
};

// Int8Analysis is used to pass information between PIR Passes on whether to
// enable INT8 quantization
struct Int8Analysis {
  bool enable_int8;
};

}  // namespace pass
}  // namespace pir

IR_DECLARE_EXPLICIT_TYPE_ID(pir::pass::QuantAnalysis)
IR_DECLARE_EXPLICIT_TYPE_ID(pir::pass::Int8Analysis)

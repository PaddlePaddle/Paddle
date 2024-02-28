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
#include <optional>
#include "paddle/pir/include/pass/pass.h"

namespace cinn::dialect::ir {

// This is a helper pass for preparing dynamic-shape inputs to cinn backend even
// in static shape GroupOp. Returns std::nullopt if
// FLAGS_cinn_convert_static_dim_to_dynamic_dim not set or invalid.
std::optional<std::unique_ptr<::pir::Pass>>
CreateConvertStaticDimToDynamicPass();

}  // namespace cinn::dialect::ir

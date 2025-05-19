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

namespace ap::memory {

class CirclableRefListBase;

}

namespace ap::axpr {

struct Value;

}

namespace ap::paddle {

std::optional<std::unique_ptr<::pir::Pass>> CreateApGenericAbstractDrrPass(
    const std::weak_ptr<ap::memory::CirclableRefListBase>& circlable_ref_list);
std::optional<std::unique_ptr<::pir::Pass>> CreateApGenericClassicDrrPass(
    const std::weak_ptr<ap::memory::CirclableRefListBase>& circlable_ref_list);

std::optional<std::unique_ptr<::pir::Pass>> CreateAccessTopoDrrPass(
    const std::weak_ptr<ap::memory::CirclableRefListBase>& circlable_ref_list,
    const std::string& drr_pass_tag,
    std::optional<int64_t> steps_limit);

std::optional<std::unique_ptr<::pir::Pass>> CreateCustomAccessTopoDrrPass(
    const std::weak_ptr<ap::memory::CirclableRefListBase>& circlable_ref_list,
    const ap::axpr::Value& drr_pass,
    std::optional<int64_t> steps_limit,
    const ap::axpr::Value& mut_matched_pattern_as_programs);

}  // namespace ap::paddle

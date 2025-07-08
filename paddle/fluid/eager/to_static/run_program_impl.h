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

#include <unordered_set>
#include <vector>
#include "paddle/fluid/eager/eager_tensor.h"

#include "paddle/pir/include/core/block.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/core/value.h"

namespace egr::to_static {
namespace details {
void GcScope(paddle::framework::Scope *scope,
             const std::unordered_set<std::string_view> &persistent_names = {});

}  // namespace details

std::vector<paddle::Tensor> RunProgramImpl(
    const std::vector<paddle::Tensor> &x,
    const std::vector<paddle::Tensor> &params,
    std::vector<paddle::framework::Scope *> &step_scope,  // NOLINT
    bool require_any_grad,
    const paddle::framework::AttributeMap &attrs,
    const int64_t &place_hash_key);
void RunProgramGradImpl(
    const std::vector<paddle::Tensor> &out_grad,
    const std::vector<paddle::framework::Scope *> &step_scope,  // NOLINT
    const paddle::framework::AttributeMap &attrs,
    std::vector<paddle::Tensor> *x_grad,
    std::vector<paddle::Tensor> *params_grad,
    const int64_t &place_hash_key);
void LegacyRunProgramImpl(
    const std::vector<paddle::Tensor> &x,
    const std::vector<paddle::Tensor> &params,
    std::vector<paddle::Tensor *> &out,                   // NOLINT
    std::vector<paddle::framework::Scope *> &step_scope,  // NOLINT
    bool require_any_grad,
    const paddle::framework::AttributeMap &attrs,
    const int64_t &place_hash_key);
void LegacyRunProgramGradImpl(
    const std::vector<paddle::Tensor> &out_grad,
    const std::vector<paddle::framework::Scope *> &step_scope,  // NOLINT
    const paddle::framework::AttributeMap &attrs,
    std::vector<paddle::Tensor *> &x_grad,       // NOLINT
    std::vector<paddle::Tensor *> &params_grad,  // NOLINT
    const int64_t &place_hash_key);
}  // namespace egr::to_static

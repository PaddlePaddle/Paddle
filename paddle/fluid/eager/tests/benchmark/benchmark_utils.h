// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/eager/eager_tensor.h"
#include "paddle/fluid/imperative/layer.h"
#include "paddle/pten/api/all.h"
#include "paddle/pten/include/core.h"
namespace egr {

/* ---- Eager Scale ---- */
void benchmark_eager_scale_accuracy_check(const egr::EagerTensor& tensor);
void benchmark_eager_scale(const egr::EagerTensor& tensor);

/* ---- Eager MatMul ---- */
void benchmark_eager_intermediate_matmul_accuracy_check(
    const egr::EagerTensor& X, const egr::EagerTensor& Y);
void benchmark_eager_intermediate_matmul(const egr::EagerTensor& X,
                                         const egr::EagerTensor& Y);

}  // namespace egr

namespace paddle {
namespace imperative {
/* ---- Fluid Scale ---- */
// TODO(jiabin): Change this and remove nolint
void benchmark_fluid_scale_accuracy_check(
    const std::shared_ptr<imperative::VarBase>& X,    // NOLINT
    const std::shared_ptr<imperative::VarBase>& Out,  // NOLINT
    const paddle::platform::Place& place);

// TODO(jiabin): Change this and remove nolint
void benchmark_fluid_scale(
    const std::shared_ptr<imperative::VarBase>& X,    // NOLINT
    const std::shared_ptr<imperative::VarBase>& Out,  // NOLINT
    const paddle::platform::Place& place);

/* ---- Fluid MatMul ---- */
void benchmark_fluid_matmul_accuracy_check(
    const std::shared_ptr<imperative::VarBase>& X,    // NOLINT
    const std::shared_ptr<imperative::VarBase>& Y,    // NOLINT
    const std::shared_ptr<imperative::VarBase>& Out,  // NOLINT
    const paddle::platform::Place& place);

void benchmark_fluid_matmul(
    const std::shared_ptr<imperative::VarBase>& X,
    const std::shared_ptr<imperative::VarBase>& Y,    // NOLINT
    const std::shared_ptr<imperative::VarBase>& Out,  // NOLINT
    const paddle::platform::Place& place);

}  // namespace imperative
}  // namespace paddle

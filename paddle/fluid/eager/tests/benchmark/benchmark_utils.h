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

#include "paddle/fluid/imperative/layer.h"
#include "paddle/top/api/all.h"

namespace egr {

void benchmark_eager_accuracy_check(const pt::Tensor& tensor);
void benchmark_eager(const pt::Tensor& tensor);

}  // namespace egr

namespace paddle {
namespace imperative {
// TODO(jiabin): Change this and remove nolint
void benchmark_fluid_accuracy_check(
    std::shared_ptr<imperative::VarBase>& X,    // NOLINT
    std::shared_ptr<imperative::VarBase>& Out,  // NOLINT
    const paddle::platform::Place& place);
// TODO(jiabin): Change this and remove nolint
void benchmark_fluid(std::shared_ptr<imperative::VarBase>& X,    // NOLINT
                     std::shared_ptr<imperative::VarBase>& Out,  // NOLINT
                     const paddle::platform::Place& place);

}  // namespace imperative
}  // namespace paddle

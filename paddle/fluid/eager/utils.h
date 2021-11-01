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
#include "paddle/fluid/eager/function_api.h"
#include "paddle/pten/hapi/all.h"
namespace egr {
std::vector<std::shared_ptr<egr::EagerTensor>> SyncToVars(
    const egr::EagerTensor& tensor);
std::vector<std::shared_ptr<egr::EagerTensor>> SyncToVars(
    const std::vector<egr::EagerTensor>& tensors);
std::vector<std::shared_ptr<egr::EagerTensor>> SyncToTensors(
    const egr::EagerTensor& tensor);
std::vector<std::shared_ptr<egr::EagerTensor>> SyncToTensors(
    const std::vector<egr::EagerTensor>& tensors);
std::vector<std::shared_ptr<EagerTensor>> ConstructDuplicableOutput(
    const size_t num);
std::vector<egr::EagerTensor> GetOutputs(
    const std::vector<std::shared_ptr<EagerTensor>>& outs);
egr::EagerTensor GetOutput(const std::shared_ptr<EagerTensor>& outs);
}  // namespace egr

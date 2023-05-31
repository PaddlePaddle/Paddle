// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include <absl/container/flat_hash_map.h>

#include <string>
#include <vector>

namespace cinn {
namespace hlir {
namespace pe {

void InputX86Param(
    absl::flat_hash_map<std::string,
                        absl::flat_hash_map<std::string, std::vector<int>>>
        *model_data,
    const std::string &key,
    const absl::flat_hash_map<std::string, std::vector<int>> &schedule_data);

absl::flat_hash_map<std::string,
                    absl::flat_hash_map<std::string, std::vector<int>>>
CreateX86Params();
void LoadResNet18Params(
    absl::flat_hash_map<std::string,
                        absl::flat_hash_map<std::string, std::vector<int>>>
        *model_data);
void LoadResNet50Params(
    absl::flat_hash_map<std::string,
                        absl::flat_hash_map<std::string, std::vector<int>>>
        *model_data);
void LoadMobileNetV1Params(
    absl::flat_hash_map<std::string,
                        absl::flat_hash_map<std::string, std::vector<int>>>
        *model_data);
void LoadMobileNetV2Params(
    absl::flat_hash_map<std::string,
                        absl::flat_hash_map<std::string, std::vector<int>>>
        *model_data);
void LoadFaceDetParams(
    absl::flat_hash_map<std::string,
                        absl::flat_hash_map<std::string, std::vector<int>>>
        *model_data);
void LoadEfficientNetParams(
    absl::flat_hash_map<std::string,
                        absl::flat_hash_map<std::string, std::vector<int>>>
        *model_data);
void LoadSqueezeNetParams(
    absl::flat_hash_map<std::string,
                        absl::flat_hash_map<std::string, std::vector<int>>>
        *model_data);

void CreateX86Params(
    absl::flat_hash_map<std::string,
                        absl::flat_hash_map<std::string, std::vector<int>>>
        *model_data);

}  // namespace pe
}  // namespace hlir
}  // namespace cinn

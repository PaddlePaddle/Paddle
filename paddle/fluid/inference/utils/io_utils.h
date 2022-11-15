// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>
#include <vector>

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/inference/api/paddle_api.h"
#include "paddle/fluid/inference/utils/shape_range_info.pb.h"

namespace paddle {
struct PaddleTensor;
}  // namespace paddle

namespace paddle {
namespace inference {

constexpr uint32_t kCurPDTensorVersion = 0;

void SerializePDTensorToStream(std::ostream* os, const PaddleTensor& tensor);
void DeserializePDTensorToStream(std::istream& is, PaddleTensor* tensor);

void SerializePDTensorsToStream(std::ostream* os,
                                const std::vector<PaddleTensor>& tensors);
void DeserializePDTensorsToStream(std::istream& is,
                                  std::vector<PaddleTensor>* tensors);

void SerializePDTensorsToFile(const std::string& path,
                              const std::vector<PaddleTensor>& tensors);
void DeserializePDTensorsToFile(const std::string& path,
                                std::vector<PaddleTensor>* tensors);
void SerializeShapeRangeInfo(
    const std::string& path,
    const std::map<std::string, std::vector<int32_t>>& min_shape,
    const std::map<std::string, std::vector<int32_t>>& max_shape,
    const std::map<std::string, std::vector<int32_t>>& opt_shape,
    const std::map<std::string, std::vector<int32_t>>& min_value,
    const std::map<std::string, std::vector<int32_t>>& max_value,
    const std::map<std::string, std::vector<int32_t>>& opt_value);
void DeserializeShapeRangeInfo(
    const std::string& path,
    std::map<std::string, std::vector<int32_t>>* min_shape,
    std::map<std::string, std::vector<int32_t>>* max_shape,
    std::map<std::string, std::vector<int32_t>>* opt_shape,
    std::map<std::string, std::vector<int32_t>>* min_value,
    std::map<std::string, std::vector<int32_t>>* max_value,
    std::map<std::string, std::vector<int32_t>>* opt_value);
void UpdateShapeRangeInfo(
    const std::string& path,
    const std::map<std::string, std::vector<int32_t>>& min_shape,
    const std::map<std::string, std::vector<int32_t>>& max_shape,
    const std::map<std::string, std::vector<int32_t>>& opt_shape,
    const std::vector<std::string>& names);
}  // namespace inference
}  // namespace paddle

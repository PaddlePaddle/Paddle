// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
#include <unordered_map>
#include <vector>

#include <NvInfer.h>

namespace infrt {
namespace backends {
namespace tensorrt {

// Build default params
constexpr int32_t max_batch_not_provided{0};
constexpr int32_t default_workspace{16};
// Inference default params
constexpr int32_t default_batch{1};
constexpr int32_t batch_not_provided{0};

enum class PrecisionConstraints { kNONE, kOBEY, kPREFER };

enum class SparsityFlag { kDISABLE, kENABLE, kFORCE };

using ShapeRange =
    std::array<std::vector<int32_t>,
               nvinfer1::EnumMax<nvinfer1::OptProfileSelector>()>;

using IOFormat = std::pair<nvinfer1::DataType, nvinfer1::TensorFormats>;

struct BuildOptions {
  // Set max batch size.
  int32_t max_batch{max_batch_not_provided};

  // Set workspace size in megabytes (default = 16)
  int32_t workspace{default_workspace};

  // Enable tf32 precision, in addition to fp32 (default = disabled)
  bool tf32{false};

  // Enable fp16 precision, in addition to fp32 (default = disabled)
  bool fp16{false};

  // Enable int8 precision, in addition to fp32 (default = disabled)
  bool int8{false};

  // Control precision constraints. (default = none)
  // Precision Constaints: = none, obey, prefer
  //     none = no constraints
  //     prefer = meet precision constraints if possible
  //     obey = meet precision constraints or fail otherwise
  PrecisionConstraints precision_constraints{PrecisionConstraints::kNONE};

  // Save the serialized engine.
  bool save{false};

  // Load a serialized engine.
  bool load{false};

  // Build with dynamic shapes using a profile with the min, max and opt shapes
  // provided
  std::unordered_map<std::string, ShapeRange> shapes;

  // Type and format of each of the input tensors (default = all inputs in
  // fp32:chw)
  std::vector<IOFormat> input_formats;

  // Type and format of each of the output tensors (default = all outputs in
  // fp32:chw)
  std::vector<IOFormat> output_formats;
};

struct InferenceOptions {
  int32_t batch{batch_not_provided};
  std::unordered_map<std::string, std::vector<int32_t>> shapes;
};

}  // namespace tensorrt
}  // namespace backends
}  // namespace infrt

// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/inference/tensorrt/engine.h"

namespace paddle {
namespace inference {
namespace tensorrt {

inline nvinfer1::PluginFieldType GetPluginFieldType(nvinfer1::DataType type) {
  switch (type) {
    case nvinfer1::DataType::kBOOL:
      return nvinfer1::PluginFieldType::kCHAR;
    case nvinfer1::DataType::kFLOAT:
      return nvinfer1::PluginFieldType::kFLOAT32;
    case nvinfer1::DataType::kHALF:
      return nvinfer1::PluginFieldType::kFLOAT16;
    case nvinfer1::DataType::kINT32:
      return nvinfer1::PluginFieldType::kINT32;
    case nvinfer1::DataType::kINT8:
      return nvinfer1::PluginFieldType::kINT8;
    default:
      return nvinfer1::PluginFieldType::kUNKNOWN;
  }
}

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

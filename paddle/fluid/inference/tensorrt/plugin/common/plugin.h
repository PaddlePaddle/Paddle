// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
// SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION &
// AFFILIATES. All rights reserved.
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

#ifndef PADDLE_FLUID_INFERENCE_TENSORRT_PLUGIN_COMMON_PLUGIN_H_
#define PADDLE_FLUID_INFERENCE_TENSORRT_PLUGIN_COMMON_PLUGIN_H_
#include <cuda_runtime.h>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include "NvInfer.h"
#include "NvInferPlugin.h"

typedef enum {
  STATUS_SUCCESS = 0,
  STATUS_FAILURE = 1,
  STATUS_BAD_PARAM = 2,
  STATUS_NOT_SUPPORTED = 3,
  STATUS_NOT_INITIALIZED = 4
} pluginStatus_t;

namespace nvinfer1 {

class BasePlugin : public IPluginV2 {
 protected:
  void setPluginNamespace(const char* libNamespace) noexcept override {
    mNamespace = libNamespace;
  }
  const char* getPluginNamespace() const noexcept override {
    return mNamespace.c_str();
  }
  std::string mNamespace;
};

class BaseCreator : public IPluginCreator {
 public:
  void setPluginNamespace(const char* libNamespace) noexcept override {
    mNamespace = libNamespace;
  }
  const char* getPluginNamespace() const noexcept override {
    return mNamespace.c_str();
  }

 protected:
  std::string mNamespace;
};

}  // namespace nvinfer1
#endif  // PADDLE_FLUID_INFERENCE_TENSORRT_PLUGIN_COMMON_PLUGIN_H_

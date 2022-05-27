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

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvInferRuntimeCommon.h>
#include <glog/logging.h>

#include <algorithm>
#include <cassert>
#include <functional>
#include <memory>
#include <unordered_map>

#include "paddle/phi/core/dense_tensor.h"

namespace infrt {
namespace backends {
namespace tensorrt {

#define IS_TRT_VERSION_GE(version)                       \
  ((NV_TENSORRT_MAJOR * 1000 + NV_TENSORRT_MINOR * 100 + \
    NV_TENSORRT_PATCH * 10 + NV_TENSORRT_BUILD) >= version)

#define IS_TRT_VERSION_LT(version)                       \
  ((NV_TENSORRT_MAJOR * 1000 + NV_TENSORRT_MINOR * 100 + \
    NV_TENSORRT_PATCH * 10 + NV_TENSORRT_BUILD) < version)

#define TRT_VERSION                                    \
  NV_TENSORRT_MAJOR * 1000 + NV_TENSORRT_MINOR * 100 + \
      NV_TENSORRT_PATCH * 10 + NV_TENSORRT_BUILD

inline nvinfer1::Dims VecToDims(const std::vector<int>& vec) {
  int limit = static_cast<int>(nvinfer1::Dims::MAX_DIMS);
  if (static_cast<int>(vec.size()) > limit) {
    assert(false);
  }
  // Pick first nvinfer1::Dims::MAX_DIMS elements
  nvinfer1::Dims dims;
  dims.nbDims = std::min(static_cast<int>(vec.size()), limit);
  std::copy_n(vec.begin(), dims.nbDims, std::begin(dims.d));
  return dims;
}

template <typename T>
struct TrtDestroyer {
  void operator()(T* t) { t->destroy(); }
};

template <typename T>
using TrtUniquePtr = std::unique_ptr<T, TrtDestroyer<T>>;

class TrtLogger : public nvinfer1::ILogger {
 public:
  void log(nvinfer1::ILogger::Severity severity,
           const char* msg) noexcept override {
    switch (severity) {
      case Severity::kVERBOSE:
        VLOG(3) << msg;
        break;
      case Severity::kINFO:
        VLOG(2) << msg;
        break;
      case Severity::kWARNING:
        LOG(WARNING) << msg;
        break;
      case Severity::kINTERNAL_ERROR:
      case Severity::kERROR:
        LOG(ERROR) << msg;
        break;
      default:
        break;
    }
  }
  nvinfer1::ILogger& GetTrtLogger() noexcept { return *this; }
  ~TrtLogger() override = default;
};

struct Binding {
  bool is_input{false};
  nvinfer1::DataType data_type{nvinfer1::DataType::kFLOAT};
  ::phi::DenseTensor* buffer{nullptr};
  std::string name;
};

class Bindings {
 public:
  Bindings() = default;

  void AddBinding(int32_t b,
                  const std::string& name,
                  bool is_input,
                  ::phi::DenseTensor* buffer,
                  nvinfer1::DataType data_type) {
    while (bindings_.size() <= static_cast<size_t>(b)) {
      bindings_.emplace_back();
    }
    names_[name] = b;
    bindings_[b].buffer = buffer;
    bindings_[b].is_input = is_input;
    bindings_[b].data_type = data_type;
    bindings_[b].name = name;
  }

  std::vector<Binding> GetInputBindings() {
    return GetBindings([](const Binding& b) -> bool { return b.is_input; });
  }

  std::vector<Binding> GetOutputBindings() {
    return GetBindings([](const Binding& b) -> bool { return !b.is_input; });
  }

  std::vector<Binding> GetBindings() {
    return GetBindings([](const Binding& b) -> bool { return true; });
  }

  std::vector<Binding> GetBindings(
      std::function<bool(const Binding& b)> predicate) {
    std::vector<Binding> bindings;
    for (const auto& b : bindings_) {
      if (predicate(b)) {
        bindings.push_back(b);
      }
    }
    return bindings;
  }

 private:
  std::unordered_map<std::string, int32_t> names_;
  std::vector<Binding> bindings_;
};

}  // namespace tensorrt
}  // namespace backends
}  // namespace infrt

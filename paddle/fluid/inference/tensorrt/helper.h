/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#pragma once

#include <NvInfer.h>
#include <cuda.h>
#include <glog/logging.h>

#include <string>
#include <utility>
#include <vector>

#include "paddle/fluid/platform/dynload/tensorrt.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/common/data_type.h"

namespace paddle {
namespace inference {
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

#if IS_TRT_VERSION_GE(8000)
#define TRT_NOEXCEPT noexcept
#else
#define TRT_NOEXCEPT
#endif

namespace dy = paddle::platform::dynload;

// TensorRT data type to size
const int kDataTypeSize[] = {
    4,  // kFLOAT
    2,  // kHALF
    1,  // kINT8
    4   // kINT32
};

// The following two API are implemented in TensorRT's header file, cannot load
// from the dynamic library. So create our own implementation and directly
// trigger the method from the dynamic library.
static nvinfer1::IBuilder* createInferBuilder(nvinfer1::ILogger* logger) {
  return static_cast<nvinfer1::IBuilder*>(
      dy::createInferBuilder_INTERNAL(logger, NV_TENSORRT_VERSION));
}
static nvinfer1::IRuntime* createInferRuntime(nvinfer1::ILogger* logger) {
  return static_cast<nvinfer1::IRuntime*>(
      dy::createInferRuntime_INTERNAL(logger, NV_TENSORRT_VERSION));
}
#if IS_TRT_VERSION_GE(6000)
static nvinfer1::IPluginRegistry* GetPluginRegistry() {
  return static_cast<nvinfer1::IPluginRegistry*>(dy::getPluginRegistry());
}
static int GetInferLibVersion() {
  return static_cast<int>(dy::getInferLibVersion());
}
#else
static int GetInferLibVersion() { return 0; }
#endif

static std::tuple<int, int, int> GetTrtRuntimeVersion() {
  int ver = GetInferLibVersion();
  int major = ver / 1000;
  ver -= major * 1000;
  int minor = ver / 100;
  int patch = ver - minor * 100;
  return std::tuple<int, int, int>{major, minor, patch};
}

static std::tuple<int, int, int> GetTrtCompileVersion() {
  return std::tuple<int, int, int>{
      NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH};
}

// A logger for create TensorRT infer builder.
class NaiveLogger : public nvinfer1::ILogger {
 public:
  void log(nvinfer1::ILogger::Severity severity,
           const char* msg) TRT_NOEXCEPT override {
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

  static nvinfer1::ILogger& Global() {
    static nvinfer1::ILogger* x = new NaiveLogger;
    return *x;
  }

  ~NaiveLogger() override {}
};

class NaiveProfiler : public nvinfer1::IProfiler {
 public:
  typedef std::pair<std::string, float> Record;
  std::vector<Record> mProfile;

  virtual void reportLayerTime(const char* layerName, float ms) TRT_NOEXCEPT {
    auto record =
        std::find_if(mProfile.begin(), mProfile.end(), [&](const Record& r) {
          return r.first == layerName;
        });
    if (record == mProfile.end())
      mProfile.push_back(std::make_pair(layerName, ms));
    else
      record->second += ms;
  }

  void printLayerTimes() {
    float totalTime = 0;
    for (size_t i = 0; i < mProfile.size(); i++) {
      printf(
          "%-40.40s %4.3fms\n", mProfile[i].first.c_str(), mProfile[i].second);
      totalTime += mProfile[i].second;
    }
    printf("Time over all layers: %4.3f\n", totalTime);
  }
};

inline size_t ProductDim(const nvinfer1::Dims& dims) {
  size_t v = 1;
  for (int i = 0; i < dims.nbDims; i++) {
    v *= dims.d[i];
  }
  return v;
}

inline void PrintITensorShape(nvinfer1::ITensor* X) {
  auto dims = X->getDimensions();
  auto name = X->getName();
  std::cout << "ITensor " << name << " shape: [";
  for (int i = 0; i < dims.nbDims; i++) {
    if (i == dims.nbDims - 1)
      std::cout << dims.d[i];
    else
      std::cout << dims.d[i] << ", ";
  }
  std::cout << "]\n";
}

template <typename T>
inline std::string Vec2Str(const std::vector<T>& vec) {
  std::ostringstream os;
  os << "(";
  for (size_t i = 0; i < vec.size() - 1; ++i) {
    os << vec[i] << ",";
  }
  os << vec[vec.size() - 1] << ")";
  return os.str();
}

static inline nvinfer1::DataType PhiType2NvType(phi::DataType type) {
  nvinfer1::DataType nv_type = nvinfer1::DataType::kFLOAT;
  switch (type) {
    case phi::DataType::FLOAT32:
      nv_type = nvinfer1::DataType::kFLOAT;
      break;
    case phi::DataType::FLOAT16:
      nv_type = nvinfer1::DataType::kHALF;
      break;
    case phi::DataType::INT32:
    case phi::DataType::INT64:
      nv_type = nvinfer1::DataType::kINT32;
      break;
    case phi::DataType::INT8:
      nv_type = nvinfer1::DataType::kINT8;
      break;
#if IS_TRT_VERSION_GE(7000)
    case phi::DataType::BOOL:
      nv_type = nvinfer1::DataType::kBOOL;
      break;
#endif
    default:
      paddle::platform::errors::InvalidArgument(
          "phi::DataType not supported data type %s.", type);
      break;
  }
  return nv_type;
}

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

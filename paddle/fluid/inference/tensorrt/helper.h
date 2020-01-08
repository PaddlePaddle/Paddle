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

namespace paddle {
namespace inference {
namespace tensorrt {

using FluidDT = framework::proto::VarType_Type;
using TRT_DT = nvinfer1::DataType;

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

namespace {  // NOLINT

TRT_DT FluidDataType2TRT(FluidDT type) {
  switch (type) {
    case FluidDT::VarType_Type_FP32:
      return TRT_DT::kFLOAT;
    case FluidDT::VarType_Type_INT32:
      return TRT_DT::kINT32;
    default:
      return TRT_DT::kINT32;
  }
  PADDLE_THROW(platform::errors::InvalidArgument(
      "unknown fluid datatype in TRT op converter"));
  return TRT_DT::kINT32;
}

// The T can be int32 or int64 type.
template <typename T>
nvinfer1::Dims Vec2TRT_Dims(const std::vector<T>& shape, std::string input,
                            bool with_dynamic_shape = false) {
  PADDLE_ENFORCE_GT(shape.size(), 1UL,
                    platform::errors::InvalidArgument(
                        "TensorRT's tensor input requires at least 2 "
                        "dimensions, but input %s has %d dims.",
                        input, shape.size()));
  PADDLE_ENFORCE_LE(shape.size(), 4UL,
                    platform::errors::InvalidArgument(
                        "TensorRT's tensor input requires at most 4 "
                        "dimensions, but input %s has %d dims.",
                        input, shape.size()));
  if (!with_dynamic_shape) {
    if (shape.size() == 4UL) {
      return nvinfer1::DimsCHW(shape[1], shape[2], shape[3]);
    } else if (shape.size() == 3UL) {
      return nvinfer1::Dims2(shape[1], shape[2]);
    }
    return nvinfer1::DimsCHW(shape[1], 1, 1);
  } else {
    if (shape.size() == 4UL) {
      return nvinfer1::DimsNCHW(shape[0], shape[1], shape[2], shape[3]);
    } else if (shape.size() == 3UL) {
      return nvinfer1::Dims3(shape[0], shape[1], shape[2]);
    }
    return nvinfer1::Dims4(shape[0], shape[1], 1, 1);
  }
}
}  // NOLINT

// A logger for create TensorRT infer builder.
class NaiveLogger : public nvinfer1::ILogger {
 public:
  void log(nvinfer1::ILogger::Severity severity, const char* msg) override {
    switch (severity) {
      case Severity::kINFO:
        VLOG(3) << msg;
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

  virtual void reportLayerTime(const char* layerName, float ms) {
    auto record =
        std::find_if(mProfile.begin(), mProfile.end(),
                     [&](const Record& r) { return r.first == layerName; });
    if (record == mProfile.end())
      mProfile.push_back(std::make_pair(layerName, ms));
    else
      record->second += ms;
  }

  void printLayerTimes() {
    float totalTime = 0;
    for (size_t i = 0; i < mProfile.size(); i++) {
      printf("%-40.40s %4.3fms\n", mProfile[i].first.c_str(),
             mProfile[i].second);
      totalTime += mProfile[i].second;
    }
    printf("Time over all layers: %4.3f\n", totalTime);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

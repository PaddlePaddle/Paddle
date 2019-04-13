// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include <iostream>

namespace paddle {
namespace lite {

enum class TargetType : int { kHost = 0, kX86, kCUDA, kLastAsPlaceHolder };
enum class PrecisionType : int { kFloat = 0, kInt8, kLastAsPlaceHolder };
enum class DataLayoutType : int { kNCHW = 0, kLastAsPlaceHolder };

// Some helper macro to get a specific TargetType.
#define TARGET(item__) paddle::lite::TargetType::item__
#define TARGET_VAL(item__) static_cast<int>(TARGET(item__))
// Some helper macro to get a specific PrecisionType.
#define PRECISION(item__) paddle::lite::PrecisionType::item__
#define PRECISION_VAL(item__) static_cast<int>(PRECISION(item__))
#define DATALAYOUT(item__) paddle::lite::DataLayoutType::item__

/*
 * Place specifies the execution context of a Kernel or input/output for a
 * kernel. It is used to make the analysis of the MIR more clear and accurate.
 */
struct Place {
  TargetType target{TARGET(kHost)};
  PrecisionType precision{PRECISION(kFloat)};
  DataLayoutType layout{DATALAYOUT(kNCHW)};

  Place() = default;
  Place(TargetType target, PrecisionType precision,
        DataLayoutType layout = DATALAYOUT(kNCHW))
      : target(target), precision(precision), layout(layout) {}
};

constexpr const int kNumPrecisions =
    PRECISION_VAL(kLastAsPlaceHolder) - PRECISION_VAL(kFloat);
constexpr const int kNumTargets =
    TARGET_VAL(kLastAsPlaceHolder) - TARGET_VAL(kHost);

static const std::string target2string[] = {"host", "x86", "cuda"};
static const std::string& TargetToStr(TargetType target) {
  return target2string[static_cast<int>(target)];
}

static const std::string precision2string[] = {"float", "int8"};
static const std::string& PrecisionToStr(PrecisionType precision) {
  return precision2string[static_cast<int>(precision)];
}

static const std::string datalayout2string[] = {"NCHW"};
static const std::string& DataLayoutToStr(DataLayoutType x) {
  return datalayout2string[static_cast<int>(x)];
}

// Event sync for multi-stream devices like CUDA and OpenCL.
// For the devices without support of stream, leave it empty.
template <TargetType Target>
class Event {};

// Memory copy directions.
enum class IoDirection {
  HtoH = 0,  // Host to host
  HtoD,      // Host to device
  DtoH,      // Device to host
};

// This interface should be specified by each kind of target.
template <TargetType Target>
class TargetWrapper {
 public:
  using stream_t = int;
  using event_t = Event<Target>;

  static size_t num_devices() { return 0; }
  static size_t maximum_stream() { return 0; }

  static void CreateStream(stream_t* stream) {}
  static void DestroyStream(const stream_t& stream) {}

  static void CreateEvent(event_t* event) {}
  static void DestroyEvent(const event_t& event) {}

  static void RecordEvent(const event_t& event) {}
  static void SyncEvent(const event_t& event) {}

  static void StreamSync(const stream_t& stream) {}

  static void* Malloc(size_t size) { return new char[size]; }
  static void Free(void* ptr) { delete[] static_cast<char*>(ptr); }

  static void MemcpySync(void* dst, void* src, size_t size, IoDirection dir) {}
  static void MemcpyAsync(void* dst, void* src, size_t size,
                          const stream_t& stream, IoDirection dir) {
    MemcpySync(dst, src, size, dir);
  }
};

}  // namespace lite
}  // namespace paddle

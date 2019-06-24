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
#include <sstream>
#include <string>
#include "paddle/fluid/lite/utils/cp_logging.h"
#ifdef LITE_WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

namespace paddle {
namespace lite {

enum class TargetType : int {
  kUnk = 0,
  kHost,
  kX86,
  kCUDA,
  kARM,
  kOpenCL,
  kAny,  // any target
  NUM,   // number of fields.
};
enum class PrecisionType : int {
  kUnk = 0,
  kFloat,
  kInt8,
  kInt32,
  kAny,  // any precision
  NUM,   // number of fields.
};
enum class DataLayoutType : int {
  kUnk = 0,
  kNCHW,
  kAny,  // any data layout
  NUM,   // number of fields.
};

static size_t PrecisionTypeLength(PrecisionType type) {
  switch (type) {
    case PrecisionType::kFloat:
      return 4;
    case PrecisionType::kInt8:
      return 1;
    case PrecisionType::kInt32:
      return 4;
    default:
      return 4;
  }
}

// Some helper macro to get a specific TargetType.
#define TARGET(item__) paddle::lite::TargetType::item__
// Some helper macro to get a specific PrecisionType.
#define PRECISION(item__) paddle::lite::PrecisionType::item__
#define DATALAYOUT(item__) paddle::lite::DataLayoutType::item__

static const std::string& TargetToStr(TargetType target) {
  static const std::string target2string[] = {"unk", "host",   "x86", "cuda",
                                              "arm", "opencl", "any"};
  auto x = static_cast<int>(target);
  CHECK_LT(x, static_cast<int>(TARGET(NUM)));
  return target2string[x];
}

static const std::string& PrecisionToStr(PrecisionType precision) {
  static const std::string precision2string[] = {"unk", "float", "int8_t",
                                                 "any"};
  auto x = static_cast<int>(precision);
  CHECK_LT(x, static_cast<int>(PRECISION(NUM)));
  return precision2string[x];
}

static const std::string& DataLayoutToStr(DataLayoutType layout) {
  static const std::string datalayout2string[] = {"unk", "NCHW", "any"};
  auto x = static_cast<int>(layout);
  CHECK_LT(x, static_cast<int>(DATALAYOUT(NUM)));
  return datalayout2string[x];
}

static const std::string& TargetRepr(TargetType target) {
  static const std::string target2string[] = {
      "kUnk", "kHost", "kX86", "kCUDA", "kARM", "kOpenCL", "kAny"};
  auto x = static_cast<int>(target);
  CHECK_LT(x, static_cast<int>(TARGET(NUM)));
  return target2string[x];
}

static const std::string& PrecisionRepr(PrecisionType precision) {
  static const std::string precision2string[] = {"kUnk", "kFloat", "kInt8",
                                                 "kInt32", "kAny"};
  auto x = static_cast<int>(precision);
  CHECK_LT(x, static_cast<int>(PRECISION(NUM)));
  return precision2string[x];
}

static const std::string& DataLayoutRepr(DataLayoutType layout) {
  static const std::string datalayout2string[] = {"kUnk", "kNCHW", "kAny"};
  auto x = static_cast<int>(layout);
  CHECK_LT(x, static_cast<int>(DATALAYOUT(NUM)));
  return datalayout2string[x];
}

/*
 * Place specifies the execution context of a Kernel or input/output for a
 * kernel. It is used to make the analysis of the MIR more clear and accurate.
 */
struct Place {
  TargetType target{TARGET(kUnk)};
  PrecisionType precision{PRECISION(kUnk)};
  DataLayoutType layout{DATALAYOUT(kUnk)};
  int16_t device{0};  // device ID

  Place() = default;
  Place(TargetType target, PrecisionType precision,
        DataLayoutType layout = DATALAYOUT(kNCHW), int16_t device = 0)
      : target(target), precision(precision), layout(layout), device(device) {}

  bool is_valid() const {
    return target != TARGET(kUnk) && precision != PRECISION(kUnk) &&
           layout != DATALAYOUT(kUnk);
  }

  size_t hash() const;

  bool operator==(const Place& other) const {
    return target == other.target && precision == other.precision &&
           layout == other.layout && device == other.device;
  }

  bool operator!=(const Place& other) const { return !(*this == other); }

  friend bool operator<(const Place& a, const Place& b);

  friend std::ostream& operator<<(std::ostream& os, const Place& other) {
    os << other.DebugString();
    return os;
  }

  std::string DebugString() const;
};

// Memory copy directions.
enum class IoDirection {
  HtoH = 0,  // Host to host
  HtoD,      // Host to device
  DtoH,      // Device to host
  DtoD,      // Device to device
};

// This interface should be specified by each kind of target.
template <TargetType Target, typename StreamTy = int, typename EventTy = int>
class TargetWrapper {
 public:
  using stream_t = StreamTy;
  using event_t = EventTy;

  static size_t num_devices() { return 0; }
  static size_t maximum_stream() { return 0; }

  static void CreateStream(stream_t* stream) {}
  static void DestroyStream(const stream_t& stream) {}

  static void CreateEvent(event_t* event) {}
  static void DestroyEvent(const event_t& event) {}

  static void RecordEvent(const event_t& event) {}
  static void SyncEvent(const event_t& event) {}

  static void StreamSync(const stream_t& stream) {}

  static void* Malloc(size_t size) {
    LOG(FATAL) << "Unimplemented malloc for " << TargetToStr(Target);
    return nullptr;
  }
  static void Free(void* ptr) { LOG(FATAL) << "Unimplemented"; }

  static void MemcpySync(void* dst, const void* src, size_t size,
                         IoDirection dir) {
    LOG(FATAL) << "Unimplemented";
  }
  static void MemcpyAsync(void* dst, const void* src, size_t size,
                          IoDirection dir, const stream_t& stream) {
    MemcpySync(dst, src, size, dir);
  }
};

// This interface should be specified by each kind of target.
using TargetWrapperHost = TargetWrapper<TARGET(kHost)>;
using TargetWrapperX86 = TargetWrapperHost;
template <>
class TargetWrapper<TARGET(kHost)> {
 public:
  using stream_t = int;
  using event_t = int;

  static size_t num_devices() { return 0; }
  static size_t maximum_stream() { return 0; }

  static void CreateStream(stream_t* stream) {}
  static void DestroyStream(const stream_t& stream) {}

  static void CreateEvent(event_t* event) {}
  static void DestroyEvent(const event_t& event) {}

  static void RecordEvent(const event_t& event) {}
  static void SyncEvent(const event_t& event) {}

  static void StreamSync(const stream_t& stream) {}

  static void* Malloc(size_t size);
  static void Free(void* ptr);

  static void MemcpySync(void* dst, const void* src, size_t size,
                         IoDirection dir);
  static void MemcpyAsync(void* dst, const void* src, size_t size,
                          IoDirection dir, const stream_t& stream) {
    MemcpySync(dst, src, size, dir);
  }
};

#ifdef LITE_WITH_CUDA
using TargetWrapperCuda =
    TargetWrapper<TARGET(kCUDA), cudaStream_t, cudaEvent_t>;
// This interface should be specified by each kind of target.
template <>
class TargetWrapper<TARGET(kCUDA), cudaStream_t, cudaEvent_t> {
 public:
  using stream_t = cudaStream_t;
  using event_t = cudaEvent_t;

  static size_t num_devices() { return 0; }
  static size_t maximum_stream() { return 0; }

  static void CreateStream(stream_t* stream) {}
  static void DestroyStream(const stream_t& stream) {}

  static void CreateEvent(event_t* event) {}
  static void DestroyEvent(const event_t& event) {}

  static void RecordEvent(const event_t& event) {}
  static void SyncEvent(const event_t& event) {}

  static void StreamSync(const stream_t& stream) {}

  static void* Malloc(size_t size);
  static void Free(void* ptr);

  static void MemcpySync(void* dst, const void* src, size_t size,
                         IoDirection dir);
  static void MemcpyAsync(void* dst, const void* src, size_t size,
                          IoDirection dir, const stream_t& stream);
};
#endif  // LITE_WITH_CUDA

template <TargetType Target>
void CopySync(void* dst, void* src, size_t size, IoDirection dir) {
  switch (Target) {
    case TARGET(kX86):
    case TARGET(kHost):
    case TARGET(kARM):
      TargetWrapperX86::MemcpySync(dst, src, size, IoDirection::HtoH);
      break;
#ifdef LITE_WITH_CUDA
    case TARGET(kCUDA):
      TargetWrapperCuda::MemcpySync(dst, src, size, dir);
#endif
  }
}

}  // namespace lite
}  // namespace paddle

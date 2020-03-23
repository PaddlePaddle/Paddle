/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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

#include <cuda_runtime.h>
#include <atomic>
#include <cstdint>
#include "paddle/fluid/platform/macros.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/stream/stream_internal.h"

namespace paddle {
namespace platform {
namespace stream {

namespace internal {
class StreamInterface;
}

constexpr int kStreamsPerDevCtx = 8;

enum class Priority : uint8_t {
  NIL = 0x0,
  HIGH = 0x1,
  NORMAL = 0x2,
};

class CUDAStream final : public internal::StreamInterface {
 public:
  CUDAStream() = default;
  CUDAStream(const Place& place,
             const enum Priority& priority = Priority::NIL) {
    Init(place, priority);
  }
  virtual ~CUDAStream() { Destroy(); }

  bool Init(const Place& place, const enum Priority& priority = Priority::NIL);

  Place place() const { return place_; }
  cudaStream_t stream() const { return stream_; }
  const Priority& priority() { return priority_; }
  bool IsIdle() const;
  void Destroy();

 private:
  Place place_;
  cudaStream_t stream_{nullptr};
  Priority priority_{Priority::NIL};
  std::once_flag once_flag_;

  DISABLE_COPY_AND_ASSIGN(CUDAStream);
};

class CUDAStreamPool final {
 public:
  explicit CUDAStreamPool(const Place& place) { Init(place); }
  const CUDAStream& GetStream(const enum Priority& priority = Priority::NORMAL);
  const std::array<CUDAStream, kStreamsPerDevCtx>& GetStreams(
      const enum Priority& priority);

 private:
  std::atomic<uint32_t> normal_priority_counters_{0};
  std::atomic<uint32_t> high_priority_counters_{0};
  std::array<CUDAStream, kStreamsPerDevCtx> normal_priority_streams_;
  std::array<CUDAStream, kStreamsPerDevCtx> high_priority_streams_;
  CUDAStream null_stream_;

  void Init(const Place& place);

  DISABLE_COPY_AND_ASSIGN(CUDAStreamPool);
};

}  // namespace stream
}  // namespace platform
}  // namespace paddle

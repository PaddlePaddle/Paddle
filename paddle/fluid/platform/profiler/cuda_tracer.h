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

#include <cstdint>
#include <mutex>
#include <vector>

#include "paddle/common/macros.h"
#include "paddle/fluid/platform/profiler/tracer_base.h"
#include "paddle/phi/backends/dynload/cupti.h"

namespace paddle {
namespace platform {

// Based on CUDA CUPTI
class CudaTracer : public TracerBase {
 public:
  // Singleton. CUPTI imposes this restriction.
  static CudaTracer& GetInstance() {
    static CudaTracer instance;
    return instance;
  }

  void PrepareTracing() override;

  void StartTracing() override;

  void StopTracing() override;

  void CollectTraceData(TraceEventCollector* collector) override;

 private:
  struct ActivityBuffer {
    ActivityBuffer(uint8_t* addr, size_t size) : addr(addr), valid_size(size) {}
    uint8_t* addr;
    size_t valid_size;
  };

  CudaTracer();

  DISABLE_COPY_AND_ASSIGN(CudaTracer);

  void EnableCuptiActivity();

  void DisableCuptiActivity();

  int ProcessCuptiActivity(TraceEventCollector* collector);

#ifdef PADDLE_WITH_CUPTI
  // Used by CUPTI Activity API to request buffer
  static void CUPTIAPI BufferRequestedCallback(uint8_t** buffer,
                                               size_t* size,
                                               size_t* max_num_records);

  // Used by CUPTI Activity API to commit a completed buffer
  static void CUPTIAPI BufferCompletedCallback(CUcontext ctx,
                                               uint32_t stream_id,
                                               uint8_t* buffer,
                                               size_t size,
                                               size_t valid_size);
#endif

  void AllocateBuffer(uint8_t** buffer, size_t* size);

  void ProduceBuffer(uint8_t* buffer, size_t valid_size);

  std::vector<ActivityBuffer> ConsumeBuffers();

  void ReleaseBuffer(uint8_t* buffer);

  uint64_t tracing_start_ns_ = UINT64_MAX;
  std::mutex activity_buffer_lock_;
  std::vector<ActivityBuffer> activity_buffers_;
};

}  // namespace platform
}  // namespace paddle

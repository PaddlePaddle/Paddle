// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include <c10/core/Event.h>

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#endif

#include "gtest/gtest.h"
#include "test/cpp/compat/cuda_test_utils.h"

TEST(EventTest, CpuEventDefaultProperties) {
  c10::Event event(c10::DeviceType::CPU);
  EXPECT_EQ(event.device_type(), c10::DeviceType::CPU);
  EXPECT_EQ(event.device_index(), -1);
  EXPECT_EQ(event.flag(), c10::EventFlag::PYTORCH_DEFAULT);
  EXPECT_FALSE(event.was_marked_for_recording());
  EXPECT_TRUE(event.query());
  EXPECT_EQ(event.eventId(), nullptr);
}

TEST(EventTest, CpuEventRecordThrows) {
  c10::Event event(c10::DeviceType::CPU);
  c10::Stream stream(c10::Stream::DEFAULT,
                     c10::Device(c10::DeviceType::CPU, 0));
  EXPECT_THROW(event.record(stream), std::exception);
  EXPECT_THROW(event.recordOnce(stream), std::exception);
}

#ifdef PADDLE_WITH_CUDA
using RawEventRecordMethod = void (c10::Event::*)(const cudaStream_t&);
[[maybe_unused]] static RawEventRecordMethod g_raw_event_record_method =
    &c10::Event::record;
#endif

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
TEST(EventTest, CudaEventLazyCreateAndRecord) {
  SKIP_IF_CUDA_RUNTIME_UNAVAILABLE();
  c10::Event event(c10::DeviceType::CUDA);
  auto stream = c10::cuda::getCurrentCUDAStream();

  EXPECT_EQ(event.device_index(), -1);
  EXPECT_EQ(event.eventId(), nullptr);
  EXPECT_FALSE(event.was_marked_for_recording());

  EXPECT_NO_THROW(event.record(stream));
  EXPECT_EQ(event.device_index(), stream.device_index());
  EXPECT_NE(event.eventId(), nullptr);
  EXPECT_TRUE(event.was_marked_for_recording());

  EXPECT_NO_THROW(event.synchronize());
  EXPECT_TRUE(event.query());
}

TEST(EventTest, CudaEventElapsedTimeRequiresTimingFlag) {
  SKIP_IF_CUDA_RUNTIME_UNAVAILABLE();
  auto stream = c10::cuda::getCurrentCUDAStream();
  c10::Event start(c10::DeviceType::CUDA);
  c10::Event end(c10::DeviceType::CUDA);

  start.record(stream);
  end.record(stream);
  end.synchronize();

  EXPECT_THROW(start.elapsedTime(end), std::exception);
}

TEST(EventTest, CudaEventElapsedTimeWithTimingEnabled) {
  SKIP_IF_CUDA_RUNTIME_UNAVAILABLE();
  auto stream = c10::cuda::getCurrentCUDAStream();
  c10::Event start(c10::DeviceType::CUDA, c10::EventFlag::BACKEND_DEFAULT);
  c10::Event end(c10::DeviceType::CUDA, c10::EventFlag::BACKEND_DEFAULT);

  start.record(stream);
  end.record(stream);
  end.synchronize();

  double elapsed_ms = -1.0;
  EXPECT_NO_THROW(elapsed_ms = start.elapsedTime(end));
  EXPECT_GE(elapsed_ms, 0.0);
}

#ifdef PADDLE_WITH_CUDA
TEST(EventTest, CudaEventRawStreamRecordCompatibility) {
  SKIP_IF_CUDA_RUNTIME_UNAVAILABLE();
  auto stream = c10::cuda::getCurrentCUDAStream();
  c10::Event event(c10::DeviceType::CUDA);
  EXPECT_NO_THROW(event.record(stream.raw_stream()));
  EXPECT_EQ(event.device_index(), stream.device_index());
  EXPECT_TRUE(event.was_marked_for_recording());
}
#endif

TEST(EventTest, CudaEventRejectsDifferentDeviceRecord) {
  SKIP_IF_CUDA_RUNTIME_UNAVAILABLE();
  if (c10::cuda::device_count() < 2) {
    return;
  }

  c10::Event event(c10::DeviceType::CUDA, c10::EventFlag::BACKEND_DEFAULT);
  auto stream0 = c10::cuda::getDefaultCUDAStream(0);
  auto stream1 = c10::cuda::getDefaultCUDAStream(1);

  EXPECT_NO_THROW(event.record(stream0));
  EXPECT_THROW(event.record(stream1), std::exception);
}
#endif  // PADDLE_WITH_CUDA || PADDLE_WITH_HIP

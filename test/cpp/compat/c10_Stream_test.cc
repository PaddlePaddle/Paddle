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

#include <ATen/cuda/CUDAContext.h>
#include <c10/core/Stream.h>
#include <c10/cuda/CUDAFunctions.h>

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include <c10/cuda/CUDAStream.h>
#endif

#include <atomic>
#include <chrono>
#include <thread>

#include "gtest/gtest.h"
#include "paddle/phi/api/include/context_pool.h"
#include "paddle/phi/backends/gpu/gpu_context.h"

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
namespace {

using StreamCallbackGate = std::atomic<bool>;

#ifdef PADDLE_WITH_HIP
void BlockingStreamCallback(hipStream_t /*stream*/,
                            hipError_t /*status*/,
                            void* user_data) {
  auto* gate = static_cast<StreamCallbackGate*>(user_data);
  while (!gate->load(std::memory_order_acquire)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

void CreateRawStream(hipStream_t* stream) {
  C10_CUDA_CHECK(hipStreamCreate(stream));
}

void DestroyRawStream(hipStream_t stream) {
  C10_CUDA_CHECK(hipStreamDestroy(stream));
}

void ClearLastStreamError() { (void)hipGetLastError(); }
#else
void CUDART_CB BlockingStreamCallback(void* user_data) {
  auto* gate = static_cast<StreamCallbackGate*>(user_data);
  while (!gate->load(std::memory_order_acquire)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

void ClearLastStreamError() { (void)cudaGetLastError(); }
#endif

}  // namespace
#endif

// Test device_count() works in both CPU and CUDA builds
TEST(StreamTest, DeviceCount) {
  c10::DeviceIndex count = c10::cuda::device_count();
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  // In CUDA builds, should return actual device count (>= 0)
  EXPECT_GE(count, 0);
#else
  // In CPU-only builds, should return 0
  EXPECT_EQ(count, 0);
#endif
}

// ==================== native_handle ====================

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
// CUDA stream: native_handle() should return the underlying cudaStream_t
// encoded as void*. For the default (null) stream the id is 0, so the
// pointer is nullptr; for a real stream it must be non-null.
TEST(StreamTest, NativeHandleCudaDefaultStream) {
  if (!at::cuda::is_available()) {
    return;
  }
  c10::Stream s = c10::cuda::getDefaultCUDAStream().unwrap();
  // Default stream encodes nullptr (id == 0), so native_handle() == nullptr.
  EXPECT_EQ(s.native_handle(), nullptr);
}

TEST(StreamTest, NativeHandleCudaCurrentStream) {
  if (!at::cuda::is_available()) {
    return;
  }
  auto cuda_stream = c10::cuda::getCurrentCUDAStream();
  c10::Stream s = cuda_stream.unwrap();
  // getCurrentCUDAStream wraps the real phi stream handle; calling
  // native_handle() must not throw.
  EXPECT_NO_THROW({ (void)s.native_handle(); });
}
#endif  // PADDLE_WITH_CUDA || PADDLE_WITH_HIP

// CPU stream: native_handle() is not supported and must throw.
TEST(StreamTest, NativeHandleCpuStreamThrows) {
  c10::Stream cpu_stream(c10::Stream::DEFAULT,
                         c10::Device(c10::DeviceType::CPU, 0));
  EXPECT_THROW({ (void)cpu_stream.native_handle(); }, std::exception);
}

// ==================== query ====================

// CPU stream is always ready.
TEST(StreamTest, QueryCpuStreamReturnsTrue) {
  c10::Stream cpu_stream(c10::Stream::DEFAULT,
                         c10::Device(c10::DeviceType::CPU, 0));
  EXPECT_TRUE(cpu_stream.query());
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
// A freshly-obtained CUDA stream with no pending work must report ready.
TEST(StreamTest, QueryCudaStreamReady) {
  if (!at::cuda::is_available()) {
    return;
  }
  auto cuda_stream = c10::cuda::getCurrentCUDAStream();
  c10::Stream s = cuda_stream.unwrap();
  // synchronize first to ensure no pending work, then query should be true.
  EXPECT_NO_THROW(s.synchronize());
  EXPECT_TRUE(s.query());
}

#endif  // PADDLE_WITH_CUDA || PADDLE_WITH_HIP

// ==================== synchronize ====================

// CPU stream: synchronize() is a no-op and must not throw.
TEST(StreamTest, SynchronizeCpuStream) {
  c10::Stream cpu_stream(c10::Stream::DEFAULT,
                         c10::Device(c10::DeviceType::CPU, 0));
  EXPECT_NO_THROW(cpu_stream.synchronize());
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
// CUDA stream: synchronize() must complete without error.
TEST(StreamTest, SynchronizeCudaStream) {
  if (!at::cuda::is_available()) {
    return;
  }
  auto cuda_stream = c10::cuda::getCurrentCUDAStream();
  c10::Stream s = cuda_stream.unwrap();
  EXPECT_NO_THROW(s.synchronize());
}
#endif  // PADDLE_WITH_CUDA || PADDLE_WITH_HIP

// ==================== getDefaultCUDAStream ====================

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
// getDefaultCUDAStream must always return the null stream (id == 0),
// which corresponds to cudaStreamDefault on the device.
TEST(CUDAStreamTest, DefaultStreamIsNullStream) {
  if (!at::cuda::is_available()) {
    return;
  }
  auto default_stream = c10::cuda::getDefaultCUDAStream();
  // id == 0 encodes cudaStreamDefault (the null stream, handle nullptr).
  EXPECT_EQ(default_stream.id(), static_cast<c10::StreamId>(0));
}

// getDefaultCUDAStream must be stable: calling it twice returns equal streams.
TEST(CUDAStreamTest, DefaultStreamIsStable) {
  if (!at::cuda::is_available()) {
    return;
  }
  auto s1 = c10::cuda::getDefaultCUDAStream();
  auto s2 = c10::cuda::getDefaultCUDAStream();
  EXPECT_EQ(s1, s2);
}

TEST(CUDAStreamTest, GetStreamFromPoolBoolOverloadPreservesHighPriority) {
  if (!at::cuda::is_available()) {
    return;
  }
  auto low_priority_stream =
      c10::cuda::getStreamFromPool(/*isHighPriority=*/false);
  auto high_priority_stream =
      c10::cuda::getStreamFromPool(/*isHighPriority=*/true);
  auto explicit_high_priority_stream = c10::cuda::getStreamFromPool(-1);

  const int low_priority = low_priority_stream.priority();
  const int high_priority = high_priority_stream.priority();
  const int explicit_high_priority = explicit_high_priority_stream.priority();

  if (low_priority == explicit_high_priority) {
    return;
  }

  EXPECT_EQ(high_priority, explicit_high_priority);
  EXPECT_NE(high_priority, low_priority);
}

// After setCurrentCUDAStream redirects the current stream,
// getDefaultCUDAStream must still return the null stream.
TEST(CUDAStreamTest, DefaultStreamUnaffectedBySetCurrentCUDAStream) {
  if (!at::cuda::is_available()) {
    return;
  }
  // Snapshot the current stream before we touch it so we can
  // restore it afterward and avoid polluting subsequent tests.
  auto original_stream = c10::cuda::getCurrentCUDAStream();

  // Obtain a non-default stream from the pool.
  auto pool_stream = c10::cuda::getStreamFromPool(/*isHighPriority=*/false);

  // Redirect the current stream.
  c10::cuda::setCurrentCUDAStream(pool_stream);

  auto default_stream = c10::cuda::getDefaultCUDAStream();
  auto current_stream = c10::cuda::getCurrentCUDAStream();
  auto place = phi::GPUPlace(current_stream.device_index());

  // Default stream is still null; current stream has changed.
  EXPECT_EQ(default_stream.id(), static_cast<c10::StreamId>(0));
  EXPECT_NE(default_stream, current_stream);
  EXPECT_EQ(paddle::GetCurrentCUDAStream(place)->raw_stream(),
            current_stream.stream());

  // Restore the original current stream.
  c10::cuda::setCurrentCUDAStream(original_stream);
  EXPECT_EQ(paddle::GetCurrentCUDAStream(place)->raw_stream(),
            original_stream.stream());
}

#endif  // PADDLE_WITH_CUDA || PADDLE_WITH_HIP

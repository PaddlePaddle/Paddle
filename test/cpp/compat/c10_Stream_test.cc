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

#include <c10/core/Stream.h>

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#endif

#include "gtest/gtest.h"

// ==================== native_handle ====================

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
// CUDA stream: native_handle() should return the underlying cudaStream_t
// encoded as void*. For the default (null) stream the id is 0, so the
// pointer is nullptr; for a real stream it must be non-null.
TEST(StreamTest, NativeHandleCudaDefaultStream) {
  c10::DeviceIndex dev = c10::cuda::current_device();
  c10::Stream s(c10::Stream::DEFAULT, c10::Device(c10::DeviceType::CUDA, dev));
  // Default stream encodes nullptr (id == 0), so native_handle() == nullptr.
  EXPECT_EQ(s.native_handle(), nullptr);
}

TEST(StreamTest, NativeHandleCudaCurrentStream) {
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
  auto cuda_stream = c10::cuda::getCurrentCUDAStream();
  c10::Stream s = cuda_stream.unwrap();
  EXPECT_NO_THROW(s.synchronize());
}
#endif  // PADDLE_WITH_CUDA || PADDLE_WITH_HIP

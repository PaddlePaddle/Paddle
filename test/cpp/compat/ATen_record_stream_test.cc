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

#include <ATen/Functions.h>
#include <ATen/core/TensorBody.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/record_stream.h>
#include <c10/core/Device.h>
#include <c10/core/Stream.h>
#include "ATen/ATen.h"
#include "gtest/gtest.h"
#include "torch/all.h"
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#endif

class RecordStreamTest : public ::testing::Test {
 protected:
  void SetUp() override {
    cpu_tensor =
        at::ones({4}, at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    if (at::cuda::is_available()) {
      cuda_tensor = at::ones(
          {4}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
    }
#endif
  }

  at::Tensor cpu_tensor;
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  at::Tensor cuda_tensor;
#endif
};

// --- Happy path: CUDA tensor + current CUDA stream should succeed ---
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
using RecordCudaStreamMethod = void (at::Tensor::*)(at::cuda::CUDAStream) const;
[[maybe_unused]] static RecordCudaStreamMethod g_record_cuda_stream_method =
    &at::Tensor::record_stream;

// Raw stream type is platform-specific:
// - CUDA: cudaStream_t (CUstream_st*)
// - HIP: hipStream_t (ihipStream_t*)
// Only test the raw stream overload on CUDA builds where cudaStream_t is
// consistently defined. HIP builds use hipStream_t which is a different type.
#if defined(PADDLE_WITH_CUDA)
using RecordRawCudaStreamMethod = void (at::Tensor::*)(cudaStream_t) const;
[[maybe_unused]] static RecordRawCudaStreamMethod
    g_record_raw_cuda_stream_method = &at::Tensor::record_stream;
#endif

TEST_F(RecordStreamTest, CudaTensorCurrentCudaStream) {
  if (!at::cuda::is_available()) {
    return;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  // record_stream should not throw
  EXPECT_NO_THROW(cuda_tensor.record_stream(stream));
}

// --- Happy path: CUDA tensor + default CUDA stream should succeed ---
TEST_F(RecordStreamTest, CudaTensorDefaultCudaStream) {
  if (!at::cuda::is_available()) {
    return;
  }
  c10::Stream default_stream = c10::cuda::getDefaultCUDAStream().unwrap();
  EXPECT_NO_THROW(cuda_tensor.record_stream(default_stream));
}

TEST_F(RecordStreamTest, CudaTensorRawCudaStream) {
  if (!at::cuda::is_available()) {
    return;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  EXPECT_NO_THROW(cuda_tensor.record_stream(stream.raw_stream()));
}
#endif  // PADDLE_WITH_CUDA || PADDLE_WITH_HIP

// --- Error path: CPU tensor + CPU stream (record_stream does not support CPU
// tensors) ---
TEST_F(RecordStreamTest, CpuTensorCpuStream) {
  c10::Stream cpu_stream(c10::Stream::DEFAULT,
                         c10::Device(c10::DeviceType::CPU, 0));
  EXPECT_THROW(cpu_tensor.record_stream(cpu_stream), std::exception);
}

// --- Error path: CPU tensor + CUDA stream (record_stream does not support CPU
// tensors) ---
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
TEST_F(RecordStreamTest, CpuTensorCudaStream) {
  if (!at::cuda::is_available()) {
    return;
  }
  auto cuda_stream = at::cuda::getCurrentCUDAStream();
  EXPECT_THROW(cpu_tensor.record_stream(cuda_stream), std::exception);
}
#endif  // PADDLE_WITH_CUDA || PADDLE_WITH_HIP

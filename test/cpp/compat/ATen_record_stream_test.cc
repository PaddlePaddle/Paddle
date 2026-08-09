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

#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>

#include "ATen/ATen.h"
#include "gtest/gtest.h"
#include "paddle/phi/core/memory/allocation/allocator_facade.h"
#include "torch/all.h"
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#endif

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
#else
void CUDART_CB BlockingStreamCallback(void* user_data) {
  auto* gate = static_cast<StreamCallbackGate*>(user_data);
  while (!gate->load(std::memory_order_acquire)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}
#endif

class CallbackGateRelease final {
 public:
  explicit CallbackGateRelease(StreamCallbackGate* gate) : gate_(gate) {}
  ~CallbackGateRelease() { gate_->store(true, std::memory_order_release); }

 private:
  StreamCallbackGate* gate_;
};

}  // namespace
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
using RecordStreamMethod = void (at::Tensor::*)(at::Stream) const;
[[maybe_unused]] static RecordStreamMethod g_record_stream_method =
    &at::Tensor::record_stream;

TEST_F(RecordStreamTest, CudaTensorCurrentCudaStream) {
  if (!at::cuda::is_available()) {
    return;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  // record_stream should not throw
  EXPECT_NO_THROW(cuda_tensor.record_stream(stream));
}

TEST_F(RecordStreamTest, StorageHolderViewKeepsNonDefaultStreamAlive) {
  if (!at::cuda::is_available()) {
    return;
  }
  auto& facade = paddle::memory::allocation::AllocatorFacade::Instance();
  if (!facade.IsStreamSafeCUDAAllocatorUsed() &&
      !facade.IsCUDAMallocAsyncAllocatorUsed()) {
    return;
  }

  auto current_stream = at::cuda::getCurrentCUDAStream();
  auto other_stream = at::cuda::getStreamFromPool(
      /*isHighPriority=*/false, current_stream.device_index());
  for (int i = 0; i < 32 && other_stream == current_stream; ++i) {
    other_stream = at::cuda::getStreamFromPool(
        /*isHighPriority=*/false, current_stream.device_index());
  }
  ASSERT_NE(other_stream, current_stream);

  StreamCallbackGate gate{false};
  CallbackGateRelease release_gate(&gate);
  void* original_ptr = nullptr;
  constexpr int64_t kElements = 1 << 20;
  constexpr size_t kBytes = kElements * sizeof(float);
  {
    at::Tensor tensor = at::empty(
        {kElements}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
    (void)tensor.storage();
    auto dense = std::dynamic_pointer_cast<phi::DenseTensor>(
        tensor._PD_GetInner().impl());
    ASSERT_NE(dense, nullptr);
    ASSERT_NE(
        std::dynamic_pointer_cast<c10::StorageHolderView>(dense->Holder()),
        nullptr);
    original_ptr = tensor.data_ptr();

#ifdef PADDLE_WITH_HIP
    C10_CUDA_CHECK(hipStreamAddCallback(
        other_stream.stream(), BlockingStreamCallback, &gate, 0));
    C10_CUDA_CHECK(
        hipMemsetAsync(original_ptr, 0, kBytes, other_stream.stream()));
#else
    C10_CUDA_CHECK(cudaLaunchHostFunc(
        other_stream.stream(), BlockingStreamCallback, &gate));
    C10_CUDA_CHECK(
        cudaMemsetAsync(original_ptr, 0, kBytes, other_stream.stream()));
#endif
    EXPECT_NO_THROW(tensor.record_stream(other_stream.unwrap()));
  }

  at::Tensor replacement = at::empty(
      {kElements}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
  EXPECT_NE(replacement.data_ptr(), original_ptr)
      << "record_stream must defer reuse while non-default stream work is "
         "pending";

  gate.store(true, std::memory_order_release);
  EXPECT_NO_THROW(other_stream.synchronize());
}

TEST_F(RecordStreamTest, StorageHolderViewWithoutAllocationFailsLoudly) {
  if (!at::cuda::is_available()) {
    return;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  at::Tensor tensor =
      at::empty({4}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
  c10::Storage storage = tensor.storage();
  auto dense =
      std::dynamic_pointer_cast<phi::DenseTensor>(tensor._PD_GetInner().impl());
  ASSERT_NE(dense, nullptr);
  ASSERT_NE(std::dynamic_pointer_cast<c10::StorageHolderView>(dense->Holder()),
            nullptr);

  storage.set_data_ptr_noswap(
      c10::DataPtr(reinterpret_cast<void*>(static_cast<uintptr_t>(1)),
                   c10::Device(c10::DeviceType::CUDA, stream.device_index())));
  EXPECT_THROW(tensor.record_stream(stream.unwrap()), std::exception);
}

TEST_F(RecordStreamTest, RejectsStreamFromAnotherDevice) {
  if (c10::cuda::device_count() < 2) {
    return;
  }
  c10::cuda::CUDAGuard guard(0);
  at::Tensor tensor = at::empty(
      {4},
      at::TensorOptions().dtype(at::kFloat).device(c10::Device(at::kCUDA, 0)));
  auto other_device_stream = at::cuda::getStreamFromPool(
      /*isHighPriority=*/false, /*device_index=*/1);

  EXPECT_THROW(tensor.record_stream(other_device_stream.unwrap()),
               std::exception);
}

// --- Happy path: CUDA tensor + default CUDA stream should succeed ---
TEST_F(RecordStreamTest, CudaTensorDefaultCudaStream) {
  if (!at::cuda::is_available()) {
    return;
  }
  c10::Stream default_stream = c10::cuda::getDefaultCUDAStream().unwrap();
  EXPECT_NO_THROW(cuda_tensor.record_stream(default_stream));
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

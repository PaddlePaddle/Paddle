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

#include <list>
#include <memory>
#include <queue>
#include <vector>

#include "ThreadPool.h"
#include "paddle/phi/core/framework/reader.h"
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/core/platform/device/gpu/gpu_resource_pool.h"
#endif

#ifdef PADDLE_WITH_XPU
#include "paddle/phi/core/platform/device/xpu/xpu_info.h"
#include "paddle/phi/core/platform/device/xpu/xpu_resource_pool.h"
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
#include "paddle/phi/backends/event.h"
#include "paddle/phi/backends/stream.h"
#endif
namespace paddle {
namespace operators {
namespace reader {

class BufferedReader : public framework::DecoratedReader {
  using TensorVec = phi::TensorArray;
  using VecFuture = std::future<TensorVec>;

 public:
  BufferedReader(const std::shared_ptr<framework::ReaderBase>& reader,
                 const phi::Place& place,
                 size_t buffer_size,
                 bool pin_memory = false);

  ~BufferedReader() override;

  phi::Place GetPlace() const { return place_; }

 private:
  void ReadTillBufferFullAsync();

  void ReadAsync(size_t i);

 protected:
  void ShutdownImpl() override;
  void StartImpl() override;
  void ReadNextImpl(phi::TensorArray* out) override;

 private:
  ThreadPool thread_pool_;
  phi::Place place_;
  const size_t buffer_size_;
  bool pin_memory_;

  std::queue<std::future<size_t>> position_;

  // The buffer for reading data.
  // NOTE: the simplest way to implement buffered reader is do not use any
  // buffer, just read async and create futures as buffer size. However, to
  // malloc tensors every time is extremely slow. Here we store all data in
  // buffers and prevent alloc every time.
  std::vector<TensorVec> cpu_buffer_;
  std::vector<TensorVec> cuda_buffer_;
  std::vector<TensorVec> xpu_buffer_;
  std::vector<TensorVec> custom_device_buffer_;
  size_t prev_pos_{-1UL};
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  gpuStream_t compute_stream_;
  std::shared_ptr<platform::CudaStreamObject> stream_ = nullptr;
  std::vector<std::shared_ptr<platform::CudaEventObject>> events_{};
#endif

#ifdef PADDLE_WITH_XPU
  xpuStream compute_stream_;
  std::shared_ptr<platform::XpuStreamObject> stream_ = nullptr;
  std::vector<std::shared_ptr<platform::XpuEventObject>> events_{};
#endif

#ifdef PADDLE_WITH_CUSTOM_DEVICE
  std::shared_ptr<phi::stream::Stream> custom_device_compute_stream_ = nullptr;
  std::shared_ptr<phi::stream::Stream> custom_device_stream_ = nullptr;
  std::vector<std::shared_ptr<phi::event::Event>> custom_device_events_{};
#endif
};

}  // namespace reader
}  // namespace operators
}  // namespace paddle

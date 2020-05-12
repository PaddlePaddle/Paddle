// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/reader/buffered_reader.h"
#include <memory>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/platform/profiler.h"

namespace paddle {
namespace operators {
namespace reader {
BufferedReader::~BufferedReader() {
  VLOG(1) << "~BufferedReader";
  reader_->Shutdown();
  while (!position_.empty()) {
    auto &front = position_.front();
    if (front.valid()) {
      front.wait();
    }
    position_.pop();
  }
}

BufferedReader::BufferedReader(
    const std::shared_ptr<framework::ReaderBase> &reader,
    const platform::Place &place, size_t buffer_size)
    : framework::DecoratedReader(reader),
      thread_pool_(1),
      place_(place),
      buffer_size_(buffer_size) {
  VLOG(1) << "BufferedReader";
#ifdef PADDLE_WITH_CUDA
  if (platform::is_gpu_place(place_)) {
    int dev_idx = BOOST_GET_CONST(platform::CUDAPlace, place_).device;
    compute_stream_ =
        ((platform::CUDADeviceContext *)(platform::DeviceContextPool::Instance()
                                             .Get(place_)))
            ->stream();
    events_.resize(buffer_size);
    for (auto &event : events_) {
      event = platform::CudaEventResourcePool::Instance().New(dev_idx);
    }
    stream_ = platform::CudaStreamResourcePool::Instance().New(dev_idx);
  }
#endif
  cpu_buffer_.resize(buffer_size);
  gpu_buffer_.resize(buffer_size);
  ReadTillBufferFullAsync();
}

void BufferedReader::ReadTillBufferFullAsync() {
  PADDLE_ENFORCE_EQ(position_.size(), 0U);
  for (size_t i = 0; i < buffer_size_; ++i) {
    ReadAsync(i);
  }
}

void BufferedReader::ReadAsync(size_t i) {
  position_.emplace(thread_pool_.enqueue([this, i]() -> size_t {
    TensorVec &cpu = cpu_buffer_[i];
    reader_->ReadNext(&cpu);

    if (cpu.empty()) {
      return -1UL;
    }

#ifdef PADDLE_WITH_CUDA
    // NOTE(liangdun): using async copy instead of TensorCopySync
    // TensorCopySync would block other stream, because TensorCopySync
    // issues the copying command to the default stream, it will make two
    // commands from different streams cannot run concurrently.
    if (platform::is_gpu_place(place_)) {
      TensorVec &gpu = gpu_buffer_[i];
      if (gpu.empty()) {
        gpu.resize(cpu.size());
      } else {
        PADDLE_ENFORCE_EQ(gpu.size(), cpu.size(),
                          "Input tensor number not matched");
      }

      std::vector<void *> gpu_ptrs;
      gpu_ptrs.reserve(cpu.size());
      for (size_t i = 0; i < cpu.size(); ++i) {
        gpu[i].Resize(cpu[i].dims());
        gpu[i].set_layout(cpu[i].layout());
        gpu_ptrs.emplace_back(gpu[i].mutable_data(place_, cpu[i].type()));
      }

      // NOTE(zjl): cudaStreamWaitEvent() must be called after all
      // gpu[i].mutable_data() is called, since some ops release
      // gpu memory immediately without waiting gpu kernel ends
      platform::SetDeviceId(
          BOOST_GET_CONST(platform::CUDAPlace, place_).device);
      PADDLE_ENFORCE_CUDA_SUCCESS(
          cudaEventRecord(events_[i].get(), compute_stream_));
      PADDLE_ENFORCE_CUDA_SUCCESS(
          cudaStreamWaitEvent(stream_.get(), events_[i].get(), 0));

      platform::RecordEvent record_event("BufferedReader:MemoryCopy");
      for (size_t i = 0; i < cpu.size(); ++i) {
        auto cpu_place = cpu[i].place();
        auto cpu_ptr = cpu[i].data<void>();
        auto gpu_ptr = gpu_ptrs[i];
        auto size =
            cpu[i].numel() * paddle::framework::SizeOfType(cpu[i].type());
        if (platform::is_cuda_pinned_place(cpu_place)) {
          memory::Copy(BOOST_GET_CONST(platform::CUDAPlace, place_), gpu_ptr,
                       BOOST_GET_CONST(platform::CUDAPinnedPlace, cpu_place),
                       cpu_ptr, size, stream_.get());
        } else if ((platform::is_gpu_place(cpu_place))) {
          memory::Copy(BOOST_GET_CONST(platform::CUDAPlace, place_), gpu_ptr,
                       BOOST_GET_CONST(platform::CUDAPlace, cpu_place), cpu_ptr,
                       size, stream_.get());
        } else {
          platform::CUDAPinnedPlace cuda_pinned_place;
          framework::LoDTensor cuda_pinned_tensor;
          cuda_pinned_tensor.Resize(cpu[i].dims());
          auto cuda_pinned_ptr =
              cuda_pinned_tensor.mutable_data(cuda_pinned_place, cpu[i].type());
          memory::Copy(cuda_pinned_place, cuda_pinned_ptr,
                       BOOST_GET_CONST(platform::CPUPlace, cpu_place), cpu_ptr,
                       size);
          memory::Copy(BOOST_GET_CONST(platform::CUDAPlace, place_), gpu_ptr,
                       cuda_pinned_place, cuda_pinned_ptr, size, stream_.get());
          PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamSynchronize(stream_.get()));
        }
        gpu[i].set_lod(cpu[i].lod());
      }
      PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamSynchronize(stream_.get()));
    }
#endif
    return i;
  }));
}

void BufferedReader::ShutdownImpl() {
  VLOG(1) << "ShutdownImpl";
  reader_->Shutdown();
  while (!position_.empty()) {
    position_.pop();
  }
  prev_pos_ = -1UL;
}

void BufferedReader::StartImpl() {
  reader_->Start();
  ReadTillBufferFullAsync();
}

void BufferedReader::ReadNextImpl(std::vector<framework::LoDTensor> *out) {
  if (position_.empty()) {
    out->clear();
    return;
  }
  size_t i = position_.front().get();
  position_.pop();

  if (i == -1UL) {
    ReadNextImpl(out);
    return;
  }

  *out = std::move(platform::is_gpu_place(place_) ? gpu_buffer_[i]
                                                  : cpu_buffer_[i]);

  // Do not push current position into ReadAsync. Push the previous position
  // Since all computation in fluid are async, change the data of
  // current position may cause data error.
  if (prev_pos_ != -1Ul) {
    ReadAsync(prev_pos_);
  }
  prev_pos_ = i;
}

}  // namespace reader
}  // namespace operators
}  // namespace paddle

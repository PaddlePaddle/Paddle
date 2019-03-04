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
#include <vector>
#include "paddle/fluid/framework/data_type.h"

namespace paddle {
namespace operators {
namespace reader {
BufferedReader::~BufferedReader() {
  reader_->Shutdown();
  while (!position_.empty()) {
    position_.front().wait();
    position_.pop();
  }
#ifdef PADDLE_WITH_CUDA
  if (platform::is_gpu_place(place_)) {
    platform::SetDeviceId(boost::get<platform::CUDAPlace>(place_).device);
    PADDLE_ENFORCE(cudaStreamDestroy(stream));
    for (auto &event : events) PADDLE_ENFORCE(cudaEventDestroy(event));
  }
#endif
}

BufferedReader::BufferedReader(
    const std::shared_ptr<framework::ReaderBase> &reader,
    const platform::Place &place, size_t buffer_size)
    : framework::DecoratedReader(reader),
      thread_pool_(1),
      place_(place),
      buffer_size_(buffer_size) {
#ifdef PADDLE_WITH_CUDA
  if (platform::is_gpu_place(place_)) {
    platform::SetDeviceId(boost::get<platform::CUDAPlace>(place_).device);
    compute_stream =
        ((platform::CUDADeviceContext *)(platform::DeviceContextPool::Instance()
                                             .Get(place_)))
            ->stream();
    events.resize(buffer_size);
    for (auto &event : events)
      PADDLE_ENFORCE(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    PADDLE_ENFORCE(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
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
#ifdef PADDLE_WITH_CUDA
  if (platform::is_gpu_place(place_)) {
    platform::SetDeviceId(boost::get<platform::CUDAPlace>(place_).device);
    PADDLE_ENFORCE(cudaEventRecord(events[i], compute_stream));
  }
#endif
  position_.emplace(thread_pool_.enqueue([this, i]() -> size_t {
    TensorVec &cpu = cpu_buffer_[i];
    reader_->ReadNext(&cpu);

    if (cpu.empty()) {
      return -1UL;
    }

#ifdef PADDLE_WITH_CUDA
    // NOTE(liangdun): using async copy instead of TensorCopySync
    // TensorCopySync would block other stream
    if (platform::is_gpu_place(place_)) {
      platform::SetDeviceId(boost::get<platform::CUDAPlace>(place_).device);
      PADDLE_ENFORCE(cudaStreamWaitEvent(stream, events[i], 0));
      TensorVec &gpu = gpu_buffer_[i];
      gpu.resize(cpu.size());
      for (size_t i = 0; i < cpu.size(); ++i) {
        gpu[i].Resize(cpu[i].dims());
        gpu[i].set_layout(cpu[i].layout());
        auto cpu_place = cpu[i].place();
        auto cpu_ptr = cpu[i].data<void>();
        auto gpu_ptr = gpu[i].mutable_data(place_, cpu[i].type());
        auto size =
            cpu[i].numel() * paddle::framework::SizeOfType(cpu[i].type());
        if (platform::is_cuda_pinned_place(cpu_place))
          memory::Copy(boost::get<platform::CUDAPlace>(place_), gpu_ptr,
                       boost::get<platform::CUDAPinnedPlace>(cpu_place),
                       cpu_ptr, size, stream);
        else if ((platform::is_gpu_place(cpu_place)))
          memory::Copy(boost::get<platform::CUDAPlace>(place_), gpu_ptr,
                       boost::get<platform::CUDAPlace>(cpu_place), cpu_ptr,
                       size, stream);
        else
          // if cpu place is not pinned, async copy is slower than sync copy,
          // so we use sync copy instead.
          memory::Copy(boost::get<platform::CUDAPlace>(place_), gpu_ptr,
                       boost::get<platform::CPUPlace>(cpu_place), cpu_ptr, size,
                       0);
        gpu[i].set_lod(cpu[i].lod());
      }
      PADDLE_ENFORCE(cudaStreamSynchronize(stream));
    }
#endif
    return i;
  }));
}

void BufferedReader::ShutdownImpl() {
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

  *out = platform::is_gpu_place(place_) ? gpu_buffer_[i] : cpu_buffer_[i];

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

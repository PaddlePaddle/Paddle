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
  is_same_place_ = false;
  cpu_buffer_.resize(buffer_size);
  cuda_pinned_buffer_.resize(buffer_size);
  ReadTillBufferFullAsync();
}

void BufferedReader::ReadTillBufferFullAsync() {
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
    if (platform::is_gpu_place(place_)) {
      // NOTE: [Copy processing of different input devices]
      // We may accept input tensor in three different devices:
      //   - CPUPlace
      //   - CUDAPinnedPlace
      //   - CUDAPlace
      // CUDA Stream Synchronizing is slow, in order to avoid Synchronizing
      // in BufferedReader thread, we do data copy as follows:
      //   - If src Tensor on CPU memory, we copy it to CUDAPinned memory
      //   - IF src Tensor on CUDAPinned memory, we use it directly
      //   - IF src Tensor on CUDA memory, we use it directly
      platform::CUDAPinnedPlace cuda_pinned_place;
      TensorVec &cuda_pinned = cuda_pinned_buffer_[i];
      if (cuda_pinned.empty()) {
        cuda_pinned.resize(cpu.size());
      } else {
        PADDLE_ENFORCE_EQ(
            cuda_pinned.size(), cpu.size(),
            platform::errors::InvalidArgument(
                "Input tensor number on GPU and CPU devices are not matched."));
      }

      std::vector<void *> cuda_pinned_ptrs;
      cuda_pinned_ptrs.reserve(cpu.size());
      platform::RecordEvent record_event("BufferedReader:MemoryCopy");
      for (size_t i = 0; i < cpu.size(); ++i) {
        if (platform::is_cpu_place(cpu[i].place())) {
          cuda_pinned[i].Resize(cpu[i].dims());
          cuda_pinned[i].set_layout(cpu[i].layout());
          cuda_pinned_ptrs.emplace_back(
              cuda_pinned[i].mutable_data(cuda_pinned_place, cpu[i].type()));
          auto size =
              cpu[i].numel() * paddle::framework::SizeOfType(cpu[i].type());

          memory::Copy(cuda_pinned_place, cuda_pinned_ptrs[i],
                       BOOST_GET_CONST(platform::CPUPlace, cpu[i].place()),
                       cpu[i].data<void>(), size);
          cuda_pinned[i].set_lod(cpu[i].lod());
        } else {
          // we set same place flag & use cpu[i] directly
          is_same_place_ = true;
        }
      }
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

  *out = std::move((platform::is_gpu_place(place_) && !is_same_place_)
                       ? cuda_pinned_buffer_[i]
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

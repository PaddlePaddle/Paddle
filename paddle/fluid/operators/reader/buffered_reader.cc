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
#include <vector>

namespace paddle {
namespace operators {
namespace reader {
BufferedReader::~BufferedReader() { reader_->Shutdown(); }
BufferedReader::BufferedReader(
    const std::shared_ptr<framework::ReaderBase> &reader,
    const platform::Place &place, size_t buffer_size)
    : framework::DecoratedReader(reader),
      thread_pool_(1),
      place_(place),
      buffer_size_(buffer_size) {
  cpu_buffer_.resize(buffer_size);
  gpu_buffer_.resize(buffer_size);
  AppendFutureToBatchSize();
}
void BufferedReader::AppendFutureToBatchSize() {
  PADDLE_ENFORCE_EQ(position_.size(), 0U);
  for (size_t i = 0; i < buffer_size_; ++i) {
    AppendFuture(i);
  }
}
void BufferedReader::AppendFuture(size_t i) {
  position_.emplace(thread_pool_.enqueue([this, i]() -> size_t {
    TensorVec &cpu = cpu_buffer_[i];
    reader_->ReadNext(&cpu);

    if (cpu.empty()) {
      return -1UL;
    }

    if (platform::is_gpu_place(place_)) {
      TensorVec &gpu = gpu_buffer_[i];
      gpu.resize(cpu.size());
      for (size_t i = 0; i < cpu.size(); ++i) {
        framework::TensorCopySync(cpu[i], place_, &gpu[i]);
      }
    }
    return i;
  }));
}
void BufferedReader::ShutdownImpl() {
  reader_->Shutdown();
  while (!position_.empty()) {
    position_.pop();
  }
}
void BufferedReader::StartImpl() {
  reader_->Start();
  AppendFutureToBatchSize();
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
  AppendFuture(i);
}

}  // namespace reader
}  // namespace operators
}  // namespace paddle

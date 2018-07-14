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
BufferedReader::~BufferedReader() {
  reader_->Shutdown();
  buffer_.clear();
}
BufferedReader::BufferedReader(
    const std::shared_ptr<framework::ReaderBase> &reader,
    const platform::Place &place, size_t buffer_size)
    : framework::DecoratedReader(reader),
      thread_pool_(1),
      place_(place),
      buffer_size_(buffer_size) {
  AppendFutureToBatchSize();
}
void BufferedReader::AppendFutureToBatchSize() {
  while (buffer_.size() < buffer_size_) {
    AppendFuture();
  }
}
void BufferedReader::AppendFuture() {
  buffer_.emplace_back(thread_pool_.enqueue([this] {
    TensorVec cpu_buffer;
    reader_->ReadNext(&cpu_buffer);
    if (platform::is_gpu_place(place_)) {
      TensorVec gpu_buffer;

      for (size_t i = 0; i < cpu_buffer.size(); ++i) {
        gpu_buffer.emplace_back();
        framework::TensorCopySync(cpu_buffer[i], place_, &gpu_buffer.back());
      }

      cpu_buffer = gpu_buffer;
    }
    return cpu_buffer;
  }));
}
void BufferedReader::ShutdownImpl() {
  reader_->Shutdown();
  buffer_.clear();
}
void BufferedReader::StartImpl() {
  reader_->Start();
  AppendFutureToBatchSize();
}
void BufferedReader::ReadNextImpl(std::vector<framework::LoDTensor> *out) {
  PADDLE_ENFORCE_EQ(buffer_.size(), buffer_size_);
  *out = buffer_.front().get();
  buffer_.pop_front();
  AppendFuture();
}

}  // namespace reader
}  // namespace operators
}  // namespace paddle

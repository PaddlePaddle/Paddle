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

#pragma once

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/reader.h"
#include "paddle/fluid/framework/threadpool.h"
#include "paddle/fluid/operators/reader/lod_tensor_blocking_queue.h"

namespace paddle {
namespace operators {
namespace reader {

class CTRReader : public framework::FileReader {
 public:
  explicit CTRReader(const std::shared_ptr<LoDTensorBlockingQueue>& queue,
                     int batch_size, int thread_num,
                     const std::vector<std::string>& slots,
                     const std::vector<std::string>& file_list)
      : framework::FileReader() {
    thread_num_ = thread_num;
    batch_size_ = batch_size;
    PADDLE_ENFORCE(queue != nullptr, "LoDTensorBlockingQueue must not be null");
    queue_ = queue;
    slots_ = slots;
    file_list_ = file_list;
  }

  ~CTRReader() { queue_->Close(); }

  void ReadNext(std::vector<framework::LoDTensor>* out) override {
    bool success;
    *out = queue_->Pop(&success);
    if (!success) out->clear();
  }

  void Shutdown() override { queue_->Close(); }

  void Start() override {
    queue_->ReOpen();
    //    for (int i = 0; i < thread_num_; i++) {
    //      read_threads_.emplace_back(
    //          new std::thread(std::bind(&CTRReader::ReadThread, this,
    //          file_list_,
    //                                    slots_, batch_size_, queue_)));
    //    }
  }

 private:
  void ReadThread(const std::vector<std::string>& file_list,
                  const std::vector<std::string>& slots, int batch_size,
                  std::shared_ptr<LoDTensorBlockingQueue>* queue);

 private:
  std::shared_ptr<LoDTensorBlockingQueue> queue_;
  std::vector<std::unique_ptr<std::thread>> read_threads_;
  int thread_num_;
  int batch_size_;
  std::vector<std::string> slots_;
  std::vector<std::string> file_list_;
};

}  // namespace reader
}  // namespace operators
}  // namespace paddle

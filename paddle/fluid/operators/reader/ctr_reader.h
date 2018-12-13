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

#include <sys/time.h>

#include <algorithm>
#include <chrono>  // NOLINT
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

enum ReaderThreadStatus { Running, Stopped };

void ReadThread(const std::vector<std::string>& file_list,
                const std::vector<std::string>& slots, int batch_size,
                int thread_id, std::vector<ReaderThreadStatus>* thread_status,
                std::shared_ptr<LoDTensorBlockingQueue> queue);

// monitor all running thread, if they are all stopped,
// then push an empty data into LoDTensorBlockingQueue
void MonitorThread(std::vector<ReaderThreadStatus>* thread_status,
                   std::shared_ptr<LoDTensorBlockingQueue> queue);

class CTRReader : public framework::FileReader {
 public:
  explicit CTRReader(const std::shared_ptr<LoDTensorBlockingQueue>& queue,
                     int batch_size, size_t thread_num,
                     const std::vector<std::string>& slots,
                     const std::vector<std::string>& file_list)
      : batch_size_(batch_size), slots_(slots), file_list_(file_list) {
    PADDLE_ENFORCE_GT(thread_num, 0, "thread num should be larger then 0!");
    PADDLE_ENFORCE(queue != nullptr, "LoDTensorBlockingQueue must not be null");
    PADDLE_ENFORCE_GT(file_list.size(), 0, "file list should not be empty");
    thread_num_ = std::min<size_t>(file_list_.size(), thread_num);
    queue_ = queue;
    SplitFiles();
    for (size_t i = 0; i < thread_num_; ++i) {
      read_thread_status_.push_back(Stopped);
    }
  }

  ~CTRReader() {}

  void ReadNext(std::vector<framework::LoDTensor>* out) override {
    bool success;
    *out = queue_->Pop(&success);
    if (!success) out->clear();
  }

  void Shutdown() override {
    VLOG(3) << "Shutdown reader";
    if (status_ == ReaderStatus::kStopped) {
      return;
    }
    // shutdown should stop all the reader thread
    for (auto& read_thread : read_threads_) {
      read_thread->join();
    }
    monitor_thread_->join();

    read_threads_.clear();
    monitor_thread_.reset(nullptr);
    queue_->Close();
    status_ = ReaderStatus::kStopped;
  }

  void Start() override {
    VLOG(3) << "Start reader";
    PADDLE_ENFORCE_EQ(read_threads_.size(), 0, "read thread should be empty!");
    queue_->ReOpen();
    VLOG(3) << "reopen success";
    VLOG(3) << "thread_num " << thread_num_;
    for (size_t thread_id = 0; thread_id < thread_num_; thread_id++) {
      read_threads_.emplace_back(new std::thread(std::bind(
          &ReadThread, file_groups_[thread_id], slots_, batch_size_,
          static_cast<int>(thread_id), &read_thread_status_, queue_)));
    }
    monitor_thread_.reset(new std::thread(
        std::bind(&MonitorThread, &read_thread_status_, queue_)));
    status_ = ReaderStatus::kRunning;
  }

 private:
  void SplitFiles() {
    file_groups_.resize(thread_num_);
    for (size_t i = 0; i < file_list_.size(); ++i) {
      auto& file_name = file_list_[i];
      std::ifstream f(file_name.c_str());
      PADDLE_ENFORCE(f.good(), "file %s not exist!", file_name);
      file_groups_[i % thread_num_].push_back(file_name);
    }
  }

 private:
  size_t thread_num_;
  const int batch_size_;
  const std::vector<std::string> slots_;
  const std::vector<std::string> file_list_;
  std::shared_ptr<LoDTensorBlockingQueue> queue_;
  std::vector<std::unique_ptr<std::thread>> read_threads_;
  std::unique_ptr<std::thread> monitor_thread_;
  std::vector<ReaderThreadStatus> read_thread_status_;
  std::vector<std::vector<std::string>> file_groups_;
};

}  // namespace reader
}  // namespace operators
}  // namespace paddle

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

struct DataDesc {
  DataDesc(int batch_size, const std::vector<std::string>& file_names,
           const std::string& file_type, const std::string& file_format,
           const std::vector<int>& dense_slot_index,
           const std::vector<int>& sparse_slot_index,
           const std::vector<std::string>& sparse_slot_ids)
      : batch_size_(batch_size),
        file_names_(file_names),
        file_type_(file_type),
        file_format_(file_format),
        dense_slot_index_(dense_slot_index),
        sparse_slot_index_(sparse_slot_index),
        sparse_slot_ids_(sparse_slot_ids) {}

  const int batch_size_;
  const std::vector<std::string> file_names_;
  const std::string file_type_;    // gzip or plain
  const std::string file_format_;  // csv or svm
  // used for csv data format
  const std::vector<int> dense_slot_index_;
  const std::vector<int> sparse_slot_index_;
  // used for svm data format
  const std::vector<std::string> sparse_slot_ids_;
};

inline std::ostream& operator<<(std::ostream& os, const DataDesc& data_desc) {
  os << "data_desc:\n";
  os << "\tbatch_size -> " << data_desc.batch_size_ << "\n";
  os << "\tfile_type -> " << data_desc.file_type_ << "\n";
  os << "\tfile_format -> " << data_desc.file_format_ << "\n";
  os << "\tfile_names -> {";
  for (auto& file_name : data_desc.file_names_) {
    os << file_name << ",";
  }
  os << "}\n";
  os << "\tdense_slot_index -> {";
  for (auto& slot : data_desc.dense_slot_index_) {
    os << slot << ",";
  }
  os << "}\n";
  os << "\tsparse_slot_index_ -> {";
  for (auto& slot : data_desc.sparse_slot_index_) {
    os << slot << ",";
  }
  os << "}\n";
  os << "\tsparse_slot_ids_ -> {";
  for (auto& slot : data_desc.sparse_slot_ids_) {
    os << slot << ",";
  }
  os << "}\n";

  return os;
}

void ReadThread(const std::vector<std::string>& file_list,
                const DataDesc& data_desc, const int thread_id,
                std::vector<ReaderThreadStatus>* thread_status,
                std::shared_ptr<LoDTensorBlockingQueues> queue);

// monitor all running thread, if they are all stopped,
// then push an empty data into LoDTensorBlockingQueue
void MonitorThread(std::vector<ReaderThreadStatus>* thread_status,
                   std::shared_ptr<LoDTensorBlockingQueues> queue);

class CTRReader : public framework::FileReader {
 public:
  CTRReader(const std::shared_ptr<LoDTensorBlockingQueues>& queue,
            int thread_num, const DataDesc& data_desc)
      : data_desc_(data_desc) {
    PADDLE_ENFORCE_GT(thread_num, 0, "thread num should be larger then 0!");
    PADDLE_ENFORCE(queue != nullptr, "LoDTensorBlockingQueue must not be null");
    PADDLE_ENFORCE_EQ(queue->Size(), thread_num,
                      "thread num muse equal queue size now");
    PADDLE_ENFORCE_GE(data_desc_.file_names_.size(), thread_num,
                      "file list must larger or equal than thread_num");
    thread_num_ = std::min<size_t>(data_desc_.file_names_.size(), thread_num);
    queue_ = queue;
    SplitFiles();
    for (size_t i = 0; i < thread_num_; ++i) {
      read_thread_status_.push_back(Stopped);
    }
  }

  ~CTRReader() { Shutdown(); }

  void ReadNext(std::vector<framework::LoDTensor>* out) override {
    bool success;

    VLOG(1) << "CTR Reader ReadNext: " << GetHashThreadId();

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

    if (monitor_thread_) {
      monitor_thread_->join();
    }

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
    for (int thread_id = 0; thread_id < thread_num_; thread_id++) {
      read_threads_.emplace_back(new std::thread(std::bind(
          &ReadThread, file_groups_[thread_id], data_desc_,
          static_cast<int>(thread_id), &read_thread_status_, queue_)));
    }
    monitor_thread_.reset(new std::thread(
        std::bind(&MonitorThread, &read_thread_status_, queue_)));
    status_ = ReaderStatus::kRunning;
  }

 private:
  void SplitFiles() {
    file_groups_.resize(thread_num_);
    for (size_t i = 0; i < data_desc_.file_names_.size(); ++i) {
      auto& file_name = data_desc_.file_names_[i];
      std::ifstream f(file_name.c_str());
      PADDLE_ENFORCE(f.good(), "file %s not exist!", file_name);
      file_groups_[i % thread_num_].push_back(file_name);
    }
  }

 private:
  size_t thread_num_;
  const DataDesc data_desc_;
  std::shared_ptr<LoDTensorBlockingQueues> queue_;
  std::vector<std::unique_ptr<std::thread>> read_threads_;
  std::unique_ptr<std::thread> monitor_thread_;
  std::vector<ReaderThreadStatus> read_thread_status_;
  std::vector<std::vector<std::string>> file_groups_;
};

}  // namespace reader
}  // namespace operators
}  // namespace paddle

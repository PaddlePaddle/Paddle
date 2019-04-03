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
#include "paddle/fluid/operators/reader/blocking_queue.h"
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

void ReadThread(const DataDesc& data_desc, int thread_id,
                std::vector<ReaderThreadStatus>& thread_status,       // NOLINT
                std::shared_ptr<BlockingQueue<std::string>>& reader,  // NOLINT
                std::shared_ptr<LoDTensorBlockingQueues>& queue);     // NOLINT

// monitor all running thread, if they are all stopped,
// then push an empty data into LoDTensorBlockingQueue
void MonitorThread(
    std::vector<ReaderThreadStatus>& thread_status,                  // NOLINT
    std::vector<std::shared_ptr<LoDTensorBlockingQueues>>& queues);  // NOLINT

class CTRReader : public framework::FileReader {
 public:
  CTRReader(const std::vector<std::shared_ptr<LoDTensorBlockingQueues>>& queues,
            const DataDesc& data_desc)
      : data_desc_(data_desc) {
    PADDLE_ENFORCE(!queues.empty(), "LoDTensorBlockingQueue must not be null");

    queue_ = queues;
    thread_num_ = queues.size();
    parallelism_ = queues[0]->Queues();

    read_files_ = std::shared_ptr<BlockingQueue<std::string>>(
        new BlockingQueue<std::string>(data_desc.file_names_.size()));

    for (const auto& file_name : data_desc_.file_names_) {
      std::ifstream f(file_name.c_str());
      PADDLE_ENFORCE(f.good(), "file %s not exist!", file_name);
      read_files_->Send(file_name);
    }

    for (auto i = 0; i < thread_num_; ++i) {
      read_thread_status_.push_back(Stopped);
    }
  }

  ~CTRReader() override { Shutdown(); }

  void ReadNext(std::vector<framework::LoDTensor>* out) override {
    auto queue_id = AppendOrGetHashId();

    bool success;
    *out = queue_[queue_id]->Pop(&success);
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

    for (auto& q_ : queue_) {
      q_->Close();
    }

    status_ = ReaderStatus::kStopped;
  }

  void Start() override {
    VLOG(3) << "Start reader";
    PADDLE_ENFORCE_EQ(read_threads_.size(), 0, "read thread should be empty!");

    for (auto& q_ : queue_) {
      q_->ReOpen();
    }

    VLOG(3) << "reopen success";
    VLOG(3) << "thread_num " << thread_num_;
    for (int thread_id = 0; thread_id < thread_num_; thread_id++) {
      read_threads_.emplace_back(
          new std::thread(ReadThread, std::ref(data_desc_), thread_id,
                          std::ref(read_thread_status_), std::ref(read_files_),
                          std::ref(queue_[thread_id])));
    }
    monitor_thread_.reset(new std::thread(
        MonitorThread, std::ref(read_thread_status_), std::ref(queue_)));
    status_ = ReaderStatus::kRunning;
  }

 private:
  size_t AppendOrGetHashId() {
    std::lock_guard<std::mutex> lock(mutex_);

    size_t hash_thread_id = GetHashThreadId();
    size_t queue_id = -1;

    auto id_search = pop_maps.find(hash_thread_id);
    if (id_search != pop_maps.end()) {
      queue_id = id_search->second;
    } else {
      queue_id = pop_maps.size();
      pop_maps.insert({hash_thread_id, queue_id});
    }
    VLOG(1) << "ReadNext queue: " << queue_id << " hash ID: " << hash_thread_id;
    return queue_id;
  }

 private:
  int thread_num_;
  int parallelism_;
  const DataDesc data_desc_;

  mutable std::mutex mutex_;
  std::unordered_map<size_t, size_t> pop_maps;

  std::vector<std::shared_ptr<LoDTensorBlockingQueues>> queue_;
  std::vector<std::unique_ptr<std::thread>> read_threads_;
  std::unique_ptr<std::thread> monitor_thread_;
  std::vector<ReaderThreadStatus> read_thread_status_;
  std::shared_ptr<BlockingQueue<std::string>> read_files_;
};

}  // namespace reader
}  // namespace operators
}  // namespace paddle

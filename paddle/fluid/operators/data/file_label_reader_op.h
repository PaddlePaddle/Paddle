// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include <fstream>
#include <string>
#include <vector>

#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/operators/reader/lod_tensor_blocking_queue.h"

namespace paddle {
namespace operators {
namespace data {
using LoDTensorArray = framework::LoDTensorArray;
using LoDTensorBlockingQueue = operators::reader::LoDTensorBlockingQueue;
using LoDTensorBlockingQueueHolder = operators::reader::LoDTensorBlockingQueueHolder;

enum BufferStatus {
  kBufferStatusSuccess = 0,
  kBufferStatusErrorClosed,
  kBufferStatusEmpty
};

template <typename T>
class Buffer final {
 public:
  explicit Buffer(size_t max_len = 2) : max_len_(max_len), is_closed_(false) {}
  ~Buffer() = default;

  BufferStatus Push(const T& item);
  BufferStatus Pull(T* item);
  BufferStatus TryReceive(T* item);
  void Close();

 private:
  std::queue<T> queue_;
  mutable std::mutex mutex_;
  size_t max_len_;
  bool is_closed_;
  std::condition_variable cond_;
};

template <typename T>
BufferStatus Buffer<T>::Push(const T& item) {
  std::unique_lock<std::mutex> lock(mutex_);
  cond_.wait(lock, [this]() { return queue_.size() < max_len_ || is_closed_; });
  if (is_closed_) {
    return kBufferStatusErrorClosed;
  }

  queue_.push(item);
  cond_.notify_one();
  return kBufferStatusSuccess;
}

template <typename T>
BufferStatus Buffer<T>::Pull(T* item) {
  std::unique_lock<std::mutex> lock(mutex_);
  cond_.wait(lock, [this]() { return (!queue_.empty()) || is_closed_; });
  if (queue_.empty()) {
    return kBufferStatusErrorClosed;
  }
  *item = queue_.front();
  queue_.pop();
  if (queue_.size() < max_len_) {
    cond_.notify_all();
  }
  return kBufferStatusSuccess;
}

template <typename T>
void Buffer<T>::Close() {
  std::unique_lock<std::mutex> lock(mutex_);
  is_closed_ = true;
  cond_.notify_all();
}

class FileDataReader {
 public:
  explicit FileDataReader(const framework::ExecutionContext& ctx,
                          LoDTensorBlockingQueue* queue, LoDTensorBlockingQueue* label_queue)
              : queue_(queue), label_queue_(label_queue){
    std::vector<std::string> files =
        ctx.Attr<std::vector<std::string>>("files");
    std::vector<int> labels = ctx.Attr<std::vector<int>>("labels");
    rank_ = ctx.Attr<int>("rank");
    world_size_ = ctx.Attr<int>("world_size");
 
    batch_size_ = ctx.Attr<int>("batch_size");
    current_epoch_ = 0;
    current_iter_ = 0;
    // iters_per_epoch_ = labels.size() / (batch_size_ * world_size_);
    auto total_batch_size = batch_size_ * world_size_;
    iters_per_epoch_ = (labels.size() + total_batch_size) / total_batch_size;
    std::cout << "DEBUGGGGG iters per epoch: " << iters_per_epoch_ << std::endl;
    is_closed_ = false;
    for (int i = 0, n = files.size(); i < n; i++)
      image_label_pairs_.emplace_back(std::move(files[i]), labels[i]);
    StartLoadThread();
  }

  int GetStartIndex() {
    int start_idx =
        batch_size_ * world_size_ * (current_iter_ % iters_per_epoch_) +
        rank_ * batch_size_;
    current_iter_++;
    return start_idx;
  }

  framework::LoDTensor ReadSample(const std::string filename) {
    std::ifstream input(filename.c_str(),
                        std::ios::in | std::ios::binary | std::ios::ate);
    std::streamsize file_size = input.tellg();

    input.seekg(0, std::ios::beg);

    // auto* out = ctx.Output<framework::LoDTensor>("Out");
    framework::LoDTensor out;
    std::vector<int64_t> out_shape = {file_size};
    out.Resize(framework::make_ddim(out_shape));

    uint8_t* data = out.mutable_data<uint8_t>(platform::CPUPlace());

    input.read(reinterpret_cast<char*>(data), file_size);
    return out;
  }

  void StartLoadThread() {
    if (load_thrd_.joinable()) {
      return;
    }

    load_thrd_ = std::thread([this] {
      while (!is_closed_.load()) LoadBatch();
    });
  }

  void ShutDown() {
    if (queue_ && !queue_->IsClosed()) queue_->Close();
    if (label_queue_ && !label_queue_->IsClosed()) label_queue_->Close();

    is_closed_.store(true);
    if (load_thrd_.joinable()) {
      load_thrd_.join();
    }
  }


  std::pair<LoDTensorArray, std::vector<int>> Read() {
    LoDTensorArray ret;
    std::vector<int> label;
    ret.reserve(batch_size_);
    int start_index = GetStartIndex();
    for (int32_t i = start_index; i < start_index + batch_size_; ++i) {
      if (static_cast<size_t>(i) >= image_label_pairs_.size()) {
        // FIXME(dkp): refine close pipeline
        while (queue_->Size()) sleep(0.5);
        queue_->Close();
        while (label_queue_->Size()) sleep(0.5);
        label_queue_->Close();

        is_closed_.store(true);
        break;
      }
      else{
        std::cout << "index / size: " << i << "  / " <<  image_label_pairs_.size() << std::endl;
      }
      i %= image_label_pairs_.size();
      framework::LoDTensor tmp = ReadSample(image_label_pairs_[i].first);
      ret.push_back(std::move(tmp));
      label.push_back(image_label_pairs_[i].second);
    }
    return std::make_pair(ret, label);
  }

  
  void LoadBatch() {
    
    auto batch_data = std::move(Read());
    queue_->Push(batch_data.first);
    framework::LoDTensor label_tensor;
    LoDTensorArray label_array;
    // auto& label_tensor = label.GetMutable<framework::LoDTensor>();
    label_tensor.Resize(
        framework::make_ddim({static_cast<int64_t>(batch_data.first.size())}));
    platform::CPUPlace cpu;
    auto* label_data = label_tensor.mutable_data<int>(cpu);
    for (size_t i = 0; i < batch_data.first.size(); ++i) {
      label_data[i] = batch_data.second[i];
    }
    label_array.push_back(label_tensor);
    label_queue_->Push(label_array);
  }

 private:
  int batch_size_;
  std::string file_root_, file_list_;
  std::vector<std::pair<std::string, int>> image_label_pairs_;
  int current_epoch_;
  int current_iter_;
  int rank_;
  int world_size_;
  int iters_per_epoch_;
  std::atomic<bool> is_closed_;
  Buffer<LoDTensorArray> batch_buffer_;
  std::thread load_thrd_;
  LoDTensorBlockingQueue* queue_;
  LoDTensorBlockingQueue* label_queue_;
};


class ReaderManager {
  // PipelineManager is a signleton manager for Pipeline, we
  // create single Pipeline for a program id
 private:
  DISABLE_COPY_AND_ASSIGN(ReaderManager);

  static ReaderManager *rm_instance_ptr_;
  static std::mutex m_;

  std::map<int64_t, std::unique_ptr<FileDataReader>> prog_id_to_reader_;

 public:
  static ReaderManager *Instance() {
    if (rm_instance_ptr_ == nullptr) {
      std::lock_guard<std::mutex> lk(m_);
      if (rm_instance_ptr_ == nullptr) {
        rm_instance_ptr_ = new ReaderManager;
      }
    }
    return rm_instance_ptr_;
  }

  // FileDataReader* GetReader(
  void GetReader(
      int64_t program_id, const framework::ExecutionContext& ctx,
             LoDTensorBlockingQueue* queue, LoDTensorBlockingQueue* label_queue) {
    auto iter = prog_id_to_reader_.find(program_id);
    if (iter == prog_id_to_reader_.end()) {
      prog_id_to_reader_[program_id] = std::unique_ptr<FileDataReader>(new FileDataReader(ctx, queue, label_queue));
      // return prog_id_to_reader_[program_id].get();
    } else {
      // return iter->second.get();
    }
  }

  void ShutDown() {
    auto iter = prog_id_to_reader_.begin();
    while (iter != prog_id_to_reader_.end()){
      if(iter->second.get()){
        iter->second->ShutDown();
      }
      iter++;
    }
    prog_id_to_reader_.clear();
  }

  ReaderManager() { VLOG(1) << "ReaderManager init"; }

  ~ReaderManager() {
    VLOG(1) << "~ReaderManager";
    ShutDown();
  }
};

}  // namespace data
}  // namespace operators
}  // namespace paddle

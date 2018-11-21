/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifndef PADDLE_FLUID_FRAMEWORK_DATA_FEED_H_
#define PADDLE_FLUID_FRAMEWORK_DATA_FEED_H_

#include <memory>
#include <set>
#include <map>
#include <string>
#include <thread>               // NOLINT
#include <vector>
#include <queue>
#include <mutex>                // NOLINT
#include <unordered_map>
#include <unordered_set>
#include <condition_variable>   // NOLINT
#include <fstream>
#include <deque>
#include <atomic>

#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/data_feed.pb.h"

namespace paddle {
namespace framework {

class MixTensor {
 public:
  MixTensor(){}
  MixTensor(LoDTensor* lodtensor) {
    is_dense_ = false;
    lodtensor_ = lodtensor;
  }
  MixTensor(Tensor* tensor) {
    is_dense_ = true;
    tensor_ = tensor;
  }
  bool IsDense() {return is_dense_;}
  LoDTensor* GetLoDTensor(){
    if (is_dense_) {
      LOG(ERROR) << "error: let a dense var return a LoDTensor ptr";
      exit(-1);
    }
    return lodtensor_;
  }
  Tensor* GetTensor(){
    if (!is_dense_) {
      LOG(ERROR) << "error: let a sparse var return a Tensor ptr";
      exit(-1);
    }
    return tensor_;
  }
 private:
  bool is_dense_;
  LoDTensor* lodtensor_;
  Tensor* tensor_;
};

template<typename T>
class BlockingQueue {
 public:
  explicit BlockingQueue(size_t capacity = 32)
      : capacity_(capacity), closed_(false) {
    size_.store(0);
  }
  
  void ReCap(size_t capacity) {
    capacity_ = capacity;
  }

  bool Send(const T& elem) {
    int c = -1;
    {
      std::unique_lock<std::mutex> lock(send_mutex_);
      send_cv_.wait(lock, [&] {return size_.load() < capacity_ || closed_;});
      if (closed_) {
        VLOG(5)
            << "WARNING: Sending an element to a closed reader::BlokcingQueue.";
        return false;
      }
      queue_.push_back(elem);
      c = size_.load();
      size_.fetch_add(1);
    }
    if (c + 1 < capacity_) {
      send_cv_.notify_one();
    }

    if (c == 0) {
      std::unique_lock<std::mutex> lock(receive_mutex_);
      receive_cv_.notify_one();
    }
    return true;
  }

  bool Receive(T* elem) {
    int c = -1;
    {
      std::unique_lock<std::mutex> lock(receive_mutex_);
      receive_cv_.wait(lock, [&] {return size_.load() != 0 || closed_;});
      if (size_.load() != 0) {
        *elem = queue_.front();
        queue_.pop_front();
        c = size_.load();
        size_.fetch_sub(1);
      } else {
        return false;
      }
    }
    if (c > 1) {
      receive_cv_.notify_one();
    }
    if (c == capacity_) {
      std::unique_lock<std::mutex> lock(send_mutex_);
      send_cv_.notify_one();
    }
    return true;
  }

  void Close() {
    std::lock_guard<std::mutex> lock1(send_mutex_);
    std::lock_guard<std::mutex> lock2(receive_mutex_);
    closed_ = true;
    send_cv_.notify_all();
    receive_cv_.notify_all();
  }

 private:
  size_t capacity_;
  std::atomic_size_t size_;
  bool closed_;
  std::deque<T> queue_;

  mutable std::mutex send_mutex_;
  mutable std::mutex receive_mutex_;
  mutable std::condition_variable send_cv_;
  mutable std::condition_variable receive_cv_;
};

class DataFeed {
 public:
  DataFeed() {}
  virtual ~DataFeed() {}
  virtual void Init(paddle::framework::DataFeedDesc& data_feed_desc) = 0;
  // for some datafeeds may not be able to implement this interface
  virtual bool CheckFile(const char* filename) {
    LOG(ERROR) << "error: The function CheckFile is not implemented";
    return false;
  }
  virtual bool SetFileList(const std::vector<std::string>& files); 
  virtual bool Start() = 0;
  virtual int Next() = 0;
  virtual void SetBatchSize(int batch) { default_batch_size_ = batch; }
  virtual int GetBatchSize() { return batch_size_; }
  // for subclass with queue
  virtual void SetQueueSize(int queue_size) {
    LOG(ERROR) << "error: The function SetQueueSize is not implemented";
    exit(-1);
  }
  // for subclass with buffer
  virtual void SetBufferSize(int buffer_size) {
    LOG(ERROR) << "error: The function SetBufferSize is not implemented";
    exit(-1);
  }
  virtual const std::vector<std::string>& GetAllSlotAlias() {return all_slots_;}
  virtual const std::vector<std::string>& GetUseSlotAlias() {return use_slots_;}
  std::vector<MixTensor>& GetFeedVec() {return feed_vec_;}
  virtual void AddFeedVar(Variable* var, const std::string& name);
 protected:
  // Check if it is executed in this order:
  //   Init -> SetFileList/BindingMemory -> Start -> Next
  virtual void CheckInit();
  virtual void CheckSetFileList();
  virtual void CheckStart();
  virtual bool PickOneFile(std::string& filename);
  
  static std::vector<std::string> filelist_;
  static size_t file_idx_;
  static std::mutex mutex_for_pick_file_;
  
  std::vector<std::string> use_slots_;
  std::vector<bool> use_slots_is_dense_;

  std::vector<std::string> all_slots_;
  std::vector<std::string> all_slots_type_;
  std::vector<int> use_slots_index_; // -1: not used; >=0: the index of use_slots_
  
  std::vector<MixTensor> feed_vec_;
  
  int default_batch_size_;
  int batch_size_;

  bool finish_init_;
  static bool finish_set_filelist_;
  bool finish_binding_memory_;
  bool finish_start_;
};

template<typename T>
class PrivateQueueDataFeed : public DataFeed {
 public:
  PrivateQueueDataFeed() {}
  virtual ~PrivateQueueDataFeed() {}
  virtual void Init(paddle::framework::DataFeedDesc& data_feed_desc) = 0;
  virtual bool Start();
  virtual int Next(); // no buffer
  virtual void SetQueueSize(int queue_size);

 protected:
  virtual void ReadThread();
  virtual bool ParseOneInstance(T& instance) = 0;
  virtual void AddInstanceToInsVec(T& vec_ins, T& instance, int index) = 0;
  virtual void PutToFeedVec(T& ins_vec) = 0;

  std::thread read_thread_; // the thread for read files
  /* using ifstream one line and one line parse is faster 
   * than using fread one buffer and one buffer parse.
   *   for 601M JingPai data:
   *     ifstream one line and one line parse: 6034 ms
   *     fread one buffer and one buffer parse: 7097 ms */
  std::ifstream file_;
  size_t queue_size_;
  BlockingQueue<T> queue_;
};

class MultiSlotType {
 public:
  MultiSlotType() {}
  ~MultiSlotType() {}
  void Init(std::string& type) {
    CheckType(type);
    if (type_[0] == 'f') {
      float_feasign_.clear();
    } else if (type_[0] == 'u') {
      uint64_feasign_.clear();
    }
    type_ = type;
  }
  void InitOffset() {
    offset_.resize(1);
    // LoDTensor' lod is counted from 0, the size of lod 
    // is one size larger than the size of data.
    offset_[0] = 0;
  }
  std::vector<size_t>& GetOffset() {
    return offset_;
  }
  void AddValue(float v) {
    CheckFloat();
    float_feasign_.push_back(v);
  }
  void AddValue(uint64_t v) {
    CheckUint64();
    uint64_feasign_.push_back(v);
  }
  void AddIns(MultiSlotType& ins) {
    if (ins.GetType()[0] == 'f') { //float
      CheckFloat();
      auto& vec = ins.GetFloatData();
      offset_.push_back(offset_.back() + vec.size());
      float_feasign_.insert(float_feasign_.end(), vec.begin(), vec.end());
    } else if (ins.GetType()[0] == 'u') { //uint64
      CheckUint64();
      auto& vec = ins.GetUint64Data();
      offset_.push_back(offset_.back() + vec.size());
      uint64_feasign_.insert(uint64_feasign_.end(), vec.begin(), vec.end());
    }
  }
  std::vector<float>& GetFloatData() {
    return float_feasign_;
  }
  std::vector<uint64_t>& GetUint64Data() {
    return uint64_feasign_;
  }
  std::string& GetType() {
    return type_;
  }
 private:
  void CheckType(std::string& type) {
    if (type != "uint64" && type != "float") {
      // check in here
      LOG(ERROR) << "error: here is no this type";
      exit(-1);
    }
  }
  void CheckFloat() {
    if (type_[0] != 'f') { //float
      LOG(ERROR) << "error: add " << type_ << " value to float slot";
      exit(-1);
    }
  }
  void CheckUint64() {
    if (type_[0] != 'u') { //uint64
      LOG(ERROR) << "error: add " << type_ << " value to uint64 slot";
      exit(-1);
    }
  }
  std::vector<float> float_feasign_;
  std::vector<uint64_t> uint64_feasign_;
  std::string type_;
  std::vector<size_t> offset_;
};

class MultiSlotDataFeed : public PrivateQueueDataFeed<std::vector<MultiSlotType>> {
 public:
  MultiSlotDataFeed() {}
  virtual ~MultiSlotDataFeed() {}
  virtual void Init(paddle::framework::DataFeedDesc& data_feed_desc);
  virtual bool CheckFile(const char* filename);
 protected:
  virtual void AddInstanceToInsVec(std::vector<MultiSlotType>& vec_ins, 
      std::vector<MultiSlotType>& instance, int index);
  virtual bool ParseOneInstance(std::vector<MultiSlotType>& instance);
  virtual void PutToFeedVec(std::vector<MultiSlotType>& ins_vec);
};

}   // namespace framework
}   // namespace paddle

#endif  // PADDLE_FLUID_FRAMEWORK_DATA_FEED_H_
/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */

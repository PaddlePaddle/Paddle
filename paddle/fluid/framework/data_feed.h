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

#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"

namespace paddle {
namespace framework {
struct Gauc {
  int show, click;
  uint64_t fea;
  std::string lineid;
};

struct Instance {
  std::vector<std::vector<uint64_t>> feed_vec_buffer;
  std::vector<std::vector<int>> feed_vec_lod;
  std::vector<float> other_label;
  std::vector<Gauc> gauc_vec;
};

class DataFeed {
 public:
  DataFeed() : default_batch_size_(1), batch_size_(0), thread_id_(0) {}
  virtual ~DataFeed() {}
  virtual void Init() = 0;
  /*
  * This function will be used to check file format.
  * Considering that this function may be used alone,
  * it does not check anything.
  * */
  virtual bool CheckFile(const char* filename) = 0;
  virtual bool SetFile(const char* filename) = 0;
  virtual bool ReadBatch() = 0;
  virtual const std::vector<uint16_t>& GetAllSlotIds() {
    return all_slot_ids_;
  }

  virtual const std::vector<uint16_t>& GetUseSlotIds() {
    return use_slot_ids_;
  }

  virtual const std::vector<std::string>& GetUseSlotAlias() {
    return use_slot_alias_;
  }

  virtual void AddFeedVar(Variable* var,
                            const std::string& name) = 0;
  virtual void BindScope(Scope* scope) = 0;
  virtual void SetBatchSize(int batch) { default_batch_size_ = batch; }
  virtual int GetBatchSize() { return batch_size_; }
  virtual void SetBufferSize(int buffer_size) {}
  virtual unsigned int GetCurrentEpoch() = 0;
  virtual const char *PickOneFile() = 0;
  virtual void UpdateEpochNum() = 0;
  virtual void Start() = 0;
  virtual int Next() = 0;

  std::vector<LoDTensor*>& GetFeedVec() {
    return feed_vec_;
  }

  virtual std::vector<LoDTensor*>& GetFeedVec(const Instance& ins) {
    LOG(ERROR) << "use defalut get_feed_vec";
    return feed_vec_;
  }

  int GetThreadId() {return thread_id_;}
  void SetThreadId(int thread_id) {thread_id_ = thread_id;}

 protected:
  std::vector<uint16_t> all_slot_ids_;
  std::vector<uint16_t> use_slot_ids_;
  std::vector<std::string> use_slot_alias_;
  std::vector<LoDTensor*> feed_vec_;
  int default_batch_size_;
  int batch_size_;
  int thread_id_;
};

class MultiSlotDataFeed : public DataFeed {
 public:
  MultiSlotDataFeed();
  MultiSlotDataFeed(const MultiSlotDataFeed& data_feed);

 public:
  virtual ~MultiSlotDataFeed() {}
  virtual void Init();
  virtual bool ReadBatch();
  virtual void AddFeedVar(Variable* feed, const std::string& name);
  virtual void BindScope(Scope* scope) {}
  virtual bool SetFile(const char* filename);
  virtual bool CheckFile(const char* filename) {
    // TODO(xxx)
    return false;
  }
  void SetBatchSize(int batch) {batch_size_ = batch;}
  unsigned int GetCurrentEpoch() {return s_current_epoch_;}
  void UpdateEpochNum();
  void Start();
  int Next();

 public:
  void SetFieldNames(const std::vector<std::string>& field_names);

 public:
  static void SetFileList(const char* filelist);

 private:
  const char* PickOneFile();

 private:
  char* file_content_buffer_;
  char* file_content_buffer_ptr_;
  int* batch_id_buffer_;
  int* label_ptr_;
  int file_size_;
  std::vector<std::string> field_names_;
  std::shared_ptr<char> file_content_buffer_host_;
  std::shared_ptr<int> batch_id_host_;
  std::shared_ptr<int> label_host_;

  static std::vector<std::string> s_filelist_;
  static std::mutex s_locker_for_pick_file_;
  static unsigned int s_current_file_idx_;
  static size_t s_current_finished_file_cnt_;
  static unsigned int s_current_epoch_;
  static int s_current_save_epoch_;
  static std::mutex s_locker_epoch_start_;
  static std::condition_variable s_condition_epoch_start_;
  static bool s_epoch_start_flag_;
};

}   // namespace framework
}   // namespace paddle

#endif  // PADDLE_FLUID_FRAMEWORK_DATA_FEED_H_
/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */

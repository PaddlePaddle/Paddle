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

#pragma once

#include <fstream>
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <thread>  // NOLINT
#include <vector>

#include "paddle/fluid/framework/data_feed.pb.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/reader.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/operators/reader/blocking_queue.h"
#include "paddle/fluid/string/string_helper.h"

namespace paddle {
namespace framework {

// DataFeed is the base virtual class for all ohther DataFeeds.
// It is used to read files and parse the data for subsequent trainer.
// Example:
//   DataFeed* reader =
//   paddle::framework::DataFeedFactory::CreateDataFeed(data_feed_name);
//   reader->Init(data_feed_desc); // data_feed_desc is a protobuf object
//   reader->SetFileList(filelist);
//   const std::vector<std::string> & use_slot_alias =
//   reader->GetUseSlotAlias();
//   for (auto name: use_slot_alias){ // for binding memory
//     reader->AddFeedVar(scope->Var(name), name);
//   }
//   reader->Start();
//   while (reader->Next()) {
//      // trainer do something
//   }
class DataFeed {
 public:
  DataFeed() {}
  virtual ~DataFeed() {}
  virtual void Init(const paddle::framework::DataFeedDesc& data_feed_desc) = 0;
  virtual bool CheckFile(const char* filename) {
    PADDLE_THROW("This function(CheckFile) is not implemented.");
  }
  // Set filelist for DataFeed.
  // Pay attention that it must init all readers before call this function.
  // Otherwise, Init() function will init finish_set_filelist_ flag.
  virtual bool SetFileList(const std::vector<std::string>& files);
  virtual bool Start() = 0;
  // The trainer calls the Next() function, and the DataFeed will load a new
  // batch to the feed_vec. The return value of this function is the batch
  // size of the current batch.
  virtual int Next() = 0;
  // Get all slots' alias which defined in protofile
  virtual const std::vector<std::string>& GetAllSlotAlias() {
    return all_slots_;
  }
  // Get used slots' alias which defined in protofile
  virtual const std::vector<std::string>& GetUseSlotAlias() {
    return use_slots_;
  }
  // This function is used for binding feed_vec memory
  virtual void AddFeedVar(Variable* var, const std::string& name);

 protected:
  // The following three functions are used to check if it is executed in this
  // order:
  //   Init() -> SetFileList() -> Start() -> Next()
  virtual void CheckInit();
  virtual void CheckSetFileList();
  virtual void CheckStart();
  virtual void SetBatchSize(
      int batch);  // batch size will be set in Init() function
  // This function is used to pick one file from the global filelist(thread
  // safe).
  virtual bool PickOneFile(std::string* filename);

  static std::vector<std::string> filelist_;
  static size_t file_idx_;
  static std::mutex mutex_for_pick_file_;

  // the alias of used slots, and its order is determined by
  // data_feed_desc(proto object)
  std::vector<std::string> use_slots_;
  std::vector<bool> use_slots_is_dense_;

  // the alias of all slots, and its order is determined by data_feed_desc(proto
  // object)
  std::vector<std::string> all_slots_;
  std::vector<std::string> all_slots_type_;
  std::vector<int>
      use_slots_index_;  // -1: not used; >=0: the index of use_slots_

  // The data read by DataFeed will be stored here
  std::vector<LoDTensor*> feed_vec_;

  // the batch size defined by user
  int default_batch_size_;
  // current batch size
  int batch_size_;

  bool finish_init_;
  static bool finish_set_filelist_;
  bool finish_start_;
  std::string pipe_command_;
};

// PrivateQueueDataFeed is the base virtual class for ohther DataFeeds.
// It use a read-thread to read file and parse data to a private-queue
// (thread level), and get data from this queue when trainer call Next().
template <typename T>
class PrivateQueueDataFeed : public DataFeed {
 public:
  PrivateQueueDataFeed() {}
  virtual ~PrivateQueueDataFeed() {}
  virtual void Init(const paddle::framework::DataFeedDesc& data_feed_desc) = 0;
  virtual bool Start();
  virtual int Next();

 protected:
  // The thread implementation function for reading file and parse.
  virtual void ReadThread();
  // This function is used to set private-queue size, and the most
  // efficient when the queue size is close to the batch size.
  virtual void SetQueueSize(int queue_size);
  // The reading and parsing method called in the ReadThread.
  virtual bool ParseOneInstance(T* instance) = 0;
  virtual bool ParseOneInstanceFromPipe(T* instance) = 0;
  // This function is used to put instance to vec_ins
  virtual void AddInstanceToInsVec(T* vec_ins, const T& instance,
                                   int index) = 0;
  // This function is used to put ins_vec to feed_vec
  virtual void PutToFeedVec(const T& ins_vec) = 0;

  // The thread for read files
  std::thread read_thread_;
  // using ifstream one line and one line parse is faster
  // than using fread one buffer and one buffer parse.
  //   for a 601M real data:
  //     ifstream one line and one line parse: 6034 ms
  //     fread one buffer and one buffer parse: 7097 ms
  std::ifstream file_;
  std::shared_ptr<FILE> fp_;
  size_t queue_size_;
  string::LineFileReader reader_;
  // The queue for store parsed data
  std::unique_ptr<paddle::operators::reader::BlockingQueue<T>> queue_;
};

// This class define the data type of instance(ins_vec) in MultiSlotDataFeed
class MultiSlotType {
 public:
  MultiSlotType() {}
  ~MultiSlotType() {}
  void Init(const std::string& type) {
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
  const std::vector<size_t>& GetOffset() const { return offset_; }
  void AddValue(const float v) {
    CheckFloat();
    float_feasign_.push_back(v);
  }
  void AddValue(const uint64_t v) {
    CheckUint64();
    uint64_feasign_.push_back(v);
  }
  void AddIns(const MultiSlotType& ins) {
    if (ins.GetType()[0] == 'f') {  // float
      CheckFloat();
      auto& vec = ins.GetFloatData();
      offset_.push_back(offset_.back() + vec.size());
      float_feasign_.insert(float_feasign_.end(), vec.begin(), vec.end());
    } else if (ins.GetType()[0] == 'u') {  // uint64
      CheckUint64();
      auto& vec = ins.GetUint64Data();
      offset_.push_back(offset_.back() + vec.size());
      uint64_feasign_.insert(uint64_feasign_.end(), vec.begin(), vec.end());
    }
  }
  const std::vector<float>& GetFloatData() const { return float_feasign_; }
  const std::vector<uint64_t>& GetUint64Data() const { return uint64_feasign_; }
  const std::string& GetType() const { return type_; }

 private:
  void CheckType(const std::string& type) const {
    PADDLE_ENFORCE((type == "uint64") || (type == "float"),
                   "There is no this type<%s>.", type);
  }
  void CheckFloat() const {
    PADDLE_ENFORCE(type_[0] == 'f', "Add %s value to float slot.", type_);
  }
  void CheckUint64() const {
    PADDLE_ENFORCE(type_[0] == 'u', "Add %s value to uint64 slot.", type_);
  }
  std::vector<float> float_feasign_;
  std::vector<uint64_t> uint64_feasign_;
  std::string type_;
  std::vector<size_t> offset_;
};

// This DataFeed is used to feed multi-slot type data.
// The format of multi-slot type data:
//   [n feasign_0 feasign_1 ... feasign_n]*
class MultiSlotDataFeed
    : public PrivateQueueDataFeed<std::vector<MultiSlotType>> {
 public:
  MultiSlotDataFeed() {}
  virtual ~MultiSlotDataFeed() {}
  virtual void Init(const paddle::framework::DataFeedDesc& data_feed_desc);
  virtual bool CheckFile(const char* filename);
  // virtual void ReadThread();

 protected:
  virtual void ReadThread();
  virtual void AddInstanceToInsVec(std::vector<MultiSlotType>* vec_ins,
                                   const std::vector<MultiSlotType>& instance,
                                   int index);
  virtual bool ParseOneInstance(std::vector<MultiSlotType>* instance);
  virtual bool ParseOneInstanceFromPipe(std::vector<MultiSlotType>* instance);
  virtual void PutToFeedVec(const std::vector<MultiSlotType>& ins_vec);

 private:
  BatchGenerator batch_gen_;
};
}  // namespace framework
}  // namespace paddle

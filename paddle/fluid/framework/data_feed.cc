/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <utility>
#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"

#include "gflags/gflags.h"
#include "paddle/fluid/framework/feed_fetch_method.h"
#include "paddle/fluid/framework/feed_fetch_type.h"
#include "paddle/fluid/framework/lod_rank_table.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/reader.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/profiler.h"
#include "paddle/fluid/framework/data_feed.h"


namespace paddle {
namespace framework {

std::vector<std::string> DataFeed::filelist_;
size_t DataFeed::file_idx_;
std::mutex DataFeed::mutex_for_pick_file_;

void DataFeed::AddFeedVar(Variable* var, const std::string& name) {
  CheckInit();
  for (size_t i = 0; i < use_slots_.size(); ++i) {
    if (name == use_slots_[i]) {
      if (use_slots_is_dense_[i]) {
        feed_vec_[i] = MixTensor(var->GetMutable<Tensor>());
      } else {
        feed_vec_[i] = MixTensor(var->GetMutable<LoDTensor>());
      }
    }
  }
}

bool DataFeed::SetFileList(const std::vector<std::string>& files) {
  CheckInit();
  if (files.size() == 0) {
    LOG(ERROR) << "error: you have set an empty filelist";
    return false;
  }
  filelist_.assign(files.begin(), files.end());
  file_idx_ = 0;

  finish_set_filelist_ = true;
  return true;
}

bool DataFeed::PickOneFile(std::string& filename) {
  std::unique_lock<std::mutex> lock(mutex_for_pick_file_);
  if (file_idx_ == filelist_.size()) {
    return false;
  }
  filename = filelist_[file_idx_++];
  return true;
}

void DataFeed::CheckInit() {
  if (finish_init_) {return;}
  LOG(ERROR) << "error: initialization did not succeed";
  exit(-1);
}

void DataFeed::CheckSetFileList() {
  if (finish_set_filelist_) {return;}
  LOG(ERROR) << "error: set filelist did not succeed";
  exit(-1);
}

void DataFeed::CheckStart() {
  if (finish_start_) {return;}
  LOG(ERROR) << "error: Datafeed has not started running yet";
  exit(-1);
}

template<typename T>
void PrivateQueueDataFeed<T>::SetQueueSize(int queue_size) {
  CheckInit();
  if (queue_size <= 0) {
    LOG(ERROR) << "error: illegal queue size: " << queue_size;
    return;
  }
  queue_size_ = queue_size;
  queue_.ReCap(queue_size_);
}

template<typename T>
bool PrivateQueueDataFeed<T>::Start() {
  CheckSetFileList();
  read_thread_ = std::thread(&PrivateQueueDataFeed::ReadThread, this);
  read_thread_.detach();

  finish_start_ = true;
  return true;
} 

template<typename T>
void PrivateQueueDataFeed<T>::ReadThread(){
  std::string filename;
  while (PickOneFile(filename)) {
    file_.open(filename.c_str()); // is_text_feed
    if (!file_.good()) {
      LOG(ERROR) << "error: open file<" << filename << "> fail";
      continue;
    }
    T instance;
    while (ParseOneInstance(instance)) {
      queue_.Send(instance);
    }
    file_.close();
  }
  queue_.Close();
}

template<typename T>
int PrivateQueueDataFeed<T>::Next(){
  CheckStart();
  int index = 0;
  T instance;
  T ins_vec;
  while (index < default_batch_size_) {
    if (!queue_.Receive(&instance)) {
      break;
    }
    AddInstanceToInsVec(ins_vec, instance, index++);
  }
  batch_size_ = index;
  if (batch_size_ != 0) {
    PutToFeedVec(ins_vec);
  }
  return batch_size_;
}

void MultiSlotDataFeed::Init(paddle::framework::DataFeedDesc& data_feed_desc) {
  finish_init_ = false;
  finish_set_filelist_ = false;
  finish_start_ = false;
  
  if (!data_feed_desc.has_multi_slot_desc()){
    LOG(ERROR) << "error: multi_slot_desc has not been set";
    exit(-1);
  }
  paddle::framework::MultiSlotDesc multi_slot_desc = data_feed_desc.multi_slot_desc();
  SetBatchSize(data_feed_desc.batch());
  size_t all_slot_num = multi_slot_desc.slots_size();
  all_slots_.resize(all_slot_num);
  all_slots_type_.resize(all_slot_num);
  use_slots_index_.resize(all_slot_num);
  use_slots_.clear();
  use_slots_is_dense_.clear();
  for (size_t i = 0; i < all_slot_num; ++i) {
    auto& slot = multi_slot_desc.slots(i);
    all_slots_[i] = slot.name();
    all_slots_type_[i] = slot.type();
    use_slots_index_[i] = slot.use() ? use_slots_.size() : -1;
    if (slot.use()) {
      use_slots_.push_back(all_slots_[i]);
      use_slots_is_dense_.push_back(slot.dense());
    }
  }
  feed_vec_.resize(use_slots_.size());

  finish_init_ = true;
}

bool MultiSlotDataFeed::CheckFile(const char* filename) {
  // check with protobuf ?
  std::cerr << "Check error" << std::endl;
  return false;
}

bool MultiSlotDataFeed::ParseOneInstance(std::vector<MultiSlotType>& instance) {
  std::string line;
  if (getline(file_, line)) {
    int use_slots_num = use_slots_.size();
    instance.resize(use_slots_num);
    //parse line
    const char* str = line.c_str();
    char* endptr = (char*)str;
    int pos = 0;
    for (size_t i = 0; i < use_slots_index_.size(); ++i) {
      int idx = use_slots_index_[i];
      int num = (int)strtol(&str[pos], &endptr, 10);
      if (num == 0) {
        LOG(ERROR) << "error: the number of ids can not be zero, you need padding it";
        exit(-1);
      }
      if (idx != -1) {
        instance[idx].Init(all_slots_type_[i]);
        if (instance[idx].GetType()[0] == 'f') { // float
          for (int j = 0; j < num; ++j) {
            float feasign = (float)strtof(endptr, &endptr);
            instance[idx].AddValue(feasign);
          }
        } else if (instance[idx].GetType()[0] == 'u'){ // uint64
          for (int j = 0; j < num; ++j) {
            uint64_t feasign = (uint64_t)strtoull(endptr, &endptr, 10);
            instance[idx].AddValue(feasign);
          }
        }
        pos = endptr - str;
      } else {
        for (int j = 0; j <= num; ++j) {
          pos = line.find_first_of(' ', pos + 1);
        }
      }
    }
  } else {
    return false;
  }
  return true;
}

void MultiSlotDataFeed::AddInstanceToInsVec(std::vector<MultiSlotType>& ins_vec,
   std::vector<MultiSlotType>& instance, int index) {
  if (index == 0) {
    ins_vec.resize(instance.size());
    for (size_t i = 0; i < instance.size(); ++i) {
      ins_vec[i].Init(instance[i].GetType());
      ins_vec[i].InitOffset();
    }
  }
  for (size_t i = 0; i < instance.size(); ++i){
    ins_vec[i].AddIns(instance[i]);
  }
}

void MultiSlotDataFeed::PutToFeedVec(std::vector<MultiSlotType>& ins_vec) {
  for (size_t i = 0; i < use_slots_.size(); ++i) {
    auto& type = ins_vec[i].GetType();
    auto& offset = ins_vec[i].GetOffset();
    int total_instance = static_cast<int>(offset.back());
    if (type[0] == 'f') { // float
      auto& feasign = ins_vec[i].GetFloatData();
      if (feed_vec_[i].IsDense()) {
        int size_in_each_batch = total_instance / batch_size_;
        float* tensor_ptr = feed_vec_[i].GetTensor()->
          mutable_data<float>({batch_size_, size_in_each_batch}, platform::CPUPlace());
        memcpy(tensor_ptr, &feasign[0], total_instance * sizeof(float));
      } else {
        float* tensor_ptr = feed_vec_[i].GetLoDTensor()->
          mutable_data<float>({total_instance, 1}, platform::CPUPlace());
        memcpy(tensor_ptr, &feasign[0], total_instance * sizeof(float));
        LoD data_lod{offset};
        feed_vec_[i].GetLoDTensor()->set_lod(data_lod);
      }
    } else if (type[0] == 'u') { // uint64
      // no uint64_t type in paddle
      auto& feasign = ins_vec[i].GetUint64Data();
      if (feed_vec_[i].IsDense()) {
        int size_in_each_batch = total_instance / batch_size_;
        int64_t* tensor_ptr = feed_vec_[i].GetTensor()->
          mutable_data<int64_t>({batch_size_, size_in_each_batch}, platform::CPUPlace());
        memcpy(tensor_ptr, &feasign[0], total_instance * sizeof(int64_t));
      } else {
        int64_t* tensor_ptr = feed_vec_[i].GetLoDTensor()->
          mutable_data<int64_t>({total_instance, 1}, platform::CPUPlace());
        memcpy(tensor_ptr, &feasign[0], total_instance * sizeof(int64_t));
        LoD data_lod{offset};
        feed_vec_[i].GetLoDTensor()->set_lod(data_lod);
      }
    }
  }
}

}   // namespace framework
}   // namespace paddle
/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */


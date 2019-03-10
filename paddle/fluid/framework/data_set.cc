/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License. */

#include "paddle/fluid/framework/data_set.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"
#include "paddle/fluid/framework/data_feed_factory.h"

namespace paddle {
namespace framework {

Dataset::Dataset() { thread_num_ = 1; }

void Dataset::SetFileList(const std::vector<std::string>& filelist) {
  VLOG(3) << "filelist size: " << filelist.size();
  filelist_ = filelist;
  int file_cnt = filelist_.size();
  if (thread_num_ > file_cnt) {
    VLOG(1) << "DataSet thread num = " << thread_num_
            << ", file num = " << file_cnt
            << ". Changing DataSet thread num = " << file_cnt;
    thread_num_ = file_cnt;
  }
}

// buggy here, a user should set filelist first before this function
// not user friendly
void Dataset::SetThreadNum(int thread_num) {
  int file_cnt = filelist_.size();
  if (file_cnt != 0 && thread_num > file_cnt) {
    VLOG(1) << "DataSet thread num = " << thread_num
            << ", file num = " << file_cnt
            << ". Changing DataSet thread num = " << file_cnt;
    thread_num = file_cnt;
  }
  thread_num_ = thread_num;
}

void Dataset::SetTrainerNum(int trainer_num) { trainer_num_ = trainer_num; }

void Dataset::SetDataFeedDesc(const std::string& data_feed_desc_str) {
  google::protobuf::TextFormat::ParseFromString(data_feed_desc_str,
                                                &data_feed_desc_);
}

const std::vector<std::shared_ptr<paddle::framework::DataFeed>>&
Dataset::GetReaders() {
  return readers_;
}

void Dataset::LoadIntoMemory() {
  if (readers_.size() == 0) {
    CreateReaders();
  }
  std::vector<std::thread> load_threads;
  for (int64_t i = 0; i < thread_num_; ++i) {
    load_threads.push_back(std::thread(
        &paddle::framework::DataFeed::LoadIntoMemory, readers_[i].get()));
  }
  for (std::thread& t : load_threads) {
    t.join();
  }
}

void Dataset::LocalShuffle() {
  if (readers_.size() == 0) {
    CreateReaders();
  }
  std::vector<std::thread> local_shuffle_threads;
  for (int64_t i = 0; i < thread_num_; ++i) {
    local_shuffle_threads.push_back(std::thread(
        &paddle::framework::DataFeed::LocalShuffle, readers_[i].get()));
  }
  for (std::thread& t : local_shuffle_threads) {
    t.join();
  }
}

// todo global shuffle
void Dataset::GlobalShuffle() {
  /*
  auto fleet_ptr = FleetWrapper::GetInstance();
  fleet_ptr->registe_client2client_msg_handler(0,
    [this](int msg_type, int client_id, const std::string& msg) -> int {
    return this->ReceiveFromClient(msg_type, client_id, msg);
  });
  if (readers_.size() == 0) {
    CreateReaders();
  }
  std::vector<std::thread> global_shuffle_threads;
  for (int64_t i = 0; i < thread_num_; ++i) {
    global_shuffle_threads.push_back(std::thread(&paddle::framework::DataFeed::GlobalShuffle,
                                     readers_[i].get(), trainer_num_));
  }
  for (std::thread& t : global_shuffle_threads) {
    t.join();
  }*/
}

void Dataset::CreateReaders() {
  VLOG(3) << "Calling CreateReaders()";
  CHECK(thread_num_ > 0) << "thread_num should > 0";
  VLOG(3) << "thread_num in Readers: " << thread_num_;
  VLOG(3) << "readers size: " << readers_.size();
  if (readers_.size() != 0) {
    return;
  }
  VLOG(3) << "data feed class name: " << data_feed_desc_.name();
  for (int64_t i = 0; i < thread_num_; ++i) {
    readers_.push_back(DataFeedFactory::CreateDataFeed(data_feed_desc_.name()));
    readers_.back()->Init(data_feed_desc_);
  }
  VLOG(3) << "Filelist size in readers: " << filelist_.size();
  readers_[0]->SetFileList(filelist_);
}

int Dataset::ReceiveFromClient(int msg_type, int client_id,
                               const std::string& msg) {
  // can also use hash
  // int64_t index = paddle::ps::local_random_engine()() % thread_num_;
  int64_t index = 0;
  readers_[index]->PutInsToChannel(msg);
  return 0;
}

}  // end namespace framework
}  // end namespace paddle

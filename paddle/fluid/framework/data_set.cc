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
#include <random>
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"
#include "paddle/fluid/framework/data_feed_factory.h"

namespace paddle {
namespace framework {

template <typename T>
DatasetImpl<T>::DatasetImpl() {
  thread_num_ = 1;
}

template <typename T>
void DatasetImpl<T>::SetFileList(const std::vector<std::string>& filelist) {
  VLOG(3) << "filelist size: " << filelist.size();
  filelist_ = filelist;
  /*
  int file_cnt = filelist_.size();
  if (thread_num_ > file_cnt) {
    VLOG(1) << "DataSet thread num = " << thread_num_
            << ", file num = " << file_cnt
            << ". Changing DataSet thread num = " << file_cnt;
    thread_num_ = file_cnt;
  }*/
}

// buggy here, a user should set filelist first before this function
// not user friendly
template <typename T>
void DatasetImpl<T>::SetThreadNum(int thread_num) {
  int file_cnt = filelist_.size();
  if (file_cnt != 0 && thread_num > file_cnt) {
    VLOG(1) << "DataSet thread num = " << thread_num
            << ", file num = " << file_cnt
            << ". Changing DataSet thread num = " << file_cnt;
    thread_num = file_cnt;
  }
  thread_num_ = thread_num;
}

template <typename T>
void DatasetImpl<T>::SetTrainerNum(int trainer_num) {
  trainer_num_ = trainer_num;
}

template <typename T>
void DatasetImpl<T>::SetDataFeedDesc(const std::string& data_feed_desc_str) {
  google::protobuf::TextFormat::ParseFromString(data_feed_desc_str,
                                                &data_feed_desc_);
}

template <typename T>
std::vector<std::shared_ptr<paddle::framework::DataFeed>>&
DatasetImpl<T>::GetReaders() {
  return readers_;
}

template <typename T>
void DatasetImpl<T>::LoadIntoMemory() {
  VLOG(3) << "DatasetImpl<T>::LoadIntoMemory() begin";
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
  VLOG(3) << "DatasetImpl<T>::LoadIntoMemory() end";
}

template <typename T>
void DatasetImpl<T>::LocalShuffle() {
  VLOG(3) << "DatasetImpl<T>::LocalShuffle() begin";
  if (readers_.size() == 0) {
    CreateReaders();
  }
  // if it is not InMemory, memory_data_ is empty
  std::random_shuffle(memory_data_.begin(), memory_data_.end());

  std::vector<std::thread> local_shuffle_threads;
  for (int64_t i = 0; i < thread_num_; ++i) {
    local_shuffle_threads.push_back(std::thread(
        &paddle::framework::DataFeed::LocalShuffle, readers_[i].get()));
  }
  for (std::thread& t : local_shuffle_threads) {
    t.join();
  }
  std::vector<T>().swap(memory_data_);
  VLOG(3) << "DatasetImpl<T>::LocalShuffle() end";
}

template <typename T>
void DatasetImpl<T>::GlobalShuffle() {
  VLOG(3) << "DatasetImpl<T>::GlobalShuffle() begin";
  if (readers_.size() == 0) {
    CreateReaders();
  }
  // if it is not InMemory, memory_data_ is empty
  std::random_shuffle(memory_data_.begin(), memory_data_.end());
  auto fleet_ptr = FleetWrapper::GetInstance();
  VLOG(3) << "RegisterClientToClientMsgHandler";
  fleet_ptr->RegisterClientToClientMsgHandler(
      0, [this](int msg_type, int client_id, const std::string& msg) -> int {
        return this->ReceiveFromClient(msg_type, client_id, msg);
      });
  VLOG(3) << "start global shuffle threads";
  std::vector<std::thread> global_shuffle_threads;
  for (int i = 0; i < thread_num_; ++i) {
    global_shuffle_threads.push_back(std::thread(
        &paddle::framework::DataFeed::GlobalShuffle, readers_[i].get()));
  }
  for (std::thread& t : global_shuffle_threads) {
    t.join();
  }
  VLOG(3) << "DatasetImpl<T>::GlobalShuffle() end";
}

template <typename T>
void DatasetImpl<T>::CreateReaders() {
  VLOG(3) << "Calling CreateReaders()";
  CHECK(thread_num_ > 0) << "thread_num should > 0";
  VLOG(3) << "thread_num in Readers: " << thread_num_;
  VLOG(3) << "readers size: " << readers_.size();
  if (readers_.size() != 0) {
    return;
  }
  VLOG(3) << "data feed class name: " << data_feed_desc_.name();
  for (int i = 0; i < thread_num_; ++i) {
    readers_.push_back(DataFeedFactory::CreateDataFeed(data_feed_desc_.name()));
    readers_.back()->Init(data_feed_desc_);
    readers_.back()->SetMemoryData(&memory_data_);
    readers_.back()->SetMemoryDataMutex(&mutex_for_update_memory_data_);
    readers_.back()->SetThreadId(i);
    readers_.back()->SetThreadNum(thread_num_);
    readers_.back()->SetTrainerNum(trainer_num_);
  }
  VLOG(3) << "Filelist size in readers: " << filelist_.size();
  readers_[0]->SetFileList(filelist_);
}

template <typename T>
void DatasetImpl<T>::DestroyReaders() {
  VLOG(3) << "Calling DestroyReaders()";
  // clear memory_data_ before fill it
  // because if LoadIntoMemory but no Shuffle,
  // memory_data_ has empty data which has been std::move to channel
  if (memory_data_.size() != 0) {
    std::vector<T>().swap(memory_data_);
  }
  std::vector<std::thread> fill_threads;
  for (int i = 0; i < thread_num_; ++i) {
    fill_threads.push_back(
        std::thread(&paddle::framework::DataFeed::FillChannelToMemoryData,
                    readers_[i].get()));
  }
  for (std::thread& t : fill_threads) {
    t.join();
  }
  std::vector<std::shared_ptr<paddle::framework::DataFeed>>().swap(readers_);
  LOG(WARNING) << "readers size: " << readers_.size();
}

template <typename T>
int DatasetImpl<T>::ReceiveFromClient(int msg_type, int client_id,
                                      const std::string& msg) {
  // todo random
  // int64_t index = paddle::ps::local_random_engine()() % thread_num_;
  int64_t index = 0;
  readers_[index]->PutInsToChannel(msg);
  return 0;
}

// explicit instantiation
template class DatasetImpl<std::vector<MultiSlotType>>;

}  // end namespace framework
}  // end namespace paddle

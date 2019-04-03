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
#include "paddle/fluid/framework/io/fs.h"
#include "paddle/fluid/platform/timer.h"

#if defined _WIN32 || defined __APPLE__
#else
#define _LINUX
#endif

namespace paddle {
namespace framework {

// constructor
template <typename T>
DatasetImpl<T>::DatasetImpl() {
  thread_num_ = 1;
  trainer_num_ = 1;
  file_idx_ = 0;
}

// set filelist, file_idx_ will reset to zero.
template <typename T>
void DatasetImpl<T>::SetFileList(const std::vector<std::string>& filelist) {
  VLOG(3) << "filelist size: " << filelist.size();
  filelist_ = filelist;
  file_idx_ = 0;
}

// set expect thread num. actually it may change
template <typename T>
void DatasetImpl<T>::SetThreadNum(int thread_num) {
  VLOG(3) << "SetThreadNum thread_num=" << thread_num;
  thread_num_ = thread_num;
}

// if you run distributed, and want to do global shuffle,
// set this before global shuffle.
// be sure you call CreateReaders before SetTrainerId
template <typename T>
void DatasetImpl<T>::SetTrainerId(int trainer_id) {
  trainer_id_ = trainer_id;
  for (auto reader : readers_) {
    reader->SetTrainerId(trainer_id);
  }
}

// if you run distributed, and want to do global shuffle,
// set this before global shuffle.
// be sure you call CreateReaders before SetTrainerNum
template <typename T>
void DatasetImpl<T>::SetTrainerNum(int trainer_num) {
  trainer_num_ = trainer_num;
  // should inform reader of trainer_num directly
  for (auto reader : readers_) {
    reader->SetTrainerNum(trainer_num);
  }
}

// if you run distributed, and want to do global shuffle,
// set this before global shuffle.
// be sure you call CreateReaders before SetFleetSendBatchSize
template <typename T>
void DatasetImpl<T>::SetFleetSendBatchSize(int64_t size) {
  fleet_send_batch_size_ = size;
  for (auto reader : readers_) {
    reader->SetFleetSendBatchSize(size);
  }
}

template <typename T>
void DatasetImpl<T>::SetHdfsConfig(const std::string& fs_name,
                                   const std::string& fs_ugi) {
  fs_name_ = fs_name;
  fs_ugi_ = fs_ugi;
  std::string cmd = std::string("hadoop fs");
  cmd += " -D fs.default.name=" + fs_name;
  cmd += " -D hadoop.job.ugi=" + fs_ugi;
  paddle::framework::hdfs_set_command(cmd);
}

template <typename T>
void DatasetImpl<T>::SetDataFeedDesc(const std::string& data_feed_desc_str) {
  google::protobuf::TextFormat::ParseFromString(data_feed_desc_str,
                                                &data_feed_desc_);
}

// readers_.size() may not be equal to thread_num_,
// it changes when filelist_.size() < thread_num_
template <typename T>
std::vector<std::shared_ptr<paddle::framework::DataFeed>>&
DatasetImpl<T>::GetReaders() {
  return readers_;
}

// if sent message between workers, should first call this function
template <typename T>
void DatasetImpl<T>::RegisterClientToClientMsgHandler() {
  auto fleet_ptr = FleetWrapper::GetInstance();
  VLOG(3) << "RegisterClientToClientMsgHandler";
  fleet_ptr->RegisterClientToClientMsgHandler(
      0, [this](int msg_type, int client_id, const std::string& msg) -> int {
        return this->ReceiveFromClient(msg_type, client_id, msg);
      });
  VLOG(3) << "RegisterClientToClientMsgHandler done";
}

// load data into memory, Dataset hold this memory,
// which will later be fed into readers' channel
template <typename T>
void DatasetImpl<T>::LoadIntoMemory() {
  VLOG(3) << "DatasetImpl<T>::LoadIntoMemory() begin";
  platform::Timer timeline;
  timeline.Start();
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
  timeline.Pause();
  VLOG(3) << "DatasetImpl<T>::LoadIntoMemory() end"
          << ", memory data size=" << memory_data_.size()
          << ", cost time=" << timeline.ElapsedSec() << " seconds";
}

// release memory data
template <typename T>
void DatasetImpl<T>::ReleaseMemory() {
  VLOG(3) << "DatasetImpl<T>::ReleaseMemory() begin";
  std::vector<T>().swap(memory_data_);
  VLOG(3) << "DatasetImpl<T>::ReleaseMemory() end";
}

// do local shuffle
template <typename T>
void DatasetImpl<T>::LocalShuffle() {
  VLOG(3) << "DatasetImpl<T>::LocalShuffle() begin";
  platform::Timer timeline;
  timeline.Start();
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
  timeline.Pause();
  VLOG(3) << "DatasetImpl<T>::LocalShuffle() end, cost time="
          << timeline.ElapsedSec() << " seconds";
}

template <typename T>
void DatasetImpl<T>::GlobalShuffle() {
  VLOG(3) << "DatasetImpl<T>::GlobalShuffle() begin";
  platform::Timer timeline;
  timeline.Start();
  if (readers_.size() == 0) {
    CreateReaders();
  }
  // if it is not InMemory, memory_data_ is empty
  std::random_shuffle(memory_data_.begin(), memory_data_.end());
  VLOG(3) << "start global shuffle threads";
  std::vector<std::thread> global_shuffle_threads;
  for (int i = 0; i < thread_num_; ++i) {
    global_shuffle_threads.push_back(std::thread(
        &paddle::framework::DataFeed::GlobalShuffle, readers_[i].get()));
  }
  for (std::thread& t : global_shuffle_threads) {
    t.join();
  }
  std::vector<T>().swap(memory_data_);
  timeline.Pause();
  VLOG(3) << "DatasetImpl<T>::GlobalShuffle() end, cost time="
          << timeline.ElapsedSec() << " seconds";
}

template <typename T>
void DatasetImpl<T>::CreateReaders() {
  VLOG(3) << "Calling CreateReaders()";
  CHECK(thread_num_ > 0) << "thread_num should > 0";
  int file_cnt = filelist_.size();
  int memory_data_size = memory_data_.size();
  if (memory_data_size != 0 && thread_num_ > memory_data_size) {
    VLOG(3) << "Dataset thread num = " << thread_num_
            << ", memory data size = " << memory_data_size
            << ". Changing Dataset thread num = " << memory_data_size;
    thread_num_ = memory_data_size;
  } else if (file_cnt != 0 && thread_num_ > file_cnt) {
    VLOG(3) << "Dataset thread num = " << thread_num_
            << ", file num = " << file_cnt
            << ". Changing Dataset thread num = " << file_cnt;
    thread_num_ = file_cnt;
  }
  VLOG(3) << "thread_num in Readers: " << thread_num_;
  VLOG(3) << "readers size: " << readers_.size();
  VLOG(3) << "Filelist size in readers: " << filelist_.size();
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
    readers_.back()->SetFileListMutex(&mutex_for_pick_file_);
    readers_.back()->SetFileListIndex(&file_idx_);
    readers_.back()->SetFileList(filelist_);
  }
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
  VLOG(3) << "readers size: " << readers_.size();
  // if memory_data_ is empty, which means it's not InMemory mode,
  // so the next epoch should read all data again
  if (memory_data_.size() == 0) {
    file_idx_ = 0;
  }
}

template <typename T>
int DatasetImpl<T>::ReceiveFromClient(int msg_type, int client_id,
                                      const std::string& msg) {
#ifdef _LINUX
  VLOG(3) << "ReceiveFromClient msg_type=" << msg_type
          << ", client_id=" << client_id << ", msg length=" << msg.length();
  auto fleet_ptr = FleetWrapper::GetInstance();
  int64_t index = rand_r(&rand_seed) % thread_num_;
  VLOG(3) << "ramdom index=" << index;
  readers_[index]->PutInsToChannel(msg);
#endif
  return 0;
}

// explicit instantiation
template class DatasetImpl<std::vector<MultiSlotType>>;

}  // end namespace framework
}  // end namespace paddle

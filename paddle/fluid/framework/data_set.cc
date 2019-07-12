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
  VLOG(3) << "DatasetImpl<T>::DatasetImpl() constructor";
  thread_num_ = 1;
  trainer_num_ = 1;
  channel_num_ = 1;
  file_idx_ = 0;
  cur_channel_ = 0;
  fleet_send_batch_size_ = 80000;
  fleet_send_sleep_seconds_ = 2;
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
// be sure you call CreateReaders before SetTrainerNum
template <typename T>
void DatasetImpl<T>::SetTrainerNum(int trainer_num) {
  trainer_num_ = trainer_num;
}

// if you run distributed, and want to do global shuffle,
// set this before global shuffle.
// be sure you call CreateReaders before SetFleetSendBatchSize
template <typename T>
void DatasetImpl<T>::SetFleetSendBatchSize(int64_t size) {
  fleet_send_batch_size_ = size;
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

template <typename T>
void DatasetImpl<T>::SetChannelNum(int channel_num) {
  channel_num_ = channel_num;
}

template <typename T>
std::vector<paddle::framework::DataFeed*> DatasetImpl<T>::GetReaders() {
  std::vector<paddle::framework::DataFeed*> ret;
  ret.reserve(readers_.size());
  for (auto i : readers_) {
    ret.push_back(i.get());
  }
  return ret;
}

template <typename T>
void DatasetImpl<T>::CreateChannel() {
  if (input_channel_ == nullptr) {
    input_channel_ = paddle::framework::MakeChannel<T>();
  }
  if (multi_output_channel_.size() == 0) {
    multi_output_channel_.reserve(channel_num_);
    for (int i = 0; i < channel_num_; ++i) {
      multi_output_channel_.push_back(paddle::framework::MakeChannel<T>());
    }
  }
  if (multi_consume_channel_.size() == 0) {
    multi_consume_channel_.reserve(channel_num_);
    for (int i = 0; i < channel_num_; ++i) {
      multi_consume_channel_.push_back(paddle::framework::MakeChannel<T>());
    }
  }
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
  std::vector<std::thread> load_threads;
  for (int64_t i = 0; i < thread_num_; ++i) {
    load_threads.push_back(std::thread(
        &paddle::framework::DataFeed::LoadIntoMemory, readers_[i].get()));
  }
  for (std::thread& t : load_threads) {
    t.join();
  }
  input_channel_->Close();
  int64_t in_chan_size = input_channel_->Size();
  input_channel_->SetBlockSize(in_chan_size / thread_num_ + 1);
  timeline.Pause();
  VLOG(3) << "DatasetImpl<T>::LoadIntoMemory() end"
          << ", memory data size=" << input_channel_->Size()
          << ", cost time=" << timeline.ElapsedSec() << " seconds";
}

template <typename T>
void DatasetImpl<T>::PreLoadIntoMemory() {
  VLOG(3) << "DatasetImpl<T>::PreLoadIntoMemory() begin";
  preload_threads_.clear();
  for (int64_t i = 0; i < thread_num_; ++i) {
    preload_threads_.push_back(std::thread(
        &paddle::framework::DataFeed::LoadIntoMemory, readers_[i].get()));
  }
  VLOG(3) << "DatasetImpl<T>::PreLoadIntoMemory() end";
}

template <typename T>
void DatasetImpl<T>::WaitPreLoadDone() {
  VLOG(3) << "DatasetImpl<T>::WaitPreLoadDone() begin";
  for (std::thread& t : preload_threads_) {
    t.join();
  }
  input_channel_->Close();
  int64_t in_chan_size = input_channel_->Size();
  input_channel_->SetBlockSize(in_chan_size / thread_num_ + 1);
  VLOG(3) << "DatasetImpl<T>::WaitPreLoadDone() end";
}

// release memory data
template <typename T>
void DatasetImpl<T>::ReleaseMemory() {
  VLOG(3) << "DatasetImpl<T>::ReleaseMemory() begin";
  if (input_channel_) {
    input_channel_->Clear();
    input_channel_ = nullptr;
  }
  for (size_t i = 0; i < multi_output_channel_.size(); ++i) {
    if (!multi_output_channel_[i]) {
      continue;
    }
    multi_output_channel_[i]->Clear();
    multi_output_channel_[i] = nullptr;
  }
  std::vector<paddle::framework::Channel<T>>().swap(multi_output_channel_);
  for (size_t i = 0; i < multi_consume_channel_.size(); ++i) {
    if (!multi_consume_channel_[i]) {
      continue;
    }
    multi_consume_channel_[i]->Clear();
    multi_consume_channel_[i] = nullptr;
  }
  std::vector<paddle::framework::Channel<T>>().swap(multi_consume_channel_);
  std::vector<std::shared_ptr<paddle::framework::DataFeed>>().swap(readers_);
  VLOG(3) << "DatasetImpl<T>::ReleaseMemory() end";
}

// do local shuffle
template <typename T>
void DatasetImpl<T>::LocalShuffle() {
  VLOG(3) << "DatasetImpl<T>::LocalShuffle() begin";
  platform::Timer timeline;
  timeline.Start();

  if (!input_channel_ || input_channel_->Size() == 0) {
    VLOG(3) << "DatasetImpl<T>::LocalShuffle() end, no data to shuffle";
    return;
  }
  auto fleet_ptr = FleetWrapper::GetInstance();
  input_channel_->Close();
  std::vector<T> data;
  input_channel_->ReadAll(data);
  std::shuffle(data.begin(), data.end(), fleet_ptr->LocalRandomEngine());
  input_channel_->Open();
  input_channel_->Write(std::move(data));
  data.clear();
  data.shrink_to_fit();
  input_channel_->Close();

  timeline.Pause();
  VLOG(3) << "DatasetImpl<T>::LocalShuffle() end, cost time="
          << timeline.ElapsedSec() << " seconds";
}

template <typename T>
void DatasetImpl<T>::GlobalShuffle() {
  VLOG(3) << "DatasetImpl<T>::GlobalShuffle() begin";
  platform::Timer timeline;
  timeline.Start();
  auto fleet_ptr = FleetWrapper::GetInstance();

  if (!input_channel_ || input_channel_->Size() == 0) {
    VLOG(3) << "DatasetImpl<T>::GlobalShuffle() end, no data to shuffle";
    return;
  }

  // local shuffle
  input_channel_->Close();
  std::vector<T> data;
  input_channel_->ReadAll(data);
  std::shuffle(data.begin(), data.end(), fleet_ptr->LocalRandomEngine());
  input_channel_->Open();
  input_channel_->Write(std::move(data));
  data.clear();
  data.shrink_to_fit();

  input_channel_->Close();
  input_channel_->SetBlockSize(fleet_send_batch_size_);
  VLOG(3) << "DatasetImpl<T>::GlobalShuffle() input_channel_ size "
          << input_channel_->Size();

  auto global_shuffle_func = [this]() {
    auto fleet_ptr = FleetWrapper::GetInstance();
    std::vector<T> data;
    while (this->input_channel_->Read(data)) {
      std::vector<paddle::framework::BinaryArchive> ars(this->trainer_num_);
      for (auto& t : data) {
        auto client_id = fleet_ptr->LocalRandomEngine()() % this->trainer_num_;
        ars[client_id] << t;
      }
      std::vector<std::future<int32_t>> total_status;
      std::vector<int> send_index(this->trainer_num_);
      for (int i = 0; i < this->trainer_num_; ++i) {
        send_index[i] = i;
      }
      std::shuffle(send_index.begin(), send_index.end(),
                   fleet_ptr->LocalRandomEngine());
      for (auto index = 0u; index < this->trainer_num_; ++index) {
        int i = send_index[index];
        if (ars[i].Length() == 0) {
          continue;
        }
        std::string msg(ars[i].Buffer(), ars[i].Length());
        auto ret = fleet_ptr->SendClientToClientMsg(0, i, msg);
        total_status.push_back(std::move(ret));
      }
      for (auto& t : total_status) {
        t.wait();
      }
      ars.clear();
      ars.shrink_to_fit();
      data.clear();
      data.shrink_to_fit();
      sleep(this->fleet_send_sleep_seconds_);
    }
  };

  VLOG(3) << "start global shuffle threads";
  std::vector<std::thread> global_shuffle_threads;
  for (int i = 0; i < thread_num_; ++i) {
    global_shuffle_threads.push_back(std::thread(global_shuffle_func));
  }
  for (std::thread& t : global_shuffle_threads) {
    t.join();
  }
  global_shuffle_threads.clear();
  global_shuffle_threads.shrink_to_fit();
  input_channel_->Clear();
  timeline.Pause();
  VLOG(3) << "DatasetImpl<T>::GlobalShuffle() end, cost time="
          << timeline.ElapsedSec() << " seconds";
}

template <typename T>
void DatasetImpl<T>::CreateReaders() {
  VLOG(3) << "Calling CreateReaders()";
  VLOG(3) << "thread num in Dataset: " << thread_num_;
  VLOG(3) << "Filelist size in Dataset: " << filelist_.size();
  VLOG(3) << "channel num in Dataset: " << channel_num_;
  CHECK(thread_num_ > 0) << "thread num should > 0";
  CHECK(thread_num_ <= filelist_.size())
      << "thread num should <= filelist size";
  CHECK(channel_num_ > 0) << "channel num should > 0";
  CHECK(channel_num_ <= thread_num_) << "channel num should <= thread num";
  VLOG(3) << "readers size: " << readers_.size();
  if (readers_.size() != 0) {
    VLOG(3) << "readers_.size() = " << readers_.size()
            << ", will not create again";
    return;
  }
  VLOG(3) << "data feed class name: " << data_feed_desc_.name();
  int channel_idx = 0;
  for (int i = 0; i < thread_num_; ++i) {
    readers_.push_back(DataFeedFactory::CreateDataFeed(data_feed_desc_.name()));
    readers_[i]->Init(data_feed_desc_);
    readers_[i]->SetThreadId(i);
    readers_[i]->SetThreadNum(thread_num_);
    readers_[i]->SetFileListMutex(&mutex_for_pick_file_);
    readers_[i]->SetFileListIndex(&file_idx_);
    readers_[i]->SetFileList(filelist_);
    if (input_channel_ != nullptr) {
      readers_[i]->SetInputChannel(input_channel_.get());
    }
    if (cur_channel_ == 0 && channel_idx < multi_output_channel_.size()) {
      readers_[i]->SetOutputChannel(multi_output_channel_[channel_idx].get());
      readers_[i]->SetConsumeChannel(multi_consume_channel_[channel_idx].get());
    } else if (channel_idx < multi_output_channel_.size()) {
      readers_[i]->SetOutputChannel(multi_consume_channel_[channel_idx].get());
      readers_[i]->SetConsumeChannel(multi_output_channel_[channel_idx].get());
    }
    ++channel_idx;
    if (channel_idx >= channel_num_) {
      channel_idx = 0;
    }
  }
  VLOG(3) << "readers size: " << readers_.size();
}

template <typename T>
void DatasetImpl<T>::DestroyReaders() {
  VLOG(3) << "Calling DestroyReaders()";
  std::vector<std::shared_ptr<paddle::framework::DataFeed>>().swap(readers_);
  VLOG(3) << "readers size: " << readers_.size();
  file_idx_ = 0;
  cur_channel_ = 1 - cur_channel_;
}

template <typename T>
int64_t DatasetImpl<T>::GetMemoryDataSize() {
  return input_channel_->Size();
}

template <typename T>
int64_t DatasetImpl<T>::GetShuffleDataSize() {
  int64_t sum = 0;
  for (size_t i = 0; i < multi_output_channel_.size(); ++i) {
    sum += multi_output_channel_[i]->Size() + multi_consume_channel_[i]->Size();
  }
  return sum;
}

template <typename T>
int DatasetImpl<T>::ReceiveFromClient(int msg_type, int client_id,
                                      const std::string& msg) {
#ifdef _LINUX
  VLOG(3) << "ReceiveFromClient msg_type=" << msg_type
          << ", client_id=" << client_id << ", msg length=" << msg.length();
  if (msg.length() == 0) {
    return 0;
  }
  paddle::framework::BinaryArchive ar;
  ar.SetReadBuffer(const_cast<char*>(msg.c_str()), msg.length(), nullptr);
  if (ar.Cursor() == ar.Finish()) {
    return 0;
  }
  std::vector<T> data;
  while (ar.Cursor() < ar.Finish()) {
    data.push_back(ar.Get<T>());
  }
  CHECK(ar.Cursor() == ar.Finish());

  auto fleet_ptr = FleetWrapper::GetInstance();
  int64_t index = fleet_ptr->LocalRandomEngine()() % channel_num_;
  VLOG(3) << "ramdom index=" << index;
  multi_output_channel_[index]->Write(std::move(data));

  data.clear();
  data.shrink_to_fit();
#endif
  return 0;
}

// explicit instantiation
template class DatasetImpl<std::vector<MultiSlotType>>;
template class DatasetImpl<Record>;

}  // end namespace framework
}  // end namespace paddle

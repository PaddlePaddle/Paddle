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
#include <algorithm>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"
#include "paddle/fluid/framework/data_feed_factory.h"
#include "paddle/fluid/framework/fleet/fleet_wrapper.h"
#include "paddle/fluid/framework/io/fs.h"
#include "paddle/fluid/platform/timer.h"
#include "xxhash.h"  // NOLINT

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
  fleet_send_batch_size_ = 1024;
  fleet_send_sleep_seconds_ = 0;
  merge_by_insid_ = false;
  merge_by_sid_ = true;
  enable_pv_merge_ = false;
  merge_size_ = 2;
  parse_ins_id_ = false;
  parse_content_ = false;
  parse_logkey_ = false;
  preload_thread_num_ = 0;
  global_index_ = 0;
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
void DatasetImpl<T>::SetDownloadCmd(const std::string& download_cmd) {
  paddle::framework::set_download_command(download_cmd);
}

template <typename T>
std::string DatasetImpl<T>::GetDownloadCmd() {
  return paddle::framework::download_cmd();
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
void DatasetImpl<T>::SetParseInsId(bool parse_ins_id) {
  parse_ins_id_ = parse_ins_id;
}

template <typename T>
void DatasetImpl<T>::SetParseContent(bool parse_content) {
  parse_content_ = parse_content;
}

template <typename T>
void DatasetImpl<T>::SetParseLogKey(bool parse_logkey) {
  parse_logkey_ = parse_logkey;
}

template <typename T>
void DatasetImpl<T>::SetMergeByInsId(int merge_size) {
  merge_by_insid_ = true;
  parse_ins_id_ = true;
  merge_size_ = merge_size;
}

template <typename T>
void DatasetImpl<T>::SetMergeBySid(bool is_merge) {
  merge_by_sid_ = is_merge;
}

template <typename T>
void DatasetImpl<T>::SetEnablePvMerge(bool enable_pv_merge) {
  enable_pv_merge_ = enable_pv_merge;
}

template <typename T>
void DatasetImpl<T>::SetGenerateUniqueFeasign(bool gen_uni_feasigns) {
  gen_uni_feasigns_ = gen_uni_feasigns;
  VLOG(3) << "Set generate unique feasigns: " << gen_uni_feasigns;
}

template <typename T>
void DatasetImpl<T>::SetFeaEval(bool fea_eval, int record_candidate_size) {
  slots_shuffle_fea_eval_ = fea_eval;
  slots_shuffle_rclist_.ReSize(record_candidate_size);
  VLOG(3) << "SetFeaEval fea eval mode: " << fea_eval
          << " with record candidate size: " << record_candidate_size;
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
  if (input_pv_channel_ == nullptr) {
    input_pv_channel_ = paddle::framework::MakeChannel<PvInstance>();
  }
  if (multi_pv_output_.size() == 0) {
    multi_pv_output_.reserve(channel_num_);
    for (int i = 0; i < channel_num_; ++i) {
      multi_pv_output_.push_back(paddle::framework::MakeChannel<PvInstance>());
    }
  }
  if (multi_pv_consume_.size() == 0) {
    multi_pv_consume_.reserve(channel_num_);
    for (int i = 0; i < channel_num_; ++i) {
      multi_pv_consume_.push_back(paddle::framework::MakeChannel<PvInstance>());
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
  if (preload_thread_num_ != 0) {
    CHECK(static_cast<size_t>(preload_thread_num_) == preload_readers_.size());
    preload_threads_.clear();
    for (int64_t i = 0; i < preload_thread_num_; ++i) {
      preload_threads_.push_back(
          std::thread(&paddle::framework::DataFeed::LoadIntoMemory,
                      preload_readers_[i].get()));
    }
  } else {
    CHECK(static_cast<size_t>(thread_num_) == readers_.size());
    preload_threads_.clear();
    for (int64_t i = 0; i < thread_num_; ++i) {
      preload_threads_.push_back(std::thread(
          &paddle::framework::DataFeed::LoadIntoMemory, readers_[i].get()));
    }
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
  if (input_pv_channel_) {
    input_pv_channel_->Clear();
    input_pv_channel_ = nullptr;
  }
  for (size_t i = 0; i < multi_pv_output_.size(); ++i) {
    if (!multi_pv_output_[i]) {
      continue;
    }
    multi_pv_output_[i]->Clear();
    multi_pv_output_[i] = nullptr;
  }
  std::vector<paddle::framework::Channel<PvInstance>>().swap(multi_pv_output_);
  for (size_t i = 0; i < multi_pv_consume_.size(); ++i) {
    if (!multi_pv_consume_[i]) {
      continue;
    }
    multi_pv_consume_[i]->Clear();
    multi_pv_consume_[i] = nullptr;
  }
  std::vector<paddle::framework::Channel<PvInstance>>().swap(multi_pv_consume_);

  std::vector<std::shared_ptr<paddle::framework::DataFeed>>().swap(readers_);
  input_records_.clear();
  std::vector<T>().swap(input_records_);
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
void DatasetImpl<T>::GlobalShuffle(int thread_num) {
#ifdef PADDLE_WITH_PSLIB
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

  auto get_client_id = [this, fleet_ptr](const T& data) -> size_t {
    if (!this->merge_by_insid_) {
      return fleet_ptr->LocalRandomEngine()() % this->trainer_num_;
    } else {
      return XXH64(data.ins_id_.data(), data.ins_id_.length(), 0) %
             this->trainer_num_;
    }
  };

  auto global_shuffle_func = [this, get_client_id]() {
    auto fleet_ptr = FleetWrapper::GetInstance();
    std::vector<T> data;
    while (this->input_channel_->Read(data)) {
      std::vector<paddle::framework::BinaryArchive> ars(this->trainer_num_);
      for (auto& t : data) {
        auto client_id = get_client_id(t);
        ars[client_id] << t;
      }
      std::vector<std::future<int32_t>> total_status;
      std::vector<int> send_index(this->trainer_num_);
      for (int i = 0; i < this->trainer_num_; ++i) {
        send_index[i] = i;
      }
      std::shuffle(send_index.begin(), send_index.end(),
                   fleet_ptr->LocalRandomEngine());
      for (int index = 0; index < this->trainer_num_; ++index) {
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
      // currently we find bottleneck is server not able to handle large data
      // in time, so we can remove this sleep and set fleet_send_batch_size to
      // 1024, and set server thread to 24.
      if (fleet_send_sleep_seconds_ != 0) {
        sleep(this->fleet_send_sleep_seconds_);
      }
    }
  };

  std::vector<std::thread> global_shuffle_threads;
  if (thread_num == -1) {
    thread_num = thread_num_;
  }
  VLOG(3) << "start global shuffle threads, num = " << thread_num;
  for (int i = 0; i < thread_num; ++i) {
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
#endif
}

template <typename T>
void DatasetImpl<T>::DynamicAdjustChannelNum(int channel_num,
                                             bool discard_remaining_ins) {
  if (channel_num_ == channel_num) {
    VLOG(3) << "DatasetImpl<T>::DynamicAdjustChannelNum channel_num_="
            << channel_num_ << ", channel_num_=channel_num, no need to adjust";
    return;
  }
  VLOG(3) << "adjust channel num from " << channel_num_ << " to "
          << channel_num;
  channel_num_ = channel_num;
  std::vector<paddle::framework::Channel<T>>* origin_channels = nullptr;
  std::vector<paddle::framework::Channel<T>>* other_channels = nullptr;
  std::vector<paddle::framework::Channel<PvInstance>>* origin_pv_channels =
      nullptr;
  std::vector<paddle::framework::Channel<PvInstance>>* other_pv_channels =
      nullptr;

  // find out which channel (output or consume) has data
  int cur_channel = 0;
  uint64_t output_channels_data_size = 0;
  uint64_t consume_channels_data_size = 0;
  CHECK(multi_output_channel_.size() == multi_consume_channel_.size());
  for (size_t i = 0; i < multi_output_channel_.size(); ++i) {
    output_channels_data_size += multi_output_channel_[i]->Size();
    consume_channels_data_size += multi_consume_channel_[i]->Size();
  }
  if (output_channels_data_size != 0) {
    CHECK(consume_channels_data_size == 0);  // NOLINT
    cur_channel = 0;
  } else {
    CHECK(output_channels_data_size == 0);  // NOLINT
    cur_channel = 1;
  }
  if (cur_channel == 0) {
    origin_channels = &multi_output_channel_;
    other_channels = &multi_consume_channel_;
    origin_pv_channels = &multi_pv_output_;
    other_pv_channels = &multi_pv_consume_;
  } else {
    origin_channels = &multi_consume_channel_;
    other_channels = &multi_output_channel_;
    origin_pv_channels = &multi_pv_consume_;
    other_pv_channels = &multi_pv_output_;
  }
  CHECK(origin_channels != nullptr);     // NOLINT
  CHECK(other_channels != nullptr);      // NOLINT
  CHECK(origin_pv_channels != nullptr);  // NOLINT
  CHECK(other_pv_channels != nullptr);   // NOLINT

  paddle::framework::Channel<T> total_data_channel =
      paddle::framework::MakeChannel<T>();
  std::vector<paddle::framework::Channel<T>> new_channels;
  std::vector<paddle::framework::Channel<T>> new_other_channels;
  std::vector<paddle::framework::Channel<PvInstance>> new_pv_channels;
  std::vector<paddle::framework::Channel<PvInstance>> new_other_pv_channels;

  std::vector<T> local_vec;
  for (size_t i = 0; i < origin_channels->size(); ++i) {
    local_vec.clear();
    (*origin_channels)[i]->Close();
    (*origin_channels)[i]->ReadAll(local_vec);
    total_data_channel->Write(std::move(local_vec));
  }
  total_data_channel->Close();
  if (static_cast<int>(total_data_channel->Size()) >= channel_num) {
    total_data_channel->SetBlockSize(total_data_channel->Size() / channel_num +
                                     (discard_remaining_ins ? 0 : 1));
  }
  if (static_cast<int>(input_channel_->Size()) >= channel_num) {
    input_channel_->SetBlockSize(input_channel_->Size() / channel_num +
                                 (discard_remaining_ins ? 0 : 1));
  }
  if (static_cast<int>(input_pv_channel_->Size()) >= channel_num) {
    input_pv_channel_->SetBlockSize(input_pv_channel_->Size() / channel_num +
                                    (discard_remaining_ins ? 0 : 1));
    VLOG(3) << "now input_pv_channle block size is "
            << input_pv_channel_->BlockSize();
  }

  for (int i = 0; i < channel_num; ++i) {
    local_vec.clear();
    total_data_channel->Read(local_vec);
    new_other_channels.push_back(paddle::framework::MakeChannel<T>());
    new_channels.push_back(paddle::framework::MakeChannel<T>());
    new_channels[i]->Write(std::move(local_vec));
    new_other_pv_channels.push_back(
        paddle::framework::MakeChannel<PvInstance>());
    new_pv_channels.push_back(paddle::framework::MakeChannel<PvInstance>());
  }

  total_data_channel->Clear();
  origin_channels->clear();
  other_channels->clear();
  *origin_channels = new_channels;
  *other_channels = new_other_channels;

  origin_pv_channels->clear();
  other_pv_channels->clear();
  *origin_pv_channels = new_pv_channels;
  *other_pv_channels = new_other_pv_channels;

  new_channels.clear();
  new_other_channels.clear();
  std::vector<paddle::framework::Channel<T>>().swap(new_channels);
  std::vector<paddle::framework::Channel<T>>().swap(new_other_channels);

  new_pv_channels.clear();
  new_other_pv_channels.clear();
  std::vector<paddle::framework::Channel<PvInstance>>().swap(new_pv_channels);
  std::vector<paddle::framework::Channel<PvInstance>>().swap(
      new_other_pv_channels);

  local_vec.clear();
  std::vector<T>().swap(local_vec);
  VLOG(3) << "adjust channel num done";
}

template <typename T>
void DatasetImpl<T>::DynamicAdjustReadersNum(int thread_num) {
  if (thread_num_ == thread_num) {
    VLOG(3) << "DatasetImpl<T>::DynamicAdjustReadersNum thread_num_="
            << thread_num_ << ", thread_num_=thread_num, no need to adjust";
    return;
  }
  VLOG(3) << "adjust readers num from " << thread_num_ << " to " << thread_num;
  thread_num_ = thread_num;
  std::vector<std::shared_ptr<paddle::framework::DataFeed>>().swap(readers_);
  CreateReaders();
  VLOG(3) << "adjust readers num done";
}

template <typename T>
void DatasetImpl<T>::SetFleetSendSleepSeconds(int seconds) {
  fleet_send_sleep_seconds_ = seconds;
}

template <typename T>
void DatasetImpl<T>::CreateReaders() {
  VLOG(3) << "Calling CreateReaders()";
  VLOG(3) << "thread num in Dataset: " << thread_num_;
  VLOG(3) << "Filelist size in Dataset: " << filelist_.size();
  VLOG(3) << "channel num in Dataset: " << channel_num_;
  CHECK(thread_num_ > 0) << "thread num should > 0";
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
    readers_[i]->SetParseInsId(parse_ins_id_);
    readers_[i]->SetParseContent(parse_content_);
    readers_[i]->SetParseLogKey(parse_logkey_);
    readers_[i]->SetEnablePvMerge(enable_pv_merge_);
    // Notice: it is only valid for untest of test_paddlebox_datafeed.
    // In fact, it does not affect the train process when paddle is
    // complied with Box_Ps.
    readers_[i]->SetCurrentPhase(current_phase_);
    if (input_channel_ != nullptr) {
      readers_[i]->SetInputChannel(input_channel_.get());
    }
    if (input_pv_channel_ != nullptr) {
      readers_[i]->SetInputPvChannel(input_pv_channel_.get());
    }
    if (cur_channel_ == 0 &&
        static_cast<size_t>(channel_idx) < multi_output_channel_.size()) {
      readers_[i]->SetOutputChannel(multi_output_channel_[channel_idx].get());
      readers_[i]->SetConsumeChannel(multi_consume_channel_[channel_idx].get());
      readers_[i]->SetOutputPvChannel(multi_pv_output_[channel_idx].get());
      readers_[i]->SetConsumePvChannel(multi_pv_consume_[channel_idx].get());
    } else if (static_cast<size_t>(channel_idx) <
               multi_output_channel_.size()) {
      readers_[i]->SetOutputChannel(multi_consume_channel_[channel_idx].get());
      readers_[i]->SetConsumeChannel(multi_output_channel_[channel_idx].get());
      readers_[i]->SetOutputPvChannel(multi_pv_consume_[channel_idx].get());
      readers_[i]->SetConsumePvChannel(multi_pv_output_[channel_idx].get());
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
  VLOG(3) << "readers size1: " << readers_.size();
  std::vector<std::shared_ptr<paddle::framework::DataFeed>>().swap(readers_);
  VLOG(3) << "readers size: " << readers_.size();
  file_idx_ = 0;
  cur_channel_ = 1 - cur_channel_;
}

template <typename T>
void DatasetImpl<T>::SetPreLoadThreadNum(int thread_num) {
  preload_thread_num_ = thread_num;
}

template <typename T>
void DatasetImpl<T>::CreatePreLoadReaders() {
  VLOG(3) << "Begin CreatePreLoadReaders";
  if (preload_thread_num_ == 0) {
    preload_thread_num_ = thread_num_;
  }
  CHECK(preload_thread_num_ > 0) << "thread num should > 0";
  CHECK(input_channel_ != nullptr);
  preload_readers_.clear();
  for (int i = 0; i < preload_thread_num_; ++i) {
    preload_readers_.push_back(
        DataFeedFactory::CreateDataFeed(data_feed_desc_.name()));
    preload_readers_[i]->Init(data_feed_desc_);
    preload_readers_[i]->SetThreadId(i);
    preload_readers_[i]->SetThreadNum(preload_thread_num_);
    preload_readers_[i]->SetFileListMutex(&mutex_for_pick_file_);
    preload_readers_[i]->SetFileListIndex(&file_idx_);
    preload_readers_[i]->SetFileList(filelist_);
    preload_readers_[i]->SetParseInsId(parse_ins_id_);
    preload_readers_[i]->SetParseContent(parse_content_);
    preload_readers_[i]->SetParseLogKey(parse_logkey_);
    preload_readers_[i]->SetEnablePvMerge(enable_pv_merge_);
    preload_readers_[i]->SetInputChannel(input_channel_.get());
    preload_readers_[i]->SetOutputChannel(nullptr);
    preload_readers_[i]->SetConsumeChannel(nullptr);
    preload_readers_[i]->SetOutputPvChannel(nullptr);
    preload_readers_[i]->SetConsumePvChannel(nullptr);
  }
  VLOG(3) << "End CreatePreLoadReaders";
}

template <typename T>
void DatasetImpl<T>::DestroyPreLoadReaders() {
  VLOG(3) << "Begin DestroyPreLoadReaders";
  preload_readers_.clear();
  std::vector<std::shared_ptr<paddle::framework::DataFeed>>().swap(
      preload_readers_);
  file_idx_ = 0;
  VLOG(3) << "End DestroyPreLoadReaders";
}

template <typename T>
int64_t DatasetImpl<T>::GetMemoryDataSize() {
  return input_channel_->Size();
}

template <typename T>
int64_t DatasetImpl<T>::GetPvDataSize() {
  if (enable_pv_merge_) {
    return input_pv_channel_->Size();
  } else {
    VLOG(0) << "It does not merge pv..";
    return 0;
  }
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
  // not use random because it doesn't perform well here.
  // to make sure each channel get data equally, we just put data to
  // channel one by one.
  // int64_t index = fleet_ptr->LocalRandomEngine()() % channel_num_;
  int64_t index = 0;
  {
    std::unique_lock<std::mutex> lk(global_index_mutex_);
    index = global_index_++;
  }
  index = index % channel_num_;
  VLOG(3) << "ramdom index=" << index;
  multi_output_channel_[index]->Write(std::move(data));

  data.clear();
  data.shrink_to_fit();
#endif
  return 0;
}

// explicit instantiation
template class DatasetImpl<Record>;

void MultiSlotDataset::PostprocessInstance() {
  // divide pv instance, and merge to input_channel_
  if (enable_pv_merge_) {
    auto fleet_ptr = FleetWrapper::GetInstance();
    std::shuffle(input_records_.begin(), input_records_.end(),
                 fleet_ptr->LocalRandomEngine());
    input_channel_->Open();
    input_channel_->Write(std::move(input_records_));
    for (size_t i = 0; i < multi_pv_consume_.size(); ++i) {
      multi_pv_consume_[i]->Clear();
    }
    input_channel_->Close();
    input_records_.clear();
    input_records_.shrink_to_fit();
  } else {
    input_channel_->Open();
    for (size_t i = 0; i < multi_consume_channel_.size(); ++i) {
      std::vector<Record> ins_data;
      multi_consume_channel_[i]->Close();
      multi_consume_channel_[i]->ReadAll(ins_data);
      input_channel_->Write(std::move(ins_data));
      ins_data.clear();
      ins_data.shrink_to_fit();
      multi_consume_channel_[i]->Clear();
    }
    input_channel_->Close();
    this->LocalShuffle();
  }
}

void MultiSlotDataset::SetCurrentPhase(int current_phase) {
  current_phase_ = current_phase;
}

void MultiSlotDataset::PreprocessInstance() {
  if (!input_channel_ || input_channel_->Size() == 0) {
    return;
  }
  if (!enable_pv_merge_) {  // means to use Record
    this->LocalShuffle();
  } else {  // means to use Pv
    auto fleet_ptr = FleetWrapper::GetInstance();
    input_channel_->Close();
    std::vector<PvInstance> pv_data;
    input_channel_->ReadAll(input_records_);
    int all_records_num = input_records_.size();
    std::vector<Record*> all_records;
    all_records.reserve(all_records_num);
    for (int index = 0; index < all_records_num; ++index) {
      all_records.push_back(&input_records_[index]);
    }

    std::sort(all_records.data(), all_records.data() + all_records_num,
              [](const Record* lhs, const Record* rhs) {
                return lhs->search_id < rhs->search_id;
              });
    if (merge_by_sid_) {
      uint64_t last_search_id = 0;
      for (int i = 0; i < all_records_num; ++i) {
        Record* ins = all_records[i];
        if (i == 0 || last_search_id != ins->search_id) {
          PvInstance pv_instance = make_pv_instance();
          pv_instance->merge_instance(ins);
          pv_data.push_back(pv_instance);
          last_search_id = ins->search_id;
          continue;
        }
        pv_data.back()->merge_instance(ins);
      }
    } else {
      for (int i = 0; i < all_records_num; ++i) {
        Record* ins = all_records[i];
        PvInstance pv_instance = make_pv_instance();
        pv_instance->merge_instance(ins);
        pv_data.push_back(pv_instance);
      }
    }

    std::shuffle(pv_data.begin(), pv_data.end(),
                 fleet_ptr->LocalRandomEngine());
    input_pv_channel_->Open();
    input_pv_channel_->Write(std::move(pv_data));

    pv_data.clear();
    pv_data.shrink_to_fit();
    input_pv_channel_->Close();
  }
}

void MultiSlotDataset::GenerateLocalTablesUnlock(int table_id, int feadim,
                                                 int read_thread_num,
                                                 int consume_thread_num,
                                                 int shard_num) {
  VLOG(3) << "MultiSlotDataset::GenerateUniqueFeasign begin";
  if (!gen_uni_feasigns_) {
    VLOG(3) << "generate_unique_feasign_=false, will not GenerateUniqueFeasign";
    return;
  }

  CHECK(multi_output_channel_.size() != 0);  // NOLINT
  auto fleet_ptr_ = FleetWrapper::GetInstance();
  std::vector<std::unordered_map<uint64_t, std::vector<float>>>&
      local_map_tables = fleet_ptr_->GetLocalTable();
  local_map_tables.resize(shard_num);
  // read thread
  int channel_num = multi_output_channel_.size();
  if (read_thread_num < channel_num) {
    read_thread_num = channel_num;
  }
  std::vector<std::thread> threads(read_thread_num);
  consume_task_pool_.resize(consume_thread_num);
  for (size_t i = 0; i < consume_task_pool_.size(); i++) {
    consume_task_pool_[i].reset(new ::ThreadPool(1));
  }
  auto consume_func = [&local_map_tables](int shard_id, int feadim,
                                          std::vector<uint64_t>& keys) {
    for (auto k : keys) {
      if (local_map_tables[shard_id].find(k) ==
          local_map_tables[shard_id].end()) {
        local_map_tables[shard_id][k] = std::vector<float>(feadim, 0);
      }
    }
  };
  auto gen_func = [this, &shard_num, &feadim, &local_map_tables,
                   &consume_func](int i) {
    std::vector<Record> vec_data;
    std::vector<std::vector<uint64_t>> task_keys(shard_num);
    std::vector<std::future<void>> task_futures;
    this->multi_output_channel_[i]->Close();
    this->multi_output_channel_[i]->ReadAll(vec_data);
    for (size_t j = 0; j < vec_data.size(); j++) {
      for (auto& feature : vec_data[j].uint64_feasigns_) {
        int shard = feature.sign().uint64_feasign_ % shard_num;
        task_keys[shard].push_back(feature.sign().uint64_feasign_);
      }
    }

    for (int shard_id = 0; shard_id < shard_num; shard_id++) {
      task_futures.emplace_back(consume_task_pool_[shard_id]->enqueue(
          consume_func, shard_id, feadim, task_keys[shard_id]));
    }

    multi_output_channel_[i]->Open();
    multi_output_channel_[i]->Write(std::move(vec_data));
    vec_data.clear();
    vec_data.shrink_to_fit();
    for (auto& tk : task_keys) {
      tk.clear();
      std::vector<uint64_t>().swap(tk);
    }
    task_keys.clear();
    std::vector<std::vector<uint64_t>>().swap(task_keys);
    for (auto& tf : task_futures) {
      tf.wait();
    }
  };
  for (size_t i = 0; i < threads.size(); i++) {
    threads[i] = std::thread(gen_func, i);
  }
  for (std::thread& t : threads) {
    t.join();
  }
  for (size_t i = 0; i < consume_task_pool_.size(); i++) {
    consume_task_pool_[i].reset();
  }
  consume_task_pool_.clear();
  fleet_ptr_->PullSparseToLocal(table_id, feadim);
}

void MultiSlotDataset::MergeByInsId() {
  VLOG(3) << "MultiSlotDataset::MergeByInsId begin";
  if (!merge_by_insid_) {
    VLOG(3) << "merge_by_insid=false, will not MergeByInsId";
    return;
  }
  auto multi_slot_desc = data_feed_desc_.multi_slot_desc();
  std::vector<std::string> use_slots;
  std::vector<bool> use_slots_is_dense;
  for (int i = 0; i < multi_slot_desc.slots_size(); ++i) {
    const auto& slot = multi_slot_desc.slots(i);
    if (slot.is_used()) {
      use_slots.push_back(slot.name());
      use_slots_is_dense.push_back(slot.is_dense());
    }
  }
  CHECK(multi_output_channel_.size() != 0);  // NOLINT
  auto channel_data = paddle::framework::MakeChannel<Record>();
  VLOG(3) << "multi_output_channel_.size() " << multi_output_channel_.size();
  for (size_t i = 0; i < multi_output_channel_.size(); ++i) {
    std::vector<Record> vec_data;
    multi_output_channel_[i]->Close();
    multi_output_channel_[i]->ReadAll(vec_data);
    channel_data->Write(std::move(vec_data));
    vec_data.clear();
    vec_data.shrink_to_fit();
    multi_output_channel_[i]->Clear();
  }
  channel_data->Close();
  std::vector<Record> recs;
  recs.reserve(channel_data->Size());
  channel_data->ReadAll(recs);
  channel_data->Clear();
  std::sort(recs.begin(), recs.end(), [](const Record& a, const Record& b) {
    return a.ins_id_ < b.ins_id_;
  });

  std::vector<Record> results;
  uint64_t drop_ins_num = 0;
  std::unordered_set<uint16_t> all_int64;
  std::unordered_set<uint16_t> all_float;
  std::unordered_set<uint16_t> local_uint64;
  std::unordered_set<uint16_t> local_float;
  std::unordered_map<uint16_t, std::vector<FeatureItem>> all_dense_uint64;
  std::unordered_map<uint16_t, std::vector<FeatureItem>> all_dense_float;
  std::unordered_map<uint16_t, std::vector<FeatureItem>> local_dense_uint64;
  std::unordered_map<uint16_t, std::vector<FeatureItem>> local_dense_float;
  std::unordered_map<uint16_t, bool> dense_empty;

  VLOG(3) << "recs.size() " << recs.size();
  for (size_t i = 0; i < recs.size();) {
    size_t j = i + 1;
    while (j < recs.size() && recs[j].ins_id_ == recs[i].ins_id_) {
      j++;
    }
    if (merge_size_ > 0 && j - i != merge_size_) {
      drop_ins_num += j - i;
      LOG(WARNING) << "drop ins " << recs[i].ins_id_ << " size=" << j - i
                   << ", because merge_size=" << merge_size_;
      i = j;
      continue;
    }

    all_int64.clear();
    all_float.clear();
    all_dense_uint64.clear();
    all_dense_float.clear();
    bool has_conflict_slot = false;
    uint16_t conflict_slot = 0;

    Record rec;
    rec.ins_id_ = recs[i].ins_id_;
    rec.content_ = recs[i].content_;

    for (size_t k = i; k < j; k++) {
      dense_empty.clear();
      local_dense_uint64.clear();
      local_dense_float.clear();
      for (auto& feature : recs[k].uint64_feasigns_) {
        uint16_t slot = feature.slot();
        if (!use_slots_is_dense[slot]) {
          continue;
        }
        local_dense_uint64[slot].push_back(feature);
        if (feature.sign().uint64_feasign_ != 0) {
          dense_empty[slot] = false;
        } else if (dense_empty.find(slot) == dense_empty.end() &&
                   all_dense_uint64.find(slot) == all_dense_uint64.end()) {
          dense_empty[slot] = true;
        }
      }
      for (auto& feature : recs[k].float_feasigns_) {
        uint16_t slot = feature.slot();
        if (!use_slots_is_dense[slot]) {
          continue;
        }
        local_dense_float[slot].push_back(feature);
        if (fabs(feature.sign().float_feasign_) >= 1e-6) {
          dense_empty[slot] = false;
        } else if (dense_empty.find(slot) == dense_empty.end() &&
                   all_dense_float.find(slot) == all_dense_float.end()) {
          dense_empty[slot] = true;
        }
      }
      for (auto& p : dense_empty) {
        if (local_dense_uint64.find(p.first) != local_dense_uint64.end()) {
          all_dense_uint64[p.first] = std::move(local_dense_uint64[p.first]);
        } else if (local_dense_float.find(p.first) != local_dense_float.end()) {
          all_dense_float[p.first] = std::move(local_dense_float[p.first]);
        }
      }
    }
    for (auto& f : all_dense_uint64) {
      rec.uint64_feasigns_.insert(rec.uint64_feasigns_.end(), f.second.begin(),
                                  f.second.end());
    }
    for (auto& f : all_dense_float) {
      rec.float_feasigns_.insert(rec.float_feasigns_.end(), f.second.begin(),
                                 f.second.end());
    }

    for (size_t k = i; k < j; k++) {
      local_uint64.clear();
      local_float.clear();
      for (auto& feature : recs[k].uint64_feasigns_) {
        uint16_t slot = feature.slot();
        if (use_slots_is_dense[slot]) {
          continue;
        } else if (all_int64.find(slot) != all_int64.end()) {
          has_conflict_slot = true;
          conflict_slot = slot;
          break;
        }
        local_uint64.insert(slot);
        rec.uint64_feasigns_.push_back(std::move(feature));
      }
      if (has_conflict_slot) {
        break;
      }
      all_int64.insert(local_uint64.begin(), local_uint64.end());

      for (auto& feature : recs[k].float_feasigns_) {
        uint16_t slot = feature.slot();
        if (use_slots_is_dense[slot]) {
          continue;
        } else if (all_float.find(slot) != all_float.end()) {
          has_conflict_slot = true;
          conflict_slot = slot;
          break;
        }
        local_float.insert(slot);
        rec.float_feasigns_.push_back(std::move(feature));
      }
      if (has_conflict_slot) {
        break;
      }
      all_float.insert(local_float.begin(), local_float.end());
    }

    if (has_conflict_slot) {
      LOG(WARNING) << "drop ins " << recs[i].ins_id_ << " size=" << j - i
                   << ", because conflict_slot=" << use_slots[conflict_slot];
      drop_ins_num += j - i;
    } else {
      results.push_back(std::move(rec));
    }
    i = j;
  }
  std::vector<Record>().swap(recs);
  VLOG(3) << "results size " << results.size();
  LOG(WARNING) << "total drop ins num: " << drop_ins_num;
  results.shrink_to_fit();

  auto fleet_ptr = FleetWrapper::GetInstance();
  std::shuffle(results.begin(), results.end(), fleet_ptr->LocalRandomEngine());
  channel_data->Open();
  channel_data->Write(std::move(results));
  channel_data->Close();
  results.clear();
  results.shrink_to_fit();
  VLOG(3) << "channel data size " << channel_data->Size();
  channel_data->SetBlockSize(channel_data->Size() / channel_num_ + 1);
  VLOG(3) << "channel data block size " << channel_data->BlockSize();
  for (size_t i = 0; i < multi_output_channel_.size(); ++i) {
    std::vector<Record> vec_data;
    channel_data->Read(vec_data);
    multi_output_channel_[i]->Open();
    multi_output_channel_[i]->Write(std::move(vec_data));
    vec_data.clear();
    vec_data.shrink_to_fit();
  }
  CHECK(channel_data->Size() == 0);  // NOLINT
  channel_data->Clear();
  VLOG(3) << "MultiSlotDataset::MergeByInsId end";
}

void MultiSlotDataset::GetRandomData(const std::set<uint16_t>& slots_to_replace,
                                     std::vector<Record>* result) {
  int debug_erase_cnt = 0;
  int debug_push_cnt = 0;
  auto multi_slot_desc = data_feed_desc_.multi_slot_desc();
  slots_shuffle_rclist_.ReInit();
  for (const auto& rec : slots_shuffle_original_data_) {
    RecordCandidate rand_rec;
    Record new_rec = rec;
    slots_shuffle_rclist_.AddAndGet(rec, &rand_rec);
    for (auto it = new_rec.uint64_feasigns_.begin();
         it != new_rec.uint64_feasigns_.end();) {
      if (slots_to_replace.find(it->slot()) != slots_to_replace.end()) {
        it = new_rec.uint64_feasigns_.erase(it);
        debug_erase_cnt += 1;
      } else {
        ++it;
      }
    }
    for (auto slot : slots_to_replace) {
      auto range = rand_rec.feas.equal_range(slot);
      for (auto it = range.first; it != range.second; ++it) {
        new_rec.uint64_feasigns_.push_back({it->second, it->first});
        debug_push_cnt += 1;
      }
    }
    result->push_back(std::move(new_rec));
  }
  VLOG(2) << "erase feasign num: " << debug_erase_cnt
          << " repush feasign num: " << debug_push_cnt;
}

// slots shuffle to input_channel_ with needed-shuffle slots
void MultiSlotDataset::SlotsShuffle(
    const std::set<std::string>& slots_to_replace) {
  int out_channel_size = 0;
  if (cur_channel_ == 0) {
    for (size_t i = 0; i < multi_output_channel_.size(); ++i) {
      out_channel_size += multi_output_channel_[i]->Size();
    }
  } else {
    for (size_t i = 0; i < multi_consume_channel_.size(); ++i) {
      out_channel_size += multi_consume_channel_[i]->Size();
    }
  }
  VLOG(2) << "DatasetImpl<T>::SlotsShuffle() begin with input channel size: "
          << input_channel_->Size()
          << " output channel size: " << out_channel_size;
  if (!slots_shuffle_fea_eval_) {
    VLOG(3) << "DatasetImpl<T>::SlotsShuffle() end,"
               "fea eval mode off, need to set on for slots shuffle";
    return;
  }
  if ((!input_channel_ || input_channel_->Size() == 0) &&
      slots_shuffle_original_data_.size() == 0 && out_channel_size == 0) {
    VLOG(3) << "DatasetImpl<T>::SlotsShuffle() end, no data to slots shuffle";
    return;
  }
  platform::Timer timeline;
  timeline.Start();
  auto multi_slot_desc = data_feed_desc_.multi_slot_desc();
  std::set<uint16_t> index_slots;
  for (int i = 0; i < multi_slot_desc.slots_size(); ++i) {
    std::string cur_slot = multi_slot_desc.slots(i).name();
    if (slots_to_replace.find(cur_slot) != slots_to_replace.end()) {
      index_slots.insert(i);
    }
  }
  if (slots_shuffle_original_data_.size() == 0) {
    // before first slots shuffle, instances could be in
    // input_channel, oupput_channel or consume_channel
    if (input_channel_ && input_channel_->Size() != 0) {
      slots_shuffle_original_data_.reserve(input_channel_->Size());
      input_channel_->Close();
      input_channel_->ReadAll(slots_shuffle_original_data_);
    } else {
      CHECK(out_channel_size > 0);  // NOLINT
      if (cur_channel_ == 0) {
        for (size_t i = 0; i < multi_output_channel_.size(); ++i) {
          std::vector<Record> vec_data;
          multi_output_channel_[i]->Close();
          multi_output_channel_[i]->ReadAll(vec_data);
          slots_shuffle_original_data_.reserve(
              slots_shuffle_original_data_.size() + vec_data.size());
          slots_shuffle_original_data_.insert(
              slots_shuffle_original_data_.end(),
              std::make_move_iterator(vec_data.begin()),
              std::make_move_iterator(vec_data.end()));
          vec_data.clear();
          vec_data.shrink_to_fit();
          multi_output_channel_[i]->Clear();
        }
      } else {
        for (size_t i = 0; i < multi_consume_channel_.size(); ++i) {
          std::vector<Record> vec_data;
          multi_consume_channel_[i]->Close();
          multi_consume_channel_[i]->ReadAll(vec_data);
          slots_shuffle_original_data_.reserve(
              slots_shuffle_original_data_.size() + vec_data.size());
          slots_shuffle_original_data_.insert(
              slots_shuffle_original_data_.end(),
              std::make_move_iterator(vec_data.begin()),
              std::make_move_iterator(vec_data.end()));
          vec_data.clear();
          vec_data.shrink_to_fit();
          multi_consume_channel_[i]->Clear();
        }
      }
    }
  } else {
    // if already have original data for slots shuffle, clear channel
    input_channel_->Clear();
    if (cur_channel_ == 0) {
      for (size_t i = 0; i < multi_output_channel_.size(); ++i) {
        if (!multi_output_channel_[i]) {
          continue;
        }
        multi_output_channel_[i]->Clear();
      }
    } else {
      for (size_t i = 0; i < multi_consume_channel_.size(); ++i) {
        if (!multi_consume_channel_[i]) {
          continue;
        }
        multi_consume_channel_[i]->Clear();
      }
    }
  }
  int end_size = 0;
  if (cur_channel_ == 0) {
    for (size_t i = 0; i < multi_output_channel_.size(); ++i) {
      if (!multi_output_channel_[i]) {
        continue;
      }
      end_size += multi_output_channel_[i]->Size();
    }
  } else {
    for (size_t i = 0; i < multi_consume_channel_.size(); ++i) {
      if (!multi_consume_channel_[i]) {
        continue;
      }
      end_size += multi_consume_channel_[i]->Size();
    }
  }
  CHECK(input_channel_->Size() == 0)
      << "input channel should be empty before slots shuffle";
  std::vector<Record> random_data;
  random_data.clear();
  // get slots shuffled random_data
  GetRandomData(index_slots, &random_data);
  input_channel_->Open();
  input_channel_->Write(std::move(random_data));
  random_data.clear();
  random_data.shrink_to_fit();
  input_channel_->Close();
  cur_channel_ = 0;

  timeline.Pause();
  VLOG(2) << "DatasetImpl<T>::SlotsShuffle() end"
          << ", memory data size for slots shuffle=" << input_channel_->Size()
          << ", cost time=" << timeline.ElapsedSec() << " seconds";
}

}  // end namespace framework
}  // end namespace paddle

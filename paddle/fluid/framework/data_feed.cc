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

#include "paddle/fluid/framework/data_feed.h"
#include <stdio_ext.h>
#include <utility>
#include "gflags/gflags.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"
#include "io/fs.h"
#include "io/shell.h"
#include "paddle/fluid/framework/feed_fetch_method.h"
#include "paddle/fluid/framework/feed_fetch_type.h"
#include "paddle/fluid/platform/timer.h"

namespace paddle {
namespace framework {

void DataFeed::AddFeedVar(Variable* var, const std::string& name) {
  CheckInit();
  for (size_t i = 0; i < use_slots_.size(); ++i) {
    if (name == use_slots_[i]) {
      feed_vec_[i] = var->GetMutable<LoDTensor>();
    }
  }
}

bool DataFeed::SetFileList(const std::vector<std::string>& files) {
  std::unique_lock<std::mutex> lock(*mutex_for_pick_file_);
  CheckInit();
  // Do not set finish_set_filelist_ flag,
  // since a user may set file many times after init reader
  /*
  if (finish_set_filelist_) {
    VLOG(3) << "info: you have set the filelist.";
    return false;
  }
  */
  // PADDLE_ENFORCE(files.size(), "You have set an empty filelist.");
  filelist_.assign(files.begin(), files.end());

  finish_set_filelist_ = true;
  return true;
}

void DataFeed::SetBatchSize(int batch_size) {
  PADDLE_ENFORCE(batch_size > 0, "Illegal batch size: %d.", batch_size);
  default_batch_size_ = batch_size;
}

bool DataFeed::PickOneFile(std::string* filename) {
  PADDLE_ENFORCE(mutex_for_pick_file_ != nullptr,
                 "should call SetFileListMutex before PickOneFile");
  PADDLE_ENFORCE(file_idx_ != nullptr,
                 "should call SetFileListIndex before PickOneFile");
  std::unique_lock<std::mutex> lock(*mutex_for_pick_file_);
  if (*file_idx_ == filelist_.size()) {
    VLOG(3) << "DataFeed::PickOneFile no more file to pick";
    return false;
  }
  VLOG(3) << "file_idx_=" << *file_idx_;
  *filename = filelist_[(*file_idx_)++];
  // LOG(ERROR) << "pick file:" << *filename;
  return true;
}

void DataFeed::CheckInit() {
  PADDLE_ENFORCE(finish_init_, "Initialization did not succeed.");
}

void DataFeed::CheckSetFileList() {
  PADDLE_ENFORCE(finish_set_filelist_, "Set filelist did not succeed.");
}

void DataFeed::CheckStart() {
  PADDLE_ENFORCE(finish_start_, "Datafeed has not started running yet.");
}

template <typename T>
void PrivateQueueDataFeed<T>::SetQueueSize(int queue_size) {
  PADDLE_ENFORCE(queue_size > 0, "Illegal queue size: %d.", queue_size);
  queue_size_ = queue_size;
  queue_ = std::unique_ptr<paddle::operators::reader::BlockingQueue<T>>(
      new paddle::operators::reader::BlockingQueue<T>(queue_size_));
}

template <typename T>
bool PrivateQueueDataFeed<T>::Start() {
  CheckSetFileList();
  read_thread_ = std::thread(&PrivateQueueDataFeed::ReadThread, this);
  read_thread_.detach();

  finish_start_ = true;
  return true;
}

template <typename T>
void PrivateQueueDataFeed<T>::ReadThread() {
  std::string filename;
  while (PickOneFile(&filename)) {
    int err_no = 0;
    fp_ = fs_open_read(filename, &err_no, pipe_command_);
    __fsetlocking(&*fp_, FSETLOCKING_BYCALLER);
    thread_local string::LineFileReader reader;
    T instance;
    while (ParseOneInstanceFromPipe(&instance)) {
      queue_->Send(instance);
    }
  }
  queue_->Close();
}

template <typename T>
int PrivateQueueDataFeed<T>::Next() {
  CheckStart();
  int index = 0;
  T instance;
  T ins_vec;
  while (index < default_batch_size_) {
    if (!queue_->Receive(&instance)) {
      break;
    }
    AddInstanceToInsVec(&ins_vec, instance, index++);
  }
  batch_size_ = index;
  if (batch_size_ != 0) {
    PutToFeedVec(ins_vec);
  }
  return batch_size_;
}

// explicit instantiation
template class PrivateQueueDataFeed<std::vector<MultiSlotType>>;

template <typename T>
InMemoryDataFeed<T>::InMemoryDataFeed() {
  cur_channel_ = 0;
  shuffled_ins_ = std::make_shared<paddle::framework::BlockingQueue<T>>();
  shuffled_ins_out_ = std::make_shared<paddle::framework::BlockingQueue<T>>();
  fleet_send_batch_size_ = 80000;
  memory_data_ = nullptr;
  mutex_for_update_memory_data_ = nullptr;
  this->file_idx_ = nullptr;
  this->mutex_for_pick_file_ = nullptr;
}

template <typename T>
bool InMemoryDataFeed<T>::Start() {
  DataFeed::CheckSetFileList();
  if (shuffled_ins_->Size() == 0 && shuffled_ins_out_->Size() == 0) {
    FillMemoryDataToChannel();
  }
  DataFeed::finish_start_ = true;
  return true;
}

template <typename T>
int InMemoryDataFeed<T>::Next() {
  DataFeed::CheckStart();
  std::shared_ptr<paddle::framework::BlockingQueue<T>> in_channel = nullptr;
  std::shared_ptr<paddle::framework::BlockingQueue<T>> out_channel = nullptr;
  if (cur_channel_ == 0) {
    in_channel = shuffled_ins_;
    out_channel = shuffled_ins_out_;
  } else {
    in_channel = shuffled_ins_out_;
    out_channel = shuffled_ins_;
  }
  CHECK(in_channel != nullptr);
  CHECK(out_channel != nullptr);
  VLOG(3) << "in_channel size=" << in_channel->Size()
          << ", out_channel size=" << out_channel->Size()
          << ", thread_id=" << thread_id_;
  int index = 0;
  T instance;
  T ins_vec;
  while (index < DataFeed::default_batch_size_) {
    if (in_channel->Size() == 0) {
      break;
    }
    in_channel->Pop(&instance);

    AddInstanceToInsVec(&ins_vec, instance, index++);
    out_channel->Push(std::move(instance));
  }
  DataFeed::batch_size_ = index;
  VLOG(3) << "batch_size_=" << DataFeed::batch_size_
          << ", thread_id=" << thread_id_;
  if (DataFeed::batch_size_ != 0) {
    PutToFeedVec(ins_vec);
  } else {
    cur_channel_ = 1 - cur_channel_;
  }
  return DataFeed::batch_size_;
}

template <typename T>
void InMemoryDataFeed<T>::SetMemoryData(void* memory_data) {
  memory_data_ = static_cast<std::vector<T>*>(memory_data);
}

template <typename T>
void InMemoryDataFeed<T>::SetMemoryDataMutex(std::mutex* mutex) {
  mutex_for_update_memory_data_ = mutex;
}

template <typename T>
void InMemoryDataFeed<T>::SetThreadId(int thread_id) {
  thread_id_ = thread_id;
}

template <typename T>
void InMemoryDataFeed<T>::SetThreadNum(int thread_num) {
  thread_num_ = thread_num;
}

template <typename T>
void InMemoryDataFeed<T>::SetTrainerNum(int trainer_num) {
  trainer_num_ = trainer_num;
}

template <typename T>
void InMemoryDataFeed<T>::PutInsToChannel(const std::string& ins_str) {
  std::vector<T> ins;
  DeserializeIns(&ins, ins_str);
  shuffled_ins_->Extend(std::move(ins));
  VLOG(3) << "PutInsToChannel put ins num=" << ins.size()
          << " to channel, channel size=" << shuffled_ins_->Size()
          << " thread_id=" << thread_id_;
}

template <typename T>
void InMemoryDataFeed<T>::FillMemoryDataToChannel() {
  VLOG(3) << "FillMemoryDataToChannel, thread_id=" << thread_id_;
  auto interval = GetMemoryDataInterval();
  VLOG(3) << "memory data size=" << memory_data_->size()
          << ", fill data from  [" << interval.first << ", "
          << interval.second << "), thread_id=" << thread_id_;
  for (int64_t i = interval.first; i < interval.second; ++i) {
    T& t = (*memory_data_)[i];
    shuffled_ins_->Push(std::move(t));
  }
}

template <typename T>
void InMemoryDataFeed<T>::FillChannelToMemoryData() {
  VLOG(3) << "FillChannelToMemoryData, thread_id=" << thread_id_;
  std::vector<T> local_vec;
  std::shared_ptr<paddle::framework::BlockingQueue<T>> channel = nullptr;
  std::shared_ptr<paddle::framework::BlockingQueue<T>> pre_channel = nullptr;
  if (cur_channel_ == 0) {
    channel = shuffled_ins_;
    pre_channel = shuffled_ins_out_;
  } else {
    channel = shuffled_ins_out_;
    pre_channel = shuffled_ins_;
  }
  CHECK(channel != nullptr);
  CHECK(pre_channel != nullptr);
  CHECK_EQ(pre_channel->Size(), 0);
  local_vec.resize(channel->Size());
  for (int64_t i = 0; i < local_vec.size(); ++i) {
    channel->Pop(&local_vec[i]);
  }
  VLOG(3) << "local_vec size=" << local_vec.size()
          <<", thread_id=" << thread_id_;
  {
    std::lock_guard<std::mutex> g(*mutex_for_update_memory_data_);
    VLOG(3) << "before insert, memory_data_ size=" << memory_data_->size()
            << ", thread_id=" << thread_id_;
    memory_data_->insert(memory_data_->end(), local_vec.begin(),
        local_vec.end());
    VLOG(3) << "after insert memory_data_ size=" << memory_data_->size()
            << ", thread_id=" << thread_id_;
  }
  std::vector<T>().swap(local_vec);
}

template <typename T>
void InMemoryDataFeed<T>::LoadIntoMemory() {
  VLOG(3) << "LoadIntoMemory() begin, thread_id=" << thread_id_;
  std::vector<T> local_vec;
  std::string filename;
  while (DataFeed::PickOneFile(&filename)) {
    VLOG(3) << "PickOneFile, filename=" << filename
            << ", thread_id=" << thread_id_;
    int err_no = 0;
    PrivateQueueDataFeed<T>::fp_ =
        fs_open_read(filename, &err_no, PrivateQueueDataFeed<T>::pipe_command_);
    CHECK(PrivateQueueDataFeed<T>::fp_ != nullptr);
    __fsetlocking(&*PrivateQueueDataFeed<T>::fp_, FSETLOCKING_BYCALLER);
    T instance;
    platform::Timer timeline;
    timeline.Start();
    while (ParseOneInstanceFromPipe(&instance)) {
      local_vec.push_back(instance);
    }
    timeline.Pause();
    VLOG(3) << "LoadIntoMemory() read all lines, file="
            << filename << ", cost time=" << timeline.ElapsedSec()
            << " seconds, thread_id=" << thread_id_;
    {
      std::lock_guard<std::mutex> lock(*mutex_for_update_memory_data_);
      timeline.Start();
      memory_data_->insert(memory_data_->end(),
                           std::make_move_iterator(local_vec.begin()),
                           std::make_move_iterator(local_vec.end()));
      timeline.Pause();
      VLOG(3) << "LoadIntoMemory() memory_data insert, cost time="
              << timeline.ElapsedSec() << " seconds, thread_id="
              << thread_id_;
    }
    local_vec.clear();
  }
  std::vector<T>().swap(local_vec);
  VLOG(3) << "LoadIntoMemory() end, thread_id=" << thread_id_;
}

template <typename T>
void InMemoryDataFeed<T>::LocalShuffle() {
  VLOG(3) << "LocalShuffle() begin, thread_id=" << thread_id_;
  FillMemoryDataToChannel();
  VLOG(3) << "LocalShuffle() end, thread_id=" << thread_id_;
}

template <typename T>
void InMemoryDataFeed<T>::GlobalShuffle() {
  VLOG(3) << "GlobalShuffle() begin, thread_id=" << thread_id_;
  auto fleet_ptr = FleetWrapper::GetInstance();
  std::vector<std::vector<T*>> send_vec(trainer_num_);
  for (auto& vec : send_vec) {
    vec.reserve(fleet_send_batch_size_);
  }
  std::vector<std::future<int32_t>> total_status;
  auto interval = GetMemoryDataInterval();
  VLOG(3) << "global shuffle data from  [" << interval.first << ", "
          << interval.second << "), thread_id=" << thread_id_;
  for (int64_t i = interval.first; i < interval.second; ++i) {
    // if get ins id, can also use hash
    // std::string ins_id = memory_data_[i].ins_id;
    int64_t random_num = fleet_ptr->LocalRandomEngine()();
    int64_t node_id = random_num % trainer_num_;
    send_vec[node_id].push_back(&((*memory_data_)[i]));
    if (i % fleet_send_batch_size_ == 0 && i != 0) {
      for (int j = 0; j < send_vec.size(); ++j) {
        std::string send_str;
        SerializeIns(send_vec[j], &send_str);
        VLOG(3) << "send str_length=" << send_str.length()
                << ", ins num=" << send_vec[j].size() << " to node_id="
                << j << ", thread_id=" << thread_id_;
        auto ret = fleet_ptr->SendClientToClientMsg(0, j, send_str);
        VLOG(3) << "end send, thread_id=" << thread_id_;
        send_vec[j].clear();
        total_status.push_back(std::move(ret));
      }
    }
  }
  for (int j = 0; j < send_vec.size(); ++j) {
    if (send_vec[j].size() != 0) {
      std::string send_str;
      SerializeIns(send_vec[j], &send_str);
      VLOG(3) << "send str_length=" << send_str.length()
              << " to node_id=" << j << ", thread_id=" << thread_id_;
      auto ret = fleet_ptr->SendClientToClientMsg(0, j, send_str);
      VLOG(3) << "end send, thread_id=" << thread_id_;
      total_status.push_back(std::move(ret));
    }
    std::vector<T*>().swap(send_vec[j]);
  }
  for (auto& t : total_status) {
    t.wait();
  }
  VLOG(3) << "GlobalShuffle() end, thread_id=" << thread_id_;
}

template <typename T>
std::pair<int64_t, int64_t> InMemoryDataFeed<T>::GetMemoryDataInterval() {
  int64_t start = 0;
  int64_t end = 0;
  int64_t size = memory_data_->size();
  for (int64_t i = 0; i <= static_cast<int64_t>(thread_id_); ++i) {
    int64_t len = size / static_cast<int64_t>(thread_num_) +
                  (i < (size % static_cast<int64_t>(thread_num_)));
    start = end;
    end += len;
  }
  return std::make_pair(start, end);
}

// explicit instantiation
template class InMemoryDataFeed<std::vector<MultiSlotType>>;

void MultiSlotDataFeed::Init(
    const paddle::framework::DataFeedDesc& data_feed_desc) {
  finish_init_ = false;
  finish_set_filelist_ = false;
  finish_start_ = false;

  PADDLE_ENFORCE(data_feed_desc.has_multi_slot_desc(),
                 "Multi_slot_desc has not been set.");
  paddle::framework::MultiSlotDesc multi_slot_desc =
      data_feed_desc.multi_slot_desc();
  SetBatchSize(data_feed_desc.batch_size());
  SetQueueSize(data_feed_desc.batch_size());
  size_t all_slot_num = multi_slot_desc.slots_size();
  all_slots_.resize(all_slot_num);
  all_slots_type_.resize(all_slot_num);
  use_slots_index_.resize(all_slot_num);
  use_slots_.clear();
  use_slots_is_dense_.clear();
  for (size_t i = 0; i < all_slot_num; ++i) {
    const auto& slot = multi_slot_desc.slots(i);
    all_slots_[i] = slot.name();
    all_slots_type_[i] = slot.type();
    use_slots_index_[i] = slot.is_used() ? use_slots_.size() : -1;
    if (slot.is_used()) {
      use_slots_.push_back(all_slots_[i]);
      use_slots_is_dense_.push_back(slot.is_dense());
    }
  }
  feed_vec_.resize(use_slots_.size());
  pipe_command_ = data_feed_desc.pipe_command();
  finish_init_ = true;
}

void MultiSlotDataFeed::ReadThread() {
  std::string filename;
  while (PickOneFile(&filename)) {
    int err_no = 0;
    fp_ = fs_open_read(filename, &err_no, pipe_command_);
    CHECK(fp_ != nullptr);
    __fsetlocking(&*fp_, FSETLOCKING_BYCALLER);
    thread_local string::LineFileReader reader;
    std::vector<MultiSlotType> instance;
    int ins_num = 0;
    while (ParseOneInstanceFromPipe(&instance)) {
      ins_num++;
      queue_->Send(instance);
    }
    VLOG(3) << "filename: " << filename << " inst num: " << ins_num;
  }
  queue_->Close();
}

bool MultiSlotDataFeed::CheckFile(const char* filename) {
  CheckInit();  // get info of slots
  std::ifstream fin(filename);
  if (!fin.good()) {
    VLOG(1) << "error: open file<" << filename << "> fail";
    return false;
  }
  std::string line;
  int instance_cout = 0;
  std::string all_slots_alias = "";
  for (const auto& alias : all_slots_) {
    all_slots_alias += alias + " ";
  }
  std::string use_slots_alias = "";
  for (const auto& alias : use_slots_) {
    use_slots_alias += alias + " ";
  }
  VLOG(3) << "total slots num: " << all_slots_.size();
  VLOG(3) << "total slots alias: " << all_slots_alias;
  VLOG(3) << "used slots num: " << use_slots_.size();
  VLOG(3) << "used slots alias: " << use_slots_alias;
  while (getline(fin, line)) {
    ++instance_cout;
    const char* str = line.c_str();
    char* endptr = const_cast<char*>(str);
    int len = line.length();
    for (size_t i = 0; i < all_slots_.size(); ++i) {
      int num = strtol(endptr, &endptr, 10);
      if (num < 0) {
        VLOG(0) << "error: the number of ids is a negative number: " << num;
        VLOG(0) << "please check line<" << instance_cout << "> in file<"
                << filename << ">";
        return false;
      } else if (num == 0) {
        VLOG(0)
            << "error: the number of ids can not be zero, you need "
               "padding it in data generator; or if there is something wrong"
               " with the data, please check if the data contains unresolvable "
               "characters.";
        VLOG(0) << "please check line<" << instance_cout << "> in file<"
                << filename << ">";
        return false;
      } else if (errno == ERANGE || num > INT_MAX) {
        VLOG(0) << "error: the number of ids greater than INT_MAX";
        VLOG(0) << "please check line<" << instance_cout << "> in file<"
                << filename << ">";
        return false;
      }
      if (all_slots_type_[i] == "float") {
        for (int i = 0; i < num; ++i) {
          strtof(endptr, &endptr);
          if (errno == ERANGE) {
            VLOG(0) << "error: the value is out of the range of "
                       "representable values for float";
            VLOG(0) << "please check line<" << instance_cout << "> in file<"
                    << filename << ">";
            return false;
          }
          if (i + 1 != num && endptr - str == len) {
            VLOG(0) << "error: there is a wrong with the number of ids.";
            VLOG(0) << "please check line<" << instance_cout << "> in file<"
                    << filename << ">";
            return false;
          }
        }
      } else if (all_slots_type_[i] == "uint64") {
        for (int i = 0; i < num; ++i) {
          strtoull(endptr, &endptr, 10);
          if (errno == ERANGE) {
            VLOG(0) << "error: the value is out of the range of "
                       "representable values for uint64_t";
            VLOG(0) << "please check line<" << instance_cout << "> in file<"
                    << filename << ">";
            return false;
          }
          if (i + 1 != num && endptr - str == len) {
            VLOG(0) << "error: there is a wrong with the number of ids.";
            VLOG(0) << "please check line<" << instance_cout << "> in file<"
                    << filename << ">";
            return false;
          }
        }
      } else {
        VLOG(0) << "error: this type<" << all_slots_type_[i]
                << "> is not supported";
        return false;
      }
    }
    // It may be added '\t' character to the end of the output of reduce
    // task when processes data by Hadoop(when the output of the reduce
    // task of Hadoop has only one field, it will add a '\t' at the end
    // of the line by default, and you can use this option to avoid it:
    // `-D mapred.textoutputformat.ignoreseparator=true`), which does
    // not affect the correctness of the data. Therefore, it should be
    // judged that the data is not normal when the end of each line of
    // data contains characters which are not spaces.
    while (endptr - str != len) {
      if (!isspace(*(endptr++))) {
        VLOG(0)
            << "error: there is some extra characters at the end of the line.";
        VLOG(0) << "please check line<" << instance_cout << "> in file<"
                << filename << ">";
        return false;
      }
    }
  }
  VLOG(3) << "instances cout: " << instance_cout;
  VLOG(3) << "The file format is correct";
  return true;
}

bool MultiSlotDataFeed::ParseOneInstanceFromPipe(
    std::vector<MultiSlotType>* instance) {
  thread_local string::LineFileReader reader;

  if (!reader.getline(&*(fp_.get()))) {
    return false;
  } else {
    int use_slots_num = use_slots_.size();
    instance->resize(use_slots_num);

    const char* str = reader.get();
    std::string line = std::string(str);
    // VLOG(3) << line;
    char* endptr = const_cast<char*>(str);
    int pos = 0;
    for (size_t i = 0; i < use_slots_index_.size(); ++i) {
      int idx = use_slots_index_[i];
      int num = strtol(&str[pos], &endptr, 10);
      PADDLE_ENFORCE(
          num,
          "The number of ids can not be zero, you need padding "
          "it in data generator; or if there is something wrong with "
          "the data, please check if the data contains unresolvable "
          "characters.\nplease check this error line: %s",
          str);
      if (idx != -1) {
        (*instance)[idx].Init(all_slots_type_[i]);
        if ((*instance)[idx].GetType()[0] == 'f') {  // float
          for (int j = 0; j < num; ++j) {
            float feasign = strtof(endptr, &endptr);
            (*instance)[idx].AddValue(feasign);
          }
        } else if ((*instance)[idx].GetType()[0] == 'u') {  // uint64
          for (int j = 0; j < num; ++j) {
            uint64_t feasign = (uint64_t)strtoull(endptr, &endptr, 10);
            (*instance)[idx].AddValue(feasign);
          }
        }
        pos = endptr - str;
      } else {
        for (int j = 0; j <= num; ++j) {
          // pos = line.find_first_of(' ', pos + 1);
          while (line[pos + 1] != ' ') {
            pos++;
          }
        }
      }
    }
    return true;
  }
}

bool MultiSlotDataFeed::ParseOneInstance(std::vector<MultiSlotType>* instance) {
  std::string line;
  if (getline(file_, line)) {
    int use_slots_num = use_slots_.size();
    instance->resize(use_slots_num);
    // parse line
    const char* str = line.c_str();
    char* endptr = const_cast<char*>(str);
    int pos = 0;
    for (size_t i = 0; i < use_slots_index_.size(); ++i) {
      int idx = use_slots_index_[i];
      int num = strtol(&str[pos], &endptr, 10);
      PADDLE_ENFORCE(
          num,
          "The number of ids can not be zero, you need padding "
          "it in data generator; or if there is something wrong with "
          "the data, please check if the data contains unresolvable "
          "characters.\nplease check this error line: %s",
          str);

      if (idx != -1) {
        (*instance)[idx].Init(all_slots_type_[i]);
        if ((*instance)[idx].GetType()[0] == 'f') {  // float
          for (int j = 0; j < num; ++j) {
            float feasign = strtof(endptr, &endptr);
            (*instance)[idx].AddValue(feasign);
          }
        } else if ((*instance)[idx].GetType()[0] == 'u') {  // uint64
          for (int j = 0; j < num; ++j) {
            uint64_t feasign = (uint64_t)strtoull(endptr, &endptr, 10);
            (*instance)[idx].AddValue(feasign);
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

void MultiSlotDataFeed::AddInstanceToInsVec(
    std::vector<MultiSlotType>* ins_vec,
    const std::vector<MultiSlotType>& instance, int index) {
  if (index == 0) {
    ins_vec->resize(instance.size());
    for (size_t i = 0; i < instance.size(); ++i) {
      (*ins_vec)[i].Init(instance[i].GetType());
      (*ins_vec)[i].InitOffset();
    }
  }

  for (size_t i = 0; i < instance.size(); ++i) {
    (*ins_vec)[i].AddIns(instance[i]);
  }
}

void MultiSlotDataFeed::PutToFeedVec(
    const std::vector<MultiSlotType>& ins_vec) {
  for (size_t i = 0; i < use_slots_.size(); ++i) {
    const auto& type = ins_vec[i].GetType();
    const auto& offset = ins_vec[i].GetOffset();
    int total_instance = static_cast<int>(offset.back());

    if (type[0] == 'f') {  // float
      const auto& feasign = ins_vec[i].GetFloatData();
      float* tensor_ptr = feed_vec_[i]->mutable_data<float>(
          {total_instance, 1}, platform::CPUPlace());
      memcpy(tensor_ptr, &feasign[0], total_instance * sizeof(float));
    } else if (type[0] == 'u') {  // uint64
      // no uint64_t type in paddlepaddle
      const auto& feasign = ins_vec[i].GetUint64Data();
      int64_t* tensor_ptr = feed_vec_[i]->mutable_data<int64_t>(
          {total_instance, 1}, platform::CPUPlace());
      memcpy(tensor_ptr, &feasign[0], total_instance * sizeof(int64_t));
    }

    LoD data_lod{offset};
    feed_vec_[i]->set_lod(data_lod);
    if (use_slots_is_dense_[i]) {
      int dim = total_instance / batch_size_;
      feed_vec_[i]->Resize({batch_size_, dim});
    }
  }
}

void MultiSlotInMemoryDataFeed::Init(
    const paddle::framework::DataFeedDesc& data_feed_desc) {
  finish_init_ = false;
  finish_set_filelist_ = false;
  finish_start_ = false;

  PADDLE_ENFORCE(data_feed_desc.has_multi_slot_desc(),
                 "Multi_slot_desc has not been set.");
  paddle::framework::MultiSlotDesc multi_slot_desc =
      data_feed_desc.multi_slot_desc();
  SetBatchSize(data_feed_desc.batch_size());
  SetQueueSize(data_feed_desc.batch_size());
  size_t all_slot_num = multi_slot_desc.slots_size();
  all_slots_.resize(all_slot_num);
  all_slots_type_.resize(all_slot_num);
  use_slots_index_.resize(all_slot_num);
  use_slots_.clear();
  use_slots_is_dense_.clear();
  for (size_t i = 0; i < all_slot_num; ++i) {
    const auto& slot = multi_slot_desc.slots(i);
    all_slots_[i] = slot.name();
    all_slots_type_[i] = slot.type();
    use_slots_index_[i] = slot.is_used() ? use_slots_.size() : -1;
    if (slot.is_used()) {
      use_slots_.push_back(all_slots_[i]);
      use_slots_is_dense_.push_back(slot.is_dense());
    }
  }
  feed_vec_.resize(use_slots_.size());
  pipe_command_ = data_feed_desc.pipe_command();
  finish_init_ = true;
}

bool MultiSlotInMemoryDataFeed::ParseOneInstanceFromPipe(
    std::vector<MultiSlotType>* instance) {
  thread_local string::LineFileReader reader;

  if (!reader.getline(&*(fp_.get()))) {
    return false;
  } else {
    int use_slots_num = use_slots_.size();
    instance->resize(use_slots_num);

    const char* str = reader.get();
    std::string line = std::string(str);
    // VLOG(3) << line;
    char* endptr = const_cast<char*>(str);
    int pos = 0;
    for (size_t i = 0; i < use_slots_index_.size(); ++i) {
      int idx = use_slots_index_[i];
      int num = strtol(&str[pos], &endptr, 10);
      PADDLE_ENFORCE(
          num,
          "The number of ids can not be zero, you need padding "
          "it in data generator; or if there is something wrong with "
          "the data, please check if the data contains unresolvable "
          "characters.\nplease check this error line: %s",
          str);
      if (idx != -1) {
        (*instance)[idx].Init(all_slots_type_[i]);
        if ((*instance)[idx].GetType()[0] == 'f') {  // float
          for (int j = 0; j < num; ++j) {
            float feasign = strtof(endptr, &endptr);
            (*instance)[idx].AddValue(feasign);
          }
        } else if ((*instance)[idx].GetType()[0] == 'u') {  // uint64
          for (int j = 0; j < num; ++j) {
            uint64_t feasign = (uint64_t)strtoull(endptr, &endptr, 10);
            (*instance)[idx].AddValue(feasign);
          }
        }
        pos = endptr - str;
      } else {
        for (int j = 0; j <= num; ++j) {
          // pos = line.find_first_of(' ', pos + 1);
          while (line[pos + 1] != ' ') {
            pos++;
          }
        }
      }
    }
    return true;
  }
}

bool MultiSlotInMemoryDataFeed::ParseOneInstance(
    std::vector<MultiSlotType>* instance) {
  std::string line;
  if (getline(file_, line)) {
    int use_slots_num = use_slots_.size();
    instance->resize(use_slots_num);
    VLOG(3) << line;
    // parse line
    const char* str = line.c_str();
    char* endptr = const_cast<char*>(str);
    int pos = 0;
    for (size_t i = 0; i < use_slots_index_.size(); ++i) {
      int idx = use_slots_index_[i];
      int num = strtol(&str[pos], &endptr, 10);
      PADDLE_ENFORCE(
          num,
          "The number of ids can not be zero, you need padding "
          "it in data generator; or if there is something wrong with "
          "the data, please check if the data contains unresolvable "
          "characters.\nplease check this error line: %s",
          str);

      if (idx != -1) {
        (*instance)[idx].Init(all_slots_type_[i]);
        if ((*instance)[idx].GetType()[0] == 'f') {  // float
          for (int j = 0; j < num; ++j) {
            float feasign = strtof(endptr, &endptr);
            (*instance)[idx].AddValue(feasign);
          }
        } else if ((*instance)[idx].GetType()[0] == 'u') {  // uint64
          for (int j = 0; j < num; ++j) {
            uint64_t feasign = (uint64_t)strtoull(endptr, &endptr, 10);
            (*instance)[idx].AddValue(feasign);
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

void MultiSlotInMemoryDataFeed::AddInstanceToInsVec(
    std::vector<MultiSlotType>* ins_vec,
    const std::vector<MultiSlotType>& instance, int index) {
  if (index == 0) {
    ins_vec->resize(instance.size());
    for (size_t i = 0; i < instance.size(); ++i) {
      (*ins_vec)[i].Init(instance[i].GetType());
      (*ins_vec)[i].InitOffset();
    }
  }

  for (size_t i = 0; i < instance.size(); ++i) {
    (*ins_vec)[i].AddIns(instance[i]);
  }
}

void MultiSlotInMemoryDataFeed::PutToFeedVec(
    const std::vector<MultiSlotType>& ins_vec) {
  for (size_t i = 0; i < use_slots_.size(); ++i) {
    const auto& type = ins_vec[i].GetType();
    const auto& offset = ins_vec[i].GetOffset();
    int total_instance = static_cast<int>(offset.back());

    if (type[0] == 'f') {  // float
      const auto& feasign = ins_vec[i].GetFloatData();
      float* tensor_ptr = feed_vec_[i]->mutable_data<float>(
          {total_instance, 1}, platform::CPUPlace());
      memcpy(tensor_ptr, &feasign[0], total_instance * sizeof(float));
    } else if (type[0] == 'u') {  // uint64
      // no uint64_t type in paddlepaddle
      const auto& feasign = ins_vec[i].GetUint64Data();
      int64_t* tensor_ptr = feed_vec_[i]->mutable_data<int64_t>(
          {total_instance, 1}, platform::CPUPlace());
      memcpy(tensor_ptr, &feasign[0], total_instance * sizeof(int64_t));
    }

    LoD data_lod{offset};
    feed_vec_[i]->set_lod(data_lod);
    if (use_slots_is_dense_[i]) {
      int dim = total_instance / batch_size_;
      feed_vec_[i]->Resize({batch_size_, dim});
    }
  }
}

// todo serialize ins in global shuffle
void MultiSlotInMemoryDataFeed::SerializeIns(
    const std::vector<std::vector<MultiSlotType>*>& ins,
    std::string* str) {
  auto fleet_ptr = FleetWrapper::GetInstance();
  fleet_ptr->Serialize(ins, str);
}
// todo deserialize ins in global shuffle
void MultiSlotInMemoryDataFeed::DeserializeIns(
    std::vector<std::vector<MultiSlotType>>* ins,
    const std::string& str) {
  auto fleet_ptr = FleetWrapper::GetInstance();
  fleet_ptr->Deserialize(ins, str);
}

}  // namespace framework
}  // namespace paddle

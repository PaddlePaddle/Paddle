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

#if defined _WIN32 || defined __APPLE__
#else
#define _LINUX
#endif

#include "paddle/fluid/framework/data_feed.h"
#ifdef _LINUX
#include <stdio_ext.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#endif
#include <map>
#include <utility>
#include "gflags/gflags.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"
#include "io/fs.h"
#include "io/shell.h"
#include "paddle/fluid/framework/feed_fetch_method.h"
#include "paddle/fluid/framework/feed_fetch_type.h"
#include "paddle/fluid/framework/fleet/box_wrapper.h"
#include "paddle/fluid/framework/fleet/fleet_wrapper.h"
#include "paddle/fluid/platform/monitor.h"
#include "paddle/fluid/platform/timer.h"
#include "paddle/fluid/string/string_helper.h"
#ifdef PADDLE_WITH_BOX_PS
#include <dlfcn.h>
extern "C" {
typedef paddle::framework::ISlotParser* (*MyPadBoxGetObject)(void);
typedef void (*MyPadBoxFreeObject)(paddle::framework::ISlotParser*);
}
#endif

// USE_INT_STAT(STAT_total_feasign_num_in_mem);
namespace paddle {
namespace framework {
using platform::Timer;

void RecordCandidateList::ReSize(size_t length) {
  mutex_.lock();
  capacity_ = length;
  CHECK(capacity_ > 0);  // NOLINT
  candidate_list_.clear();
  candidate_list_.resize(capacity_);
  full_ = false;
  cur_size_ = 0;
  total_size_ = 0;
  mutex_.unlock();
}

void RecordCandidateList::ReInit() {
  mutex_.lock();
  full_ = false;
  cur_size_ = 0;
  total_size_ = 0;
  mutex_.unlock();
}

void RecordCandidateList::AddAndGet(const Record& record,
                                    RecordCandidate* result) {
  mutex_.lock();
  size_t index = 0;
  ++total_size_;
  auto fleet_ptr = FleetWrapper::GetInstance();
  if (!full_) {
    candidate_list_[cur_size_++] = record;
    full_ = (cur_size_ == capacity_);
  } else {
    CHECK(cur_size_ == capacity_);
    index = fleet_ptr->LocalRandomEngine()() % total_size_;
    if (index < capacity_) {
      candidate_list_[index] = record;
    }
  }
  index = fleet_ptr->LocalRandomEngine()() % cur_size_;
  *result = candidate_list_[index];
  mutex_.unlock();
}

void DataFeed::AddFeedVar(Variable* var, const std::string& name) {
  CheckInit();
  for (size_t i = 0; i < use_slots_.size(); ++i) {
    if (name == use_slots_[i]) {
      if (var == nullptr) {
        feed_vec_[i] = nullptr;
      } else {
        feed_vec_[i] = var->GetMutable<LoDTensor>();
      }
    }
  }
}

bool DataFeed::SetFileList(const std::vector<std::string>& files) {
  std::unique_lock<std::mutex> lock(*mutex_for_pick_file_);
  CheckInit();
  // Do not set finish_set_filelist_ flag,
  // since a user may set file many times after init reader
  filelist_.assign(files.begin(), files.end());

  finish_set_filelist_ = true;
  return true;
}

void DataFeed::SetBatchSize(int batch_size) {
  PADDLE_ENFORCE_GT(batch_size, 0,
                    platform::errors::InvalidArgument(
                        "Batch size %d is illegal.", batch_size));
  default_batch_size_ = batch_size;
}

bool DataFeed::PickOneFile(std::string* filename) {
  PADDLE_ENFORCE_NOT_NULL(
      mutex_for_pick_file_,
      platform::errors::PreconditionNotMet(
          "You should call SetFileListMutex before PickOneFile"));
  PADDLE_ENFORCE_NOT_NULL(
      file_idx_, platform::errors::PreconditionNotMet(
                     "You should call SetFileListIndex before PickOneFile"));
  std::unique_lock<std::mutex> lock(*mutex_for_pick_file_);
  if (*file_idx_ == filelist_.size()) {
    VLOG(3) << "DataFeed::PickOneFile no more file to pick";
    return false;
  }
  VLOG(3) << "file_idx_=" << *file_idx_;
  *filename = filelist_[(*file_idx_)++];
  return true;
}

void DataFeed::CheckInit() {
  PADDLE_ENFORCE(finish_init_, "Initialization did not succeed.");
}

void DataFeed::CheckSetFileList() {
  PADDLE_ENFORCE(finish_set_filelist_, "Set filelist did not succeed.");
}

void DataFeed::CheckStart() {
  PADDLE_ENFORCE_EQ(finish_start_, true,
                    platform::errors::PreconditionNotMet(
                        "Datafeed has not started running yet."));
}

void DataFeed::AssignFeedVar(const Scope& scope) {
  CheckInit();
  for (size_t i = 0; i < use_slots_.size(); ++i) {
    feed_vec_[i] = scope.FindVar(use_slots_[i])->GetMutable<LoDTensor>();
  }
}

void DataFeed::CopyToFeedTensor(void* dst, const void* src, size_t size) {
  if (platform::is_cpu_place(this->place_)) {
    memcpy(dst, src, size);
  } else {
#ifdef PADDLE_WITH_CUDA
    cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
#else
    PADDLE_THROW("Not supported GPU, Please compile WITH_GPU option");
#endif
  }
}

template <typename T>
void PrivateQueueDataFeed<T>::SetQueueSize(int queue_size) {
  PADDLE_ENFORCE(queue_size > 0, "Illegal queue size: %d.", queue_size);
  queue_size_ = queue_size;
  queue_ = paddle::framework::MakeChannel<T>();
  queue_->SetCapacity(queue_size);
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
#ifdef _LINUX
  std::string filename;
  while (PickOneFile(&filename)) {
    int err_no = 0;
    fp_ = fs_open_read(filename, &err_no, pipe_command_);
    __fsetlocking(&*fp_, FSETLOCKING_BYCALLER);
    T instance;
    while (ParseOneInstanceFromPipe(&instance)) {
      queue_->Put(instance);
    }
  }
  queue_->Close();
#endif
}

template <typename T>
int PrivateQueueDataFeed<T>::Next() {
#ifdef _LINUX
  CheckStart();
  int index = 0;
  T ins_vec;
  while (index < default_batch_size_) {
    T instance;
    if (!queue_->Get(instance)) {
      break;
    }
    AddInstanceToInsVec(&ins_vec, instance, index++);
  }
  batch_size_ = index;
  if (batch_size_ != 0) {
    PutToFeedVec(ins_vec);
  }
  return batch_size_;
#else
  return 0;
#endif
}

// explicit instantiation
template class PrivateQueueDataFeed<std::vector<MultiSlotType>>;

template <typename T>
InMemoryDataFeed<T>::InMemoryDataFeed() {
  this->file_idx_ = nullptr;
  this->mutex_for_pick_file_ = nullptr;
  this->fp_ = nullptr;
  this->thread_id_ = 0;
  this->thread_num_ = 1;
  this->parse_ins_id_ = false;
  this->parse_content_ = false;
  this->parse_logkey_ = false;
  this->enable_pv_merge_ = false;
  this->current_phase_ = 1;  // 1:join ;0:update
  this->input_channel_ = nullptr;
  this->output_channel_ = nullptr;
  this->consume_channel_ = nullptr;
}

template <typename T>
bool InMemoryDataFeed<T>::Start() {
#ifdef _LINUX
  this->CheckSetFileList();
  if (output_channel_->Size() == 0 && input_channel_->Size() != 0) {
    std::vector<T> data;
    input_channel_->Read(data);
    output_channel_->Write(std::move(data));
  }
#endif
  this->finish_start_ = true;
  return true;
}

template <typename T>
int InMemoryDataFeed<T>::Next() {
#ifdef _LINUX
  this->CheckStart();
  CHECK(output_channel_ != nullptr);
  CHECK(consume_channel_ != nullptr);
  VLOG(3) << "output_channel_ size=" << output_channel_->Size()
          << ", consume_channel_ size=" << consume_channel_->Size()
          << ", thread_id=" << thread_id_;
  int index = 0;
  T instance;
  std::vector<T> ins_vec;
  // ins_vec.clear();
  ins_vec.reserve(this->default_batch_size_);
  while (index < this->default_batch_size_) {
    if (output_channel_->Size() == 0) {
      break;
    }
    output_channel_->Get(instance);
    ins_vec.push_back(instance);
    ++index;
    consume_channel_->Put(std::move(instance));
  }
  this->batch_size_ = index;
  VLOG(3) << "batch_size_=" << this->batch_size_
          << ", thread_id=" << thread_id_;
  if (this->batch_size_ != 0) {
    PutToFeedVec(ins_vec);
  } else {
    VLOG(3) << "finish reading, output_channel_ size="
            << output_channel_->Size()
            << ", consume_channel_ size=" << consume_channel_->Size()
            << ", thread_id=" << thread_id_;
  }
  return this->batch_size_;
#else
  return 0;
#endif
}

template <typename T>
void InMemoryDataFeed<T>::SetInputChannel(void* channel) {
  input_channel_ = static_cast<paddle::framework::ChannelObject<T>*>(channel);
}

template <typename T>
void InMemoryDataFeed<T>::SetInputPtrChannel(void* channel) {
  input_ptr_channel_ =
      static_cast<paddle::framework::ChannelObject<T*>*>(channel);
}

template <typename T>
void InMemoryDataFeed<T>::SetOutputChannel(void* channel) {
  output_channel_ = static_cast<paddle::framework::ChannelObject<T>*>(channel);
}

template <typename T>
void InMemoryDataFeed<T>::SetConsumeChannel(void* channel) {
  consume_channel_ = static_cast<paddle::framework::ChannelObject<T>*>(channel);
}

template <typename T>
void InMemoryDataFeed<T>::SetOutputPtrChannel(void* channel) {
  output_ptr_channel_ =
      static_cast<paddle::framework::ChannelObject<T*>*>(channel);
}

template <typename T>
void InMemoryDataFeed<T>::SetConsumePtrChannel(void* channel) {
  consume_ptr_channel_ =
      static_cast<paddle::framework::ChannelObject<T*>*>(channel);
}

template <typename T>
void InMemoryDataFeed<T>::SetInputPvChannel(void* channel) {
  input_pv_channel_ =
      static_cast<paddle::framework::ChannelObject<PvInstance>*>(channel);
}

template <typename T>
void InMemoryDataFeed<T>::SetOutputPvChannel(void* channel) {
  output_pv_channel_ =
      static_cast<paddle::framework::ChannelObject<PvInstance>*>(channel);
}

template <typename T>
void InMemoryDataFeed<T>::SetConsumePvChannel(void* channel) {
  consume_pv_channel_ =
      static_cast<paddle::framework::ChannelObject<PvInstance>*>(channel);
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
void InMemoryDataFeed<T>::SetParseContent(bool parse_content) {
  parse_content_ = parse_content;
}

template <typename T>
void InMemoryDataFeed<T>::SetParseLogKey(bool parse_logkey) {
  parse_logkey_ = parse_logkey;
}

template <typename T>
void InMemoryDataFeed<T>::SetEnablePvMerge(bool enable_pv_merge) {
  enable_pv_merge_ = enable_pv_merge;
}

template <typename T>
void InMemoryDataFeed<T>::SetCurrentPhase(int current_phase) {
  current_phase_ = current_phase;
}

template <typename T>
void InMemoryDataFeed<T>::SetParseInsId(bool parse_ins_id) {
  parse_ins_id_ = parse_ins_id;
}

template <typename T>
void InMemoryDataFeed<T>::LoadIntoMemory() {
#ifdef _LINUX
  VLOG(3) << "LoadIntoMemory() begin, thread_id=" << thread_id_;
  std::string filename;
  while (this->PickOneFile(&filename)) {
    VLOG(3) << "PickOneFile, filename=" << filename
            << ", thread_id=" << thread_id_;
#ifdef PADDLE_WITH_BOX_PS
    if (BoxWrapper::GetInstance()->UseAfsApi()) {
      this->fp_ = BoxWrapper::GetInstance()->OpenReadFile(filename,
                                                          this->pipe_command_);
    } else {
#endif
      int err_no = 0;
      this->fp_ = fs_open_read(filename, &err_no, this->pipe_command_);
#ifdef PADDLE_WITH_BOX_PS
    }
#endif
    CHECK(this->fp_ != nullptr);
    __fsetlocking(&*(this->fp_), FSETLOCKING_BYCALLER);
    paddle::framework::ChannelWriter<T> writer(input_channel_);
    T instance;
    platform::Timer timeline;
    timeline.Start();
    while (ParseOneInstanceFromPipe(&instance)) {
      writer << std::move(instance);
      instance = T();
    }
    STAT_ADD(STAT_total_feasign_num_in_mem, fea_num_);
    {
      std::lock_guard<std::mutex> flock(*mutex_for_fea_num_);
      *total_fea_num_ += fea_num_;
      fea_num_ = 0;
    }
    writer.Flush();
    timeline.Pause();
    VLOG(3) << "LoadIntoMemory() read all lines, file=" << filename
            << ", cost time=" << timeline.ElapsedSec()
            << " seconds, thread_id=" << thread_id_;
  }
  VLOG(3) << "LoadIntoMemory() end, thread_id=" << thread_id_;
#endif
}

// explicit instantiation
template class InMemoryDataFeed<Record>;

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
  // temporarily set queue size = batch size * 100
  SetQueueSize(data_feed_desc.batch_size() * 100);
  size_t all_slot_num = multi_slot_desc.slots_size();
  all_slots_.resize(all_slot_num);
  all_slots_type_.resize(all_slot_num);
  use_slots_index_.resize(all_slot_num);
  total_dims_without_inductive_.resize(all_slot_num);
  inductive_shape_index_.resize(all_slot_num);
  use_slots_.clear();
  use_slots_is_dense_.clear();
  for (size_t i = 0; i < all_slot_num; ++i) {
    const auto& slot = multi_slot_desc.slots(i);
    all_slots_[i] = slot.name();
    all_slots_type_[i] = slot.type();
    use_slots_index_[i] = slot.is_used() ? use_slots_.size() : -1;
    total_dims_without_inductive_[i] = 1;
    inductive_shape_index_[i] = -1;
    if (slot.is_used()) {
      use_slots_.push_back(all_slots_[i]);
      use_slots_is_dense_.push_back(slot.is_dense());
      std::vector<int> local_shape;
      if (slot.is_dense()) {
        for (int j = 0; j < slot.shape_size(); ++j) {
          if (slot.shape(j) > 0) {
            total_dims_without_inductive_[i] *= slot.shape(j);
          }
          if (slot.shape(j) == -1) {
            inductive_shape_index_[i] = j;
          }
        }
      }
      for (int j = 0; j < slot.shape_size(); ++j) {
        local_shape.push_back(slot.shape(j));
      }
      use_slots_shape_.push_back(local_shape);
    }
  }
  feed_vec_.resize(use_slots_.size());
  pipe_command_ = data_feed_desc.pipe_command();
  finish_init_ = true;
}

void MultiSlotDataFeed::ReadThread() {
#ifdef _LINUX
  std::string filename;
  while (PickOneFile(&filename)) {
    int err_no = 0;
    fp_ = fs_open_read(filename, &err_no, pipe_command_);
    CHECK(fp_ != nullptr);
    __fsetlocking(&*fp_, FSETLOCKING_BYCALLER);
    std::vector<MultiSlotType> instance;
    int ins_num = 0;
    while (ParseOneInstanceFromPipe(&instance)) {
      ins_num++;
      queue_->Put(instance);
    }
    VLOG(3) << "filename: " << filename << " inst num: " << ins_num;
  }
  queue_->Close();
#endif
}

bool MultiSlotDataFeed::CheckFile(const char* filename) {
#ifdef _LINUX
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
      auto num = strtol(endptr, &endptr, 10);
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
#endif
  return true;
}

bool MultiSlotDataFeed::ParseOneInstanceFromPipe(
    std::vector<MultiSlotType>* instance) {
#ifdef _LINUX
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
      PADDLE_ENFORCE_NE(
          num, 0,
          platform::errors::InvalidArgument(
              "The number of ids can not be zero, you need padding "
              "it in data generator; or if there is something wrong with "
              "the data, please check if the data contains unresolvable "
              "characters.\nplease check this error line: %s",
              str));
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
#else
  return true;
#endif
}

bool MultiSlotDataFeed::ParseOneInstance(std::vector<MultiSlotType>* instance) {
#ifdef _LINUX
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
#endif
  return false;
}

void MultiSlotDataFeed::AddInstanceToInsVec(
    std::vector<MultiSlotType>* ins_vec,
    const std::vector<MultiSlotType>& instance, int index) {
#ifdef _LINUX
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
#endif
}

void MultiSlotDataFeed::PutToFeedVec(
    const std::vector<MultiSlotType>& ins_vec) {
#ifdef _LINUX
  for (size_t i = 0; i < use_slots_.size(); ++i) {
    if (feed_vec_[i] == nullptr) {
      continue;
    }
    const auto& type = ins_vec[i].GetType();
    const auto& offset = ins_vec[i].GetOffset();
    int total_instance = static_cast<int>(offset.back());

    if (type[0] == 'f') {  // float
      const auto& feasign = ins_vec[i].GetFloatData();
      float* tensor_ptr =
          feed_vec_[i]->mutable_data<float>({total_instance, 1}, this->place_);
      CopyToFeedTensor(tensor_ptr, &feasign[0], total_instance * sizeof(float));
    } else if (type[0] == 'u') {  // uint64
      // no uint64_t type in paddlepaddle
      const auto& feasign = ins_vec[i].GetUint64Data();
      int64_t* tensor_ptr = feed_vec_[i]->mutable_data<int64_t>(
          {total_instance, 1}, this->place_);
      CopyToFeedTensor(tensor_ptr, &feasign[0],
                       total_instance * sizeof(int64_t));
    }

    LoD data_lod{offset};
    feed_vec_[i]->set_lod(data_lod);
    if (use_slots_is_dense_[i]) {
      if (inductive_shape_index_[i] != -1) {
        use_slots_shape_[i][inductive_shape_index_[i]] =
            total_instance / total_dims_without_inductive_[i];
      }
      feed_vec_[i]->Resize(framework::make_ddim(use_slots_shape_[i]));
    }
  }
#endif
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
  size_t all_slot_num = multi_slot_desc.slots_size();
  all_slots_.resize(all_slot_num);
  all_slots_type_.resize(all_slot_num);
  use_slots_index_.resize(all_slot_num);
  total_dims_without_inductive_.resize(all_slot_num);
  inductive_shape_index_.resize(all_slot_num);
  use_slots_.clear();
  use_slots_is_dense_.clear();
  for (size_t i = 0; i < all_slot_num; ++i) {
    const auto& slot = multi_slot_desc.slots(i);
    all_slots_[i] = slot.name();
    all_slots_type_[i] = slot.type();
    use_slots_index_[i] = slot.is_used() ? use_slots_.size() : -1;
    total_dims_without_inductive_[i] = 1;
    inductive_shape_index_[i] = -1;
    if (slot.is_used()) {
      use_slots_.push_back(all_slots_[i]);
      use_slots_is_dense_.push_back(slot.is_dense());
      std::vector<int> local_shape;
      if (slot.is_dense()) {
        for (int j = 0; j < slot.shape_size(); ++j) {
          if (slot.shape(j) > 0) {
            total_dims_without_inductive_[i] *= slot.shape(j);
          }
          if (slot.shape(j) == -1) {
            inductive_shape_index_[i] = j;
          }
        }
      }
      for (int j = 0; j < slot.shape_size(); ++j) {
        local_shape.push_back(slot.shape(j));
      }
      use_slots_shape_.push_back(local_shape);
    }
  }
  feed_vec_.resize(use_slots_.size());
  const int kEstimatedFeasignNumPerSlot = 5;  // Magic Number
  for (size_t i = 0; i < all_slot_num; i++) {
    batch_float_feasigns_.push_back(std::vector<float>());
    batch_uint64_feasigns_.push_back(std::vector<uint64_t>());
    batch_float_feasigns_[i].reserve(default_batch_size_ *
                                     kEstimatedFeasignNumPerSlot);
    batch_uint64_feasigns_[i].reserve(default_batch_size_ *
                                      kEstimatedFeasignNumPerSlot);
    offset_.push_back(std::vector<size_t>());
    offset_[i].reserve(default_batch_size_ +
                       1);  // Each lod info will prepend a zero
  }
  visit_.resize(all_slot_num, false);
  offset.reserve(all_slot_num * (1 + default_batch_size_));
  offset_sum.reserve(all_slot_num * (1 + default_batch_size_));
  ins_vec_.reserve(1 + default_batch_size_);
  pipe_command_ = data_feed_desc.pipe_command();
  finish_init_ = true;
  input_type_ = data_feed_desc.input_type();
}

void MultiSlotInMemoryDataFeed::GetMsgFromLogKey(const std::string& log_key,
                                                 uint64_t* search_id,
                                                 uint32_t* cmatch,
                                                 uint32_t* rank) {
  std::string searchid_str = log_key.substr(16, 16);
  *search_id = (uint64_t)strtoull(searchid_str.c_str(), NULL, 16);

  std::string cmatch_str = log_key.substr(11, 3);
  *cmatch = (uint32_t)strtoul(cmatch_str.c_str(), NULL, 16);

  std::string rank_str = log_key.substr(14, 2);
  *rank = (uint32_t)strtoul(rank_str.c_str(), NULL, 16);
}

bool MultiSlotInMemoryDataFeed::ParseOneInstanceFromPipe(Record* instance) {
#ifdef _LINUX
  thread_local string::LineFileReader reader;

  if (!reader.getline(&*(fp_.get()))) {
    return false;
  } else {
    const char* str = reader.get();
    std::string line = std::string(str);
    // VLOG(3) << line;
    char* endptr = const_cast<char*>(str);
    int pos = 0;
    if (parse_ins_id_) {
      int num = strtol(&str[pos], &endptr, 10);
      CHECK(num == 1);  // NOLINT
      pos = endptr - str + 1;
      size_t len = 0;
      while (str[pos + len] != ' ') {
        ++len;
      }
      instance->ins_id_ = std::string(str + pos, len);
      pos += len + 1;
      VLOG(3) << "ins_id " << instance->ins_id_;
    }
    if (parse_content_) {
      int num = strtol(&str[pos], &endptr, 10);
      CHECK(num == 1);  // NOLINT
      pos = endptr - str + 1;
      size_t len = 0;
      while (str[pos + len] != ' ') {
        ++len;
      }
      instance->content_ = std::string(str + pos, len);
      pos += len + 1;
      VLOG(3) << "content " << instance->content_;
    }
    if (parse_logkey_) {
      int num = strtol(&str[pos], &endptr, 10);
      CHECK(num == 1);  // NOLINT
      pos = endptr - str + 1;
      size_t len = 0;
      while (str[pos + len] != ' ') {
        ++len;
      }
      // parse_logkey
      std::string log_key = std::string(str + pos, len);
      uint64_t search_id;
      uint32_t cmatch;
      uint32_t rank;
      GetMsgFromLogKey(log_key, &search_id, &cmatch, &rank);

      instance->ins_id_ = log_key;
      instance->search_id = search_id;
      instance->cmatch = cmatch;
      instance->rank = rank;
      pos += len + 1;
    }
    // Object pool may be a better method
    instance->uint64_feasigns_.reserve(1000);
    instance->float_feasigns_.reserve(41);
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
        if (all_slots_type_[i][0] == 'f') {  // float
          for (int j = 0; j < num; ++j) {
            float feasign = strtof(endptr, &endptr);
            // if float feasign is equal to zero, ignore it
            // except when slot is dense
            if (fabs(feasign) < 1e-6 && !use_slots_is_dense_[i]) {
              continue;
            }
            FeatureKey f;
            f.float_feasign_ = feasign;
            instance->float_feasigns_.push_back(FeatureItem(f, idx));
          }
        } else if (all_slots_type_[i][0] == 'u') {  // uint64
          for (int j = 0; j < num; ++j) {
            uint64_t feasign = (uint64_t)strtoull(endptr, &endptr, 10);
            // if uint64 feasign is equal to zero, ignore it
            // except when slot is dense
            if (feasign == 0 && !use_slots_is_dense_[i]) {
              continue;
            }
            FeatureKey f;
            f.uint64_feasign_ = feasign;
            instance->uint64_feasigns_.push_back(FeatureItem(f, idx));
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
    instance->float_feasigns_.shrink_to_fit();
    instance->uint64_feasigns_.shrink_to_fit();
    fea_num_ += instance->uint64_feasigns_.size();
    return true;
  }
#else
  return false;
#endif
}

bool MultiSlotInMemoryDataFeed::ParseOneInstance(Record* instance) {
#ifdef _LINUX
  std::string line;
  if (getline(file_, line)) {
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
        if (all_slots_type_[i][0] == 'f') {  // float
          for (int j = 0; j < num; ++j) {
            float feasign = strtof(endptr, &endptr);
            if (fabs(feasign) < 1e-6) {
              continue;
            }
            FeatureKey f;
            f.float_feasign_ = feasign;
            instance->float_feasigns_.push_back(FeatureItem(f, idx));
          }
        } else if (all_slots_type_[i][0] == 'u') {  // uint64
          for (int j = 0; j < num; ++j) {
            uint64_t feasign = (uint64_t)strtoull(endptr, &endptr, 10);
            if (feasign == 0) {
              continue;
            }
            FeatureKey f;
            f.uint64_feasign_ = feasign;
            instance->uint64_feasigns_.push_back(FeatureItem(f, idx));
          }
        }
        pos = endptr - str;
      } else {
        for (int j = 0; j <= num; ++j) {
          pos = line.find_first_of(' ', pos + 1);
        }
      }
    }
    instance->float_feasigns_.shrink_to_fit();
    instance->uint64_feasigns_.shrink_to_fit();
    return true;
  } else {
    return false;
  }
#endif
  return false;
}

void MultiSlotInMemoryDataFeed::PutToFeedVec(
    const std::vector<Record>& ins_vec) {
#ifdef _LINUX
  for (size_t i = 0; i < batch_float_feasigns_.size(); ++i) {
    batch_float_feasigns_[i].clear();
    batch_uint64_feasigns_[i].clear();
    offset_[i].clear();
    offset_[i].push_back(0);
  }
  ins_content_vec_.clear();
  ins_content_vec_.reserve(ins_vec.size());
  ins_id_vec_.clear();
  ins_id_vec_.reserve(ins_vec.size());
  for (size_t i = 0; i < ins_vec.size(); ++i) {
    auto& r = ins_vec[i];
    ins_id_vec_.push_back(r.ins_id_);
    ins_content_vec_.push_back(r.content_);
    for (auto& item : r.float_feasigns_) {
      batch_float_feasigns_[item.slot()].push_back(item.sign().float_feasign_);
      visit_[item.slot()] = true;
    }
    for (auto& item : r.uint64_feasigns_) {
      batch_uint64_feasigns_[item.slot()].push_back(
          item.sign().uint64_feasign_);
      visit_[item.slot()] = true;
    }
    for (size_t j = 0; j < use_slots_.size(); ++j) {
      const auto& type = all_slots_type_[j];
      if (visit_[j]) {
        visit_[j] = false;
      } else {
        // fill slot value with default value 0
        if (type[0] == 'f') {  // float
          batch_float_feasigns_[j].push_back(0.0);
        } else if (type[0] == 'u') {  // uint64
          batch_uint64_feasigns_[j].push_back(0);
        }
      }
      // get offset of this ins in this slot
      if (type[0] == 'f') {  // float
        offset_[j].push_back(batch_float_feasigns_[j].size());
      } else if (type[0] == 'u') {  // uint64
        offset_[j].push_back(batch_uint64_feasigns_[j].size());
      }
    }
  }

  for (size_t i = 0; i < use_slots_.size(); ++i) {
    if (feed_vec_[i] == nullptr) {
      continue;
    }
    int total_instance = offset_[i].back();
    const auto& type = all_slots_type_[i];
    if (type[0] == 'f') {  // float
      float* feasign = batch_float_feasigns_[i].data();
      float* tensor_ptr =
          feed_vec_[i]->mutable_data<float>({total_instance, 1}, this->place_);
      CopyToFeedTensor(tensor_ptr, feasign, total_instance * sizeof(float));
    } else if (type[0] == 'u') {  // uint64
      // no uint64_t type in paddlepaddle
      uint64_t* feasign = batch_uint64_feasigns_[i].data();
      int64_t* tensor_ptr = feed_vec_[i]->mutable_data<int64_t>(
          {total_instance, 1}, this->place_);
      CopyToFeedTensor(tensor_ptr, feasign, total_instance * sizeof(int64_t));
    }
    auto& slot_offset = offset_[i];
    if (this->input_type_ == 0) {
      LoD data_lod{slot_offset};
      feed_vec_[i]->set_lod(data_lod);
    } else if (this->input_type_ == 1) {
      if (!use_slots_is_dense_[i]) {
        std::vector<size_t> tmp_offset;
        PADDLE_ENFORCE_EQ(slot_offset.size(), 2,
                          platform::errors::InvalidArgument(
                              "In batch reader, the sparse tensor lod size "
                              "must be 2, but received %d",
                              slot_offset.size()));
        const auto& max_size = slot_offset[1];
        tmp_offset.reserve(max_size + 1);
        for (unsigned int k = 0; k <= max_size; k++) {
          tmp_offset.emplace_back(k);
        }
        slot_offset = tmp_offset;
        LoD data_lod{slot_offset};
        feed_vec_[i]->set_lod(data_lod);
      }
    }
    if (use_slots_is_dense_[i]) {
      if (inductive_shape_index_[i] != -1) {
        use_slots_shape_[i][inductive_shape_index_[i]] =
            total_instance / total_dims_without_inductive_[i];
      }
      feed_vec_[i]->Resize(framework::make_ddim(use_slots_shape_[i]));
    }
  }
#endif
}

#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
template <typename T>
void PrivateInstantDataFeed<T>::PutToFeedVec() {
  for (size_t i = 0; i < use_slots_.size(); ++i) {
    const auto& type = ins_vec_[i].GetType();
    const auto& offset = ins_vec_[i].GetOffset();
    int total_instance = static_cast<int>(offset.back());

    if (type[0] == 'f') {  // float
      const auto& feasign = ins_vec_[i].GetFloatData();
      float* tensor_ptr =
          feed_vec_[i]->mutable_data<float>({total_instance, 1}, this->place_);
      CopyToFeedTensor(tensor_ptr, &feasign[0], total_instance * sizeof(float));
    } else if (type[0] == 'u') {  // uint64
      // no uint64_t type in paddlepaddle
      const auto& feasign = ins_vec_[i].GetUint64Data();
      int64_t* tensor_ptr = feed_vec_[i]->mutable_data<int64_t>(
          {total_instance, 1}, this->place_);
      CopyToFeedTensor(tensor_ptr, &feasign[0],
                       total_instance * sizeof(int64_t));
    }

    LoD data_lod{offset};
    feed_vec_[i]->set_lod(data_lod);
    if (use_slots_is_dense_[i]) {
      int64_t total_dims = 1;
      for (const auto e : use_slots_shape_[i]) {
        total_dims *= e;
      }
      PADDLE_ENFORCE(
          total_dims == total_instance,
          "The actual data size of slot[%s] doesn't match its declaration",
          use_slots_[i].c_str());
      feed_vec_[i]->Resize(framework::make_ddim(use_slots_shape_[i]));
    }
  }
}

template <typename T>
int PrivateInstantDataFeed<T>::Next() {
  if (ParseOneMiniBatch()) {
    PutToFeedVec();
    return ins_vec_[0].GetBatchSize();
  }
  Postprocess();

  std::string filename;
  if (!PickOneFile(&filename)) {
    return -1;
  }
  if (!Preprocess(filename)) {
    return -1;
  }

  PADDLE_ENFORCE(true == ParseOneMiniBatch(), "Fail to parse mini-batch data");
  PutToFeedVec();
  return ins_vec_[0].GetBatchSize();
}

template <typename T>
void PrivateInstantDataFeed<T>::Init(const DataFeedDesc& data_feed_desc) {
  finish_init_ = false;
  finish_set_filelist_ = false;
  finish_start_ = false;

  PADDLE_ENFORCE(data_feed_desc.has_multi_slot_desc(),
                 "Multi_slot_desc has not been set.");
  paddle::framework::MultiSlotDesc multi_slot_desc =
      data_feed_desc.multi_slot_desc();
  SetBatchSize(data_feed_desc.batch_size());
  size_t all_slot_num = multi_slot_desc.slots_size();
  all_slots_.resize(all_slot_num);
  all_slots_type_.resize(all_slot_num);
  use_slots_index_.resize(all_slot_num);
  multi_inductive_shape_index_.resize(all_slot_num);
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
      std::vector<int> local_shape;
      if (slot.is_dense()) {
        for (int j = 0; j < slot.shape_size(); ++j) {
          if (slot.shape(j) == -1) {
            multi_inductive_shape_index_[i].push_back(j);
          }
        }
      }
      for (int j = 0; j < slot.shape_size(); ++j) {
        local_shape.push_back(slot.shape(j));
      }
      use_slots_shape_.push_back(local_shape);
    }
  }
  feed_vec_.resize(use_slots_.size());
  ins_vec_.resize(use_slots_.size());

  finish_init_ = true;
}

template class PrivateInstantDataFeed<std::vector<MultiSlotType>>;

bool MultiSlotFileInstantDataFeed::Preprocess(const std::string& filename) {
  fd_ = open(filename.c_str(), O_RDONLY);
  PADDLE_ENFORCE(fd_ != -1, "Fail to open file: %s", filename.c_str());

  struct stat sb;
  fstat(fd_, &sb);
  end_ = static_cast<size_t>(sb.st_size);

  buffer_ =
      reinterpret_cast<char*>(mmap(NULL, end_, PROT_READ, MAP_PRIVATE, fd_, 0));
  PADDLE_ENFORCE(buffer_ != MAP_FAILED, strerror(errno));

  offset_ = 0;
  return true;
}

bool MultiSlotFileInstantDataFeed::Postprocess() {
  if (buffer_ != nullptr) {
    munmap(buffer_, end_);
    buffer_ = nullptr;
  }
  if (fd_ != -1) {
    close(fd_);
    fd_ = -1;
    end_ = 0;
    offset_ = 0;
  }
  return true;
}

bool MultiSlotFileInstantDataFeed::ParseOneMiniBatch() {
  if (offset_ == end_) {
    return false;
  }

  batch_size_ = 0;
  while (batch_size_ < default_batch_size_ && offset_ < end_) {
    for (size_t i = 0; i < use_slots_index_.size(); ++i) {
      int idx = use_slots_index_[i];
      char type = all_slots_type_[i][0];

      uint16_t num = *reinterpret_cast<uint16_t*>(buffer_ + offset_);
      PADDLE_ENFORCE(
          num,
          "The number of ids can not be zero, you need padding "
          "it in data generator; or if there is something wrong with "
          "the data, please check if the data contains unresolvable "
          "characters.");
      offset_ += sizeof(uint16_t);

      if (idx != -1) {
        int inductive_size = multi_inductive_shape_index_[i].size();
        if (UNLIKELY(batch_size_ == 0)) {
          ins_vec_[idx].Init(all_slots_type_[i], default_batch_size_ * num);
          ins_vec_[idx].InitOffset(default_batch_size_);
          uint64_t* inductive_shape =
              reinterpret_cast<uint64_t*>(buffer_ + offset_);
          for (int inductive_id = 0; inductive_id < inductive_size;
               ++inductive_id) {
            use_slots_shape_[i][multi_inductive_shape_index_[i][inductive_id]] =
                static_cast<int>(*(inductive_shape + inductive_id));
          }
        }
        num -= inductive_size;
        offset_ += sizeof(uint64_t) * inductive_size;

        if (type == 'f') {
          ins_vec_[idx].AppendValues(
              reinterpret_cast<float*>(buffer_ + offset_), num);
          offset_ += num * sizeof(float);
        } else if (type == 'u') {
          ins_vec_[idx].AppendValues(
              reinterpret_cast<uint64_t*>(buffer_ + offset_), num);
          offset_ += num * sizeof(uint64_t);
        }
      } else {
        if (type == 'f') {
          offset_ += num * sizeof(float);
        } else if (type == 'u') {
          offset_ += num * sizeof(uint64_t);
        }
      }
    }
    ++batch_size_;
    // OPTIMIZE: It is better to insert check codes between instances for format
    // checking
  }

  PADDLE_ENFORCE(batch_size_ == default_batch_size_ || offset_ == end_,
                 "offset_ != end_");
  return true;
}
#endif

bool PaddleBoxDataFeed::Start() {
#ifdef _LINUX
  int phase = GetCurrentPhase();  // join: 1, update: 0
  this->CheckSetFileList();
  if (enable_pv_merge_ && phase == 1) {
    // join phase : input_pv_channel to output_pv_channel
    if (output_pv_channel_->Size() == 0 && input_pv_channel_->Size() != 0) {
      std::vector<PvInstance> data;
      input_pv_channel_->Read(data);
      output_pv_channel_->Write(std::move(data));
    }
  } else {
    // input_channel to output
    if (output_channel_->Size() == 0 && input_channel_->Size() != 0) {
      std::vector<Record> data;
      input_channel_->Read(data);
      output_channel_->Write(std::move(data));
    }
    if (output_ptr_channel_->Size() == 0 && input_ptr_channel_->Size() != 0) {
      std::vector<Record*> data;
      input_ptr_channel_->Read(data);
      output_ptr_channel_->Open();
      output_ptr_channel_->Write(std::move(data));
      output_ptr_channel_->Close();
    }
  }
#endif
  this->finish_start_ = true;
  return true;
}

int PaddleBoxDataFeed::Next() {
#ifdef _LINUX
  int phase = GetCurrentPhase();  // join: 1, update: 0
  this->CheckStart();
  if (enable_pv_merge_ && phase == 1) {
    // join phase : output_pv_channel to consume_pv_channel
    CHECK(output_pv_channel_ != nullptr);
    CHECK(consume_pv_channel_ != nullptr);
    VLOG(3) << "output_pv_channel_ size=" << output_pv_channel_->Size()
            << ", consume_pv_channel_ size=" << consume_pv_channel_->Size()
            << ", thread_id=" << thread_id_;
    int index = 0;
    PvInstance pv_instance;
    // std::vector<PvInstance> pv_vec;
    // pv_vec.reserve(this->pv_batch_size_);
    pv_vec_.clear();
    while (index < this->pv_batch_size_) {
      if (output_pv_channel_->Size() == 0) {
        break;
      }

      output_pv_channel_->Get(pv_instance);
      pv_vec_.push_back(pv_instance);
      ++index;
      consume_pv_channel_->Put(std::move(pv_instance));
    }
    this->batch_size_ = index;
    VLOG(3) << "pv_batch_size_=" << this->batch_size_
            << ", thread_id=" << thread_id_;
    if (this->batch_size_ != 0) {
      PutToFeedVec(pv_vec_);
    } else {
      VLOG(3) << "finish reading, output_pv_channel_ size="
              << output_pv_channel_->Size()
              << ", consume_pv_channel_ size=" << consume_pv_channel_->Size()
              << ", thread_id=" << thread_id_;
    }
  } else {
    this->CheckStart();
    CHECK(output_channel_ != nullptr);
    CHECK(consume_channel_ != nullptr);
    VLOG(3) << "output_channel_ size=" << output_channel_->Size()
            << ", consume_channel_ size=" << consume_channel_->Size()
            << ", thread_id=" << thread_id_;
    int index = 0;
    Record* instance;
    // std::vector<T> ins_vec;
    ins_vec_.clear();
    // ins_vec.reserve(this->default_batch_size_);
    while (index < this->default_batch_size_) {
      if (output_ptr_channel_->Size() == 0) {
        break;
      }
      output_ptr_channel_->Get(instance);
      ins_vec_.push_back(instance);
      ++index;
      consume_ptr_channel_->Put(std::move(instance));
    }
    this->batch_size_ = index;
    VLOG(3) << "batch_size_=" << this->batch_size_
            << ", thread_id=" << thread_id_;
    if (this->batch_size_ != 0) {
      PutToFeedVec(ins_vec_);
    } else {
      VLOG(3) << "finish reading, output_channel_ size="
              << output_channel_->Size()
              << ", consume_channel_ size=" << consume_channel_->Size()
              << ", thread_id=" << thread_id_;
    }
  }
  return this->batch_size_;
#else
  return 0;
#endif
}

void PaddleBoxDataFeed::Init(const DataFeedDesc& data_feed_desc) {
  MultiSlotInMemoryDataFeed::Init(data_feed_desc);
  rank_offset_name_ = data_feed_desc.rank_offset();
  pv_batch_size_ = data_feed_desc.pv_batch_size();
}

void PaddleBoxDataFeed::GetRankOffset(const std::vector<PvInstance>& pv_vec,
                                      int ins_number) {
  int index = 0;
  int max_rank = 3;  // the value is setting
  int row = ins_number;
  int col = max_rank * 2 + 1;
  int pv_num = pv_vec.size();

  std::vector<int> rank_offset_mat(row * col, -1);
  rank_offset_mat.shrink_to_fit();

  for (int i = 0; i < pv_num; i++) {
    auto pv_ins = pv_vec[i];
    int ad_num = pv_ins->ads.size();
    int index_start = index;
    for (int j = 0; j < ad_num; ++j) {
      auto ins = pv_ins->ads[j];
      int rank = -1;
      if ((ins->cmatch == 222 || ins->cmatch == 223) &&
          ins->rank <= static_cast<uint32_t>(max_rank) && ins->rank != 0) {
        rank = ins->rank;
      }

      rank_offset_mat[index * col] = rank;
      if (rank > 0) {
        for (int k = 0; k < ad_num; ++k) {
          auto cur_ins = pv_ins->ads[k];
          int fast_rank = -1;
          if ((cur_ins->cmatch == 222 || cur_ins->cmatch == 223) &&
              cur_ins->rank <= static_cast<uint32_t>(max_rank) &&
              cur_ins->rank != 0) {
            fast_rank = cur_ins->rank;
          }

          if (fast_rank > 0) {
            int m = fast_rank - 1;
            rank_offset_mat[index * col + 2 * m + 1] = cur_ins->rank;
            rank_offset_mat[index * col + 2 * m + 2] = index_start + k;
          }
        }
      }
      index += 1;
    }
  }

  int* rank_offset = rank_offset_mat.data();
  int* tensor_ptr = rank_offset_->mutable_data<int>({row, col}, this->place_);
  CopyToFeedTensor(tensor_ptr, rank_offset, row * col * sizeof(int));
}

void PaddleBoxDataFeed::AssignFeedVar(const Scope& scope) {
  MultiSlotInMemoryDataFeed::AssignFeedVar(scope);
  // set rank offset memory
  int phase = GetCurrentPhase();  // join: 1, update: 0
  if (enable_pv_merge_ && phase == 1) {
    rank_offset_ = scope.FindVar(rank_offset_name_)->GetMutable<LoDTensor>();
  }
}

void PaddleBoxDataFeed::PutToFeedVec(const std::vector<PvInstance>& pv_vec) {
#ifdef _LINUX
  int ins_number = 0;
  std::vector<Record*> ins_vec;
  for (auto& pv : pv_vec) {
    ins_number += pv->ads.size();
    for (auto ins : pv->ads) {
      ins_vec.push_back(ins);
    }
  }
  GetRankOffset(pv_vec, ins_number);
  PutToFeedVec(ins_vec);
#endif
}

int PaddleBoxDataFeed::GetCurrentPhase() {
#ifdef PADDLE_WITH_BOX_PS
  auto box_ptr = paddle::framework::BoxWrapper::GetInstance();
  if (box_ptr->Mode() == 1) {  // For AucRunner
    return 1;
  } else {
    return box_ptr->Phase();
  }
#else
  LOG(WARNING) << "It should be complied with BOX_PS...";
  return current_phase_;
#endif
}

void PaddleBoxDataFeed::PutToFeedVec(const std::vector<Record*>& ins_vec) {
#if defined(PADDLE_WITH_CUDA) && defined(_LINUX)
  ins_content_vec_.clear();
  ins_content_vec_.reserve(ins_vec.size());
  ins_id_vec_.clear();
  ins_id_vec_.reserve(ins_vec.size());
  for (size_t i = 0; i < ins_vec.size(); ++i) {
    auto& r = ins_vec[i];
    ins_id_vec_.push_back(r->ins_id_);
    ins_content_vec_.push_back(r->content_);
  }

  paddle::platform::SetDeviceId(
      boost::get<platform::CUDAPlace>(this->GetPlace()).GetDeviceId());
  std::vector<size_t> ins_len(ins_vec.size(), 0);  // prefix sum of ins length
  size_t total_size = 0;
  for (size_t i = 0; i < ins_vec.size(); ++i) {
    auto& r = ins_vec[i];
    total_size += r->uint64_feasigns_.size() + r->float_feasigns_.size();
    ins_len[i] = total_size;
  }

  auto cpu_buf = memory::AllocShared(platform::CPUPlace(),
                                     total_size * sizeof(FeatureItem));
  auto gpu_buf =
      memory::AllocShared(this->GetPlace(), total_size * sizeof(FeatureItem));
  FeatureItem* fea_list_cpu = reinterpret_cast<FeatureItem*>(cpu_buf->ptr());
  FeatureItem* fea_list_gpu = reinterpret_cast<FeatureItem*>(gpu_buf->ptr());

  size_t size_off = 0;
  for (size_t i = 0; i < ins_vec.size(); ++i) {
    auto& r = ins_vec[i];
    memcpy(fea_list_cpu + size_off, r->uint64_feasigns_.data(),
           sizeof(FeatureItem) * r->uint64_feasigns_.size());
    size_off += r->uint64_feasigns_.size();
    memcpy(fea_list_cpu + size_off, r->float_feasigns_.data(),
           sizeof(FeatureItem) * r->float_feasigns_.size());
    size_off += r->float_feasigns_.size();
  }

  size_t row_size = use_slots_.size();       // slot_number
  size_t col_size = 1 + ins_vec.size();      // 1 + batch_size
  size_t offset_size = row_size * col_size;  // length of lod array

  size_t uslot_num = 0;  // slot number of uint64_t type
  std::vector<size_t> index_map(
      row_size);  // make index map from origin slot id to actual slot id
  std::vector<char> slot_type(row_size, 'u');  // Record actual slot type

  for (size_t i = 0; i < row_size; ++i) {
    const auto& type = all_slots_type_[i];
    if (type[0] == 'u') {
      uslot_num++;
    }
  }
  for (size_t i = uslot_num; i < row_size; ++i) {
    slot_type[i] = 'f';
  }
  int u_slot_id = 0;
  for (size_t i = 0; i < use_slots_.size(); ++i) {
    const auto& type = all_slots_type_[i];
    if (type[0] == 'f') {  // float
      index_map[i] = uslot_num++;
    } else if (type[0] == 'u') {
      index_map[i] = u_slot_id++;
    }
  }

  // std::vector<size_t> offset(offset_size, 0);
  // std::vector<size_t> offset_sum(offset_size, 0);
  offset.clear();
  offset_sum.clear();
  offset.resize(offset_size, 0);
  offset_sum.resize(offset_size, 0);

  // construct lod info
  int col = 1;
  for (size_t i = 0; i < total_size; ++i) {
    if (i >= ins_len[col - 1]) {
      col++;
    }
    offset[index_map[fea_list_cpu[i].slot()] * col_size + col]++;
  }
  // compute sum-lod info, for using kernel to make batch
  for (size_t i = 0; i < offset.size(); ++i) {
    int row = i / col_size;
    int col = i % col_size;
    if (col == 0) {
      continue;
    } else if (row == 0) {
      offset[i] += offset[i - 1];
      offset_sum[i] = offset[i];
    } else {
      offset_sum[i] = offset[i] + offset_sum[i - 1] + offset_sum[i - col_size] -
                      offset_sum[i - col_size - 1];
      offset[i] += offset[i - 1];
    }
  }
  Tensor offset_gpu;
  size_t* offset_gpu_data =
      reinterpret_cast<size_t*>(offset_gpu.mutable_data<int64_t>(
          {static_cast<int64_t>(offset_size), 1}, this->GetPlace()));
  cudaMemcpy(fea_list_gpu, fea_list_cpu, total_size * sizeof(FeatureItem),
             cudaMemcpyHostToDevice);
  cudaMemcpy(offset_gpu_data, offset_sum.data(), sizeof(size_t) * offset_size,
             cudaMemcpyHostToDevice);

  std::vector<void*> dest_cpu_p(row_size, nullptr);

  for (size_t i = 0; i < use_slots_.size(); ++i) {
    if (feed_vec_[i] == nullptr) {
      continue;
    }
    int total_instance = offset[(index_map[i] + 1) * col_size - 1];
    const auto& type = all_slots_type_[i];
    if (type[0] == 'f') {  // float
      float* tensor_ptr =
          feed_vec_[i]->mutable_data<float>({total_instance, 1}, this->place_);
      dest_cpu_p[index_map[i]] = static_cast<void*>(tensor_ptr);
    } else if (type[0] == 'u') {
      // printf("slot[%d]: total feanum[%d]\n", (int)i, (int)total_instance);
      int64_t* tensor_ptr = feed_vec_[i]->mutable_data<int64_t>(
          {total_instance, 1}, this->place_);
      dest_cpu_p[index_map[i]] = static_cast<void*>(tensor_ptr);
    }
  }
  // fprintf(stderr, "after create tensor\n");
  auto buf = memory::AllocShared(this->GetPlace(), row_size * sizeof(void*));
  auto type_buf =
      memory::AllocShared(this->GetPlace(), row_size * sizeof(char));
  void** dest_gpu_p = reinterpret_cast<void**>(buf->ptr());
  char* type_gpu_p = reinterpret_cast<char*>(type_buf->ptr());
  cudaMemcpy(dest_gpu_p, dest_cpu_p.data(), row_size * sizeof(void*),
             cudaMemcpyHostToDevice);
  cudaMemcpy(type_gpu_p, slot_type.data(), row_size * sizeof(char),
             cudaMemcpyHostToDevice);

  CopyForTensor(this->GetPlace(), fea_list_gpu, dest_gpu_p, offset_gpu_data,
                type_gpu_p, total_size, row_size, col_size);

  std::vector<size_t> lodinfo(col_size, 0);
  for (size_t i = 0; i < use_slots_.size(); ++i) {
    memcpy(lodinfo.data(), offset.data() + index_map[i] * col_size,
           col_size * sizeof(size_t));
    LoD lod{lodinfo};
    feed_vec_[i]->set_lod(lod);
    if (use_slots_is_dense_[i]) {
      if (inductive_shape_index_[i] != -1) {
        use_slots_shape_[i][inductive_shape_index_[i]] =
            offset[(index_map[i] + 1) * col_size - 1] /
            total_dims_without_inductive_[i];
      }
      feed_vec_[i]->Resize(framework::make_ddim(use_slots_shape_[i]));
    }
  }
#endif
}

//================================ new boxps
//=============================================
#ifdef PADDLE_WITH_BOX_PS
void SlotPaddleBoxDataFeed::Init(const DataFeedDesc& data_feed_desc) {
  finish_init_ = false;
  finish_set_filelist_ = false;
  finish_start_ = false;

  PADDLE_ENFORCE(data_feed_desc.has_multi_slot_desc(),
                 "Multi_slot_desc has not been set.");
  paddle::framework::MultiSlotDesc multi_slot_desc =
      data_feed_desc.multi_slot_desc();
  SetBatchSize(data_feed_desc.batch_size());
  size_t all_slot_num = multi_slot_desc.slots_size();

  all_slots_.resize(all_slot_num);
  all_slots_info_.resize(all_slot_num);
  used_slots_info_.resize(all_slot_num);
  use_slot_size_ = 0;
  use_slots_.clear();

  float_total_dims_size_ = 0;
  float_total_dims_without_inductives_.clear();
  for (size_t i = 0; i < all_slot_num; ++i) {
    const auto& slot = multi_slot_desc.slots(i);
    all_slots_[i] = slot.name();

    AllSlotInfo& all_slot = all_slots_info_[i];
    all_slot.slot = slot.name();
    all_slot.type = slot.type();
    all_slot.used_idx = slot.is_used() ? use_slot_size_ : -1;
    all_slot.slot_value_idx = -1;

    if (slot.is_used()) {
      UsedSlotInfo& info = used_slots_info_[use_slot_size_];
      info.idx = i;
      info.slot = slot.name();
      info.type = slot.type();
      info.dense = slot.is_dense();
      info.total_dims_without_inductive = 1;
      info.inductive_shape_index = -1;

      // record float value and uint64_t value pos
      if (info.type[0] == 'u') {
        info.slot_value_idx = uint64_use_slot_size_;
        all_slot.slot_value_idx = uint64_use_slot_size_;
        ++uint64_use_slot_size_;
      } else if (info.type[0] == 'f') {
        info.slot_value_idx = float_use_slot_size_;
        all_slot.slot_value_idx = float_use_slot_size_;
        ++float_use_slot_size_;
      }

      use_slots_.push_back(slot.name());

      if (slot.is_dense()) {
        for (int j = 0; j < slot.shape_size(); ++j) {
          if (slot.shape(j) > 0) {
            info.total_dims_without_inductive *= slot.shape(j);
          }
          if (slot.shape(j) == -1) {
            info.inductive_shape_index = j;
          }
        }
      }
      if (info.type[0] == 'f') {
        float_total_dims_without_inductives_.push_back(
            info.total_dims_without_inductive);
        float_total_dims_size_ += info.total_dims_without_inductive;
      }
      info.local_shape.clear();
      for (int j = 0; j < slot.shape_size(); ++j) {
        info.local_shape.push_back(slot.shape(j));
      }
      ++use_slot_size_;
    }
  }
  used_slots_info_.resize(use_slot_size_);

  feed_vec_.resize(used_slots_info_.size());
  const int kEstimatedFeasignNumPerSlot = 5;  // Magic Number
  for (size_t i = 0; i < all_slot_num; i++) {
    batch_float_feasigns_.push_back(std::vector<float>());
    batch_uint64_feasigns_.push_back(std::vector<uint64_t>());
    batch_float_feasigns_[i].reserve(default_batch_size_ *
                                     kEstimatedFeasignNumPerSlot);
    batch_uint64_feasigns_[i].reserve(default_batch_size_ *
                                      kEstimatedFeasignNumPerSlot);
    offset_.push_back(std::vector<size_t>());
    offset_[i].reserve(default_batch_size_ +
                       1);  // Each lod info will prepend a zero
  }
  pipe_command_ = data_feed_desc.pipe_command();
  finish_init_ = true;
  input_type_ = data_feed_desc.input_type();

  rank_offset_name_ = data_feed_desc.rank_offset();
  pv_batch_size_ = data_feed_desc.pv_batch_size();

  // fprintf(stdout, "rank_offset_name: [%s]\n", rank_offset_name_.c_str());
  size_t pos = pipe_command_.find(".so");
  if (pos != std::string::npos) {
    pos = pipe_command_.rfind('|');
    if (pos == std::string::npos) {
      parser_so_path_ = pipe_command_;
      pipe_command_.clear();
    } else {
      parser_so_path_ = pipe_command_.substr(pos + 1);
      pipe_command_ = pipe_command_.substr(0, pos);
    }
    parser_so_path_ = paddle::string::erase_spaces(parser_so_path_);
  } else {
    parser_so_path_.clear();
  }
}
void SlotPaddleBoxDataFeed::GetUsedSlotIndex(
    std::vector<int>* used_slot_index) {
  auto boxps_ptr = BoxWrapper::GetInstance();
  // get feasigns that FeedPass doesn't need
  const std::unordered_set<std::string>& slot_name_omited_in_feedpass_ =
      boxps_ptr->GetOmitedSlot();
  used_slot_index->clear();
  for (int i = 0; i < use_slot_size_; ++i) {
    auto& info = used_slots_info_[i];
    if (info.type[0] != 'u') {
      continue;
    }
    if (slot_name_omited_in_feedpass_.find(info.slot) ==
        slot_name_omited_in_feedpass_.end()) {
      used_slot_index->push_back(info.slot_value_idx);
    }
  }
}
bool SlotPaddleBoxDataFeed::Start() {
  this->CheckSetFileList();
  this->offset_index_ = 0;
  this->finish_start_ = true;
#if defined(PADDLE_WITH_CUDA) && defined(_LINUX)
  CHECK(paddle::platform::is_gpu_place(this->place_));
  pack_ = BatchGpuPackMgr().get(this->GetPlace(), used_slots_info_);
#endif
  return true;
}
int SlotPaddleBoxDataFeed::Next() {
  int phase = GetCurrentPhase();  // join: 1, update: 0
  this->CheckStart();
  if (offset_index_ >= static_cast<int>(batch_offsets_.size())) {
    return 0;
  }
  auto& batch = batch_offsets_[offset_index_++];
  if (enable_pv_merge_ && phase == 1) {
    // join phase : output_pv_channel to consume_pv_channel
    this->batch_size_ = batch.second;
    if (this->batch_size_ != 0) {
      batch_timer_.Resume();
      PutToFeedPvVec(&pv_ins_[batch.first], this->batch_size_);
      batch_timer_.Pause();
    } else {
      VLOG(3) << "finish reading, batch size zero, thread_id=" << thread_id_;
    }
    return this->batch_size_;
  } else {
    this->batch_size_ = batch.second;
    batch_timer_.Resume();
    PutToFeedSlotVec(&records_[batch.first], this->batch_size_);
    batch_timer_.Pause();
    return this->batch_size_;
  }
}
bool SlotPaddleBoxDataFeed::EnablePvMerge(void) {
  return (enable_pv_merge_ && GetCurrentPhase() == 1);
}
int SlotPaddleBoxDataFeed::GetPackInstance(SlotRecord** ins) {
  if (offset_index_ >= static_cast<int>(batch_offsets_.size())) {
    return 0;
  }
  auto& batch = batch_offsets_[offset_index_];
  *ins = &records_[batch.first];
  return batch.second;
}
int SlotPaddleBoxDataFeed::GetPackPvInstance(SlotPvInstance** pv_ins) {
  if (offset_index_ >= static_cast<int>(batch_offsets_.size())) {
    return 0;
  }
  auto& batch = batch_offsets_[offset_index_];
  *pv_ins = &pv_ins_[batch.first];
  return batch.second;
}
void SlotPaddleBoxDataFeed::AssignFeedVar(const Scope& scope) {
  CheckInit();
  for (int i = 0; i < use_slot_size_; ++i) {
    feed_vec_[i] =
        scope.FindVar(used_slots_info_[i].slot)->GetMutable<LoDTensor>();
  }
  // set rank offset memory
  int phase = GetCurrentPhase();  // join: 1, update: 0
  if (enable_pv_merge_ && phase == 1) {
    rank_offset_ = scope.FindVar(rank_offset_name_)->GetMutable<LoDTensor>();
  }
}
void SlotPaddleBoxDataFeed::PutToFeedPvVec(const SlotPvInstance* pvs, int num) {
#if defined(PADDLE_WITH_CUDA) && defined(_LINUX)
  paddle::platform::SetDeviceId(
      boost::get<platform::CUDAPlace>(place_).GetDeviceId());
  pack_->pack_pvinstance(pvs, num);
  int ins_num = pack_->ins_num();
  int pv_num = pack_->pv_num();
  GetRankOffsetGPU(pv_num, ins_num);
  BuildSlotBatchGPU(ins_num);
#else
  int ins_number = 0;
  std::vector<SlotRecord> ins_vec;
  for (int i = 0; i < num; ++i) {
    auto& pv = pvs[i];
    ins_number += pv->ads.size();
    for (auto ins : pv->ads) {
      ins_vec.push_back(ins);
    }
  }
  GetRankOffset(pvs, num, ins_number);
  PutToFeedSlotVec(&ins_vec[0], ins_number);
#endif
}

// expand values
void SlotPaddleBoxDataFeed::ExpandSlotRecord(SlotRecord* rec) {
  SlotRecord& ins = (*rec);
  if (ins->slot_float_feasigns_.slot_offsets.empty()) {
    return;
  }
  size_t total_value_size = ins->slot_float_feasigns_.slot_values.size();
  if (float_total_dims_size_ == total_value_size) {
    return;
  }

  int float_slot_num =
      static_cast<int>(float_total_dims_without_inductives_.size());
  CHECK(float_slot_num == float_use_slot_size_);
  std::vector<float> old_values;
  std::vector<uint32_t> old_offsets;
  old_values.swap(ins->slot_float_feasigns_.slot_values);
  old_offsets.swap(ins->slot_float_feasigns_.slot_offsets);

  ins->slot_float_feasigns_.slot_values.resize(float_total_dims_size_);
  ins->slot_float_feasigns_.slot_offsets.assign(float_slot_num + 1, 0);

  auto& slot_offsets = ins->slot_float_feasigns_.slot_offsets;
  auto& slot_values = ins->slot_float_feasigns_.slot_values;

  uint32_t offset = 0;
  int num = 0;
  uint32_t old_off = 0;
  int dim = 0;

  for (int i = 0; i < float_slot_num; ++i) {
    dim = float_total_dims_without_inductives_[i];
    old_off = old_offsets[i];
    num = static_cast<int>(old_offsets[i + 1] - old_off);
    if (num == 0) {
      // fill slot value with default value 0
      for (int k = 0; k < dim; ++k) {
        slot_values[k + offset] = 0.0;
      }
    } else {
      if (num == dim) {
        memcpy(&slot_values[offset], &old_values[old_off], dim * sizeof(float));
      } else {
        // position fea
        // record position index need fix values
        int pos_idx = static_cast<int>(old_values[old_off]);
        for (int k = 0; k < dim; ++k) {
          if (k == pos_idx) {
            slot_values[k + offset] = 1.0;
          } else {
            slot_values[k + offset] = 0.0;
          }
        }
      }
    }
    slot_offsets[i] = offset;
    offset += dim;
  }
  slot_offsets[float_slot_num] = offset;
  CHECK(float_total_dims_size_ == static_cast<size_t>(offset));
}

void SlotPaddleBoxDataFeed::PutToFeedSlotVec(const SlotRecord* ins_vec,
                                             int num) {
#if defined(PADDLE_WITH_CUDA) && defined(_LINUX)
  paddle::platform::SetDeviceId(
      boost::get<platform::CUDAPlace>(place_).GetDeviceId());
  pack_->pack_instance(ins_vec, num);
  BuildSlotBatchGPU(pack_->ins_num());
#else
  for (int j = 0; j < use_slot_size_; ++j) {
    auto& feed = feed_vec_[j];
    if (feed == nullptr) {
      continue;
    }

    auto& slot_offset = offset_[j];
    slot_offset.clear();
    slot_offset.reserve(num + 1);
    slot_offset.push_back(0);

    int total_instance = 0;
    auto& info = used_slots_info_[j];
    // fill slot value with default value 0
    if (info.type[0] == 'f') {  // float
      auto& batch_fea = batch_float_feasigns_[j];
      batch_fea.clear();

      for (int i = 0; i < num; ++i) {
        auto r = ins_vec[i];
        size_t fea_num = 0;
        float* slot_values =
            r->slot_float_feasigns_.get_values(info.slot_value_idx, &fea_num);
        batch_fea.resize(total_instance + fea_num);
        memcpy(&batch_fea[total_instance], slot_values,
               sizeof(float) * fea_num);
        total_instance += fea_num;
        slot_offset.push_back(total_instance);
      }

      float* feasign = batch_fea.data();
      float* tensor_ptr =
          feed->mutable_data<float>({total_instance, 1}, this->place_);
      CopyToFeedTensor(tensor_ptr, feasign, total_instance * sizeof(float));

    } else if (info.type[0] == 'u') {  // uint64
      auto& batch_fea = batch_uint64_feasigns_[j];
      batch_fea.clear();

      for (int i = 0; i < num; ++i) {
        auto r = ins_vec[i];
        size_t fea_num = 0;
        uint64_t* slot_values =
            r->slot_uint64_feasigns_.get_values(info.slot_value_idx, &fea_num);
        if (fea_num > 0) {
          batch_fea.resize(total_instance + fea_num);
          memcpy(&batch_fea[total_instance], slot_values,
                 sizeof(uint64_t) * fea_num);
          total_instance += fea_num;
        }
        slot_offset.push_back(total_instance);
      }

      // no uint64_t type in paddlepaddle
      uint64_t* feasign = batch_fea.data();
      int64_t* tensor_ptr =
          feed->mutable_data<int64_t>({total_instance, 1}, this->place_);
      CopyToFeedTensor(tensor_ptr, feasign, total_instance * sizeof(int64_t));
    }

    LoD data_lod{slot_offset};
    feed_vec_[j]->set_lod(data_lod);

    if (info.dense) {
      if (info.inductive_shape_index != -1) {
        info.local_shape[info.inductive_shape_index] =
            total_instance / info.total_dims_without_inductive;
      }
      feed->Resize(framework::make_ddim(info.local_shape));
    }
  }
#endif
}

// template<typename T>
// void print_vector_data(const std::string &name, const T *values, int size) {
//  std::ostringstream ostream;
//  for (int i = 0; i < size; ++i) {
//    ostream << " " << values[i];
//  }
//  LOG(WARNING) << "[" << name << "]" << ostream.str();
//}

void SlotPaddleBoxDataFeed::BuildSlotBatchGPU(const int ins_num) {
#if defined(PADDLE_WITH_CUDA) && defined(_LINUX)
  fill_timer_.Resume();

  int offset_cols_size = (ins_num + 1);
  size_t slot_total_num = (use_slot_size_ * offset_cols_size);

  pack_->resize_gpu_slot_offsets(slot_total_num * sizeof(size_t));

  auto& value = pack_->value();
  const UsedSlotGpuType* used_slot_gpu_types =
      static_cast<const UsedSlotGpuType*>(pack_->get_gpu_slots());
  FillSlotValueOffset(ins_num, use_slot_size_,
                      reinterpret_cast<size_t*>(pack_->gpu_slot_offsets()),
                      value.d_uint64_offset.data(), uint64_use_slot_size_,
                      value.d_float_offset.data(), float_use_slot_size_,
                      used_slot_gpu_types);
  fill_timer_.Pause();
  size_t* d_slot_offsets = reinterpret_cast<size_t*>(pack_->gpu_slot_offsets());

  offset_timer_.Resume();
  HostBuffer<size_t>& offsets = pack_->offsets();
  offsets.resize(slot_total_num);
  HostBuffer<void*>& h_tensor_ptrs = pack_->h_tensor_ptrs();
  h_tensor_ptrs.resize(use_slot_size_);
  // alloc gpu memory
  pack_->resize_tensor();

  LoDTensor& float_tensor = pack_->float_tensor();
  LoDTensor& uint64_tensor = pack_->uint64_tensor();

  int64_t float_offset = 0;
  int64_t uint64_offset = 0;
  offset_timer_.Pause();

  copy_timer_.Resume();
  // copy index
  cudaMemcpy(offsets.data(), d_slot_offsets, slot_total_num * sizeof(size_t),
             cudaMemcpyDeviceToHost);
  copy_timer_.Pause();

  data_timer_.Resume();

  for (int j = 0; j < use_slot_size_; ++j) {
    auto& feed = feed_vec_[j];
    if (feed == nullptr) {
      h_tensor_ptrs[j] = nullptr;
      continue;
    }

    size_t* off_start_ptr = &offsets[j * offset_cols_size];

    int total_instance = static_cast<int>(off_start_ptr[offset_cols_size - 1]);
    CHECK(total_instance >= 0) << "slot idx:" << j
                               << ", total instance:" << total_instance;

    auto& info = used_slots_info_[j];
    // fill slot value with default value 0
    if (info.type[0] == 'f') {  // float
      if (total_instance > 0) {
        feed->ShareDataWith(float_tensor.Slice(
            static_cast<int64_t>(float_offset),
            static_cast<int64_t>(float_offset + total_instance)));
        feed->Resize({total_instance, 1});
        float_offset += total_instance;
        h_tensor_ptrs[j] = feed->mutable_data<float>(this->place_);
      } else {
        h_tensor_ptrs[j] =
            feed->mutable_data<float>({total_instance, 1}, this->place_);
      }
    } else if (info.type[0] == 'u') {  // uint64
      if (total_instance > 0) {
        feed->ShareDataWith(uint64_tensor.Slice(
            static_cast<int64_t>(uint64_offset),
            static_cast<int64_t>(uint64_offset + total_instance)));
        feed->Resize({total_instance, 1});
        uint64_offset += total_instance;
        h_tensor_ptrs[j] = feed->mutable_data<int64_t>(this->place_);
      } else {
        h_tensor_ptrs[j] =
            feed->mutable_data<int64_t>({total_instance, 1}, this->place_);
      }
    }
    // feed->set_lod({offsets});
    LoD& lod = (*feed->mutable_lod());
    lod.resize(1);
    lod[0].resize(offset_cols_size);
    memcpy(lod[0].MutableData(platform::CPUPlace()), off_start_ptr,
           offset_cols_size * sizeof(size_t));

    if (info.dense) {
      if (info.inductive_shape_index != -1) {
        info.local_shape[info.inductive_shape_index] =
            total_instance / info.total_dims_without_inductive;
      }
      feed->Resize(framework::make_ddim(info.local_shape));
    }
  }
  data_timer_.Pause();

  trans_timer_.Resume();
  void** dest_gpu_p = reinterpret_cast<void**>(pack_->slot_buf_ptr());
  cudaMemcpy(dest_gpu_p, h_tensor_ptrs.data(), use_slot_size_ * sizeof(void*),
             cudaMemcpyHostToDevice);

  CopyForTensor(ins_num, use_slot_size_, dest_gpu_p,
                (const size_t*)pack_->gpu_slot_offsets(),
                (const uint64_t*)value.d_uint64_keys.data(),
                (const int*)value.d_uint64_offset.data(),
                (const int*)value.d_uint64_lens.data(), uint64_use_slot_size_,
                (const float*)value.d_float_keys.data(),
                (const int*)value.d_float_offset.data(),
                (const int*)value.d_float_lens.data(), float_use_slot_size_,
                used_slot_gpu_types);
  trans_timer_.Pause();
#endif
}
int SlotPaddleBoxDataFeed::GetCurrentPhase() {
  auto box_ptr = paddle::framework::BoxWrapper::GetInstance();
  if (box_ptr->Mode() == 1) {  // For AucRunner
    return 1;
  } else {
    return box_ptr->Phase();
  }
}
void SlotPaddleBoxDataFeed::GetRankOffsetGPU(const int pv_num,
                                             const int ins_num) {
#if defined(PADDLE_WITH_CUDA) && defined(_LINUX)
  int max_rank = 3;  // the value is setting
  int col = max_rank * 2 + 1;
  auto& value = pack_->value();
  int* tensor_ptr =
      rank_offset_->mutable_data<int>({ins_num, col}, this->place_);
  CopyRankOffset(tensor_ptr, ins_num, pv_num, max_rank,
                 (const int*)value.d_rank.data(),
                 (const int*)value.d_cmatch.data(),
                 (const int*)value.d_ad_offset.data(), col);
#endif
}
void SlotPaddleBoxDataFeed::GetRankOffset(const SlotPvInstance* pv_vec,
                                          int pv_num, int ins_number) {
  int index = 0;
  int max_rank = 3;  // the value is setting
  int row = ins_number;
  int col = max_rank * 2 + 1;

  std::vector<int> rank_offset_mat(row * col, -1);
  rank_offset_mat.shrink_to_fit();

  for (int i = 0; i < pv_num; i++) {
    auto pv_ins = pv_vec[i];
    int ad_num = pv_ins->ads.size();
    int index_start = index;
    for (int j = 0; j < ad_num; ++j) {
      auto ins = pv_ins->ads[j];
      int rank = -1;
      if ((ins->cmatch == 222 || ins->cmatch == 223) &&
          ins->rank <= static_cast<uint32_t>(max_rank) && ins->rank != 0) {
        rank = ins->rank;
      }

      rank_offset_mat[index * col] = rank;
      if (rank > 0) {
        for (int k = 0; k < ad_num; ++k) {
          auto cur_ins = pv_ins->ads[k];
          int fast_rank = -1;
          if ((cur_ins->cmatch == 222 || cur_ins->cmatch == 223) &&
              cur_ins->rank <= static_cast<uint32_t>(max_rank) &&
              cur_ins->rank != 0) {
            fast_rank = cur_ins->rank;
          }

          if (fast_rank > 0) {
            int m = fast_rank - 1;
            rank_offset_mat[index * col + 2 * m + 1] = cur_ins->rank;
            rank_offset_mat[index * col + 2 * m + 2] = index_start + k;
          }
        }
      }
      index += 1;
    }
  }

  int* rank_offset = rank_offset_mat.data();
  int* tensor_ptr = rank_offset_->mutable_data<int>({row, col}, this->place_);
  CopyToFeedTensor(tensor_ptr, rank_offset, row * col * sizeof(int));
}

class BufferedLineFileReader {
  static const int MAX_FILE_BUFF_SIZE = 4 * 1024 * 1024;
  class FILEReader {
   public:
    explicit FILEReader(FILE* fp) : fp_(fp) {}
    int read(char* buf, int len) { return fread(buf, sizeof(char), len, fp_); }

   private:
    FILE* fp_;
  };

 private:
  template <typename T>
  int read_lines(T* reader, std::function<void(const std::string& s)> func) {
    int lines = 0;
    size_t ret = 0;
    char* ptr = NULL;
    char* eol = NULL;
    _total_len = 0;

    std::string x;
    while ((ret = reader->read(_buff, MAX_FILE_BUFF_SIZE)) > 0) {
      _total_len += ret;
      ptr = _buff;
      eol = reinterpret_cast<char*>(memchr(ptr, '\n', ret));
      while (eol != NULL) {
        int size = static_cast<int>((eol - ptr) + 1);
        x.append(ptr, size - 1);
        ++lines;
        func(x);

        x.clear();
        ptr += size;
        ret -= size;
        eol = reinterpret_cast<char*>(memchr(ptr, '\n', ret));
      }
      if (ret > 0) {
        x.append(ptr, ret);
      }
    }
    if (!x.empty()) {
      ++lines;
      func(x);
    }
    return lines;
  }

 public:
  BufferedLineFileReader() {
    _total_len = 0;
    _buff =
        reinterpret_cast<char*>(calloc(MAX_FILE_BUFF_SIZE + 1, sizeof(char)));
  }
  ~BufferedLineFileReader() { free(_buff); }

  int read_api(boxps::PaddleDataReader* reader,
               std::function<void(const std::string& s)> func) {
    return read_lines<boxps::PaddleDataReader>(reader, func);
  }

  int read_file(FILE* fp, std::function<void(const std::string& s)> func) {
    FILEReader reader(fp);
    return read_lines<FILEReader>(&reader, func);
  }

  uint64_t file_size(void) { return _total_len; }

 private:
  char* _buff = nullptr;
  uint64_t _total_len = 0;
};

class SlotInsParserMgr {
  struct ParserInfo {
    void* hmodule = nullptr;
    paddle::framework::ISlotParser* parser = nullptr;
  };

 public:
  SlotInsParserMgr() {}
  ~SlotInsParserMgr() {
    if (obj_map_.empty()) {
      return;
    }

    mutex_.lock();
    for (auto it = obj_map_.begin(); it != obj_map_.end(); ++it) {
      auto& info = it->second;
      MyPadBoxFreeObject func_freeobj =
          (MyPadBoxFreeObject)dlsym(info.hmodule, "PadBoxFreeObject");
      if (func_freeobj) {
        func_freeobj(info.parser);
      }
      dlclose(info.hmodule);
    }
    obj_map_.clear();
    mutex_.unlock();
  }

  paddle::framework::ISlotParser* Get(const std::string& path,
                                      const std::vector<AllSlotInfo>& slots) {
    ParserInfo info;
    mutex_.lock();
    std::map<std::string, ParserInfo>::iterator itx = obj_map_.find(path);
    if (itx != obj_map_.end()) {
      mutex_.unlock();
      return itx->second.parser;
    }
    info.hmodule = dlopen(path.c_str(), RTLD_NOW);
    if (info.hmodule == nullptr) {
      mutex_.unlock();
      LOG(FATAL) << "open so path:[" << path << "] failed";
      return nullptr;
    }
    MyPadBoxGetObject func_getobj =
        (MyPadBoxGetObject)dlsym(info.hmodule, "PadBoxGetObject");
    if (func_getobj) {
      info.parser = func_getobj();
      if (!info.parser->Init(slots)) {
        mutex_.unlock();
        LOG(FATAL) << "init so path:[" << path << "] failed";
        return nullptr;
      }
    }
    obj_map_.insert(std::make_pair(path, info));
    mutex_.unlock();

    return info.parser;
  }

 private:
  std::mutex mutex_;
  std::map<std::string, ParserInfo> obj_map_;
};

SlotInsParserMgr& global_parser_pool() {
  static SlotInsParserMgr pool;
  return pool;
}

void SlotPaddleBoxDataFeed::UnrollInstance(std::vector<SlotRecord>& items) {
    paddle::framework::ISlotParser* parser =
      global_parser_pool().Get(parser_so_path_, all_slots_info_);
    CHECK(parser != nullptr);
    std::cout<<"begin merge "<<std::endl;
    if (parser->unroll_instance(
          items,items.size(), [this](std::vector<SlotRecord> & release) {
                  SlotRecordPool().put(&release);
                  release.clear();
                  release.shrink_to_fit();
            })) {
        std::cout<<"merge success"<<std::endl;
  }
}

void SlotPaddleBoxDataFeed::LoadIntoMemory() {
  VLOG(3) << "LoadIntoMemory() begin, thread_id=" << thread_id_;
  if (!parser_so_path_.empty()) {
    LoadIntoMemoryByLib();
  } else {
    LoadIntoMemoryByCommand();
  }
}

void SlotPaddleBoxDataFeed::LoadIntoMemoryByLib(void) {
  paddle::framework::ISlotParser* parser =
      global_parser_pool().Get(parser_so_path_, all_slots_info_);
  CHECK(parser != nullptr);

  boxps::PaddleDataReader* reader = nullptr;
  if (BoxWrapper::GetInstance()->UseAfsApi() && pipe_command_.empty()) {
    reader =
        boxps::PaddleDataReader::New(BoxWrapper::GetInstance()->GetFileMgr());
  }

  std::string filename;
  BufferedLineFileReader line_reader;

  int from_pool_num = 0;
  while (this->PickOneFile(&filename)) {
    VLOG(3) << "PickOneFile, filename=" << filename
            << ", thread_id=" << thread_id_;
    std::vector<SlotRecord> record_vec;
    platform::Timer timeline;
    timeline.Start();
    const int max_fetch_num = 10000;
    int offset = 0;

    SlotRecordPool().get(&record_vec, max_fetch_num);
    from_pool_num = GetTotalFeaNum(record_vec, max_fetch_num);
    auto func = [this, &parser, &record_vec, &offset, &max_fetch_num,
                 &from_pool_num, &filename](const std::string& line) {
      int old_offset = offset;
      if (!parser->ParseOneInstance(
              line, [this, &offset, &record_vec, &max_fetch_num, &old_offset](
                        std::vector<SlotRecord>& vec, int num) {
                vec.resize(num);
                if (offset + num > max_fetch_num) {
                  // Considering the prob of show expanding is low, so we don't
                  // update STAT here
                  input_channel_->WriteMove(offset, &record_vec[0]);
                  SlotRecordPool().get(&record_vec[0], offset);
                  record_vec.resize(max_fetch_num);
                  offset = 0;
                  old_offset = 0;
                }
                for (int i = 0; i < num; ++i) {
                  auto& ins = record_vec[offset + i];
                  ins->reset();
                  vec[i] = ins;
                }
                offset = offset + num;
              })) {
        offset = old_offset;
        LOG(WARNING) << "read file:[" << filename << "] item error, line:["
                     << line << "]";
      }
      if (offset >= max_fetch_num) {
        input_channel_->Write(std::move(record_vec));
        STAT_ADD(STAT_total_feasign_num_in_mem,
                 GetTotalFeaNum(record_vec, max_fetch_num) - from_pool_num);
        record_vec.clear();
        SlotRecordPool().get(&record_vec, max_fetch_num);
        from_pool_num = GetTotalFeaNum(record_vec, max_fetch_num);
        offset = 0;
      }
    };
    int lines = 0;
    if (BoxWrapper::GetInstance()->UseAfsApi() && pipe_command_.empty()) {
      while (reader->open(filename) < 0) {
        sleep(1);
      }
      lines = line_reader.read_api(reader, func);
      reader->close();
    } else {
      if (BoxWrapper::GetInstance()->UseAfsApi()) {
        this->fp_ = BoxWrapper::GetInstance()->OpenReadFile(
            filename, this->pipe_command_);
      } else {
        int err_no = 0;
        this->fp_ = fs_open_read(filename, &err_no, this->pipe_command_);
      }
      CHECK(this->fp_ != nullptr);
      __fsetlocking(&*(this->fp_), FSETLOCKING_BYCALLER);
      lines = line_reader.read_file(this->fp_.get(), func);
    }
    if (offset > 0) {
      input_channel_->WriteMove(offset, &record_vec[0]);
      STAT_ADD(STAT_total_feasign_num_in_mem,
               GetTotalFeaNum(record_vec, max_fetch_num) - from_pool_num);
      if (offset < max_fetch_num) {
        SlotRecordPool().put(&record_vec[offset], (max_fetch_num - offset));
      }
    } else {
      SlotRecordPool().put(&record_vec);
    }
    record_vec.clear();
    timeline.Pause();
    VLOG(3) << "LoadIntoMemoryByLib() read all lines, file=" << filename
            << ", cost time=" << timeline.ElapsedSec()
            << " seconds, thread_id=" << thread_id_ << ", count=" << lines
            << ", filesize=" << line_reader.file_size() / 1024.0 / 1024.0
            << "MB";
  }
  if (reader != nullptr) {
    delete reader;
  }

  VLOG(3) << "LoadIntoMemoryByLib() end, thread_id=" << thread_id_
          << ", total size: " << line_reader.file_size();
}

void SlotPaddleBoxDataFeed::LoadIntoMemoryByCommand(void) {
  std::string filename;
  BufferedLineFileReader line_reader;
  while (this->PickOneFile(&filename)) {
    VLOG(3) << "PickOneFile, filename=" << filename
            << ", thread_id=" << thread_id_;
    if (BoxWrapper::GetInstance()->UseAfsApi()) {
      this->fp_ = BoxWrapper::GetInstance()->OpenReadFile(filename,
                                                          this->pipe_command_);
    } else {
      int err_no = 0;
      this->fp_ = fs_open_read(filename, &err_no, this->pipe_command_);
    }
    CHECK(this->fp_ != nullptr);
    __fsetlocking(&*(this->fp_), FSETLOCKING_BYCALLER);

    std::vector<SlotRecord> record_vec;
    platform::Timer timeline;
    timeline.Start();
    int max_fetch_num = 10000;
    SlotRecordPool().get(&record_vec, max_fetch_num);

    int offset = 0;
    line_reader.read_file(
        this->fp_.get(), [this, &record_vec, &offset, &max_fetch_num,
                          &filename](const std::string& line) {
          if (ParseOneInstance(line, &record_vec[offset])) {
            ++offset;
          } else {
            LOG(WARNING) << "read file:[" << filename << "] item error, line:["
                         << line << "]";
          }
          if (offset >= max_fetch_num) {
            input_channel_->Write(std::move(record_vec));
            record_vec.clear();
            SlotRecordPool().get(&record_vec, max_fetch_num);
            offset = 0;
          }
        });
    if (offset > 0) {
      input_channel_->WriteMove(offset, &record_vec[0]);
      if (offset < max_fetch_num) {
        SlotRecordPool().put(&record_vec[offset], (max_fetch_num - offset));
      }
    } else {
      SlotRecordPool().put(&record_vec);
    }
    record_vec.clear();
    timeline.Pause();
    VLOG(3) << "LoadIntoMemory() read all lines, file=" << filename
            << ", cost time=" << timeline.ElapsedSec()
            << " seconds, thread_id=" << thread_id_;
  }
  VLOG(3) << "LoadIntoMemory() end, thread_id=" << thread_id_
          << ", total size: " << line_reader.file_size();
}

static void parser_log_key(const std::string& log_key, uint64_t* search_id,
                           uint32_t* cmatch, uint32_t* rank) {
  std::string searchid_str = log_key.substr(16, 16);
  *search_id = static_cast<uint64_t>(strtoull(searchid_str.c_str(), NULL, 16));
  std::string cmatch_str = log_key.substr(11, 3);
  *cmatch = static_cast<uint32_t>(strtoul(cmatch_str.c_str(), NULL, 16));
  std::string rank_str = log_key.substr(14, 2);
  *rank = static_cast<uint32_t>(strtoul(rank_str.c_str(), NULL, 16));
}

bool SlotPaddleBoxDataFeed::ParseOneInstance(const std::string& line,
                                             SlotRecord* ins) {
  SlotRecord& rec = (*ins);
  // parse line
  const char* str = line.c_str();
  char* endptr = const_cast<char*>(str);
  int pos = 0;

  thread_local std::vector<std::vector<float>> slot_float_feasigns;
  thread_local std::vector<std::vector<uint64_t>> slot_uint64_feasigns;
  slot_float_feasigns.resize(float_use_slot_size_);
  slot_uint64_feasigns.resize(uint64_use_slot_size_);

  if (parse_ins_id_) {
    int num = strtol(&str[pos], &endptr, 10);
    CHECK(num == 1);  // NOLINT
    pos = endptr - str + 1;
    size_t len = 0;
    while (str[pos + len] != ' ') {
      ++len;
    }
    rec->ins_id_ = std::string(str + pos, len);
    pos += len + 1;
  }
  //  if (parse_content_) {
  //    int num = strtol(&str[pos], &endptr, 10);
  //    CHECK(num == 1);  // NOLINT
  //    pos = endptr - str + 1;
  //    size_t len = 0;
  //    while (str[pos + len] != ' ') {
  //      ++len;
  //    }
  //    rec->content_ = std::string(str + pos, len);
  //    pos += len + 1;
  //  }
  if (parse_logkey_) {
    int num = strtol(&str[pos], &endptr, 10);
    CHECK(num == 1);  // NOLINT
    pos = endptr - str + 1;
    size_t len = 0;
    while (str[pos + len] != ' ') {
      ++len;
    }
    // parse_logkey
    std::string log_key = std::string(str + pos, len);
    uint64_t search_id;
    uint32_t cmatch;
    uint32_t rank;
    parser_log_key(log_key, &search_id, &cmatch, &rank);

    rec->ins_id_ = log_key;
    rec->search_id = search_id;
    rec->cmatch = cmatch;
    rec->rank = rank;
    pos += len + 1;
  }

  int float_total_slot_num = 0;
  int uint64_total_slot_num = 0;

  for (size_t i = 0; i < all_slots_info_.size(); ++i) {
    auto& info = all_slots_info_[i];
    int num = strtol(&str[pos], &endptr, 10);
    PADDLE_ENFORCE(num,
                   "The number of ids can not be zero, you need padding "
                   "it in data generator; or if there is something wrong with "
                   "the data, please check if the data contains unresolvable "
                   "characters.\nplease check this error line: %s",
                   str);
    if (info.used_idx != -1) {
      if (info.type[0] == 'f') {  // float
        auto& slot_fea = slot_float_feasigns[info.slot_value_idx];
        slot_fea.clear();
        for (int j = 0; j < num; ++j) {
          float feasign = strtof(endptr, &endptr);
          if (fabs(feasign) < 1e-6 && !used_slots_info_[info.used_idx].dense) {
            continue;
          }
          slot_fea.push_back(feasign);
          ++float_total_slot_num;
        }
      } else if (info.type[0] == 'u') {  // uint64
        auto& slot_fea = slot_uint64_feasigns[info.slot_value_idx];
        slot_fea.clear();
        for (int j = 0; j < num; ++j) {
          uint64_t feasign =
              static_cast<uint64_t>(strtoull(endptr, &endptr, 10));
          if (feasign == 0 && !used_slots_info_[info.used_idx].dense) {
            continue;
          }
          slot_fea.push_back(feasign);
          ++uint64_total_slot_num;
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
  rec->slot_float_feasigns_.add_slot_feasigns(slot_float_feasigns,
                                              float_total_slot_num);
  rec->slot_uint64_feasigns_.add_slot_feasigns(slot_uint64_feasigns,
                                               uint64_total_slot_num);

  return (uint64_total_slot_num > 0);
}

////////////////////////////// pack ////////////////////////////////////
#if defined(PADDLE_WITH_CUDA) && defined(_LINUX)
static void SetCPUAffinity(int tid) {
  std::vector<int>& cores = boxps::get_train_cores();
  if (cores.empty()) {
    VLOG(0) << "not found binding read ins thread cores";
    return;
  }

  size_t core_num = cores.size() / 2;
  if (core_num < 8) {
    return;
  }
  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(cores[core_num + (tid % core_num)], &mask);
  pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask);
}
MiniBatchGpuPack::MiniBatchGpuPack(const paddle::platform::Place& place,
                                   const std::vector<UsedSlotInfo>& infos) {
  place_ = place;
  //  paddle::platform::SetDeviceId(boost::get<platform::CUDAPlace>(place).GetDeviceId());
  //  paddle::platform::CUDADeviceContext* context =
  //      dynamic_cast<paddle::platform::CUDADeviceContext*>(platform::DeviceContextPool::Instance().Get(
  //          place));
  stream_ = dynamic_cast<platform::CUDADeviceContext*>(
                platform::DeviceContextPool::Instance().Get(
                    boost::get<platform::CUDAPlace>(place)))
                ->stream();

  ins_num_ = 0;
  pv_num_ = 0;
  used_float_num_ = 0;
  used_uint64_num_ = 0;
  enable_pv_ = false;

  used_slot_size_ = static_cast<int>(infos.size());
  for (int i = 0; i < used_slot_size_; ++i) {
    auto& info = infos[i];
    if (info.type[0] == 'u') {
      gpu_used_slots_.push_back({1, info.slot_value_idx});
      ++used_uint64_num_;
    } else {
      gpu_used_slots_.push_back({0, info.slot_value_idx});
      ++used_float_num_;
    }
  }
  copy_host2device(&gpu_slots_, gpu_used_slots_.data(), gpu_used_slots_.size());

  slot_buf_ptr_ = memory::AllocShared(place_, used_slot_size_ * sizeof(void*));
}

MiniBatchGpuPack::~MiniBatchGpuPack() {}

void MiniBatchGpuPack::reset(const paddle::platform::Place& place) {
  place_ = place;
  stream_ = dynamic_cast<platform::CUDADeviceContext*>(
                platform::DeviceContextPool::Instance().Get(
                    boost::get<platform::CUDAPlace>(place)))
                ->stream();
  ins_num_ = 0;
  pv_num_ = 0;
  enable_pv_ = false;

  pack_timer_.Reset();
  trans_timer_.Reset();
}

void MiniBatchGpuPack::pack_pvinstance(const SlotPvInstance* pv_ins, int num) {
  pv_num_ = num;
  buf_.h_ad_offset.resize(num + 1);
  buf_.h_ad_offset[0] = 0;
  size_t ins_number = 0;

  ins_vec_.clear();
  for (int i = 0; i < num; ++i) {
    auto& pv = pv_ins[i];
    ins_number += pv->ads.size();
    for (auto ins : pv->ads) {
      ins_vec_.push_back(ins);
    }
    buf_.h_ad_offset[i + 1] = ins_number;
  }
  buf_.h_rank.resize(ins_number);
  buf_.h_cmatch.resize(ins_number);
  enable_pv_ = true;

  pack_instance(&ins_vec_[0], ins_number);
}

void MiniBatchGpuPack::pack_all_data(const SlotRecord* ins_vec, int num) {
  int uint64_total_num = 0;
  int float_total_num = 0;

  buf_.h_uint64_lens.resize(num + 1);
  buf_.h_uint64_lens[0] = 0;
  buf_.h_float_lens.resize(num + 1);
  buf_.h_float_lens[0] = 0;

  if (enable_pv_) {
    for (int i = 0; i < num; ++i) {
      auto r = ins_vec[i];
      uint64_total_num += r->slot_uint64_feasigns_.slot_values.size();
      buf_.h_uint64_lens[i + 1] = uint64_total_num;
      float_total_num += r->slot_float_feasigns_.slot_values.size();
      buf_.h_float_lens[i + 1] = float_total_num;

      buf_.h_rank[i] = r->rank;
      buf_.h_cmatch[i] = r->cmatch;
    }
  } else {
    for (int i = 0; i < num; ++i) {
      auto r = ins_vec[i];
      uint64_total_num += r->slot_uint64_feasigns_.slot_values.size();
      buf_.h_uint64_lens[i + 1] = uint64_total_num;
      float_total_num += r->slot_float_feasigns_.slot_values.size();
      buf_.h_float_lens[i + 1] = float_total_num;
    }
  }

  int uint64_cols = (used_uint64_num_ + 1);
  buf_.h_uint64_offset.resize(uint64_cols * num);
  buf_.h_uint64_keys.resize(uint64_total_num);

  int float_cols = (used_float_num_ + 1);
  buf_.h_float_offset.resize(float_cols * num);
  buf_.h_float_keys.resize(float_total_num);

  size_t fea_num = 0;
  uint64_total_num = 0;
  float_total_num = 0;
  for (int i = 0; i < num; ++i) {
    auto r = ins_vec[i];
    auto& uint64_feasigns = r->slot_uint64_feasigns_;
    fea_num = uint64_feasigns.slot_values.size();
    if (fea_num > 0) {
      memcpy(&buf_.h_uint64_keys[uint64_total_num],
             uint64_feasigns.slot_values.data(), fea_num * sizeof(uint64_t));
    }
    uint64_total_num += fea_num;
    // copy uint64 offset
    memcpy(&buf_.h_uint64_offset[i * uint64_cols],
           uint64_feasigns.slot_offsets.data(), sizeof(int) * uint64_cols);

    auto& float_feasigns = r->slot_float_feasigns_;
    fea_num = float_feasigns.slot_values.size();
    memcpy(&buf_.h_float_keys[float_total_num],
           float_feasigns.slot_values.data(), fea_num * sizeof(float));
    float_total_num += fea_num;

    // copy float offset
    memcpy(&buf_.h_float_offset[i * float_cols],
           float_feasigns.slot_offsets.data(), sizeof(int) * float_cols);
  }
  CHECK(uint64_total_num == static_cast<int>(buf_.h_uint64_lens.back()))
      << "uint64 value length error";
  CHECK(float_total_num == static_cast<int>(buf_.h_float_lens.back()))
      << "float value length error";
}
void MiniBatchGpuPack::pack_uint64_data(const SlotRecord* ins_vec, int num) {
  int uint64_total_num = 0;

  buf_.h_float_lens.clear();
  buf_.h_float_keys.clear();
  buf_.h_float_offset.clear();

  buf_.h_uint64_lens.resize(num + 1);
  buf_.h_uint64_lens[0] = 0;

  if (enable_pv_) {
    for (int i = 0; i < num; ++i) {
      auto r = ins_vec[i];
      uint64_total_num += r->slot_uint64_feasigns_.slot_values.size();
      buf_.h_uint64_lens[i + 1] = uint64_total_num;

      buf_.h_rank[i] = r->rank;
      buf_.h_cmatch[i] = r->cmatch;
    }
  } else {
    for (int i = 0; i < num; ++i) {
      auto r = ins_vec[i];
      uint64_total_num += r->slot_uint64_feasigns_.slot_values.size();
      buf_.h_uint64_lens[i + 1] = uint64_total_num;
    }
  }

  int uint64_cols = (used_uint64_num_ + 1);
  buf_.h_uint64_offset.resize(uint64_cols * num);
  buf_.h_uint64_keys.resize(uint64_total_num);

  size_t fea_num = 0;
  uint64_total_num = 0;
  for (int i = 0; i < num; ++i) {
    auto r = ins_vec[i];
    auto& uint64_feasigns = r->slot_uint64_feasigns_;
    fea_num = uint64_feasigns.slot_values.size();
    if (fea_num > 0) {
      memcpy(&buf_.h_uint64_keys[uint64_total_num],
             uint64_feasigns.slot_values.data(), fea_num * sizeof(uint64_t));
    }
    uint64_total_num += fea_num;
    // copy uint64 offset
    memcpy(&buf_.h_uint64_offset[i * uint64_cols],
           uint64_feasigns.slot_offsets.data(), sizeof(int) * uint64_cols);
  }
  CHECK(uint64_total_num == static_cast<int>(buf_.h_uint64_lens.back()))
      << "uint64 value length error";
}
void MiniBatchGpuPack::pack_float_data(const SlotRecord* ins_vec, int num) {
  int float_total_num = 0;

  buf_.h_uint64_lens.clear();
  buf_.h_uint64_offset.clear();
  buf_.h_uint64_keys.clear();

  buf_.h_float_lens.resize(num + 1);
  buf_.h_float_lens[0] = 0;

  if (enable_pv_) {
    for (int i = 0; i < num; ++i) {
      auto r = ins_vec[i];
      float_total_num += r->slot_float_feasigns_.slot_values.size();
      buf_.h_float_lens[i + 1] = float_total_num;

      buf_.h_rank[i] = r->rank;
      buf_.h_cmatch[i] = r->cmatch;
    }
  } else {
    for (int i = 0; i < num; ++i) {
      auto r = ins_vec[i];
      float_total_num += r->slot_float_feasigns_.slot_values.size();
      buf_.h_float_lens[i + 1] = float_total_num;
    }
  }

  int float_cols = (used_float_num_ + 1);
  buf_.h_float_offset.resize(float_cols * num);
  buf_.h_float_keys.resize(float_total_num);

  size_t fea_num = 0;
  float_total_num = 0;
  for (int i = 0; i < num; ++i) {
    auto r = ins_vec[i];
    auto& float_feasigns = r->slot_float_feasigns_;
    fea_num = float_feasigns.slot_values.size();
    memcpy(&buf_.h_float_keys[float_total_num],
           float_feasigns.slot_values.data(), fea_num * sizeof(float));
    float_total_num += fea_num;

    // copy float offset
    memcpy(&buf_.h_float_offset[i * float_cols],
           float_feasigns.slot_offsets.data(), sizeof(int) * float_cols);
  }
  CHECK(float_total_num == static_cast<int>(buf_.h_float_lens.back()))
      << "float value length error";
}

void MiniBatchGpuPack::pack_instance(const SlotRecord* ins_vec, int num) {
  pack_timer_.Resume();
  ins_num_ = num;
  CHECK(used_uint64_num_ > 0 || used_float_num_ > 0);
  // uint64 and float
  if (used_uint64_num_ > 0 && used_float_num_ > 0) {
    pack_all_data(ins_vec, num);
  } else if (used_uint64_num_ > 0) {  // uint64
    pack_uint64_data(ins_vec, num);
  } else {  // only float
    pack_float_data(ins_vec, num);
  }
  pack_timer_.Pause();
  // to gpu
  transfer_to_gpu();
}

void MiniBatchGpuPack::transfer_to_gpu(void) {
  trans_timer_.Resume();
  if (enable_pv_) {
    copy_host2device(&value_.d_ad_offset, buf_.h_ad_offset);
    copy_host2device(&value_.d_rank, buf_.h_rank);
    copy_host2device(&value_.d_cmatch, buf_.h_cmatch);
  }
  copy_host2device(&value_.d_uint64_lens, buf_.h_uint64_lens);
  copy_host2device(&value_.d_uint64_keys, buf_.h_uint64_keys);
  copy_host2device(&value_.d_uint64_offset, buf_.h_uint64_offset);

  copy_host2device(&value_.d_float_lens, buf_.h_float_lens);
  copy_host2device(&value_.d_float_keys, buf_.h_float_keys);
  copy_host2device(&value_.d_float_offset, buf_.h_float_offset);
  cudaStreamSynchronize(stream_);
  trans_timer_.Pause();
}
#endif

#endif
}  // namespace framework
}  // namespace paddle

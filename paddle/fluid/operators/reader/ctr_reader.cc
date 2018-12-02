// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/operators/reader/ctr_reader.h"

#include <gzstream.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>

#include <algorithm>
#include <random>

namespace paddle {
namespace operators {
namespace reader {

static inline void string_split(const std::string& s, const char delimiter,
                                std::vector<std::string>* output) {
  size_t start = 0;
  size_t end = s.find_first_of(delimiter);

  while (end <= std::string::npos) {
    output->emplace_back(s.substr(start, end - start));
    if (end == std::string::npos) {
      break;
    }
    start = end + 1;
    end = s.find_first_of(delimiter, start);
  }
}

static inline void parse_line(
    const std::string& line,
    const std::unordered_map<std::string, size_t>& slot_to_index,
    int64_t* label,
    std::unordered_map<std::string, std::vector<int64_t>>* slot_to_data) {
  std::vector<std::string> ret;
  string_split(line, ' ', &ret);
  *label = std::stoi(ret[2]) > 0;

  for (size_t i = 3; i < ret.size(); ++i) {
    const std::string& item = ret[i];
    std::vector<std::string> feasign_and_slot;
    string_split(item, ':', &feasign_and_slot);
    if (feasign_and_slot.size() == 2 &&
        slot_to_index.find(feasign_and_slot[1]) != slot_to_index.end()) {
      int64_t feasign = std::strtoll(feasign_and_slot[0].c_str(), NULL, 10);
      (*slot_to_data)[feasign_and_slot[1]].push_back(feasign);
    }
  }

  // NOTE:: if the slot has no value, then fill [0] as it's data.
  for (auto& item : slot_to_index) {
    if (slot_to_data->find(item.first) == slot_to_data->end()) {
      (*slot_to_data)[item.first].push_back(0);
    }
  }
}

class Reader {
 public:
  virtual ~Reader() {}
  virtual bool HasNext() = 0;
  virtual void NextLine(std::string* line) = 0;
};

class GzipReader : public Reader {
 public:
  explicit GzipReader(const std::string& file_name)
      : gzstream_(file_name.c_str()) {}

  ~GzipReader() {}

  bool HasNext() override { return gzstream_.peek() != EOF; }

  void NextLine(std::string* line) override { std::getline(gzstream_, *line); }

 private:
  igzstream gzstream_;
};

class MultiGzipReader : public Reader {
 public:
  explicit MultiGzipReader(const std::vector<std::string>& file_list) {
    for (auto& file : file_list) {
      readers_.emplace_back(std::make_shared<GzipReader>(file));
    }
  }

  bool HasNext() override {
    if (current_reader_index_ >= readers_.size()) {
      return false;
    }
    if (!readers_[current_reader_index_]->HasNext()) {
      current_reader_index_++;
      return HasNext();
    }
    return true;
  }

  void NextLine(std::string* line) override {
    readers_[current_reader_index_]->NextLine(line);
  }

 private:
  std::vector<std::shared_ptr<GzipReader>> readers_;
  size_t current_reader_index_ = 0;
};

void MonitorThread(std::vector<ReaderThreadStatus>* thread_status,
                   std::shared_ptr<LoDTensorBlockingQueue> queue) {
  VLOG(30) << "monitor thread in";
  bool reader_thread_is_running = true;
  while (reader_thread_is_running) {
    VLOG(30) << "reader_thread_is_running";
    reader_thread_is_running = false;
    for (size_t i = 0; i < (*thread_status).size(); ++i) {
      if ((*thread_status)[i] == Running) {
        VLOG(30) << "reader is running!";
        reader_thread_is_running = true;
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  }
  VLOG(30) << "all reader thread is stopped, push empty data into queue";
  queue->Push({});
  VLOG(30) << "monitor thread exited";
}

void ReadThread(const std::vector<std::string>& file_list,
                const std::vector<std::string>& slots, int batch_size,
                int thread_id, std::vector<ReaderThreadStatus>* thread_status,
                std::shared_ptr<LoDTensorBlockingQueue> queue) {
  VLOG(30) << "[" << thread_id << "]"
           << " reader thread start! thread_id = " << thread_id;
  for (auto& file : file_list) {
    VLOG(30) << "[" << thread_id << "]"
             << " file " << file;
  }
  (*thread_status)[thread_id] = Running;
  VLOG(30) << "set status to running";

  std::unordered_map<std::string, size_t> slot_to_index;
  for (size_t i = 0; i < slots.size(); ++i) {
    slot_to_index[slots[i]] = i;
  }

  std::string line;

  std::vector<std::unordered_map<std::string, std::vector<int64_t>>> batch_data;
  std::vector<int64_t> batch_label;

  MultiGzipReader reader(file_list);

  VLOG(30) << "reader inited";

  while (reader.HasNext()) {
    batch_data.clear();
    batch_data.reserve(batch_size);

    batch_label.clear();
    batch_label.reserve(batch_size);

    // read batch_size data
    for (int i = 0; i < batch_size; ++i) {
      if (reader.HasNext()) {
        reader.NextLine(&line);
        std::unordered_map<std::string, std::vector<int64_t>> slot_to_data;
        int64_t label;
        parse_line(line, slot_to_index, &label, &slot_to_data);
        batch_data.push_back(slot_to_data);
        batch_label.push_back(label);
      } else {
        break;
      }
    }

    std::vector<framework::LoDTensor> lod_datas;

    // first insert tensor for each slots
    for (auto& slot : slots) {
      std::vector<size_t> lod_data{0};
      std::vector<int64_t> batch_feasign;

      for (size_t i = 0; i < batch_data.size(); ++i) {
        auto& feasign = batch_data[i][slot];
        lod_data.push_back(lod_data.back() + feasign.size());
        batch_feasign.insert(batch_feasign.end(), feasign.begin(),
                             feasign.end());
      }

      framework::LoDTensor lod_tensor;
      framework::LoD lod{lod_data};
      lod_tensor.set_lod(lod);
      int64_t* tensor_data = lod_tensor.mutable_data<int64_t>(
          framework::make_ddim({1, static_cast<int64_t>(batch_feasign.size())}),
          platform::CPUPlace());
      memcpy(tensor_data, batch_feasign.data(),
             batch_feasign.size() * sizeof(int64_t));
      lod_datas.push_back(lod_tensor);
    }

    // insert label tensor
    framework::LoDTensor label_tensor;
    auto* label_tensor_data = label_tensor.mutable_data<int64_t>(
        framework::make_ddim({1, static_cast<int64_t>(batch_label.size())}),
        platform::CPUPlace());
    memcpy(label_tensor_data, batch_label.data(),
           batch_label.size() * sizeof(int64_t));
    lod_datas.push_back(label_tensor);

    queue->Push(lod_datas);
    VLOG(40) << "push one data, queue_size=" << queue->Size();
  }

  (*thread_status)[thread_id] = Stopped;
  VLOG(30) << "set status to stopped, thread " << thread_id << " exited";
}

}  // namespace reader
}  // namespace operators
}  // namespace paddle

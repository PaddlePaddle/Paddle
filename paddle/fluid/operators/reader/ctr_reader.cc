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
    const std::string& line, const std::vector<std::string>& slots,
    int64_t* label,
    std::unordered_map<std::string, std::vector<int64_t>>* slots_to_data) {
  std::vector<std::string> ret;
  string_split(line, ' ', &ret);
  *label = std::stoi(ret[2]) > 0;

  for (size_t i = 3; i < ret.size(); ++i) {
    const std::string& item = ret[i];
    std::vector<std::string> slot_and_feasign;
    string_split(item, ':', &slot_and_feasign);
    if (slot_and_feasign.size() == 2) {
      const std::string& slot = slot_and_feasign[1];
      int64_t feasign = std::strtoll(slot_and_feasign[0].c_str(), NULL, 10);
      (*slots_to_data)[slot_and_feasign[1]].push_back(feasign);
    }
  }

  // NOTE:: if the slot has no value, then fill [0] as it's data.
  for (auto& slot : slots) {
    if (slots_to_data->find(slot) == slots_to_data->end()) {
      (*slots_to_data)[slot].push_back(0);
    }
  }
}

// class Reader {
// public:
//  virtual ~Reader() {}
//  virtual bool HasNext() = 0;
//  virtual void NextLine(std::string& line) = 0;
//};

class GzipReader {
 public:
  explicit GzipReader(const std::string& file_name)
      : gzstream_(file_name.c_str()) {}

  ~GzipReader() {}

  bool HasNext() { return gzstream_.peek() != EOF; }

  void NextLine(std::string* line) { std::getline(gzstream_, *line); }

 private:
  igzstream gzstream_;
};

class MultiGzipReader {
 public:
  explicit MultiGzipReader(const std::vector<std::string>& file_list) {
    for (auto& file : file_list) {
      readers_.emplace_back(std::make_shared<GzipReader>(file));
    }
  }

  bool HasNext() {
    if (current_reader_index_ >= readers_.size()) {
      return false;
    }
    if (!readers_[current_reader_index_]->HasNext()) {
      current_reader_index_++;
      return HasNext();
    }
    return true;
  }

  void NextLine(std::string* line) {
    readers_[current_reader_index_]->NextLine(line);
  }

 private:
  std::vector<std::shared_ptr<GzipReader>> readers_;
  size_t current_reader_index_ = 0;
};

void CTRReader::ReadThread(const std::vector<std::string>& file_list,
                           const std::vector<std::string>& slots,
                           int batch_size,
                           std::shared_ptr<LoDTensorBlockingQueue> queue) {
  std::string line;

  std::vector<std::unordered_map<std::string, std::vector<int64_t>>> batch_data;
  std::vector<int64_t> batch_label;

  MultiGzipReader reader(file_list);
  // read all files
  for (int i = 0; i < batch_size; ++i) {
    if (reader.HasNext()) {
      reader.NextLine(&line);
      std::unordered_map<std::string, std::vector<int64_t>> slots_to_data;
      int64_t label;
      parse_line(line, slots, &label, &slots_to_data);
      batch_data.push_back(slots_to_data);
      batch_label.push_back(label);
    } else {
      break;
    }
  }

  std::vector<framework::LoDTensor> lod_datas;
  for (auto& slot : slots) {
    for (auto& slots_to_data : batch_data) {
      std::vector<size_t> lod_data{0};
      std::vector<int64_t> batch_feasign;

      auto& feasign = slots_to_data[slot];

      lod_data.push_back(lod_data.back() + feasign.size());
      batch_feasign.insert(feasign.end(), feasign.begin(), feasign.end());
      framework::LoDTensor lod_tensor;
      framework::LoD lod{lod_data};
      lod_tensor.set_lod(lod);
      int64_t* tensor_data = lod_tensor.mutable_data<int64_t>(
          framework::make_ddim({1, static_cast<int64_t>(batch_feasign.size())}),
          platform::CPUPlace());
      memcpy(tensor_data, batch_feasign.data(), batch_feasign.size());
      lod_datas.push_back(lod_tensor);
    }
  }
  queue->Push(lod_datas);
}

}  // namespace reader
}  // namespace operators
}  // namespace paddle

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
  if (s.empty()) return;

  size_t start = 0;
  size_t end = s.find(delimiter);
  while (end != std::string::npos) {
    if (end > start) output->emplace_back(s.substr(start, end - start));
    start = end + 1;
    end = s.find(delimiter, start);
  }
  auto term = s.substr(start);
  if (!term.empty()) output->emplace_back(term);
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

// label slot1:fea_sign slot2:fea_sign slot1:fea_sign
static inline void parse_svm_line(const std::string& line) {}

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

class PlainFileReader : public Reader {
 public:
  explicit PlainFileReader(const std::string& file_name)
      : stream_(file_name.c_str()) {}

  ~PlainFileReader() {}

  bool HasNext() override { return stream_.peek() != EOF; }

  void NextLine(std::string* line) override { std::getline(stream_, *line); }

 private:
  std::ifstream stream_;
};

template <typename SingleFileReader>
class MultiFileReader : public Reader {
 public:
  explicit MultiFileReader(const std::vector<std::string>& file_list) {
    for (auto& file : file_list) {
      readers_.emplace_back(std::make_shared<SingleFileReader>(file));
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
  std::vector<std::shared_ptr<SingleFileReader>> readers_;
  size_t current_reader_index_ = 0;
};

void MonitorThread(std::vector<ReaderThreadStatus>* thread_status,
                   std::shared_ptr<LoDTensorBlockingQueue> queue) {
  VLOG(3) << "monitor thread in";
  bool reader_thread_is_running = true;
  while (reader_thread_is_running) {
    VLOG(3) << "reader_thread_is_running";
    reader_thread_is_running = false;
    for (size_t i = 0; i < (*thread_status).size(); ++i) {
      if ((*thread_status)[i] == Running) {
        VLOG(3) << "reader is running!";
        reader_thread_is_running = true;
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  }
  VLOG(3) << "all reader thread is stopped, close the queue";
  queue->Close();
  VLOG(3) << "monitor thread exited";
}

void ReadSvmData(const DataDesc& data_desc, std::shared_ptr<Reader> reader,
                 std::shared_ptr<LoDTensorBlockingQueue> queue) {
  std::unordered_map<std::string, size_t> slot_to_index;
  for (size_t i = 0; i < data_desc.sparse_slot_ids_.size(); ++i) {
    slot_to_index[data_desc.sparse_slot_ids_[i]] = i;
  }

  std::string line;

  std::vector<std::unordered_map<std::string, std::vector<int64_t>>> batch_data;
  std::vector<int64_t> batch_label;

  while (reader->HasNext()) {
    batch_data.clear();
    batch_data.reserve(data_desc.batch_size_);

    batch_label.clear();
    batch_label.reserve(data_desc.batch_size_);

    // read batch_size data
    for (int i = 0; i < data_desc.batch_size_; ++i) {
      if (reader->HasNext()) {
        reader->NextLine(&line);
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

    // first insert tensor for each sparse_slots
    for (auto& slot : data_desc.sparse_slot_ids_) {
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
          framework::make_ddim({static_cast<int64_t>(batch_feasign.size()), 1}),
          platform::CPUPlace());
      memcpy(tensor_data, batch_feasign.data(),
             batch_feasign.size() * sizeof(int64_t));
      lod_datas.push_back(lod_tensor);
    }

    // insert label tensor
    framework::LoDTensor label_tensor;
    auto* label_tensor_data = label_tensor.mutable_data<int64_t>(
        framework::make_ddim({static_cast<int64_t>(batch_label.size()), 1}),
        platform::CPUPlace());
    memcpy(label_tensor_data, batch_label.data(),
           batch_label.size() * sizeof(int64_t));
    lod_datas.push_back(label_tensor);

    queue->Push(lod_datas);
    VLOG(4) << "push one data, queue_size=" << queue->Size();
  }
}

// label dense_fea,dense_fea sparse_fea,sparse_fea
static inline void parse_csv_line(
    const std::string& line, const DataDesc& data_desc, int64_t* label,
    std::vector<std::vector<float>>* dense_datas,
    std::vector<std::vector<int64_t>>* sparse_datas) {
  std::vector<std::string> ret;
  string_split(line, ' ', &ret);
  *label = std::stol(ret[0]);
  dense_datas->resize(data_desc.dense_slot_index_.size());
  for (size_t i = 0; i < data_desc.dense_slot_index_.size(); ++i) {
    int slot_idx = data_desc.dense_slot_index_[i];
    auto& slot_data = ret[slot_idx];
    std::vector<std::string> data_in_slot_str;
    string_split(slot_data, ',', &data_in_slot_str);
    std::vector<float> data_in_slot;
    for (auto& data_str : data_in_slot_str) {
      (*dense_datas)[i].push_back(std::stof(data_str));
    }
  }
  sparse_datas->resize(data_desc.sparse_slot_index_.size());
  for (size_t i = 0; i < data_desc.sparse_slot_index_.size(); ++i) {
    int slot_idx = data_desc.sparse_slot_index_[i];
    auto& slot_data = ret[slot_idx];
    std::vector<std::string> data_in_slot_str;
    string_split(slot_data, ',', &data_in_slot_str);
    std::vector<int64_t> data_in_slot;
    for (auto& data_str : data_in_slot_str) {
      auto id = std::stol(data_str);
      (*sparse_datas)[i].push_back(id);
    }
  }
}

void ReadCsvData(const DataDesc& data_desc, std::shared_ptr<Reader> reader,
                 std::shared_ptr<LoDTensorBlockingQueue> queue) {
  std::string line;
  while (reader->HasNext()) {
    std::vector<int64_t> batch_label;
    batch_label.reserve(data_desc.batch_size_);

    std::vector<std::vector<std::vector<float>>> batch_dense_data;
    batch_dense_data.reserve(data_desc.batch_size_);

    std::vector<std::vector<std::vector<int64_t>>> batch_sparse_data;
    batch_sparse_data.reserve(data_desc.batch_size_);

    // read batch_size data
    for (int i = 0; i < data_desc.batch_size_; ++i) {
      if (reader->HasNext()) {
        reader->NextLine(&line);
        int64_t label;
        std::vector<std::vector<float>> dense_datas;
        std::vector<std::vector<int64_t>> sparse_datas;
        parse_csv_line(line, data_desc, &label, &dense_datas, &sparse_datas);
        batch_label.push_back(label);
        if (!batch_dense_data.empty()) {
          PADDLE_ENFORCE_EQ(batch_dense_data[0].size(), dense_datas.size(),
                            "dense data should have the same shape");
        }
        batch_dense_data.push_back(dense_datas);
        batch_sparse_data.push_back(sparse_datas);
      } else {
        break;
      }
    }

    // the order of output data is label, dense_datas, sparse_datas
    std::vector<framework::LoDTensor> lod_datas;

    // insert label tensor
    framework::LoDTensor label_tensor;
    auto* label_tensor_data = label_tensor.mutable_data<int64_t>(
        framework::make_ddim({static_cast<int64_t>(batch_label.size()), 1}),
        platform::CPUPlace());
    memcpy(label_tensor_data, batch_label.data(),
           batch_label.size() * sizeof(int64_t));
    lod_datas.push_back(label_tensor);

    // insert tensor for each dense_slots
    for (size_t i = 0; i < data_desc.dense_slot_index_.size(); ++i) {
      framework::LoDTensor lod_tensor;
      size_t width = batch_dense_data[0][i].size();
      auto* tensor_data = lod_tensor.mutable_data<float>(
          framework::make_ddim(
              {static_cast<int64_t>(batch_dense_data.size()),  // batch_size
               static_cast<int64_t>(width)}),
          platform::CPUPlace());

      for (size_t j = 0; j < batch_dense_data.size(); ++j) {
        auto& dense_data_row = batch_dense_data[j][i];
        memcpy(tensor_data + j * width, dense_data_row.data(),
               width * sizeof(float));
      }

      lod_datas.push_back(lod_tensor);
    }

    // insert tensor for each sparse_slots
    for (size_t i = 0; i < data_desc.sparse_slot_index_.size(); ++i) {
      std::vector<size_t> lod_data{0};
      std::vector<int64_t> batch_feasign;

      for (size_t row_idx = 0; row_idx < batch_sparse_data.size(); ++row_idx) {
        auto& sparse_ids = batch_sparse_data[row_idx][i];
        lod_data.push_back(lod_data.back() + sparse_ids.size());
        batch_feasign.insert(batch_feasign.end(), sparse_ids.begin(),
                             sparse_ids.end());
      }

      framework::LoDTensor lod_tensor;
      framework::LoD lod{lod_data};
      lod_tensor.set_lod(lod);
      int64_t* tensor_data = lod_tensor.mutable_data<int64_t>(
          framework::make_ddim({static_cast<int64_t>(batch_feasign.size()), 1}),
          platform::CPUPlace());
      memcpy(tensor_data, batch_feasign.data(),
             batch_feasign.size() * sizeof(int64_t));
      lod_datas.push_back(lod_tensor);
    }

    queue->Push(lod_datas);
    VLOG(4) << "push one data, queue_size=" << queue->Size();
  }
}

void ReadThread(const std::vector<std::string>& file_list,
                const DataDesc& data_desc, int thread_id,
                std::vector<ReaderThreadStatus>* thread_status,
                std::shared_ptr<LoDTensorBlockingQueue> queue) {
  VLOG(3) << "[" << thread_id << "]"
          << " reader thread start! thread_id = " << thread_id;
  for (auto& file : file_list) {
    VLOG(3) << "[" << thread_id << "]"
            << " file " << file;
  }
  (*thread_status)[thread_id] = Running;
  VLOG(3) << "set status to running";

  std::shared_ptr<Reader> reader;
  if (data_desc.file_type_ == "gzip") {
    reader.reset(new MultiFileReader<GzipReader>(file_list));
  } else if (data_desc.file_type_ == "plain") {
    reader.reset(new MultiFileReader<PlainFileReader>(file_list));
  } else {
    PADDLE_THROW("do not support file format %s", data_desc.file_type_);
  }

  VLOG(3) << "reader inited";

  if (data_desc.file_format_ == "svm") {
    ReadSvmData(data_desc, reader, queue);
  } else if (data_desc.file_format_ == "csv") {
    ReadCsvData(data_desc, reader, queue);
  }

  (*thread_status)[thread_id] = Stopped;
  VLOG(3) << "set status to stopped, thread " << thread_id << " exited";
}

}  // namespace reader
}  // namespace operators
}  // namespace paddle

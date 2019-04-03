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

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>

namespace paddle {
namespace operators {
namespace reader {

typedef std::unordered_map<std::string, std::vector<int64_t>> SlotMap;
typedef std::unordered_map<std::string, size_t> SlotIndex;

template <typename T>
std::string to_string(const std::vector<T>& vec) {
  std::stringstream ss;
  for (const auto& c : vec) {
    ss << c << " ";
  }
  return ss.str();
}

template <typename T>
std::vector<T> slice(const std::vector<T>& v, int m, int n) {
  std::vector<T> vec(n - m);

  if (m == n) {
    return vec;
  }
  std::copy(v.begin() + m, v.begin() + n, vec.begin());
  return vec;
}

static inline std::vector<std::string> split(const std::string& s,
                                             char delimiter) {
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream tokenStream(s);
  while (std::getline(tokenStream, token, delimiter)) {
    tokens.push_back(token);
  }
  return tokens;
}

static inline std::vector<int> bucket(const int v_size, const int b_size) {
  int remainder = v_size % b_size;
  int bucket = v_size / b_size;
  std::vector<int> ret_vec(b_size, bucket);
  for (int i = 0; i < remainder; ++i) {
    ret_vec[i] = ret_vec[i] + 1;
  }
  int cur_bucket = 0;
  for (int j = 0; j < ret_vec.size(); ++j) {
    int tmp = ret_vec[j];
    ret_vec[j] = cur_bucket;
    cur_bucket += tmp;
  }
  ret_vec.push_back(cur_bucket);
  return ret_vec;
}

static inline std::vector<int> paging(const int v_size, const int g_size) {
  int remainder = v_size % g_size;
  int paging_size = v_size / g_size + 1;

  paging_size = remainder ? paging_size + 1 : paging_size;
  std::vector<int> ret_vec(paging_size, 0);

  for (int i = 0; i < paging_size; ++i) {
    ret_vec[i] = i * g_size >= v_size ? v_size : i * g_size;
  }

  return ret_vec;
}

static inline void parse_line_to_slots(const std::string& line,
                                       const SlotIndex& slot_to_index,
                                       std::vector<SlotMap>* slot_to_datas) {
  std::vector<std::vector<int64_t>> one_data;

  std::vector<std::string> groups = split(line, ';');

  if (groups.size() < 2) {
    LOG(ERROR) << "Data format error, please check.";
    return;
  }

  std::vector<int64_t> labels;
  auto label = (int64_t)std::stoi(groups[0]);
  labels.push_back(label);

  auto pairs = split(groups[1], ' ');
  auto pos_title_num = std::stoi(pairs[0]);
  auto neg_title_num = std::stoi(pairs[1]);

  if (pos_title_num + neg_title_num != groups.size() - 3) {
    LOG(ERROR) << "Data format error, please check.";
    return;
  }

  std::vector<int64_t> query_ids;
  std::vector<std::string> query_ids_str = split(groups[2], ' ');
  std::transform(
      query_ids_str.begin(), query_ids_str.end(), std::back_inserter(query_ids),
      [](const std::string& str) { return (int64_t)std::stoi(str); });

  for (int x = 0; x < pos_title_num; ++x) {
    std::vector<std::string> pos_title_ids_str = split(groups[3 + x], ' ');
    std::vector<int64_t> pos_title_ids;
    std::transform(
        pos_title_ids_str.begin(), pos_title_ids_str.end(),
        std::back_inserter(pos_title_ids),
        [](const std::string& str) { return (int64_t)std::stoi(str); });

    for (int y = 0; y < neg_title_num; ++y) {
      std::vector<std::string> neg_title_ids_str =
          split(groups[3 + pos_title_num + y], ' ');
      std::vector<int64_t> neg_title_ids;
      std::transform(
          neg_title_ids_str.begin(), neg_title_ids_str.end(),
          std::back_inserter(neg_title_ids),
          [](const std::string& str) { return (int64_t)std::stoi(str); });

      SlotMap slot_to_data;
      slot_to_data["1"] = query_ids;
      slot_to_data["2"] = pos_title_ids;
      slot_to_data["3"] = neg_title_ids;
      slot_to_data["l"] = labels;

      slot_to_datas->push_back(slot_to_data);
    }
  }
}

static inline void parse_line(const std::string& line,
                              const SlotIndex& slot_to_index, int64_t* label,
                              SlotMap* slot_to_data) {
  std::vector<std::string> ret = split(line, ' ');
  *label = std::stoi(ret[2]) > 0;

  for (size_t i = 3; i < ret.size(); ++i) {
    const std::string& item = ret[i];
    std::vector<std::string> feasign_and_slot = split(item, ':');
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
  explicit MultiFileReader(
      std::shared_ptr<BlockingQueue<std::string>>& readers)  // NOLINT
      : readers_(readers) {}

  bool HasNext() override {
    if (current_reader != nullptr && current_reader->HasNext()) {
      return true;
    }

    if (readers_->Size() == 0) {
      return false;
    }

    std::string file;
    auto ok = readers_->Receive(&file);
    current_reader = std::make_shared<SingleFileReader>(file);
    return ok;
  }

  void NextLine(std::string* line) override { current_reader->NextLine(line); }

 private:
  std::shared_ptr<BlockingQueue<std::string>>& readers_;
  std::shared_ptr<SingleFileReader> current_reader;
};

void MonitorThread(
    std::vector<ReaderThreadStatus>* thread_status,                   // NOLINT
    std::vector<std::shared_ptr<LoDTensorBlockingQueues>>& queues) {  // NOLINT
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

  for (auto q_ : queues) {
    q_->Close();
  }

  VLOG(3) << "monitor thread exited";
}

void PushSlotToQueue(const DataDesc& data_desc, const int batch_begin,
                     const int batch_end,
                     const std::vector<SlotMap>& batch_data,
                     std::shared_ptr<BlockingQueue<BATCH>>& queue) {  // NOLINT
  std::vector<framework::LoDTensor> lod_datas;

  // first insert tensor for each sparse_slots
  for (auto& slot : data_desc.sparse_slot_ids_) {
    std::vector<size_t> lod_data{0};
    std::vector<int64_t> batch_feasign;

    for (size_t i = batch_begin; i < batch_end; ++i) {
      SlotMap slotmap = batch_data[i];
      std::vector<int64_t> feasign = slotmap.at(slot);

      lod_data.push_back(lod_data.back() + feasign.size());
      batch_feasign.insert(batch_feasign.end(), feasign.begin(), feasign.end());
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
      framework::make_ddim({static_cast<int64_t>(batch_end - batch_begin), 1}),
      platform::CPUPlace());

  for (size_t i = batch_begin; i < batch_end; ++i) {
    label_tensor_data[i - batch_begin] = batch_data[i].at("l")[0];
  }
  lod_datas.push_back(label_tensor);
  queue->Send(lod_datas);
  VLOG(2) << "push one data, queue_size=" << queue->Size()
          << " pointer=" << queue.get();
}

void ReadPairWiseData(const DataDesc& data_desc,
                      std::shared_ptr<Reader>& reader,                 // NOLINT
                      std::shared_ptr<BlockingQueue<BATCH>>& queue) {  // NOLINT
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  auto dice = std::bind(dist, mt);

  SlotIndex slot_to_index;

  for (size_t i = 0; i < data_desc.sparse_slot_ids_.size(); ++i) {
    slot_to_index[data_desc.sparse_slot_ids_[i]] = i;
  }

  std::string line;
  std::vector<SlotMap> batch_datas;

  while (reader->HasNext()) {
    for (int i = 0; i < data_desc.batch_size_; ++i) {
      if (reader->HasNext()) {
        reader->NextLine(&line);
        std::vector<SlotMap> slot_to_datas;

        parse_line_to_slots(line, slot_to_index, &slot_to_datas);

        std::copy_if(slot_to_datas.begin(), slot_to_datas.end(),
                     std::back_inserter(batch_datas),
                     [&](SlotMap& slot) { return dice() <= 0.02; });
      } else {
        break;
      }
    }

    std::vector<int> slots = paging(static_cast<const int>(batch_datas.size()),
                                    data_desc.batch_size_);

    for (int x = 1; x < slots.size() - 1; ++x) {
      PushSlotToQueue(data_desc, slots[x - 1], slots[x], batch_datas, queue);
    }

    if (slots.back() % data_desc.batch_size_ == 0 || !reader->HasNext()) {
      PushSlotToQueue(data_desc, slots[slots.size() - 2],
                      slots[slots.size() - 1], batch_datas, queue);
      batch_datas.clear();
    } else {
      batch_datas.erase(batch_datas.begin(),
                        batch_datas.begin() + slots[slots.size() - 2]);
    }
  }
}

void ReadSvmData(const DataDesc& data_desc,
                 std::shared_ptr<Reader>& reader,                 // NOLINT
                 std::shared_ptr<BlockingQueue<BATCH>>& queue) {  // NOLINT
  SlotIndex slot_to_index;
  for (size_t i = 0; i < data_desc.sparse_slot_ids_.size(); ++i) {
    slot_to_index[data_desc.sparse_slot_ids_[i]] = i;
  }

  std::string line;

  std::vector<SlotMap> batch_data;
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
        SlotMap slot_to_data;
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

    queue->Send(lod_datas);
    VLOG(4) << "push one data, queue_size=" << queue->Size();
  }
}

// label dense_fea,dense_fea sparse_fea,sparse_fea
static inline void parse_csv_line(
    const std::string& line, const DataDesc& data_desc, int64_t* label,
    std::vector<std::vector<float>>* dense_datas,
    std::vector<std::vector<int64_t>>* sparse_datas) {
  std::vector<std::string> ret = split(line, ' ');
  *label = std::stol(ret[0]);
  dense_datas->resize(data_desc.dense_slot_index_.size());
  for (size_t i = 0; i < data_desc.dense_slot_index_.size(); ++i) {
    int slot_idx = data_desc.dense_slot_index_[i];
    auto& slot_data = ret[slot_idx];
    std::vector<std::string> data_in_slot_str = split(slot_data, ',');
    std::vector<float> data_in_slot;
    for (auto& data_str : data_in_slot_str) {
      (*dense_datas)[i].push_back(std::stof(data_str));
    }
  }
  sparse_datas->resize(data_desc.sparse_slot_index_.size());
  for (size_t i = 0; i < data_desc.sparse_slot_index_.size(); ++i) {
    int slot_idx = data_desc.sparse_slot_index_[i];
    auto& slot_data = ret[slot_idx];
    std::vector<std::string> data_in_slot_str = split(slot_data, ',');
    std::vector<int64_t> data_in_slot;
    for (auto& data_str : data_in_slot_str) {
      auto id = std::stol(data_str);
      (*sparse_datas)[i].push_back(id);
    }
  }
}

void ReadCsvData(const DataDesc& data_desc,
                 std::shared_ptr<Reader>& reader,                 // NOLINT
                 std::shared_ptr<BlockingQueue<BATCH>>& queue) {  // NOLINT
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

    queue->Send(lod_datas);
    VLOG(4) << "push one data, queue_size=" << queue->Size()
            << " pointer=" << queue.get();
  }
}

void ReadThread(const DataDesc& data_desc, int thread_id,
                std::vector<ReaderThreadStatus>* thread_status,        // NOLINT
                std::shared_ptr<BlockingQueue<std::string>>& readers,  // NOLINT
                std::shared_ptr<LoDTensorBlockingQueues>& queue) {     // NOLINT
  VLOG(3) << "[" << thread_id << "]"
          << " reader thread start! thread_id = " << thread_id;
  VLOG(3) << "there are " << readers->Size() << " waiting";

  (*thread_status)[thread_id] = Running;
  VLOG(3) << "set status to running";

  std::vector<std::thread> read_threads;

  for (int x = 0; x < queue->Queues(); x++) {
    std::thread reader_t([&, x]() {
      std::shared_ptr<Reader> reader;
      if (data_desc.file_type_ == "gzip") {
        reader = std::make_shared<MultiFileReader<GzipReader>>(readers);
      } else if (data_desc.file_type_ == "plain") {
        reader = std::make_shared<MultiFileReader<PlainFileReader>>(readers);
      } else {
        PADDLE_THROW("do not support file type %s", data_desc.file_type_);
      }

      if (data_desc.file_format_ == "svm") {
        ReadSvmData(data_desc, reader, queue->Get(x));
      } else if (data_desc.file_format_ == "csv") {
        ReadCsvData(data_desc, reader, queue->Get(x));
      } else if (data_desc.file_format_ == "pw") {
        ReadPairWiseData(data_desc, reader, queue->Get(x));
      } else {
        PADDLE_THROW("do not support file format %s", data_desc.file_format_);
      }
    });
    read_threads.emplace_back(std::move(reader_t));
  }

  // shutdown should stop all the reader thread
  for (auto& read : read_threads) {
    read.join();
  }

  (*thread_status)[thread_id] = Stopped;
  VLOG(3) << "set status to stopped, thread " << thread_id << " exited";
}

}  // namespace reader
}  // namespace operators
}  // namespace paddle

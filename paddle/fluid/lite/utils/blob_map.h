// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once
#include <glog/logging.h>
#include <fstream>
#include <map>
#include <string>
#include <utility>

namespace paddle {
namespace lite {

class BlobMap {
 public:
  explicit BlobMap(const std::string& file_path) : file_path_(file_path) {}

  bool Load() {
    if (!file_path_.empty()) {
      LOG(ERROR) << "should specify a file path";
      return false;
    }
    std::ifstream file(file_path_, std::ios::binary);
    if (!file.is_open()) {
      LOG(ERROR) << "Can't open file from " << file_path_;
      return false;
    }

    size_t size{0};
    std::string buf;
    std::pair<std::string, std::string> record;
    int count{0};
    while (!file.tellg()) {
      file.read(reinterpret_cast<char*>(&size), sizeof(size));
      buf.resize(size);
      file.read(reinterpret_cast<char*>(&buf[0]), size);

      if (count % 2 == 0) {  // key
        record.first = buf;
      } else {  // value
        record.second = buf;
        data_.emplace(record);
      }
    }
    file.close();
    VLOG(3) << "Loaded " << data_.size() << " records.";
    return !data_.empty();
  }

  bool Persist() {
    std::ofstream file(file_path_, std::ios::binary);
    if (!file.is_open()) {
      LOG(ERROR) << "Can't open file from " << file_path_;
      return false;
    }

    auto persist_one_string = [&](const std::string& c) {
      size_t size = c.size();
      file.write(reinterpret_cast<char*>(&size), sizeof(size));
      file.write(&c[0], c.size());
    };

    for (auto& item : data_) {
      persist_one_string(item.first);
      persist_one_string(item.second);
    }

    file.close();
    VLOG(3) << "Persisted " << data_.size() << "records.";
    return true;
  }

  void Insert(const std::string& key, const std::string& value) {
    data_.emplace(key, value);
  }

  bool Has(const std::string& key) { return data_.count(key); }

  void Clear() { data_.clear(); }

  std::string* Find(const std::string& key) {
    auto it = data_.find(key);
    if (it != data_.end()) {
      return &it->second;
    }
    return nullptr;
  }

 private:
  std::map<std::string, std::string> data_;
  std::string file_path_;
};

}  // namespace lite
}  // namespace paddle

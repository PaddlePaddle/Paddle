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

#pragma once
#include <glog/logging.h>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <glog/logging.h>
#include <vector>
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {

typedef std::unordered_map<int64_t, std::vector<int64_t>> UUMAP;

class KV_MAPS {
 public:
  KV_MAPS() { data_ = std::make_shared<UUMAP>(); };

  static std::shared_ptr<KV_MAPS> GetInstance() {
    if (NULL == s_instance_) {
      s_instance_.reset(new paddle::framework::KV_MAPS());
    }
    return s_instance_;
  }

  static void InitInstance(const std::string& filename) {
    if (s_instance_.get() == nullptr) {
      VLOG(0) << "kv maps init";
      s_instance_.reset(new KV_MAPS());
    }
    s_instance_->InitImpl(filename);
  }

  void InsertImpl(const std::string& filename) {
    VLOG(1) << "start init implementation!";
    VLOG(1) << "filename: " << filename;
    std::ifstream fin(filename.c_str());
    PADDLE_ENFORCE(fin.good(), "Can not open %s.", filename.c_str());
    int64_t size, dimensions, feasign;
    VLOG(1) << "begin to read file";
    fin >> size >> dimensions;
    VLOG(1) << "size: " << size << "dimensions: " << dimensions;
    std::vector<int64_t> feasign_values;
    feasign_values.resize(dimensions);
    uint64_t read_v;
    for (int64_t i = 0; i < size; i++) {
      std::stringstream ss;
      fin >> read_v;
      feasign = static_cast<int64_t>(read_v);
      if (data_->count(feasign) > 0)
        continue;
      ss << "feasign: " << feasign << "feasign_value: [";
      for (int64_t j = 0; j < dimensions; j++) {
        fin >> read_v;
        feasign_values[j] = static_cast<int64_t>(read_v);
        ss << feasign_values[j] << " ";
      }
      ss << "]\n";
      VLOG(1) << ss.str();
      data_->insert(
          std::pair<int64_t, std::vector<int64_t>>(feasign, feasign_values));
    } 
  }

  void InitImpl(const std::string& filename) {
    data_->clear();;
    VLOG(1) << "start init implementation!";
    VLOG(1) << "filename: " << filename;
    std::ifstream fin(filename.c_str());
    PADDLE_ENFORCE(fin.good(), "Can not open %s.", filename.c_str());
    int64_t size, dimensions, feasign;
    VLOG(1) << "begin to read file";
    fin >> size >> dimensions;
    VLOG(1) << "size: " << size << "dimensions: " << dimensions;
    std::vector<int64_t> feasign_values;
    feasign_values.resize(dimensions);
    uint64_t read_v;
    for (int64_t i = 0; i < size; i++) {
      std::stringstream ss;
      fin >> read_v;
      feasign = static_cast<int64_t>(read_v);
      ss << "feasign: " << feasign << "feasign_value: [";
      for (int64_t j = 0; j < dimensions; j++) {
        fin >> read_v;
        feasign_values[j] = static_cast<int64_t>(read_v);
        ss << feasign_values[j] << " ";
      }
      ss << "]\n";
      VLOG(1) << ss.str();
      data_->insert(
          std::pair<int64_t, std::vector<int64_t>>(feasign, feasign_values));
    }
    is_initialized_ = true;
  }

  std::shared_ptr<UUMAP> get_data() { return data_; }

 protected:
  static bool is_initialized_;
  static std::shared_ptr<KV_MAPS> s_instance_;
  std::shared_ptr<UUMAP> data_;
};

}  // namespace framework
}  // namespace paddle

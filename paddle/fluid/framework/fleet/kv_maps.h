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
typedef std::unordered_map<std::string, std::shared_ptr<UUMAP>> UUMAP_DICT;

class KV_MAPS {
 public:
  KV_MAPS() { dict_data_ = std::make_shared<UUMAP_DICT>(); };

  static std::shared_ptr<KV_MAPS> GetInstance() {
    if (NULL == s_instance_) {
      s_instance_.reset(new paddle::framework::KV_MAPS());
    }
    return s_instance_;
  }

  static void InitInstance(std::map<std::string, std::string>& dict_filenames) {
    if (s_instance_.get() == nullptr) {
      VLOG(0) << "kv maps init";
      s_instance_.reset(new KV_MAPS());
    }
    for (auto ite = dict_filenames.begin(); ite != dict_filenames.end(); ite++) {
      s_instance_->InitImpl(ite->first, ite->second);
    }
    is_initialized_ = true;
  }

  static void InsertInstance(std::map<std::string, std::string>& dict_filenames) {
    if (s_instance_.get() == nullptr) {
      VLOG(0) << "kv maps init";
      s_instance_.reset(new KV_MAPS());
    }
    for (auto ite = dict_filenames.begin(); ite != dict_filenames.end(); ite++) {
      s_instance_->InsertImpl(ite->first, ite->second);
    }
  }

  void InitImpl(const std::string& key, const std::string& filename) {
    VLOG(0) << "Init Pass Begin for key: " << key << "; filename: " << filename;
    if (dict_data_->count(key) == 0) {
      dict_data_->insert(std::pair<std::string, std::shared_ptr<UUMAP>>(key, std::make_shared<UUMAP>()));
    }
    std::shared_ptr<UUMAP> data_ = dict_data_->at(key);
    data_->clear();;
    Impl(data_, filename);
    return;
  }

  void InsertImpl(const std::string& key, const std::string& filename) {
    VLOG(0) << "Insert Pass Begin for key: " << key << "; filename: " << filename;
    if (dict_data_->count(key) == 0) {
      dict_data_->insert(std::pair<std::string, std::shared_ptr<UUMAP>>(key, std::make_shared<UUMAP>()));
    }
    std::shared_ptr<UUMAP> data_ = dict_data_->at(key);
    Impl(data_, filename);
    return;
  }

  void Impl(std::shared_ptr<UUMAP>& data_, const std::string& filename) {
	std::ifstream fin(filename.c_str());
    PADDLE_ENFORCE(fin.good(), "Can not open %s.", filename.c_str());
    uint64_t size, dimensions, feasign;
    VLOG(0) << "begin to read file";
    fin >> size >> dimensions;
    VLOG(0) << "size: " << size << "dimensions: " << dimensions;
    std::vector<int64_t> feasign_values;
    feasign_values.resize(dimensions);
    uint64_t read_v;
    for (uint64_t i = 0; i < size; i++) {
      std::stringstream ss;
      fin >> read_v;
      feasign = static_cast<int64_t>(read_v);
      if (data_->count(feasign) > 0)
        continue;
      ss << "feasign: " << feasign << "feasign_value: [";
      for (uint64_t j = 0; j < dimensions; j++) {
        fin >> read_v;
        feasign_values[j] = static_cast<int64_t>(read_v);
        ss << feasign_values[j] << " ";
      }
      ss << "]\n";
      VLOG(1) << ss.str();
      data_->insert(
          std::pair<int64_t, std::vector<int64_t>>(feasign, feasign_values));
    }
    VLOG(0) << "End to read file";
    return;
  }

  std::shared_ptr<UUMAP> get_data(const std::string& key) { 
      if (dict_data_->count(key) == 0) {
        dict_data_->insert(std::pair<std::string, std::shared_ptr<UUMAP>>(key, std::make_shared<UUMAP>()));
      }
      return dict_data_->at(key);
  }

 protected:
  static bool is_initialized_;
  static std::shared_ptr<KV_MAPS> s_instance_;
  std::shared_ptr<UUMAP_DICT> dict_data_;
};

}  // namespace framework
}  // namespace paddle

/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifndef PADDLE_FLUID_FRAMEWORK_DATA_FEED_H_
#define PADDLE_FLUID_FRAMEWORK_DATA_FEED_H_

#include <memory>
#include <set>
#include <map>
#include <string>
#include <thread>               // NOLINT
#include <vector>
#include <queue>
#include <mutex>                // NOLINT
#include <unordered_map>
#include <unordered_set>
#include <condition_variable>   // NOLINT
#include <fstream>

#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "proto/FeedDataParameter.pb.h"

namespace paddle {
namespace framework {
typedef uint64_t FeatureKey;

struct FeatureItem {
  FeatureItem() {}
  FeatureItem(FeatureKey sign_, uint16_t slot_) {
    sign() = sign_;
    slot() = slot_;
  }

  FeatureKey& sign() {
    return *(reinterpret_cast<FeatureKey*>(sign_buffer()));
  }

  const FeatureKey& sign() const {
    return *(const FeatureKey*)sign_buffer();
  }

  uint16_t& slot() {
    return _slot;
  }

  const uint16_t& slot() const {
    return _slot;
  }

 private:
  char _sign[sizeof(FeatureKey)];
  uint16_t _slot;
  char* sign_buffer() const {
    return (char *)_sign;
  }
};

// Record(average:14031B) is smaller than Sample(average:16530B)
struct Record {
  int show, click;
  std::vector<FeatureItem> feas;
  std::string lineid;
  std::string tags;
};

struct Gauc {
  int show, click;
  uint64_t fea;
  std::string lineid;
};

struct Instance {
  std::vector<std::vector<uint64_t>> feed_vec_buffer;
  std::vector<std::vector<int>> feed_vec_lod;
  std::vector<float> other_label;
  std::vector<Gauc> gauc_vec;
};

struct Sample {
  uint64_t label;
  std::map<uint16_t, std::vector<uint64_t>> feas;

  bool from_string(const std::string& input, const std::set<uint32_t>& slots) {
    size_t end = input.find_first_of(' ');
    if (end == std::string::npos) {
      LOG(ERROR) << "[ERROR] Fail in parsing:" << input;
      return false;
    }
    label = input[end + 3] - '0';
    CHECK(label == 0 || label == 1) << "invalid label:" << label;

    std::stringstream ss(input);

    std::string token;
    uint16_t slot_id = 0;
    uint64_t feature_id = 0;
    int num_nonfeas_token = 0;
    std::ostringstream os;
    while (ss >> token) {
      size_t end = token.find_first_of(':');
      if (end == std::string::npos) {
        ++num_nonfeas_token;
        continue;
      }

      try {
        slot_id = stoi(token.substr(end + 1));
      } catch (...) {
        LOG(ERROR) << "Error in parsing slot id:" << token;
        return false;
      }

      try {
        feature_id = stoull(token.substr(0, end));
      } catch (...) {
        LOG(ERROR) << "Error in parsing feature id:" << token;
        return false;
      }

      if (slot_id <= 0) {
        LOG(ERROR) << "invalid slot:" << slot_id << " feasign:" << feature_id
                   << " line:" << input;
        return false;
      }

      if (slots.find(slot_id) == slots.end()) {
        continue;
      }

      feas[slot_id].push_back(feature_id);
    }

    if (num_nonfeas_token != 4) {
      LOG(ERROR) << "Format error. Invalid number of non-feasign token:"
                 << num_nonfeas_token;
      return false;
    }

    return true;
  }
};

struct TeacherStudentSample {
  uint64_t label;
  std::map<uint16_t, std::vector<uint64_t>> feas;
  float q_score;

  void print() {
    LOG(ERROR) << "label: " << label << " score: " << q_score;
    for (auto &slot : feas) {
      for (auto &fea : slot.second) {
        LOG(ERROR) << "slot: " << slot.first << " fea: " << fea;
      }
    }
  }

  bool from_string(const std::string& input,
      const std::set<uint32_t>& slots,
      Gauc& gauc) {   // NOLINT
    size_t end = input.find_first_of(' ');
    if (end == std::string::npos) {
      LOG(ERROR) << "[ERROR] Fail in parsing:" << input;
      return false;
    }

    label = input[end + 3] - '0';
    CHECK(label == 0 || label == 1) << "invalid label:" << label;
    gauc.show = 1;
    gauc.click = label;
    gauc.lineid = input.substr(0, end);
    gauc.fea = 0;
    size_t dnn_start = input.find("*");
    if (dnn_start == std::string::npos) {
      q_score = -1.0;
    } else {
      dnn_start += 1;
      size_t dnn_end = input.find(' ', dnn_start);
      q_score = static_cast<float>(
          atof(input.substr(dnn_start, dnn_end - dnn_start).c_str()));
    }

    size_t head_pos = input.find("\t");
    std::string head = input.substr(0, head_pos);
    std::stringstream ss(head);

    std::string token;
    uint16_t slot_id = 0;
    uint64_t feature_id = 0;
    int num_nonfeas_token = 0;
    std::ostringstream os;
    while (ss >> token) {
      size_t end = token.find_first_of(':');
      if (end == std::string::npos) {
        ++num_nonfeas_token;
        continue;
      }

      try {
        slot_id = stoi(token.substr(end + 1));
      } catch (...) {
        LOG(ERROR) << "Error in parsing slot id:" << token;
        return false;
      }

      try {
        feature_id = stoull(token.substr(0, end));
      } catch (...) {
        LOG(ERROR) << "Error in parsing feature id:" << token;
        return false;
      }

      if (slot_id <= 0) {
        LOG(ERROR) << "invalid slot:" << slot_id << " feasign:" << feature_id
                   << " line:" << input;
        return false;
      }

      if (slots.find(slot_id) == slots.end()) {
        continue;
      }

      if (slot_id == 6048) {
        gauc.fea = feature_id;
      }
      feas[slot_id].push_back(feature_id);
    }

    if (num_nonfeas_token != 4) {
      LOG(ERROR) << "Format error. Invalid number of non-feasign token:"
                 << num_nonfeas_token;
      return false;
    }
    return true;
  }
};

class DataFeed {
 public:
  DataFeed() {}
  virtual ~DataFeed() {}
  virtual void init(const datafeed::DataFeedParameter& feed_param) = 0;
  /*
  * This function will be used to check file format.
  * Considering that this function may be used alone,
  * it does not check anything.
  * */
  virtual bool check_file(const char* filename) = 0;
  virtual bool set_file(const char* filename) = 0;
  virtual bool read_batch() = 0;
  virtual const std::vector<uint16_t>& get_all_slot_ids() {
    return _all_slot_ids;
  }

  virtual const std::vector<uint16_t>& get_use_slot_ids() {
    return _use_slot_ids;
  }

  virtual const std::vector<std::string>& get_use_slot_alias() {
    return _use_slot_alias;
  }

  virtual void add_feed_var(Variable* var,
                            const std::string& name) = 0;
  virtual void bind_scope(Scope* scope) = 0;
  virtual void set_batch_size(int batch) { _default_batch_size = batch; }
  virtual int get_batch_size() { return _batch_size; }
  virtual void set_buffer_size(int buffer_size) {}

  std::vector<LoDTensor*>& get_feed_vec() {
    return _feed_vec;
  }

  virtual std::vector<LoDTensor*>& get_feed_vec(const Instance& ins) {
    LOG(ERROR) << "use defalut get_feed_vec";
    return _feed_vec;
  }

 protected:
  std::vector<uint16_t> _all_slot_ids;
  std::vector<uint16_t> _use_slot_ids;
  std::vector<std::string> _use_slot_alias;
  std::vector<LoDTensor*> _feed_vec;
  int _default_batch_size;
  int _batch_size;
};

class TextClassDataFeed : public DataFeed {
 public:
  virtual ~TextClassDataFeed() {}
  virtual void init(const datafeed::DataFeedParameter& feed_param);
  virtual bool read_batch();
  virtual void add_feed_var(Variable* feed, const std::string& name);
  virtual void bind_scope(Scope* scope) {}
  virtual bool set_file(const char* filename);

  virtual bool check_file(const char* filename) {
    // TODO(xxx)
    return false;
  }

  void set_batch_size(int batch) {_batch_size = batch;}

 private:
  int read_whole_file(const std::string& filename, char* buffer);
  char* _file_content_buffer;
  char* _file_content_buffer_ptr;
  int* _batch_id_buffer;
  int* _label_ptr;
  int _file_size;
  std::vector<std::string> _names;
  std::shared_ptr<char> _file_content_buffer_host;
  std::shared_ptr<int> _batch_id_host;
  std::shared_ptr<int> _label_host;
};

}   // namespace framework
}   // namespace paddle

#endif  // PADDLE_FLUID_FRAMEWORK_DATA_FEED_H_
/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */

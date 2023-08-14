// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include <cstring>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <vector>

#include "glog/logging.h"
#include "paddle/fluid/distributed/ps/table/graph/graph_weighted_sampler.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/string/string_helper.h"

namespace paddle {
namespace distributed {

class Node {
 public:
  Node() {}
  explicit Node(uint64_t id) : id(id) {}
  virtual ~Node() {}
  static int id_size, int_size, weight_size;
  uint64_t get_id() { return id; }
  int64_t get_py_id() { return (int64_t)id; }
  void set_id(uint64_t id) { this->id = id; }

  virtual void build_edges(bool is_weighted UNUSED) {}
  virtual void build_sampler(std::string sample_type UNUSED) {}
  virtual void add_edge(uint64_t id UNUSED, float weight UNUSED) {}
  virtual std::vector<int> sample_k(
      int k UNUSED, const std::shared_ptr<std::mt19937_64> rng UNUSED) {
    return std::vector<int>();
  }
  virtual uint64_t get_neighbor_id(int idx UNUSED) { return 0; }
  virtual float get_neighbor_weight(int idx UNUSED) { return 1.; }

  virtual int get_size(bool need_feature);
  virtual void to_buffer(char *buffer, bool need_feature);
  virtual void recover_from_buffer(char *buffer);
  virtual std::string get_feature(int idx UNUSED) { return std::string(""); }
  virtual int get_feature_ids(std::vector<uint64_t> *res UNUSED) const {
    return 0;
  }
  virtual int get_feature_ids(int slot_idx UNUSED,
                              std::vector<uint64_t> *res UNUSED) const {
    return 0;
  }
  virtual int get_feature_ids(
      int slot_idx UNUSED,
      std::vector<uint64_t> &feature_id UNUSED,      // NOLINT
      std::vector<uint8_t> &slot_id UNUSED) const {  // NOLINT
    return 0;
  }
  virtual void set_feature(int idx UNUSED, const std::string &str UNUSED) {}
  virtual void set_feature_size(int size UNUSED) {}
  virtual void shrink_to_fit() {}
  virtual int get_feature_size() { return 0; }
  virtual size_t get_neighbor_size() { return 0; }

 protected:
  uint64_t id;
  bool is_weighted;
};

class GraphNode : public Node {
 public:
  GraphNode() : Node(), sampler(nullptr), edges(nullptr) {}
  explicit GraphNode(uint64_t id)
      : Node(id), sampler(nullptr), edges(nullptr) {}
  virtual ~GraphNode();
  virtual void build_edges(bool is_weighted);
  virtual void build_sampler(std::string sample_type);
  virtual void add_edge(uint64_t id, float weight) {
    edges->add_edge(id, weight);
  }
  virtual std::vector<int> sample_k(
      int k, const std::shared_ptr<std::mt19937_64> rng) {
    return sampler->sample_k(k, rng);
  }
  virtual uint64_t get_neighbor_id(int idx) { return edges->get_id(idx); }
  virtual float get_neighbor_weight(int idx) { return edges->get_weight(idx); }
  virtual size_t get_neighbor_size() { return edges->size(); }

 protected:
  Sampler *sampler;
  GraphEdgeBlob *edges;
};

class FeatureNode : public Node {
 public:
  FeatureNode() : Node() {}
  explicit FeatureNode(uint64_t id) : Node(id) {}
  virtual ~FeatureNode() {}
  virtual int get_size(bool need_feature);
  virtual void to_buffer(char *buffer, bool need_feature);
  virtual void recover_from_buffer(char *buffer);
  virtual std::string get_feature(int idx) {
    if (idx < static_cast<int>(this->feature.size())) {
      return this->feature[idx];
    } else {
      return std::string("");
    }
  }

  virtual int get_feature_ids(std::vector<uint64_t> *res) const {
    PADDLE_ENFORCE_NOT_NULL(res,
                            paddle::platform::errors::InvalidArgument(
                                "get_feature_ids res should not be null"));
    errno = 0;
    for (auto &feature_item : feature) {
      const uint64_t *feas = (const uint64_t *)(feature_item.c_str());
      size_t num = feature_item.length() / sizeof(uint64_t);
      CHECK((feature_item.length() % sizeof(uint64_t)) == 0)
          << "bad feature_item: [" << feature_item << "]";
      size_t n = res->size();
      res->resize(n + num);
      for (size_t i = 0; i < num; ++i) {
        (*res)[n + i] = feas[i];
      }
    }
    PADDLE_ENFORCE_EQ(
        errno,
        0,
        paddle::platform::errors::InvalidArgument(
            "get_feature_ids get errno should be 0, but got %d.", errno));
    return 0;
  }

  virtual int get_feature_ids(int slot_idx, std::vector<uint64_t> *res) const {
    PADDLE_ENFORCE_NOT_NULL(res,
                            paddle::platform::errors::InvalidArgument(
                                "get_feature_ids res should not be null"));
    res->clear();
    errno = 0;
    if (slot_idx < static_cast<int>(this->feature.size())) {
      const std::string &s = this->feature[slot_idx];
      const uint64_t *feas = (const uint64_t *)(s.c_str());

      size_t num = s.length() / sizeof(uint64_t);
      CHECK((s.length() % sizeof(uint64_t)) == 0)
          << "bad feature_item: [" << s << "]";
      res->resize(num);
      for (size_t i = 0; i < num; ++i) {
        (*res)[i] = feas[i];
      }
    }
    PADDLE_ENFORCE_EQ(
        errno,
        0,
        paddle::platform::errors::InvalidArgument(
            "get_feature_ids get errno should be 0, but got %d.", errno));
    return 0;
  }

  virtual int get_feature_ids(int slot_idx,
                              std::vector<uint64_t> &feature_id,      // NOLINT
                              std::vector<uint8_t> &slot_id) const {  // NOLINT
    errno = 0;
    size_t num = 0;
    if (slot_idx < static_cast<int>(this->feature.size())) {
      const std::string &s = this->feature[slot_idx];
      const uint64_t *feas = (const uint64_t *)(s.c_str());
      num = s.length() / sizeof(uint64_t);
      CHECK((s.length() % sizeof(uint64_t)) == 0)
          << "bad feature_item: [" << s << "]";
      for (size_t i = 0; i < num; ++i) {
        feature_id.push_back(feas[i]);
        slot_id.push_back(slot_idx);
      }
    }
    PADDLE_ENFORCE_EQ(
        errno,
        0,
        paddle::platform::errors::InvalidArgument(
            "get_feature_ids get errno should be 0, but got %d.", errno));
    return num;
  }

  virtual std::string *mutable_feature(int idx) {
    if (idx >= static_cast<int>(this->feature.size())) {
      this->feature.resize(idx + 1);
    }
    return &(this->feature[idx]);
  }

  virtual void set_feature(int idx, const std::string &str) {
    if (idx >= static_cast<int>(this->feature.size())) {
      this->feature.resize(idx + 1);
    }
    this->feature[idx] = str;
  }
  virtual void set_feature_size(int size) { this->feature.resize(size); }
  virtual int get_feature_size() { return this->feature.size(); }
  virtual void shrink_to_fit() {
    feature.shrink_to_fit();
    for (auto &slot : feature) {
      slot.shrink_to_fit();
    }
  }

  template <typename T>
  static std::string parse_value_to_bytes(std::vector<std::string> feat_str) {
    T v;
    size_t Tsize = sizeof(T) * feat_str.size();
    char buffer[Tsize];
    for (size_t i = 0; i < feat_str.size(); i++) {
      std::stringstream ss(feat_str[i]);
      ss >> v;
      std::memcpy(
          buffer + sizeof(T) * i, reinterpret_cast<char *>(&v), sizeof(T));
    }
    return std::string(buffer, Tsize);
  }

  template <typename T>
  static void parse_value_to_bytes(
      std::vector<std::string>::iterator feat_str_begin,
      std::vector<std::string>::iterator feat_str_end,
      std::string *output) {
    T v;
    size_t feat_str_size = feat_str_end - feat_str_begin;
    size_t Tsize = sizeof(T) * feat_str_size;
    char buffer[Tsize];
    memset(buffer, 0, Tsize * sizeof(char));
    for (size_t i = 0; i < feat_str_size; i++) {
      std::stringstream ss(*(feat_str_begin + i));
      ss >> v;
      std::memcpy(
          buffer + sizeof(T) * i, reinterpret_cast<char *>(&v), sizeof(T));
    }
    output->assign(buffer);
  }

  template <typename T>
  static std::vector<T> parse_bytes_to_array(std::string feat_str) {
    T v;
    std::vector<T> out;
    size_t start = 0;
    const char *buffer = feat_str.data();
    while (start < feat_str.size()) {
      std::memcpy(reinterpret_cast<char *>(&v), buffer + start, sizeof(T));
      start += sizeof(T);
      out.push_back(v);
    }
    return out;
  }

  template <typename T>
  static int parse_value_to_bytes(
      std::vector<paddle::string::str_ptr>::iterator feat_str_begin,
      std::vector<paddle::string::str_ptr>::iterator feat_str_end,
      std::string *output) {
    size_t feat_str_size = feat_str_end - feat_str_begin;
    size_t Tsize = sizeof(T) * feat_str_size;
    size_t num = output->length();
    output->resize(num + Tsize);

    T *fea_ptrs = reinterpret_cast<T *>(&(*output)[num]);

    thread_local paddle::string::str_ptr_stream ss;
    for (size_t i = 0; i < feat_str_size; i++) {
      ss.reset(*(feat_str_begin + i));
      int len = ss.end - ss.ptr;
      char *old_ptr = ss.ptr;
      ss >> fea_ptrs[i];
      if (ss.ptr - old_ptr != len) {
        return -1;
      }
    }
    return 0;
  }

 protected:
  std::vector<std::string> feature;
};

}  // namespace distributed
}  // namespace paddle

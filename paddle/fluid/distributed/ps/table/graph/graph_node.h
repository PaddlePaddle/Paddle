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
#include <sstream>
#include <vector>
#include "paddle/fluid/distributed/ps/table/graph/graph_weighted_sampler.h"
namespace paddle {
namespace distributed {

class Node {
 public:
  Node() {}
  Node(uint64_t id) : id(id) {}
  virtual ~Node() {}
  static int id_size, int_size, weight_size;
  uint64_t get_id() { return id; }
  void set_id(uint64_t id) { this->id = id; }

  virtual void build_edges(bool is_weighted) {}
  virtual void build_sampler(std::string sample_type) {}
  virtual void add_edge(uint64_t id, float weight) {}
  virtual std::vector<int> sample_k(
      int k, const std::shared_ptr<std::mt19937_64> rng) {
    return std::vector<int>();
  }
  virtual uint64_t get_neighbor_id(int idx) { return 0; }
  virtual float get_neighbor_weight(int idx) { return 1.; }

  virtual int get_size(bool need_feature);
  virtual void to_buffer(char *buffer, bool need_feature);
  virtual void recover_from_buffer(char *buffer);
  virtual std::string get_feature(int idx) { return std::string(""); }
  virtual void set_feature(int idx, std::string str) {}
  virtual void set_feature_size(int size) {}
  virtual int get_feature_size() { return 0; }

 protected:
  uint64_t id;
  bool is_weighted;
};

class GraphNode : public Node {
 public:
  GraphNode() : Node(), sampler(nullptr), edges(nullptr) {}
  GraphNode(uint64_t id) : Node(id), sampler(nullptr), edges(nullptr) {}
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

 protected:
  Sampler *sampler;
  GraphEdgeBlob *edges;
};

class FeatureNode : public Node {
 public:
  FeatureNode() : Node() {}
  FeatureNode(uint64_t id) : Node(id) {}
  virtual ~FeatureNode() {}
  virtual int get_size(bool need_feature);
  virtual void to_buffer(char *buffer, bool need_feature);
  virtual void recover_from_buffer(char *buffer);
  virtual std::string get_feature(int idx) {
    if (idx < (int)this->feature.size()) {
      return this->feature[idx];
    } else {
      return std::string("");
    }
  }

  virtual void set_feature(int idx, std::string str) {
    if (idx >= (int)this->feature.size()) {
      this->feature.resize(idx + 1);
    }
    this->feature[idx] = str;
  }
  virtual void set_feature_size(int size) { this->feature.resize(size); }
  virtual int get_feature_size() { return this->feature.size(); }

  template <typename T>
  static std::string parse_value_to_bytes(std::vector<std::string> feat_str) {
    T v;
    size_t Tsize = sizeof(T) * feat_str.size();
    char buffer[Tsize];
    for (size_t i = 0; i < feat_str.size(); i++) {
      std::stringstream ss(feat_str[i]);
      ss >> v;
      std::memcpy(buffer + sizeof(T) * i, (char *)&v, sizeof(T));
    }
    return std::string(buffer, Tsize);
  }

  template <typename T>
  static std::vector<T> parse_bytes_to_array(std::string feat_str) {
    T v;
    std::vector<T> out;
    size_t start = 0;
    const char *buffer = feat_str.data();
    while (start < feat_str.size()) {
      std::memcpy((char *)&v, buffer + start, sizeof(T));
      start += sizeof(T);
      out.push_back(v);
    }
    return out;
  }

 protected:
  std::vector<std::string> feature;
};
}  // namespace distributed
}  // namespace paddle

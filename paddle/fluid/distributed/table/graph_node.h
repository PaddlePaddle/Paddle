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
#include <vector>
#include "paddle/fluid/distributed/table/weighted_sampler.h"
namespace paddle {
namespace distributed {

class Node {
 public:
  Node(){}
  Node(uint64_t id)
      : id(id) {}
  virtual ~Node() {}
  static int id_size, int_size, weight_size;
  uint64_t get_id() { return id; }
  void set_id(uint64_t id) { this->id = id; }

  virtual void build_edges(bool is_weighted) {}
  virtual void build_sampler(std::string sample_type) {}
  virtual void add_edge(uint64_t id, float weight) {}
  virtual std::vector<int> sample_k(int k) { return std::vector<int>(); }
  virtual uint64_t get_neighbor_id(int idx){ return 0; }
  virtual float get_neighbor_weight(int idx){ return 1.; }

  virtual int get_size(bool need_feature);
  virtual void to_buffer(char *buffer, bool need_feature);
  virtual void recover_from_buffer(char *buffer);
  virtual void add_feature(std::string feature) { }
  virtual std::string get_feature(int idx) { return std::string(""); }

 protected:
  uint64_t id;

};

class GraphNode: public Node {
 public:
  GraphNode(): Node(), sampler(nullptr), edges(nullptr) { }
  GraphNode(uint64_t id)
      : Node(id), sampler(nullptr), edges(nullptr) {}
  virtual ~GraphNode();
  virtual void build_edges(bool is_weighted);
  virtual void build_sampler(std::string sample_type);
  virtual void add_edge(uint64_t id, float weight) { edges->add_edge(id, weight); }
  virtual std::vector<int> sample_k(int k) { return sampler->sample_k(k); }
  virtual uint64_t get_neighbor_id(int idx){return edges->get_id(idx);}
  virtual float get_neighbor_weight(int idx){return edges->get_weight(idx);}

 protected:
  Sampler *sampler;
  GraphEdgeBlob * edges;
};


class FeatureNode: public Node{
 public:
  FeatureNode(): Node() { }
  FeatureNode(uint64_t id) : Node(id) {}
  virtual ~FeatureNode() {}
  virtual int get_size(bool need_feature);
  virtual void to_buffer(char *buffer, bool need_feature);
  virtual void recover_from_buffer(char *buffer);
  virtual void add_feature(std::string feature) { this->feature.push_back(feature); }
  virtual std::string get_feature(int idx) { return feature[idx]; }

 protected:
  std::vector<std::string> feature;
};


}
}

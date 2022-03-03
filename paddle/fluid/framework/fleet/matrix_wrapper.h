/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/data_feed.h"

namespace paddle {
namespace framework {

struct Node {
  Node() : id(0), path_number(1) {}
  ~Node() {}
  uint64_t id;
  std::vector<float> paths;
  int16_t path_number;  //层级
};

class Nodes{
 public:
  Nodes() {}
  ~Nodes() {}
  //  if (_nodes) {
  //    delete[] _nodes;
  //    _nodes = NULL;
  //  }
  //}

  int load(std::string path);
  //Node* get_node();
  //size_t get_total_node_num();
  std::unordered_map<uint64_t, std::shared_ptr<Node>> _node_map;

 private:
  // path data info
  //Node* _nodes{nullptr};
  // total number of nodes
  size_t _total_node_num{0};
  // leaf node map
};

using NodesPtr = std::shared_ptr<Nodes>;
class MatrixWrapper {
 public:
  virtual ~MatrixWrapper() {}
  MatrixWrapper() {}

  // MatrixWrapper singleton
  static std::shared_ptr<MatrixWrapper> GetInstance() {
    if (NULL == s_instance_) {
      s_instance_.reset(new paddle::framework::MatrixWrapper());
    }
    return s_instance_;
  }

  void insert(std::string item_id_path) {
    if (NULL == _nodes){
      _nodes = std::make_shared<Nodes>();
      std::cout << "create nodes\n";
    }
    _nodes->load(item_id_path);
  }

  void sample(const uint16_t sample_slot, const std::vector<uint16_t> path_slots,
              std::vector<Record>* src_datas,
              std::vector<Record>* sample_results, const uint16_t path_num);

  NodesPtr _nodes;
 private:
  static std::shared_ptr<MatrixWrapper> s_instance_;
};

}  // end namespace framework
}  // end namespace paddle

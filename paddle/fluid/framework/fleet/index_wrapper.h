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
#include "paddle/fluid/framework/index_dataset.pb.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {

class Index {
 public:
  Index() {}
  ~Index() {}
};

class TreeIndex : public Index {
 public:
  TreeIndex() {}
  ~TreeIndex() {}

  int height() {return meta_.height();}
  int branch() {return meta_.branch();}
  uint64_t total_node_nums() {return total_nodes_num_;}
  // std::vector<uint64_t> get_parent_path(uint64_t node, uint64_t ancestor, bool ret_code);
  // std::vector<uint64_t> get_itemset_given_ancestor(relation, ancestor);
  // std::vector<uint64_t> get_children_given_ancestor_and_level();
  
  std::vector<Node*> get_nodes_given_level(int level);
  std::vector<Node*> get_travel_path(uint64_t id, int start_level=-1);

  // batch operation
  std::vector<std::vector<Node*>> batch_get_travel_path(std::vector<uint64_t>& ids, int start_level=-1);
  
  int load(std::string path);
  std::unordered_map<uint64_t, Node> data_;
  std::unordered_map<uint64_t, uint64_t> id_codes_map_;
  
  uint64_t total_nodes_num_;
  TreeMeta meta_;
};

using TreePtr = std::shared_ptr<TreeIndex>;

class IndexWrapper {
 public:
  virtual ~IndexWrapper() {}
  IndexWrapper() {}

  void clear_tree() { tree_map.clear(); }

  TreePtr GetTreeIndex(const std::string name) { 
    PADDLE_ENFORCE_NE(tree_map.find(name), tree_map.end(), "");
    return tree_map[name];
  }

  void insert_tree_index(std::string name, std::string tree_path) {
    if (tree_map.find(name) != tree_map.end()) {
      return;
    }
    TreePtr tree = std::make_shared<TreeIndex>();
    tree->load(tree_path);
    tree_map.insert(std::pair<std::string, TreePtr>{name, tree});
  }

  // IndexWrapper singleton
  static std::shared_ptr<IndexWrapper> GetInstancePtr() {
    if (NULL == s_instance_) {
      s_instance_.reset(new paddle::framework::IndexWrapper());
    }
    return s_instance_;
  }

  // IndexWrapper singleton
  static IndexWrapper* GetInstance() {
    if (NULL == s_instance_) {
      s_instance_.reset(new paddle::framework::IndexWrapper());
    }
    return s_instance_.get();
  }

 private:
  static std::shared_ptr<IndexWrapper> s_instance_;
  std::unordered_map<std::string, TreePtr> tree_map;
};


}  // end namespace framework
}  // end namespace paddle
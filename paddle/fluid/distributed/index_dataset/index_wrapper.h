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
#include <cmath>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "paddle/fluid/distributed/index_dataset/index_dataset.pb.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace distributed {

class Index {
 public:
  Index() {}
  ~Index() {}
};

class TreeIndex : public Index {
 public:
  TreeIndex() {}
  ~TreeIndex() {}

  int height() { return meta_.height(); }
  int branch() { return meta_.branch(); }
  uint64_t total_node_nums() { return total_nodes_num_; }
  uint64_t tree_max_node() { return max_id_; }
  int load(const std::string path);

  std::vector<uint64_t> get_nodes_given_level(int level, bool ret_code = false);
  std::vector<std::vector<uint64_t>> get_parent_path(
      const std::vector<uint64_t>& ids, int start_level = 0,
      bool ret_code = false);
  std::vector<uint64_t> get_ancestor_given_level(
      const std::vector<uint64_t>& ids, int level, bool ret_code = false);
  std::vector<uint64_t> get_all_items();
  std::unordered_map<uint64_t, uint64_t> get_relation(
      int level, const std::vector<uint64_t>& ids);
  std::vector<uint64_t> get_children_given_ancestor_and_level(
      uint64_t ancestor, int level, bool ret_code = true);
  std::vector<uint64_t> get_travel_path(uint64_t child, uint64_t ancestor);

  std::unordered_map<uint64_t, Node> data_;
  std::unordered_map<uint64_t, uint64_t> id_codes_map_;
  uint64_t total_nodes_num_;
  TreeMeta meta_;
  uint64_t max_id_;
};

using TreePtr = std::shared_ptr<TreeIndex>;

class IndexWrapper {
 public:
  virtual ~IndexWrapper() {}
  IndexWrapper() {}

  void clear_tree() { tree_map.clear(); }

  TreePtr get_tree_index(const std::string name) {
    PADDLE_ENFORCE_NE(tree_map.find(name), tree_map.end(),
                      paddle::platform::errors::InvalidArgument(
                          "tree [%s] doesn't exist. Please insert it firstly "
                          "by API[\' insert_tree_index \'].",
                          name));
    return tree_map[name];
  }

  void insert_tree_index(const std::string name, const std::string tree_path) {
    if (tree_map.find(name) != tree_map.end()) {
      return;
    }
    TreePtr tree = std::make_shared<TreeIndex>();
    int ret = tree->load(tree_path);
    if (ret != 0) return;
    tree_map.insert(std::pair<std::string, TreePtr>{name, tree});
  }

  static std::shared_ptr<IndexWrapper> GetInstancePtr() {
    if (NULL == s_instance_) {
      s_instance_.reset(new paddle::distributed::IndexWrapper());
    }
    return s_instance_;
  }

  static IndexWrapper* GetInstance() {
    if (NULL == s_instance_) {
      s_instance_.reset(new paddle::distributed::IndexWrapper());
    }
    return s_instance_.get();
  }

 private:
  static std::shared_ptr<IndexWrapper> s_instance_;
  std::unordered_map<std::string, TreePtr> tree_map;
};

}  // end namespace distributed
}  // end namespace paddle

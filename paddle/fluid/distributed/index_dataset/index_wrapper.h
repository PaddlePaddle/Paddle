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

  int Height() { return meta_.height(); }
  int Branch() { return meta_.branch(); }
  uint64_t TotalNodeNums() { return total_nodes_num_; }
  uint64_t EmbSize() { return max_id_ + 1; }
  int Load(const std::string path);

  inline bool CheckIsValid(int code) {
    if (data_.find(code) != data_.end()) {
      return true;
    } else {
      return false;
    }
  }

  std::vector<IndexNode> GetNodes(const std::vector<uint64_t>& codes);
  std::vector<uint64_t> GetLayerCodes(int level);
  std::vector<uint64_t> GetAncestorCodes(const std::vector<uint64_t>& ids,
                                         int level);
  std::vector<uint64_t> GetChildrenCodes(uint64_t ancestor, int level);
  std::vector<uint64_t> GetTravelCodes(uint64_t id, int start_level);
  std::vector<IndexNode> GetAllLeaves();

  std::unordered_map<uint64_t, IndexNode> data_;
  std::unordered_map<uint64_t, uint64_t> id_codes_map_;
  uint64_t total_nodes_num_;
  TreeMeta meta_;
  uint64_t max_id_;
  uint64_t max_code_;
  IndexNode fake_node_;
};

using TreePtr = std::shared_ptr<TreeIndex>;

class IndexWrapper {
 public:
  virtual ~IndexWrapper() {}
  IndexWrapper() {}

  void clear_tree() { tree_map.clear(); }

  TreePtr get_tree_index(const std::string name) {
    PADDLE_ENFORCE_NE(tree_map.find(name),
                      tree_map.end(),
                      common::errors::InvalidArgument(
                          "tree [%s] doesn't exist. Please insert it firstly "
                          "by API[\' insert_tree_index \'].",
                          name));
    return tree_map[name];
  }

  void insert_tree_index(const std::string name, const std::string tree_path) {
    if (tree_map.find(name) != tree_map.end()) {
      VLOG(0) << "Tree " << name << " has already existed.";
      return;
    }
    TreePtr tree = std::make_shared<TreeIndex>();
    int ret = tree->Load(tree_path);
    PADDLE_ENFORCE_EQ(ret,
                      0,
                      common::errors::InvalidArgument(
                          "Load tree[%s] from path[%s] failed. Please "
                          "check whether the file exists.",
                          name,
                          tree_path));
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

}  // namespace distributed
}  // namespace paddle

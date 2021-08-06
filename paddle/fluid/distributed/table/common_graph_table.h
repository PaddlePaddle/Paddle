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

#include <ThreadPool.h>
#include <assert.h>
#include <pthread.h>
#include <list>
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "paddle/fluid/distributed/table/accessor.h"
#include "paddle/fluid/distributed/table/common_table.h"
#include "paddle/fluid/distributed/table/graph/graph_node.h"
#include "paddle/fluid/framework/rw_lock.h"
#include "paddle/fluid/string/string_helper.h"
namespace paddle {
namespace distributed {
class GraphShard {
 public:
  size_t get_size();
  GraphShard() {}
  GraphShard(int shard_num) { this->shard_num = shard_num; }
  ~GraphShard();
  std::vector<Node *> &get_bucket() { return bucket; }
  std::vector<Node *> get_batch(int start, int end, int step);
  std::vector<uint64_t> get_ids_by_range(int start, int end) {
    std::vector<uint64_t> res;
    for (int i = start; i < end && i < (int)bucket.size(); i++) {
      res.push_back(bucket[i]->get_id());
    }
    return res;
  }

  GraphNode *add_graph_node(uint64_t id);
  FeatureNode *add_feature_node(uint64_t id);
  Node *find_node(uint64_t id);
  void delete_node(uint64_t id);
  void clear();
  void add_neighboor(uint64_t id, uint64_t dst_id, float weight);
  std::unordered_map<uint64_t, int> get_node_location() {
    return node_location;
  }

 private:
  std::unordered_map<uint64_t, int> node_location;
  int shard_num;
  std::vector<Node *> bucket;
};
class GraphTable : public SparseTable {
 public:
  GraphTable() {}
  virtual ~GraphTable() {}
  virtual int32_t pull_graph_list(int start, int size,
                                  std::unique_ptr<char[]> &buffer,
                                  int &actual_size, bool need_feature,
                                  int step);

  virtual int32_t random_sample_neighboors(
      uint64_t *node_ids, int sample_size,
      std::vector<std::unique_ptr<char[]>> &buffers,
      std::vector<int> &actual_sizes);

  int32_t random_sample_nodes(int sample_size, std::unique_ptr<char[]> &buffers,
                              int &actual_sizes);

  virtual int32_t get_nodes_ids_by_ranges(
      std::vector<std::pair<int, int>> ranges, std::vector<uint64_t> &res);
  virtual int32_t initialize();

  int32_t load(const std::string &path, const std::string &param);

  int32_t load_edges(const std::string &path, bool reverse);

  int32_t load_nodes(const std::string &path, std::string node_type);

  int32_t add_graph_node(std::vector<uint64_t> &id_list,
                         std::vector<bool> &is_weight_list);

  int32_t remove_graph_node(std::vector<uint64_t> &id_list);

  Node *find_node(uint64_t id);

  virtual int32_t pull_sparse(float *values,
                              const PullSparseValue &pull_value) {
    return 0;
  }

  virtual int32_t push_sparse(const uint64_t *keys, const float *values,
                              size_t num) {
    return 0;
  }

  virtual int32_t clear_nodes();
  virtual void clear() {}
  virtual int32_t flush() { return 0; }
  virtual int32_t shrink(const std::string &param) { return 0; }
  //指定保存路径
  virtual int32_t save(const std::string &path, const std::string &converter) {
    return 0;
  }
  virtual int32_t initialize_shard() { return 0; }
  virtual uint32_t get_thread_pool_index_by_shard_index(uint64_t shard_index);
  virtual uint32_t get_thread_pool_index(uint64_t node_id);
  virtual std::pair<int32_t, std::string> parse_feature(std::string feat_str);

  virtual int32_t get_node_feat(const std::vector<uint64_t> &node_ids,
                                const std::vector<std::string> &feature_names,
                                std::vector<std::vector<std::string>> &res);

  virtual int32_t set_node_feat(
      const std::vector<uint64_t> &node_ids,
      const std::vector<std::string> &feature_names,
      const std::vector<std::vector<std::string>> &res);

 protected:
  std::vector<GraphShard> shards;
  size_t shard_start, shard_end, server_num, shard_num_per_table, shard_num;
  const int task_pool_size_ = 24;
  const int random_sample_nodes_ranges = 3;

  std::vector<std::string> feat_name;
  std::vector<std::string> feat_dtype;
  std::vector<int32_t> feat_shape;
  std::unordered_map<std::string, int32_t> feat_id_map;
  std::string table_name;
  std::string table_type;

  std::vector<std::shared_ptr<::ThreadPool>> _shards_task_pool;
  std::vector<std::shared_ptr<std::mt19937_64>> _shards_task_rng_pool;
};
}  // namespace distributed

};  // namespace paddle

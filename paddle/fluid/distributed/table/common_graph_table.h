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
#include "paddle/fluid/framework/rw_lock.h"
#include "paddle/fluid/string/string_helper.h"
namespace paddle {
namespace distributed {
struct pair_hash {
  inline size_t operator()(const std::pair<uint64_t, GraphNodeType> &p) const {
    return p.first * 10007 + int(p.second);
  }
};
class GraphShard {
 public:
  static int bucket_low_bound;
  static int gcd(int s, int t) {
    if (s % t == 0) return t;
    return gcd(t, s % t);
  }
  size_t get_size();
  GraphShard() {}
  GraphShard(int shard_num) {
    this->shard_num = shard_num;
    bucket_size = init_bucket_size(shard_num);
    bucket.resize(bucket_size);
  }
  std::vector<std::list<GraphNode *>> &get_bucket() { return bucket; }
  std::vector<GraphNode *> get_batch(int start, int total_size);
  int init_bucket_size(int shard_num) {
    for (int i = bucket_low_bound;; i++) {
      if (gcd(i, shard_num) == 1) return i;
    }
    return -1;
  }
  std::list<GraphNode *>::iterator add_node(GraphNode *node);
  GraphNode *find_node(uint64_t id, GraphNodeType type);
  void add_neighboor(uint64_t id, GraphNodeType type, GraphEdge *edge);
  std::unordered_map<std::pair<uint64_t, GraphNodeType>,
                     std::list<GraphNode *>::iterator, pair_hash>
  get_node_location() {
    return node_location;
  }

 private:
  std::unordered_map<std::pair<uint64_t, GraphNodeType>,
                     std::list<GraphNode *>::iterator, pair_hash>
      node_location;
  int bucket_size, shard_num;
  std::vector<std::list<GraphNode *>> bucket;
};
class GraphTable : public SparseTable {
 public:
  GraphTable() { rwlock_.reset(new framework::RWLock); }
  virtual ~GraphTable() {}
  virtual int32_t pull_graph_list(int start, int size, char *&buffer,
                                  int &actual_size);
  virtual int32_t random_sample(uint64_t node_id, GraphNodeType type,
                                int sampe_size, char *&buffer,
                                int &actual_size);
  virtual int32_t initialize();
  int32_t load(const std::string &path, const std::string &param);
  GraphNode *find_node(uint64_t id, GraphNodeType type);

  virtual int32_t pull_sparse(float *values, const uint64_t *keys, size_t num) {
    return 0;
  }
  virtual int32_t push_sparse(const uint64_t *keys, const float *values,
                              size_t num) {
    return 0;
  }
  virtual void clear() {}
  virtual int32_t flush() { return 0; }
  virtual int32_t shrink(const std::string &param) { return 0; }
  //指定保存路径
  virtual int32_t save(const std::string &path, const std::string &converter) {
    return 0;
  }
  virtual int32_t initialize_shard() { return 0; }

 protected:
  std::vector<GraphShard> shards;
  size_t shard_start, shard_end, server_num, shard_num_per_table, shard_num;
  std::unique_ptr<framework::RWLock> rwlock_{nullptr};
  const int task_pool_size_ = 7;
  std::vector<std::shared_ptr<::ThreadPool>> _shards_task_pool;
};
}
};

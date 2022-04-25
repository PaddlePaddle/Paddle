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
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <ctime>
#include <functional>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <mutex>  // NOLINT
#include <numeric>
#include <queue>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "paddle/fluid/distributed/ps/table/accessor.h"
#include "paddle/fluid/distributed/ps/table/common_table.h"
#include "paddle/fluid/distributed/ps/table/graph/class_macro.h"
#include "paddle/fluid/distributed/ps/table/graph/graph_node.h"
#include "paddle/fluid/string/string_helper.h"
#include "paddle/phi/core/utils/rw_lock.h"

#ifdef PADDLE_WITH_HETERPS
#include "paddle/fluid/distributed/ps/table/depends/rocksdb_warpper.h"
#include "paddle/fluid/framework/fleet/heter_ps/gpu_graph_node.h"
#endif
namespace paddle {
namespace distributed {
class GraphShard {
 public:
  size_t get_size();
  GraphShard() {}
  ~GraphShard();
  std::vector<Node *> &get_bucket() { return bucket; }
  std::vector<Node *> get_batch(int start, int end, int step);
  std::vector<int64_t> get_ids_by_range(int start, int end) {
    std::vector<int64_t> res;
    for (int i = start; i < end && i < (int)bucket.size(); i++) {
      res.push_back(bucket[i]->get_id());
    }
    return res;
  }

  GraphNode *add_graph_node(int64_t id);
  GraphNode *add_graph_node(Node *node);
  FeatureNode *add_feature_node(int64_t id);
  Node *find_node(int64_t id);
  void delete_node(int64_t id);
  void clear();
  void add_neighbor(int64_t id, int64_t dst_id, float weight);
  std::unordered_map<int64_t, int> &get_node_location() {
    return node_location;
  }

 private:
  std::unordered_map<int64_t, int> node_location;
  std::vector<Node *> bucket;
};

enum LRUResponse { ok = 0, blocked = 1, err = 2 };

struct SampleKey {
  int idx;
  int64_t node_key;
  size_t sample_size;
  bool is_weighted;
  SampleKey(int _idx, int64_t _node_key, size_t _sample_size,
            bool _is_weighted) {
    idx = _idx;
    node_key = _node_key;
    sample_size = _sample_size;
    is_weighted = _is_weighted;
  }
  bool operator==(const SampleKey &s) const {
    return idx == s.idx && node_key == s.node_key &&
           sample_size == s.sample_size && is_weighted == s.is_weighted;
  }
};

class SampleResult {
 public:
  size_t actual_size;
  std::shared_ptr<char> buffer;
  SampleResult(size_t _actual_size, std::shared_ptr<char> &_buffer)
      : actual_size(_actual_size), buffer(_buffer) {}
  SampleResult(size_t _actual_size, char *_buffer)
      : actual_size(_actual_size),
        buffer(_buffer, [](char *p) { delete[] p; }) {}
  ~SampleResult() {}
};

template <typename K, typename V>
class LRUNode {
 public:
  LRUNode(K _key, V _data, size_t _ttl) : key(_key), data(_data), ttl(_ttl) {
    next = pre = NULL;
  }
  K key;
  V data;
  size_t ttl;
  // time to live
  LRUNode<K, V> *pre, *next;
};
template <typename K, typename V>
class ScaledLRU;

template <typename K, typename V>
class RandomSampleLRU {
 public:
  RandomSampleLRU(ScaledLRU<K, V> *_father) {
    father = _father;
    remove_count = 0;
    node_size = 0;
    node_head = node_end = NULL;
    global_ttl = father->ttl;
    total_diff = 0;
  }

  ~RandomSampleLRU() {
    LRUNode<K, V> *p;
    while (node_head != NULL) {
      p = node_head->next;
      delete node_head;
      node_head = p;
    }
  }
  LRUResponse query(K *keys, size_t length, std::vector<std::pair<K, V>> &res) {
    if (pthread_rwlock_tryrdlock(&father->rwlock) != 0)
      return LRUResponse::blocked;
    // pthread_rwlock_rdlock(&father->rwlock);
    int init_size = node_size - remove_count;
    process_redundant(length * 3);

    for (size_t i = 0; i < length; i++) {
      auto iter = key_map.find(keys[i]);
      if (iter != key_map.end()) {
        res.emplace_back(keys[i], iter->second->data);
        iter->second->ttl--;
        if (iter->second->ttl == 0) {
          remove(iter->second);
          if (remove_count != 0) remove_count--;
        } else {
          move_to_tail(iter->second);
        }
      }
    }
    total_diff += node_size - remove_count - init_size;
    if (total_diff >= 500 || total_diff < -500) {
      father->handle_size_diff(total_diff);
      total_diff = 0;
    }
    pthread_rwlock_unlock(&father->rwlock);
    return LRUResponse::ok;
  }
  LRUResponse insert(K *keys, V *data, size_t length) {
    if (pthread_rwlock_tryrdlock(&father->rwlock) != 0)
      return LRUResponse::blocked;
    // pthread_rwlock_rdlock(&father->rwlock);
    int init_size = node_size - remove_count;
    process_redundant(length * 3);
    for (size_t i = 0; i < length; i++) {
      auto iter = key_map.find(keys[i]);
      if (iter != key_map.end()) {
        move_to_tail(iter->second);
        iter->second->ttl = global_ttl;
        iter->second->data = data[i];
      } else {
        LRUNode<K, V> *temp = new LRUNode<K, V>(keys[i], data[i], global_ttl);
        add_new(temp);
      }
    }
    total_diff += node_size - remove_count - init_size;
    if (total_diff >= 500 || total_diff < -500) {
      father->handle_size_diff(total_diff);
      total_diff = 0;
    }

    pthread_rwlock_unlock(&father->rwlock);
    return LRUResponse::ok;
  }
  void remove(LRUNode<K, V> *node) {
    fetch(node);
    node_size--;
    key_map.erase(node->key);
    delete node;
  }

  void process_redundant(int process_size) {
    int length = std::min(remove_count, process_size);
    while (length--) {
      remove(node_head);
      remove_count--;
    }
    // std::cerr<<"after remove_count = "<<remove_count<<std::endl;
  }

  void move_to_tail(LRUNode<K, V> *node) {
    fetch(node);
    place_at_tail(node);
  }

  void add_new(LRUNode<K, V> *node) {
    node->ttl = global_ttl;
    place_at_tail(node);
    node_size++;
    key_map[node->key] = node;
  }
  void place_at_tail(LRUNode<K, V> *node) {
    if (node_end == NULL) {
      node_head = node_end = node;
      node->next = node->pre = NULL;
    } else {
      node_end->next = node;
      node->pre = node_end;
      node->next = NULL;
      node_end = node;
    }
  }

  void fetch(LRUNode<K, V> *node) {
    if (node->pre) {
      node->pre->next = node->next;
    } else {
      node_head = node->next;
    }
    if (node->next) {
      node->next->pre = node->pre;
    } else {
      node_end = node->pre;
    }
  }

 private:
  std::unordered_map<K, LRUNode<K, V> *> key_map;
  ScaledLRU<K, V> *father;
  size_t global_ttl, size_limit;
  int node_size, total_diff;
  LRUNode<K, V> *node_head, *node_end;
  friend class ScaledLRU<K, V>;
  int remove_count;
};

template <typename K, typename V>
class ScaledLRU {
 public:
  ScaledLRU(size_t _shard_num, size_t size_limit, size_t _ttl)
      : size_limit(size_limit), ttl(_ttl) {
    shard_num = _shard_num;
    pthread_rwlock_init(&rwlock, NULL);
    stop = false;
    thread_pool.reset(new ::ThreadPool(1));
    global_count = 0;
    lru_pool = std::vector<RandomSampleLRU<K, V>>(shard_num,
                                                  RandomSampleLRU<K, V>(this));
    shrink_job = std::thread([this]() -> void {
      while (true) {
        {
          std::unique_lock<std::mutex> lock(mutex_);
          cv_.wait_for(lock, std::chrono::milliseconds(20000));
          if (stop) {
            return;
          }
        }
        auto status =
            thread_pool->enqueue([this]() -> int { return Shrink(); });
        status.wait();
      }
    });
    shrink_job.detach();
  }
  ~ScaledLRU() {
    std::unique_lock<std::mutex> lock(mutex_);
    stop = true;
    cv_.notify_one();
  }
  LRUResponse query(size_t index, K *keys, size_t length,
                    std::vector<std::pair<K, V>> &res) {
    return lru_pool[index].query(keys, length, res);
  }
  LRUResponse insert(size_t index, K *keys, V *data, size_t length) {
    return lru_pool[index].insert(keys, data, length);
  }
  int Shrink() {
    int node_size = 0;
    for (size_t i = 0; i < lru_pool.size(); i++) {
      node_size += lru_pool[i].node_size - lru_pool[i].remove_count;
    }

    if ((size_t)node_size <= size_t(1.1 * size_limit) + 1) return 0;
    if (pthread_rwlock_wrlock(&rwlock) == 0) {
      global_count = 0;
      for (size_t i = 0; i < lru_pool.size(); i++) {
        global_count += lru_pool[i].node_size - lru_pool[i].remove_count;
      }
      if ((size_t)global_count > size_limit) {
        size_t remove = global_count - size_limit;
        for (size_t i = 0; i < lru_pool.size(); i++) {
          lru_pool[i].total_diff = 0;
          lru_pool[i].remove_count +=
              1.0 * (lru_pool[i].node_size - lru_pool[i].remove_count) /
              global_count * remove;
        }
      }
      pthread_rwlock_unlock(&rwlock);
      return 0;
    }
    return 0;
  }

  void handle_size_diff(int diff) {
    if (diff != 0) {
      __sync_fetch_and_add(&global_count, diff);
      if (global_count > int(1.25 * size_limit)) {
        thread_pool->enqueue([this]() -> int { return Shrink(); });
      }
    }
  }

  size_t get_ttl() { return ttl; }

 private:
  pthread_rwlock_t rwlock;
  size_t shard_num;
  int global_count;
  size_t size_limit, total, hit;
  size_t ttl;
  bool stop;
  std::thread shrink_job;
  std::vector<RandomSampleLRU<K, V>> lru_pool;
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::shared_ptr<::ThreadPool> thread_pool;
  friend class RandomSampleLRU<K, V>;
};

/*
#ifdef PADDLE_WITH_HETERPS
enum GraphSamplerStatus { waiting = 0, running = 1, terminating = 2 };
class GraphTable;
class GraphSampler {
 public:
  GraphSampler() {
    status = GraphSamplerStatus::waiting;
    thread_pool.reset(new ::ThreadPool(1));
    callback = [](std::vector<paddle::framework::GpuPsCommGraph> &res) {
      return;
    };
  }
  virtual int loadData(const std::string &path){
    return 0;
  }
  virtual int run_graph_sampling() = 0;
  virtual int start_graph_sampling() {
    if (status != GraphSamplerStatus::waiting) {
      return -1;
    }
    std::promise<int> prom;
    std::future<int> fut = prom.get_future();
    graph_sample_task_over = thread_pool->enqueue([&prom, this]() {
      prom.set_value(0);
      status = GraphSamplerStatus::running;
      return run_graph_sampling();
    });
    return fut.get();
  }
  virtual void init(size_t gpu_num, GraphTable *graph_table,
                    std::vector<std::string> args) = 0;
  virtual void set_graph_sample_callback(
      std::function<void(std::vector<paddle::framework::GpuPsCommGraph> &)>
          callback) {
    this->callback = callback;
  }

  virtual int end_graph_sampling() {
    if (status == GraphSamplerStatus::running) {
      status = GraphSamplerStatus::terminating;
      return graph_sample_task_over.get();
    }
    return -1;
  }
  virtual GraphSamplerStatus get_graph_sampler_status() { return status; }

 protected:
  std::function<void(std::vector<paddle::framework::GpuPsCommGraph> &)>
      callback;
  std::shared_ptr<::ThreadPool> thread_pool;
  GraphSamplerStatus status;
  std::future<int> graph_sample_task_over;
  std::vector<paddle::framework::GpuPsCommGraph> sample_res;
};
#endif
*/

class GraphTable : public Table {
 public:
  GraphTable() {
    use_cache = false;
    shard_num = 0;
    rw_lock.reset(new pthread_rwlock_t());
  }
  virtual ~GraphTable();

  virtual void *GetShard(size_t shard_idx) { return 0; }

  static int32_t sparse_local_shard_num(uint32_t shard_num,
                                        uint32_t server_num) {
    if (shard_num % server_num == 0) {
      return shard_num / server_num;
    }
    size_t local_shard_num = shard_num / server_num + 1;
    return local_shard_num;
  }

  static size_t get_sparse_shard(uint32_t shard_num, uint32_t server_num,
                                 uint64_t key) {
    return (key % shard_num) / sparse_local_shard_num(shard_num, server_num);
  }

  virtual int32_t pull_graph_list(int type_id, int idx, int start, int size,
                                  std::unique_ptr<char[]> &buffer,
                                  int &actual_size, bool need_feature,
                                  int step);

  virtual int32_t random_sample_neighbors(
      int idx, int64_t *node_ids, int sample_size,
      std::vector<std::shared_ptr<char>> &buffers,
      std::vector<int> &actual_sizes, bool need_weight);

  int32_t random_sample_nodes(int type_id, int idx, int sample_size,
                              std::unique_ptr<char[]> &buffers,
                              int &actual_sizes);

  virtual int32_t get_nodes_ids_by_ranges(
      int type_id, int idx, std::vector<std::pair<int, int>> ranges,
      std::vector<int64_t> &res);
  virtual int32_t Initialize() { return 0; }
  virtual int32_t Initialize(const TableParameter &config,
                             const FsClientParameter &fs_config);
  virtual int32_t Initialize(const GraphParameter &config);
  int32_t Load(const std::string &path, const std::string &param);

  int32_t load_edges(const std::string &path, bool reverse,
                     const std::string &edge_type);

  int32_t load_nodes(const std::string &path, std::string node_type);

  int32_t add_graph_node(int idx, std::vector<int64_t> &id_list,
                         std::vector<bool> &is_weight_list);

  int32_t remove_graph_node(int idx, std::vector<int64_t> &id_list);

  int32_t get_server_index_by_id(int64_t id);
  Node *find_node(int type_id, int idx, int64_t id);

  virtual int32_t Pull(TableContext &context) { return 0; }
  virtual int32_t Push(TableContext &context) { return 0; }

  virtual int32_t clear_nodes(int type, int idx);
  virtual void Clear() {}
  virtual int32_t Flush() { return 0; }
  virtual int32_t Shrink(const std::string &param) { return 0; }
  //指定保存路径
  virtual int32_t Save(const std::string &path, const std::string &converter) {
    return 0;
  }
  virtual int32_t InitializeShard() { return 0; }
  virtual int32_t SetShard(size_t shard_idx, size_t server_num) {
    _shard_idx = shard_idx;
    /*
    _shard_num is not used in graph_table, this following operation is for the
    purpose of
    being compatible with base class table.
    */
    _shard_num = server_num;
    this->server_num = server_num;
    return 0;
  }
  virtual uint32_t get_thread_pool_index_by_shard_index(int64_t shard_index);
  virtual uint32_t get_thread_pool_index(int64_t node_id);
  virtual std::pair<int32_t, std::string> parse_feature(int idx,
                                                        std::string feat_str);

  virtual int32_t get_node_feat(int idx, const std::vector<int64_t> &node_ids,
                                const std::vector<std::string> &feature_names,
                                std::vector<std::vector<std::string>> &res);

  virtual int32_t set_node_feat(
      int idx, const std::vector<int64_t> &node_ids,
      const std::vector<std::string> &feature_names,
      const std::vector<std::vector<std::string>> &res);

  size_t get_server_num() { return server_num; }

  virtual int32_t make_neighbor_sample_cache(size_t size_limit, size_t ttl) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      if (use_cache == false) {
        scaled_lru.reset(new ScaledLRU<SampleKey, SampleResult>(
            task_pool_size_, size_limit, ttl));
        use_cache = true;
      }
    }
    return 0;
  }
#ifdef PADDLE_WITH_HETERPS
  // virtual int32_t start_graph_sampling() {
  //   return this->graph_sampler->start_graph_sampling();
  // }
  // virtual int32_t end_graph_sampling() {
  //   return this->graph_sampler->end_graph_sampling();
  // }
  // virtual int32_t set_graph_sample_callback(
  //     std::function<void(std::vector<paddle::framework::GpuPsCommGraph> &)>
  //         callback) {
  //   graph_sampler->set_graph_sample_callback(callback);
  //   return 0;
  // }
  virtual char *random_sample_neighbor_from_ssd(
      int idx, int64_t id, int sample_size,
      const std::shared_ptr<std::mt19937_64> rng, int &actual_size);
  virtual int32_t add_node_to_ssd(int type_id, int idx, int64_t src_id,
                                  char *data, int len);
  virtual paddle::framework::GpuPsCommGraph make_gpu_ps_graph(
      int idx, std::vector<int64_t> ids);
  // virtual GraphSampler *get_graph_sampler() { return graph_sampler.get(); }
  int search_level;
#endif
  virtual int32_t add_comm_edge(int idx, int64_t src_id, int64_t dst_id);
  virtual int32_t build_sampler(int idx, std::string sample_type = "random");
  std::vector<std::vector<GraphShard *>> edge_shards, feature_shards;
  size_t shard_start, shard_end, server_num, shard_num_per_server, shard_num;
  int task_pool_size_ = 24;
  const int random_sample_nodes_ranges = 3;

  std::vector<std::vector<std::string>> feat_name;
  std::vector<std::vector<std::string>> feat_dtype;
  std::vector<std::vector<int32_t>> feat_shape;
  std::vector<std::unordered_map<std::string, int32_t>> feat_id_map;
  std::unordered_map<std::string, int> feature_to_id, edge_to_id;
  std::vector<std::string> id_to_feature, id_to_edge;
  std::string table_name;
  std::string table_type;

  std::vector<std::shared_ptr<::ThreadPool>> _shards_task_pool;
  std::vector<std::shared_ptr<std::mt19937_64>> _shards_task_rng_pool;
  std::shared_ptr<ScaledLRU<SampleKey, SampleResult>> scaled_lru;
  std::unordered_set<int64_t> extra_nodes;
  std::unordered_map<int64_t, size_t> extra_nodes_to_thread_index;
  bool use_cache, use_duplicate_nodes;
  int cache_size_limit;
  int cache_ttl;
  mutable std::mutex mutex_;
  std::shared_ptr<pthread_rwlock_t> rw_lock;
#ifdef PADDLE_WITH_HETERPS
  // paddle::framework::GpuPsGraphTable gpu_graph_table;
  paddle::distributed::RocksDBHandler *_db;
// std::shared_ptr<::ThreadPool> graph_sample_pool;
// std::shared_ptr<GraphSampler> graph_sampler;
// REGISTER_GRAPH_FRIEND_CLASS(2, CompleteGraphSampler, BasicBfsGraphSampler)
#endif
};

/*
#ifdef PADDLE_WITH_HETERPS
REGISTER_PSCORE_REGISTERER(GraphSampler);
class CompleteGraphSampler : public GraphSampler {
 public:
  CompleteGraphSampler() {}
  ~CompleteGraphSampler() {}
  // virtual pthread_rwlock_t *export_rw_lock();
  virtual int run_graph_sampling();
  virtual void init(size_t gpu_num, GraphTable *graph_table,
                    std::vector<std::string> args_);

 protected:
  GraphTable *graph_table;
  std::vector<std::vector<paddle::framework::GpuPsGraphNode>> sample_nodes;
  std::vector<std::vector<int64_t>> sample_neighbors;
  // std::vector<GpuPsCommGraph> sample_res;
  // std::shared_ptr<std::mt19937_64> random;
  int gpu_num;
};

class BasicBfsGraphSampler : public GraphSampler {
 public:
  BasicBfsGraphSampler() {}
  ~BasicBfsGraphSampler() {}
  // virtual pthread_rwlock_t *export_rw_lock();
  virtual int run_graph_sampling();
  virtual void init(size_t gpu_num, GraphTable *graph_table,
                    std::vector<std::string> args_);

 protected:
  GraphTable *graph_table;
  // std::vector<std::vector<GpuPsGraphNode>> sample_nodes;
  std::vector<std::vector<paddle::framework::GpuPsGraphNode>> sample_nodes;
  std::vector<std::vector<int64_t>> sample_neighbors;
  size_t gpu_num;
  int init_search_size, node_num_for_each_shard, edge_num_for_each_node;
  int rounds, interval;
  std::vector<std::unordered_map<int64_t, std::vector<int64_t>>>
      sample_neighbors_map;
};
#endif
*/
}  // namespace distributed

};  // namespace paddle

namespace std {

template <>
struct hash<paddle::distributed::SampleKey> {
  size_t operator()(const paddle::distributed::SampleKey &s) const {
    return s.idx ^ s.node_key ^ s.sample_size;
  }
};
}

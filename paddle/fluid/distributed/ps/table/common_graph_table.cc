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

#include "paddle/fluid/distributed/ps/table/common_graph_table.h"
#include <time.h>
#include <algorithm>
#include <chrono>
#include <set>
#include <sstream>
#include "paddle/fluid/distributed/common/utils.h"
#include "paddle/fluid/distributed/ps/table/graph/graph_node.h"
#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/string/printf.h"
#include "paddle/fluid/string/string_helper.h"

namespace paddle {
namespace distributed {

#ifdef PADDLE_WITH_HETERPS

int CompleteGraphSampler::run_graph_sampling() {
  pthread_rwlock_t *rw_lock = graph_table->rw_lock.get();
  pthread_rwlock_rdlock(rw_lock);
  std::cout << "in graph sampling" << std::endl;
  sample_nodes.clear();
  sample_neighbors.clear();
  sample_res.clear();
  sample_nodes.resize(gpu_num);
  sample_neighbors.resize(gpu_num);
  sample_res.resize(gpu_num);
  std::vector<std::vector<std::vector<paddle::framework::GpuPsGraphNode>>>
      sample_nodes_ex(graph_table->task_pool_size_);
  std::vector<std::vector<std::vector<int64_t>>> sample_neighbors_ex(
      graph_table->task_pool_size_);
  for (int i = 0; i < graph_table->task_pool_size_; i++) {
    sample_nodes_ex[i].resize(gpu_num);
    sample_neighbors_ex[i].resize(gpu_num);
  }
  std::vector<std::future<int>> tasks;
  for (size_t i = 0; i < graph_table->shards.size(); ++i) {
    tasks.push_back(
        graph_table->_shards_task_pool[i % graph_table->task_pool_size_]
            ->enqueue([&, i, this]() -> int {
              if (this->status == GraphSamplerStatus::terminating) return 0;
              paddle::framework::GpuPsGraphNode node;
              std::vector<Node *> &v =
                  this->graph_table->shards[i]->get_bucket();
              size_t ind = i % this->graph_table->task_pool_size_;
              for (size_t j = 0; j < v.size(); j++) {
                size_t location = v[j]->get_id() % this->gpu_num;
                node.node_id = v[j]->get_id();
                node.neighbor_size = v[j]->get_neighbor_size();
                node.neighbor_offset =
                    (int)sample_neighbors_ex[ind][location].size();
                sample_nodes_ex[ind][location].emplace_back(node);
                for (int k = 0; k < node.neighbor_size; k++)
                  sample_neighbors_ex[ind][location].push_back(
                      v[j]->get_neighbor_id(k));
              }
              return 0;
            }));
  }
  for (size_t i = 0; i < tasks.size(); i++) tasks[i].get();
  tasks.clear();
  for (int i = 0; i < gpu_num; i++) {
    tasks.push_back(
        graph_table->_shards_task_pool[i % graph_table->task_pool_size_]
            ->enqueue([&, i, this]() -> int {
              if (this->status == GraphSamplerStatus::terminating) return 0;
              int total_offset = 0;
              size_t ind = i % this->graph_table->task_pool_size_;
              for (int j = 0; j < this->graph_table->task_pool_size_; j++) {
                for (size_t k = 0; k < sample_nodes_ex[j][ind].size(); k++) {
                  sample_nodes[ind].push_back(sample_nodes_ex[j][ind][k]);
                  sample_nodes[ind].back().neighbor_offset += total_offset;
                }
                size_t neighbor_size = sample_neighbors_ex[j][ind].size();
                total_offset += neighbor_size;
                for (size_t k = 0; k < neighbor_size; k++) {
                  sample_neighbors[ind].push_back(
                      sample_neighbors_ex[j][ind][k]);
                }
              }
              return 0;
            }));
  }
  for (size_t i = 0; i < tasks.size(); i++) tasks[i].get();

  if (this->status == GraphSamplerStatus::terminating) {
    pthread_rwlock_unlock(rw_lock);
    return 0;
  }
  for (int i = 0; i < gpu_num; i++) {
    sample_res[i].node_list = sample_nodes[i].data();
    sample_res[i].neighbor_list = sample_neighbors[i].data();
    sample_res[i].node_size = sample_nodes[i].size();
    sample_res[i].neighbor_size = sample_neighbors[i].size();
  }
  pthread_rwlock_unlock(rw_lock);
  if (this->status == GraphSamplerStatus::terminating) {
    return 0;
  }
  callback(sample_res);
  return 0;
}
void CompleteGraphSampler::init(size_t gpu_num, GraphTable *graph_table,
                                std::vector<std::string> args) {
  this->gpu_num = gpu_num;
  this->graph_table = graph_table;
}

int BasicBfsGraphSampler::run_graph_sampling() {
  pthread_rwlock_t *rw_lock = graph_table->rw_lock.get();
  pthread_rwlock_rdlock(rw_lock);
  while (rounds > 0 && status == GraphSamplerStatus::running) {
    for (size_t i = 0; i < sample_neighbors_map.size(); i++) {
      sample_neighbors_map[i].clear();
    }
    sample_neighbors_map.clear();
    std::vector<int> nodes_left(graph_table->shards.size(),
                                node_num_for_each_shard);
    std::promise<int> prom;
    std::future<int> fut = prom.get_future();
    sample_neighbors_map.resize(graph_table->task_pool_size_);
    int task_size = 0;
    std::vector<std::future<int>> tasks;
    int init_size = 0;
    std::function<int(int, int64_t)> bfs = [&, this](int i, int64_t id) -> int {
      if (this->status == GraphSamplerStatus::terminating) {
        int task_left = __sync_sub_and_fetch(&task_size, 1);
        if (task_left == 0) {
          prom.set_value(0);
        }
        return 0;
      }
      size_t ind = i % this->graph_table->task_pool_size_;
      if (nodes_left[i] > 0) {
        auto iter = sample_neighbors_map[ind].find(id);
        if (iter == sample_neighbors_map[ind].end()) {
          Node *node = graph_table->shards[i]->find_node(id);
          if (node != NULL) {
            nodes_left[i]--;
            sample_neighbors_map[ind][id] = std::vector<int64_t>();
            iter = sample_neighbors_map[ind].find(id);
            size_t edge_fetch_size =
                std::min((size_t) this->edge_num_for_each_node,
                         node->get_neighbor_size());
            for (size_t k = 0; k < edge_fetch_size; k++) {
              int64_t neighbor_id = node->get_neighbor_id(k);
              int node_location = neighbor_id % this->graph_table->shard_num %
                                  this->graph_table->task_pool_size_;
              __sync_add_and_fetch(&task_size, 1);
              graph_table->_shards_task_pool[node_location]->enqueue(
                  bfs, neighbor_id % this->graph_table->shard_num, neighbor_id);
              iter->second.push_back(neighbor_id);
            }
          }
        }
      }
      int task_left = __sync_sub_and_fetch(&task_size, 1);
      if (task_left == 0) {
        prom.set_value(0);
      }
      return 0;
    };
    for (size_t i = 0; i < graph_table->shards.size(); ++i) {
      std::vector<Node *> &v = graph_table->shards[i]->get_bucket();
      if (v.size() > 0) {
        int search_size = std::min(init_search_size, (int)v.size());
        for (int k = 0; k < search_size; k++) {
          init_size++;
          __sync_add_and_fetch(&task_size, 1);
          int64_t id = v[k]->get_id();
          graph_table->_shards_task_pool[i % graph_table->task_pool_size_]
              ->enqueue(bfs, i, id);
        }
      }  // if
    }
    if (init_size == 0) {
      prom.set_value(0);
    }
    fut.get();
    if (this->status == GraphSamplerStatus::terminating) {
      pthread_rwlock_unlock(rw_lock);
      return 0;
    }
    VLOG(0) << "BasicBfsGraphSampler finishes the graph searching task";
    sample_nodes.clear();
    sample_neighbors.clear();
    sample_res.clear();
    sample_nodes.resize(gpu_num);
    sample_neighbors.resize(gpu_num);
    sample_res.resize(gpu_num);
    std::vector<std::vector<std::vector<paddle::framework::GpuPsGraphNode>>>
        sample_nodes_ex(graph_table->task_pool_size_);
    std::vector<std::vector<std::vector<int64_t>>> sample_neighbors_ex(
        graph_table->task_pool_size_);
    for (int i = 0; i < graph_table->task_pool_size_; i++) {
      sample_nodes_ex[i].resize(gpu_num);
      sample_neighbors_ex[i].resize(gpu_num);
    }
    tasks.clear();
    for (size_t i = 0; i < (size_t)graph_table->task_pool_size_; ++i) {
      tasks.push_back(
          graph_table->_shards_task_pool[i]->enqueue([&, i, this]() -> int {
            if (this->status == GraphSamplerStatus::terminating) {
              return 0;
            }
            paddle::framework::GpuPsGraphNode node;
            auto iter = sample_neighbors_map[i].begin();
            size_t ind = i;
            for (; iter != sample_neighbors_map[i].end(); iter++) {
              size_t location = iter->first % this->gpu_num;
              node.node_id = iter->first;
              node.neighbor_size = iter->second.size();
              node.neighbor_offset =
                  (int)sample_neighbors_ex[ind][location].size();
              sample_nodes_ex[ind][location].emplace_back(node);
              for (auto k : iter->second)
                sample_neighbors_ex[ind][location].push_back(k);
            }
            return 0;
          }));
    }

    for (size_t i = 0; i < tasks.size(); i++) {
      tasks[i].get();
      sample_neighbors_map[i].clear();
    }
    tasks.clear();
    if (this->status == GraphSamplerStatus::terminating) {
      pthread_rwlock_unlock(rw_lock);
      return 0;
    }
    for (size_t i = 0; i < (size_t)gpu_num; i++) {
      tasks.push_back(
          graph_table->_shards_task_pool[i % graph_table->task_pool_size_]
              ->enqueue([&, i, this]() -> int {
                if (this->status == GraphSamplerStatus::terminating) {
                  pthread_rwlock_unlock(rw_lock);
                  return 0;
                }
                int total_offset = 0;
                for (int j = 0; j < this->graph_table->task_pool_size_; j++) {
                  for (size_t k = 0; k < sample_nodes_ex[j][i].size(); k++) {
                    sample_nodes[i].push_back(sample_nodes_ex[j][i][k]);
                    sample_nodes[i].back().neighbor_offset += total_offset;
                  }
                  size_t neighbor_size = sample_neighbors_ex[j][i].size();
                  total_offset += neighbor_size;
                  for (size_t k = 0; k < neighbor_size; k++) {
                    sample_neighbors[i].push_back(sample_neighbors_ex[j][i][k]);
                  }
                }
                return 0;
              }));
    }
    for (size_t i = 0; i < tasks.size(); i++) tasks[i].get();
    if (this->status == GraphSamplerStatus::terminating) {
      pthread_rwlock_unlock(rw_lock);
      return 0;
    }
    for (int i = 0; i < gpu_num; i++) {
      sample_res[i].node_list = sample_nodes[i].data();
      sample_res[i].neighbor_list = sample_neighbors[i].data();
      sample_res[i].node_size = sample_nodes[i].size();
      sample_res[i].neighbor_size = sample_neighbors[i].size();
    }
    pthread_rwlock_unlock(rw_lock);
    if (this->status == GraphSamplerStatus::terminating) {
      return 0;
    }
    callback(sample_res);
    rounds--;
    if (rounds > 0) {
      for (int i = 0;
           i < interval && this->status == GraphSamplerStatus::running; i++) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
      }
    }
  }
  return 0;
}
void BasicBfsGraphSampler::init(size_t gpu_num, GraphTable *graph_table,
                                std::vector<std::string> args) {
  this->gpu_num = gpu_num;
  this->graph_table = graph_table;
  init_search_size = args.size() > 0 ? std::stoi(args[0]) : 10;
  node_num_for_each_shard = args.size() > 1 ? std::stoi(args[1]) : 10;
  edge_num_for_each_node = args.size() > 2 ? std::stoi(args[2]) : 10;
  rounds = args.size() > 3 ? std::stoi(args[3]) : 1;
  interval = args.size() > 4 ? std::stoi(args[4]) : 60;
}

#endif

std::vector<Node *> GraphShard::get_batch(int start, int end, int step) {
  if (start < 0) start = 0;
  std::vector<Node *> res;
  for (int pos = start; pos < std::min(end, (int)bucket.size()); pos += step) {
    res.push_back(bucket[pos]);
  }
  return res;
}

size_t GraphShard::get_size() { return bucket.size(); }

int32_t GraphTable::add_graph_node(std::vector<int64_t> &id_list,
                                   std::vector<bool> &is_weight_list) {
  size_t node_size = id_list.size();
  std::vector<std::vector<std::pair<int64_t, bool>>> batch(task_pool_size_);
  for (size_t i = 0; i < node_size; i++) {
    size_t shard_id = id_list[i] % shard_num;
    if (shard_id >= shard_end || shard_id < shard_start) {
      continue;
    }
    batch[get_thread_pool_index(id_list[i])].push_back(
        {id_list[i], i < is_weight_list.size() ? is_weight_list[i] : false});
  }
  std::vector<std::future<int>> tasks;
  for (size_t i = 0; i < batch.size(); ++i) {
    if (!batch[i].size()) continue;
    tasks.push_back(_shards_task_pool[i]->enqueue([&batch, i, this]() -> int {
      for (auto &p : batch[i]) {
        size_t index = p.first % this->shard_num - this->shard_start;
        this->shards[index]->add_graph_node(p.first)->build_edges(p.second);
      }
      return 0;
    }));
  }
  for (size_t i = 0; i < tasks.size(); i++) tasks[i].get();
  return 0;
}

int32_t GraphTable::remove_graph_node(std::vector<int64_t> &id_list) {
  size_t node_size = id_list.size();
  std::vector<std::vector<int64_t>> batch(task_pool_size_);
  for (size_t i = 0; i < node_size; i++) {
    size_t shard_id = id_list[i] % shard_num;
    if (shard_id >= shard_end || shard_id < shard_start) continue;
    batch[get_thread_pool_index(id_list[i])].push_back(id_list[i]);
  }
  std::vector<std::future<int>> tasks;
  for (size_t i = 0; i < batch.size(); ++i) {
    if (!batch[i].size()) continue;
    tasks.push_back(_shards_task_pool[i]->enqueue([&batch, i, this]() -> int {
      for (auto &p : batch[i]) {
        size_t index = p % this->shard_num - this->shard_start;
        this->shards[index]->delete_node(p);
      }
      return 0;
    }));
  }
  for (size_t i = 0; i < tasks.size(); i++) tasks[i].get();
  return 0;
}

void GraphShard::clear() {
  for (size_t i = 0; i < bucket.size(); i++) {
    delete bucket[i];
  }
  bucket.clear();
  node_location.clear();
}

GraphShard::~GraphShard() { clear(); }

void GraphShard::delete_node(int64_t id) {
  auto iter = node_location.find(id);
  if (iter == node_location.end()) return;
  int pos = iter->second;
  delete bucket[pos];
  if (pos != (int)bucket.size() - 1) {
    bucket[pos] = bucket.back();
    node_location[bucket.back()->get_id()] = pos;
  }
  node_location.erase(id);
  bucket.pop_back();
}
GraphNode *GraphShard::add_graph_node(int64_t id) {
  if (node_location.find(id) == node_location.end()) {
    node_location[id] = bucket.size();
    bucket.push_back(new GraphNode(id));
  }
  return (GraphNode *)bucket[node_location[id]];
}

GraphNode *GraphShard::add_graph_node(Node *node) {
  auto id = node->get_id();
  if (node_location.find(id) == node_location.end()) {
    node_location[id] = bucket.size();
    bucket.push_back(node);
  }
  return (GraphNode *)bucket[node_location[id]];
}
FeatureNode *GraphShard::add_feature_node(int64_t id) {
  if (node_location.find(id) == node_location.end()) {
    node_location[id] = bucket.size();
    bucket.push_back(new FeatureNode(id));
  }
  return (FeatureNode *)bucket[node_location[id]];
}

void GraphShard::add_neighbor(int64_t id, int64_t dst_id, float weight) {
  find_node(id)->add_edge(dst_id, weight);
}

Node *GraphShard::find_node(int64_t id) {
  auto iter = node_location.find(id);
  return iter == node_location.end() ? nullptr : bucket[iter->second];
}

GraphTable::~GraphTable() {
  for (auto p : shards) {
    delete p;
  }
  for (auto p : extra_shards) {
    delete p;
  }
  shards.clear();
  extra_shards.clear();
}

int32_t GraphTable::load_graph_split_config(const std::string &path) {
  VLOG(4) << "in server side load graph split config\n";
  std::ifstream file(path);
  std::string line;
  while (std::getline(file, line)) {
    auto values = paddle::string::split_string<std::string>(line, "\t");
    if (values.size() < 2) continue;
    size_t index = (size_t)std::stoi(values[0]);
    if (index != _shard_idx) continue;
    auto dst_id = std::stoull(values[1]);
    extra_nodes.insert(dst_id);
  }
  if (extra_nodes.size() != 0) use_duplicate_nodes = true;
  return 0;
}

int32_t GraphTable::load(const std::string &path, const std::string &param) {
  bool load_edge = (param[0] == 'e');
  bool load_node = (param[0] == 'n');
  if (load_edge) {
    bool reverse_edge = (param[1] == '<');
    return this->load_edges(path, reverse_edge);
  }
  if (load_node) {
    std::string node_type = param.substr(1);
    return this->load_nodes(path, node_type);
  }
  return 0;
}

int32_t GraphTable::get_nodes_ids_by_ranges(
    std::vector<std::pair<int, int>> ranges, std::vector<int64_t> &res) {
  int start = 0, end, index = 0, total_size = 0;
  res.clear();
  std::vector<std::future<std::vector<int64_t>>> tasks;
  for (size_t i = 0; i < shards.size() && index < (int)ranges.size(); i++) {
    end = total_size + shards[i]->get_size();
    start = total_size;
    while (start < end && index < (int)ranges.size()) {
      if (ranges[index].second <= start)
        index++;
      else if (ranges[index].first >= end) {
        break;
      } else {
        int first = std::max(ranges[index].first, start);
        int second = std::min(ranges[index].second, end);
        start = second;
        first -= total_size;
        second -= total_size;
        tasks.push_back(_shards_task_pool[i % task_pool_size_]->enqueue(
            [this, first, second, i]() -> std::vector<int64_t> {
              return shards[i]->get_ids_by_range(first, second);
            }));
      }
    }
    total_size += shards[i]->get_size();
  }
  for (size_t i = 0; i < tasks.size(); i++) {
    auto vec = tasks[i].get();
    for (auto &id : vec) {
      res.push_back(id);
      std::swap(res[rand() % res.size()], res[(int)res.size() - 1]);
    }
  }
  return 0;
}

int32_t GraphTable::load_nodes(const std::string &path, std::string node_type) {
  auto paths = paddle::string::split_string<std::string>(path, ";");
  int64_t count = 0;
  int64_t valid_count = 0;
  for (auto path : paths) {
    std::ifstream file(path);
    std::string line;
    while (std::getline(file, line)) {
      auto values = paddle::string::split_string<std::string>(line, "\t");
      if (values.size() < 2) continue;
      auto id = std::stoull(values[1]);

      size_t shard_id = id % shard_num;
      if (shard_id >= shard_end || shard_id < shard_start) {
        VLOG(4) << "will not load " << id << " from " << path
                << ", please check id distribution";
        continue;
      }

      if (count % 1000000 == 0) {
        VLOG(0) << count << " nodes are loaded from filepath";
        VLOG(0) << line;
      }
      count++;

      std::string nt = values[0];
      if (nt != node_type) {
        continue;
      }

      size_t index = shard_id - shard_start;

      auto node = shards[index]->add_feature_node(id);

      node->set_feature_size(feat_name.size());

      for (size_t slice = 2; slice < values.size(); slice++) {
        auto feat = this->parse_feature(values[slice]);
        if (feat.first >= 0) {
          node->set_feature(feat.first, feat.second);
        } else {
          VLOG(4) << "Node feature:  " << values[slice]
                  << " not in feature_map.";
        }
      }
      valid_count++;
    }
  }

  VLOG(0) << valid_count << "/" << count << " nodes in type " << node_type
          << " are loaded successfully in " << path;
  return 0;
}

int32_t GraphTable::load_edges(const std::string &path, bool reverse_edge) {
#ifdef PADDLE_WITH_HETERPS
  if (gpups_mode) pthread_rwlock_rdlock(rw_lock.get());
#endif
  auto paths = paddle::string::split_string<std::string>(path, ";");
  int64_t count = 0;
  std::string sample_type = "random";
  bool is_weighted = false;
  int valid_count = 0;
  int extra_alloc_index = 0;
  for (auto path : paths) {
    std::ifstream file(path);
    std::string line;
    while (std::getline(file, line)) {
      auto values = paddle::string::split_string<std::string>(line, "\t");
      count++;
      if (values.size() < 2) continue;
      auto src_id = std::stoull(values[0]);
      auto dst_id = std::stoull(values[1]);
      if (reverse_edge) {
        std::swap(src_id, dst_id);
      }
      float weight = 1;
      if (values.size() == 3) {
        weight = std::stof(values[2]);
        sample_type = "weighted";
        is_weighted = true;
      }

      size_t src_shard_id = src_id % shard_num;

      if (src_shard_id >= shard_end || src_shard_id < shard_start) {
        if (use_duplicate_nodes == false ||
            extra_nodes.find(src_id) == extra_nodes.end()) {
          VLOG(4) << "will not load " << src_id << " from " << path
                  << ", please check id distribution";
          continue;
        }
        int index;
        if (extra_nodes_to_thread_index.find(src_id) !=
            extra_nodes_to_thread_index.end()) {
          index = extra_nodes_to_thread_index[src_id];
        } else {
          index = extra_alloc_index++;
          extra_alloc_index %= task_pool_size_;
          extra_nodes_to_thread_index[src_id] = index;
        }
        extra_shards[index]->add_graph_node(src_id)->build_edges(is_weighted);
        extra_shards[index]->add_neighbor(src_id, dst_id, weight);
        valid_count++;
        continue;
      }
      if (count % 1000000 == 0) {
        VLOG(0) << count << " edges are loaded from filepath";
        VLOG(0) << line;
      }

      size_t index = src_shard_id - shard_start;
      shards[index]->add_graph_node(src_id)->build_edges(is_weighted);
      shards[index]->add_neighbor(src_id, dst_id, weight);
      valid_count++;
    }
  }
  VLOG(0) << valid_count << "/" << count << " edges are loaded successfully in "
          << path;

  std::vector<int> used(task_pool_size_, 0);
  // Build Sampler j

  for (auto &shard : shards) {
    auto bucket = shard->get_bucket();
    for (size_t i = 0; i < bucket.size(); i++) {
      bucket[i]->build_sampler(sample_type);
      used[get_thread_pool_index(bucket[i]->get_id())]++;
    }
  }
  /*-----------------------
  relocate the duplicate nodes to make them distributed evenly among threads.
*/
  if (!use_duplicate_nodes) {
#ifdef PADDLE_WITH_HETERPS
    if (gpups_mode) pthread_rwlock_unlock(rw_lock.get());
#endif

    return 0;
  }
  for (auto &shard : extra_shards) {
    auto bucket = shard->get_bucket();
    for (size_t i = 0; i < bucket.size(); i++) {
      bucket[i]->build_sampler(sample_type);
    }
  }
  int size = extra_nodes_to_thread_index.size();
  if (size == 0) return 0;
  std::vector<int> index;
  for (int i = 0; i < (int)used.size(); i++) index.push_back(i);
  sort(index.begin(), index.end(),
       [&](int &a, int &b) { return used[a] < used[b]; });

  std::vector<int> alloc(index.size(), 0), has_alloc(index.size(), 0);
  int t = 1, aim = 0, mod = 0;
  for (; t < (int)used.size(); t++) {
    if ((used[index[t]] - used[index[t - 1]]) * t >= size) {
      break;
    } else {
      size -= (used[index[t]] - used[index[t - 1]]) * t;
    }
  }
  aim = used[index[t - 1]] + size / t;
  mod = size % t;
  for (int x = t - 1; x >= 0; x--) {
    alloc[index[x]] = aim;
    if (t - x <= mod) alloc[index[x]]++;
    alloc[index[x]] -= used[index[x]];
  }
  std::vector<int64_t> vec[index.size()];
  for (auto p : extra_nodes_to_thread_index) {
    has_alloc[p.second]++;
    vec[p.second].push_back(p.first);
  }
  sort(index.begin(), index.end(), [&](int &a, int &b) {
    return has_alloc[a] - alloc[a] < has_alloc[b] - alloc[b];
  });
  int left = 0, right = (int)index.size() - 1;
  while (left < right) {
    if (has_alloc[index[right]] - alloc[index[right]] == 0) break;
    int x = std::min(alloc[index[left]] - has_alloc[index[left]],
                     has_alloc[index[right]] - alloc[index[right]]);
    has_alloc[index[left]] += x;
    has_alloc[index[right]] -= x;
    int64_t id;
    while (x--) {
      id = vec[index[right]].back();
      vec[index[right]].pop_back();
      extra_nodes_to_thread_index[id] = index[left];
      vec[index[left]].push_back(id);
    }
    if (has_alloc[index[right]] - alloc[index[right]] == 0) right--;
    if (alloc[index[left]] - has_alloc[index[left]] == 0) left++;
  }
  std::vector<GraphShard *> extra_shards_copy;
  for (int i = 0; i < task_pool_size_; ++i) {
    extra_shards_copy.push_back(new GraphShard());
  }
  for (auto &shard : extra_shards) {
    auto &bucket = shard->get_bucket();
    auto &node_location = shard->get_node_location();
    while (bucket.size()) {
      Node *temp = bucket.back();
      bucket.pop_back();
      node_location.erase(temp->get_id());
      extra_shards_copy[extra_nodes_to_thread_index[temp->get_id()]]
          ->add_graph_node(temp);
    }
  }
  for (int i = 0; i < task_pool_size_; ++i) {
    delete extra_shards[i];
    extra_shards[i] = extra_shards_copy[i];
  }
#ifdef PADDLE_WITH_HETERPS
  if (gpups_mode) pthread_rwlock_unlock(rw_lock.get());
#endif
  return 0;
}

Node *GraphTable::find_node(int64_t id) {
  size_t shard_id = id % shard_num;
  if (shard_id >= shard_end || shard_id < shard_start) {
    if (use_duplicate_nodes == false || extra_nodes_to_thread_index.size() == 0)
      return nullptr;
    auto iter = extra_nodes_to_thread_index.find(id);
    if (iter == extra_nodes_to_thread_index.end())
      return nullptr;
    else {
      return extra_shards[iter->second]->find_node(id);
    }
  }
  size_t index = shard_id - shard_start;
  Node *node = shards[index]->find_node(id);
  return node;
}
uint32_t GraphTable::get_thread_pool_index(int64_t node_id) {
  if (use_duplicate_nodes == false || extra_nodes_to_thread_index.size() == 0)
    return node_id % shard_num % shard_num_per_server % task_pool_size_;
  size_t src_shard_id = node_id % shard_num;
  if (src_shard_id >= shard_end || src_shard_id < shard_start) {
    auto iter = extra_nodes_to_thread_index.find(node_id);
    if (iter != extra_nodes_to_thread_index.end()) {
      return iter->second;
    }
  }
  return src_shard_id % shard_num_per_server % task_pool_size_;
}

uint32_t GraphTable::get_thread_pool_index_by_shard_index(int64_t shard_index) {
  return shard_index % shard_num_per_server % task_pool_size_;
}

int32_t GraphTable::clear_nodes() {
  std::vector<std::future<int>> tasks;
  for (size_t i = 0; i < shards.size(); i++) {
    tasks.push_back(
        _shards_task_pool[i % task_pool_size_]->enqueue([this, i]() -> int {
          this->shards[i]->clear();
          return 0;
        }));
  }
  for (size_t i = 0; i < extra_shards.size(); i++) {
    tasks.push_back(_shards_task_pool[i]->enqueue([this, i]() -> int {
      this->extra_shards[i]->clear();
      return 0;
    }));
  }
  for (size_t i = 0; i < tasks.size(); i++) tasks[i].get();
  return 0;
}

int32_t GraphTable::random_sample_nodes(int sample_size,
                                        std::unique_ptr<char[]> &buffer,
                                        int &actual_size) {
  int total_size = 0;
  for (int i = 0; i < (int)shards.size(); i++) {
    total_size += shards[i]->get_size();
  }
  if (sample_size > total_size) sample_size = total_size;
  int range_num = random_sample_nodes_ranges;
  if (range_num > sample_size) range_num = sample_size;
  if (sample_size == 0 || range_num == 0) return 0;
  std::vector<int> ranges_len, ranges_pos;
  int remain = sample_size, last_pos = -1, num;
  std::set<int> separator_set;
  for (int i = 0; i < range_num - 1; i++) {
    while (separator_set.find(num = rand() % (sample_size - 1)) !=
           separator_set.end())
      ;
    separator_set.insert(num);
  }
  for (auto p : separator_set) {
    ranges_len.push_back(p - last_pos);
    last_pos = p;
  }
  ranges_len.push_back(sample_size - 1 - last_pos);
  remain = total_size - sample_size + range_num;
  separator_set.clear();
  for (int i = 0; i < range_num; i++) {
    while (separator_set.find(num = rand() % remain) != separator_set.end())
      ;
    separator_set.insert(num);
  }
  int used = 0, index = 0;
  last_pos = -1;
  for (auto p : separator_set) {
    used += p - last_pos - 1;
    last_pos = p;
    ranges_pos.push_back(used);
    used += ranges_len[index++];
  }
  std::vector<std::pair<int, int>> first_half, second_half;
  int start_index = rand() % total_size;
  for (size_t i = 0; i < ranges_len.size() && i < ranges_pos.size(); i++) {
    if (ranges_pos[i] + ranges_len[i] - 1 + start_index < total_size)
      first_half.push_back({ranges_pos[i] + start_index,
                            ranges_pos[i] + ranges_len[i] + start_index});
    else if (ranges_pos[i] + start_index >= total_size) {
      second_half.push_back(
          {ranges_pos[i] + start_index - total_size,
           ranges_pos[i] + ranges_len[i] + start_index - total_size});
    } else {
      first_half.push_back({ranges_pos[i] + start_index, total_size});
      second_half.push_back(
          {0, ranges_pos[i] + ranges_len[i] + start_index - total_size});
    }
  }
  for (auto &pair : first_half) second_half.push_back(pair);
  std::vector<int64_t> res;
  get_nodes_ids_by_ranges(second_half, res);
  actual_size = res.size() * sizeof(int64_t);
  buffer.reset(new char[actual_size]);
  char *pointer = buffer.get();
  memcpy(pointer, res.data(), actual_size);
  return 0;
}
int32_t GraphTable::random_sample_neighbors(
    int64_t *node_ids, int sample_size,
    std::vector<std::shared_ptr<char>> &buffers, std::vector<int> &actual_sizes,
    bool need_weight) {
  size_t node_num = buffers.size();
  std::function<void(char *)> char_del = [](char *c) { delete[] c; };
  std::vector<std::future<int>> tasks;
  std::vector<std::vector<uint32_t>> seq_id(task_pool_size_);
  std::vector<std::vector<SampleKey>> id_list(task_pool_size_);
  size_t index;
  for (size_t idx = 0; idx < node_num; ++idx) {
    index = get_thread_pool_index(node_ids[idx]);
    seq_id[index].emplace_back(idx);
    id_list[index].emplace_back(node_ids[idx], sample_size, need_weight);
  }
  for (int i = 0; i < (int)seq_id.size(); i++) {
    if (seq_id[i].size() == 0) continue;
    tasks.push_back(_shards_task_pool[i]->enqueue([&, i, this]() -> int {
      int64_t node_id;
      std::vector<std::pair<SampleKey, SampleResult>> r;
      LRUResponse response = LRUResponse::blocked;
      if (use_cache) {
        response =
            scaled_lru->query(i, id_list[i].data(), id_list[i].size(), r);
      }
      int index = 0;
      uint32_t idx;
      std::vector<SampleResult> sample_res;
      std::vector<SampleKey> sample_keys;
      auto &rng = _shards_task_rng_pool[i];
      for (size_t k = 0; k < id_list[i].size(); k++) {
        if (index < (int)r.size() &&
            r[index].first.node_key == id_list[i][k].node_key) {
          idx = seq_id[i][k];
          actual_sizes[idx] = r[index].second.actual_size;
          buffers[idx] = r[index].second.buffer;
          index++;
        } else {
          node_id = id_list[i][k].node_key;
          Node *node = find_node(node_id);
          idx = seq_id[i][k];
          int &actual_size = actual_sizes[idx];
          if (node == nullptr) {
            actual_size = 0;
            continue;
          }
          std::shared_ptr<char> &buffer = buffers[idx];
          std::vector<int> res = node->sample_k(sample_size, rng);
          actual_size =
              res.size() * (need_weight ? (Node::id_size + Node::weight_size)
                                        : Node::id_size);
          int offset = 0;
          int64_t id;
          float weight;
          char *buffer_addr = new char[actual_size];
          if (response == LRUResponse::ok) {
            sample_keys.emplace_back(node_id, sample_size, need_weight);
            sample_res.emplace_back(actual_size, buffer_addr);
            buffer = sample_res.back().buffer;
          } else {
            buffer.reset(buffer_addr, char_del);
          }
          for (int &x : res) {
            id = node->get_neighbor_id(x);
            memcpy(buffer_addr + offset, &id, Node::id_size);
            offset += Node::id_size;
            if (need_weight) {
              weight = node->get_neighbor_weight(x);
              memcpy(buffer_addr + offset, &weight, Node::weight_size);
              offset += Node::weight_size;
            }
          }
        }
      }
      if (sample_res.size()) {
        scaled_lru->insert(i, sample_keys.data(), sample_res.data(),
                           sample_keys.size());
      }
      return 0;
    }));
  }
  for (auto &t : tasks) {
    t.get();
  }
  return 0;
}

int32_t GraphTable::get_node_feat(const std::vector<int64_t> &node_ids,
                                  const std::vector<std::string> &feature_names,
                                  std::vector<std::vector<std::string>> &res) {
  size_t node_num = node_ids.size();
  std::vector<std::future<int>> tasks;
  for (size_t idx = 0; idx < node_num; ++idx) {
    int64_t node_id = node_ids[idx];
    tasks.push_back(_shards_task_pool[get_thread_pool_index(node_id)]->enqueue(
        [&, idx, node_id]() -> int {
          Node *node = find_node(node_id);

          if (node == nullptr) {
            return 0;
          }
          for (int feat_idx = 0; feat_idx < (int)feature_names.size();
               ++feat_idx) {
            const std::string &feature_name = feature_names[feat_idx];
            if (feat_id_map.find(feature_name) != feat_id_map.end()) {
              // res[feat_idx][idx] =
              // node->get_feature(feat_id_map[feature_name]);
              auto feat = node->get_feature(feat_id_map[feature_name]);
              res[feat_idx][idx] = feat;
            }
          }
          return 0;
        }));
  }
  for (size_t idx = 0; idx < node_num; ++idx) {
    tasks[idx].get();
  }
  return 0;
}

int32_t GraphTable::set_node_feat(
    const std::vector<int64_t> &node_ids,
    const std::vector<std::string> &feature_names,
    const std::vector<std::vector<std::string>> &res) {
  size_t node_num = node_ids.size();
  std::vector<std::future<int>> tasks;
  for (size_t idx = 0; idx < node_num; ++idx) {
    int64_t node_id = node_ids[idx];
    tasks.push_back(_shards_task_pool[get_thread_pool_index(node_id)]->enqueue(
        [&, idx, node_id]() -> int {
          size_t index = node_id % this->shard_num - this->shard_start;
          auto node = shards[index]->add_feature_node(node_id);
          node->set_feature_size(this->feat_name.size());
          for (int feat_idx = 0; feat_idx < (int)feature_names.size();
               ++feat_idx) {
            const std::string &feature_name = feature_names[feat_idx];
            if (feat_id_map.find(feature_name) != feat_id_map.end()) {
              node->set_feature(feat_id_map[feature_name], res[feat_idx][idx]);
            }
          }
          return 0;
        }));
  }
  for (size_t idx = 0; idx < node_num; ++idx) {
    tasks[idx].get();
  }
  return 0;
}

std::pair<int32_t, std::string> GraphTable::parse_feature(
    std::string feat_str) {
  // Return (feat_id, btyes) if name are in this->feat_name, else return (-1,
  // "")
  auto fields = paddle::string::split_string<std::string>(feat_str, " ");
  if (this->feat_id_map.count(fields[0])) {
    int32_t id = this->feat_id_map[fields[0]];
    std::string dtype = this->feat_dtype[id];
    std::vector<std::string> values(fields.begin() + 1, fields.end());
    if (dtype == "feasign") {
      return std::make_pair<int32_t, std::string>(
          int32_t(id), paddle::string::join_strings(values, ' '));
    } else if (dtype == "string") {
      return std::make_pair<int32_t, std::string>(
          int32_t(id), paddle::string::join_strings(values, ' '));
    } else if (dtype == "float32") {
      return std::make_pair<int32_t, std::string>(
          int32_t(id), FeatureNode::parse_value_to_bytes<float>(values));
    } else if (dtype == "float64") {
      return std::make_pair<int32_t, std::string>(
          int32_t(id), FeatureNode::parse_value_to_bytes<double>(values));
    } else if (dtype == "int32") {
      return std::make_pair<int32_t, std::string>(
          int32_t(id), FeatureNode::parse_value_to_bytes<int32_t>(values));
    } else if (dtype == "int64") {
      return std::make_pair<int32_t, std::string>(
          int32_t(id), FeatureNode::parse_value_to_bytes<int64_t>(values));
    }
  }
  return std::make_pair<int32_t, std::string>(-1, "");
}

int32_t GraphTable::pull_graph_list(int start, int total_size,
                                    std::unique_ptr<char[]> &buffer,
                                    int &actual_size, bool need_feature,
                                    int step) {
  if (start < 0) start = 0;
  int size = 0, cur_size;
  std::vector<std::future<std::vector<Node *>>> tasks;
  for (size_t i = 0; i < shards.size() && total_size > 0; i++) {
    cur_size = shards[i]->get_size();
    if (size + cur_size <= start) {
      size += cur_size;
      continue;
    }
    int count = std::min(1 + (size + cur_size - start - 1) / step, total_size);
    int end = start + (count - 1) * step + 1;
    tasks.push_back(_shards_task_pool[i % task_pool_size_]->enqueue(
        [this, i, start, end, step, size]() -> std::vector<Node *> {
          return this->shards[i]->get_batch(start - size, end - size, step);
        }));
    start += count * step;
    total_size -= count;
    size += cur_size;
  }
  for (size_t i = 0; i < tasks.size(); ++i) {
    tasks[i].wait();
  }
  size = 0;
  std::vector<std::vector<Node *>> res;
  for (size_t i = 0; i < tasks.size(); i++) {
    res.push_back(tasks[i].get());
    for (size_t j = 0; j < res.back().size(); j++) {
      size += res.back()[j]->get_size(need_feature);
    }
  }
  char *buffer_addr = new char[size];
  buffer.reset(buffer_addr);
  int index = 0;
  for (size_t i = 0; i < res.size(); i++) {
    for (size_t j = 0; j < res[i].size(); j++) {
      res[i][j]->to_buffer(buffer_addr + index, need_feature);
      index += res[i][j]->get_size(need_feature);
    }
  }
  actual_size = size;
  return 0;
}

int32_t GraphTable::get_server_index_by_id(int64_t id) {
  return id % shard_num / shard_num_per_server;
}
int32_t GraphTable::initialize(const TableParameter &config,
                               const FsClientParameter &fs_config) {
  LOG(INFO) << "in graphTable initialize";
  _config = config;
  if (initialize_accessor() != 0) {
    LOG(WARNING) << "Table accessor initialize failed";
    return -1;
  }

  if (_afs_client.initialize(fs_config) != 0) {
    LOG(WARNING) << "Table fs_client initialize failed";
    // return -1;
  }
  auto graph = config.graph_parameter();
  shard_num = _config.shard_num();
  LOG(INFO) << "in graphTable initialize over";
  return initialize(graph);
}
int32_t GraphTable::initialize(const GraphParameter &graph) {
#ifdef PADDLE_WITH_HETERPS
  if (graph.gpups_mode()) {
    gpups_mode = true;
    auto *sampler =
        CREATE_PSCORE_CLASS(GraphSampler, graph.gpups_graph_sample_class());
    auto slices =
        string::split_string<std::string>(graph.gpups_graph_sample_args(), ",");
    std::cout << "slices" << std::endl;
    for (auto x : slices) std::cout << x << std::endl;
    sampler->init(graph.gpu_num(), this, slices);
    graph_sampler.reset(sampler);
  }
#endif
  if (shard_num == 0) {
    server_num = 1;
    _shard_idx = 0;
    shard_num = graph.shard_num();
  }
  task_pool_size_ = graph.task_pool_size();
  use_cache = graph.use_cache();
  if (use_cache) {
    cache_size_limit = graph.cache_size_limit();
    cache_ttl = graph.cache_ttl();
    make_neighbor_sample_cache((size_t)cache_size_limit, (size_t)cache_ttl);
  }
  _shards_task_pool.resize(task_pool_size_);
  for (size_t i = 0; i < _shards_task_pool.size(); ++i) {
    _shards_task_pool[i].reset(new ::ThreadPool(1));
    _shards_task_rng_pool.push_back(paddle::framework::GetCPURandomEngine(0));
  }
  auto graph_feature = graph.graph_feature();
  // this->table_name = common.table_name();
  // this->table_type = common.name();
  this->table_name = graph.table_name();
  this->table_type = graph.table_type();
  VLOG(0) << " init graph table type " << this->table_type << " table name "
          << this->table_name;
  // int feat_conf_size = static_cast<int>(common.attributes().size());
  int feat_conf_size = static_cast<int>(graph_feature.name().size());
  for (int i = 0; i < feat_conf_size; i++) {
    // auto &f_name = common.attributes()[i];
    // auto &f_shape = common.dims()[i];
    // auto &f_dtype = common.params()[i];
    auto &f_name = graph_feature.name()[i];
    auto &f_shape = graph_feature.shape()[i];
    auto &f_dtype = graph_feature.dtype()[i];
    this->feat_name.push_back(f_name);
    this->feat_shape.push_back(f_shape);
    this->feat_dtype.push_back(f_dtype);
    this->feat_id_map[f_name] = i;
    VLOG(0) << "init graph table feat conf name:" << f_name
            << " shape:" << f_shape << " dtype:" << f_dtype;
  }
  VLOG(0) << "in init graph table shard num = " << shard_num << " shard_idx"
          << _shard_idx;
  shard_num_per_server = sparse_local_shard_num(shard_num, server_num);
  shard_start = _shard_idx * shard_num_per_server;
  shard_end = shard_start + shard_num_per_server;
  VLOG(0) << "in init graph table shard idx = " << _shard_idx << " shard_start "
          << shard_start << " shard_end " << shard_end;
  for (size_t i = 0; i < shard_num_per_server; i++) {
    shards.push_back(new GraphShard());
  }
  use_duplicate_nodes = false;
  for (int i = 0; i < task_pool_size_; i++) {
    extra_shards.push_back(new GraphShard());
  }

  return 0;
}

}  // namespace distributed
};  // namespace paddle

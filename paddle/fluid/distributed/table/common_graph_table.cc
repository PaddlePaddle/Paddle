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

#include "paddle/fluid/distributed/table/common_graph_table.h"
#include <time.h>
#include <algorithm>
#include <set>
#include <sstream>
#include "paddle/fluid/distributed/common/utils.h"
#include "paddle/fluid/string/printf.h"
#include "paddle/fluid/string/string_helper.h"
namespace paddle {
namespace distributed {

int GraphShard::bucket_low_bound = 11;

std::vector<GraphNode *> GraphShard::get_batch(int start, int total_size) {
  if (start < 0) start = 0;
  int size = 0, cur_size;
  std::vector<GraphNode *> res;
  if (total_size <= 0) return res;
  for (int i = 0; i < bucket_size; i++) {
    cur_size = bucket[i].size();
    if (size + cur_size <= start) {
      size += cur_size;
      continue;
    }
    int read = 0;
    std::list<GraphNode *>::iterator iter = bucket[i].begin();
    while (size + read < start) {
      iter++;
      read++;
    }
    read = 0;
    while (iter != bucket[i].end() && read < total_size) {
      res.push_back(*iter);
      iter++;
      read++;
    }
    if (read == total_size) break;
    size += cur_size;
    start = size;
    total_size -= read;
  }
  return res;
}

size_t GraphShard::get_size() {
  size_t res = 0;
  for (int i = 0; i < bucket_size; i++) {
    res += bucket[i].size();
  }
  return res;
}

std::list<GraphNode *>::iterator GraphShard::add_node(uint64_t id, std::string feature) {
  if (node_location.find(id) != node_location.end())
    return node_location.find(id)->second;

  int index = id % shard_num % bucket_size;
  GraphNode *node = new GraphNode(id, feature);

  std::list<GraphNode *>::iterator iter =
      bucket[index].insert(bucket[index].end(), node);

  node_location[id] = iter;
  return iter;
}

void GraphShard::add_neighboor(uint64_t id, GraphEdge *edge) {
  (*add_node(id, std::string("")))->add_edge(edge);
}

GraphNode *GraphShard::find_node(uint64_t id) {
  if (node_location.find(id) == node_location.end()) return NULL;
  return *(node_location[id]);
}

int32_t GraphTable::load(const std::string &path, const std::string &param) {
  auto cmd = paddle::string::split_string<std::string>(param, "|");
  std::set<std::string> cmd_set(cmd.begin(), cmd.end()); 
  bool reverse_edge = cmd_set.count(std::string("reverse"));
  bool load_edge = cmd_set.count(std::string("edge"));
  if(load_edge) {
      return this -> load_edges(path, reverse_edge); 
  }
  else {
      return this -> load_nodes(path);
  }
}

int32_t GraphTable::load_nodes(const std::string &path) {
  auto paths = paddle::string::split_string<std::string>(path, ";");
  for (auto path : paths) {
    std::ifstream file(path);
    std::string line;
    while (std::getline(file, line)) {
      auto values = paddle::string::split_string<std::string>(line, "\t");
      if (values.size() < 2) continue;
      auto id = std::stoull(values[1]);


      size_t shard_id = id % shard_num;
      if (shard_id >= shard_end || shard_id < shard_start) {
        VLOG(0) << "will not load " << id << " from " << path
                << ", please check id distribution";
        continue;

      }

      std::string node_type = values[0];
      std::vector<std::string > feature;
      feature.push_back(node_type);
      for(size_t slice = 2; slice < values.size(); slice ++) {
        feature.push_back(values[slice]);
      }
      auto feat = paddle::string::join_strings(feature, '\t');
      size_t index = shard_id - shard_start;
      shards[index].add_node(id, feat);

    }
  }
  return 0;
}


int32_t GraphTable::load_edges(const std::string &path, bool reverse_edge) {

  auto paths = paddle::string::split_string<std::string>(path, ";");
  int count = 0;

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
      float weight = 0;
      if (values.size() == 3) {
        weight = std::stof(values[2]);
      }

      size_t src_shard_id = src_id % shard_num;

      if (src_shard_id >= shard_end || src_shard_id < shard_start) {
        VLOG(0) << "will not load " << src_id << " from " << path
                << ", please check id distribution";
        continue;
      }

      size_t index = src_shard_id - shard_start;
      GraphEdge *edge = new GraphEdge(dst_id, weight);
      shards[index].add_neighboor(src_id, edge);
    }
  }
  VLOG(0) << "Load Finished Total Edge Count " << count;

  // Build Sampler j
  
  for (auto &shard : shards) {
    auto bucket = shard.get_bucket();
    for (int i = 0; i < bucket.size(); i++) {
      std::list<GraphNode *>::iterator iter = bucket[i].begin();
      while (iter != bucket[i].end()) {
        auto node = *iter;
        node->build_sampler();
        iter++;
      }
    }
  }
  return 0;
}

GraphNode *GraphTable::find_node(uint64_t id) {
  size_t shard_id = id % shard_num;
  if (shard_id >= shard_end || shard_id < shard_start) {
    return NULL;
  }
  size_t index = shard_id - shard_start;
  GraphNode *node = shards[index].find_node(id);
  return node;
}
uint32_t GraphTable::get_thread_pool_index(uint64_t node_id) {
  return node_id % shard_num_per_table % task_pool_size_;
}
int GraphTable::random_sample(uint64_t* node_ids, int sample_size,
                                  std::vector<char*>& buffers, std::vector<int> &actual_sizes) {
  size_t node_num = buffers.size();
  std::vector<std::future<int>> tasks;
  for (size_t idx = 0; idx < node_num; ++idx){
    uint64_t node_id = node_ids[idx];
    char* & buffer = buffers[idx];
    int& actual_size = actual_sizes[idx];
    tasks.push_back(_shards_task_pool[get_thread_pool_index(node_id)]
      ->enqueue([&]() -> int {
        GraphNode *node = find_node(node_id);
        if (node == NULL) {
          actual_size = 0;
          return 0;
        }
        std::vector<GraphEdge *> res = node->sample_k(sample_size);
        std::vector<GraphNode> node_list;
        actual_size =
            res.size() * (GraphNode::id_size + GraphNode::weight_size);
        buffer = new char[actual_size];
        int offset = 0;
        uint64_t id;
        float weight;
        for (auto &x : res) {
          id = x->get_id();
          weight = x->get_weight();
          memcpy(buffer + offset, &id, GraphNode::id_size);
          offset += GraphNode::id_size;
          memcpy(buffer + offset, &weight, GraphNode::weight_size);
          offset += GraphNode::weight_size;
        }
        return 0;
      }));
  }
  for (size_t idx = 0; idx < node_num; ++idx){
    tasks[idx].get();
  }
  return 0;
}
int32_t GraphTable::pull_graph_list(int start, int total_size, char *&buffer,
                                    int &actual_size) {
  if (start < 0) start = 0;
  int size = 0, cur_size;
  if (total_size <= 0) {
    actual_size = 0;
    return 0;
  }
  std::vector<std::future<std::vector<GraphNode *>>> tasks;
  for (size_t i = 0; i < shards.size(); i++) {
    cur_size = shards[i].get_size();
    if (size + cur_size <= start) {
      size += cur_size;
      continue;
    }
    if (size + cur_size - start >= total_size) {
      tasks.push_back(_shards_task_pool[i % task_pool_size_]->enqueue(
          [this, i, start, size, total_size]() -> std::vector<GraphNode *> {
            return this->shards[i].get_batch(start - size, total_size);
          }));
      break;
    } else {
      tasks.push_back(_shards_task_pool[i % task_pool_size_]->enqueue(
          [this, i, start, size, total_size,
           cur_size]() -> std::vector<GraphNode *> {
            return this->shards[i].get_batch(start - size,
                                             size + cur_size - start);
          }));
      total_size -= size + cur_size - start;
      size += cur_size;
      start = size;
    }
  }
  for (size_t i = 0; i < tasks.size(); ++i) {
    tasks[i].wait();
  }
  size = 0;
  std::vector<std::vector<GraphNode *>> res;
  for (size_t i = 0; i < tasks.size(); i++) {
    res.push_back(tasks[i].get());
    for (size_t j = 0; j < res.back().size(); j++) {
      size += res.back()[j]->get_size();
    }
  }
  buffer = new char[size];
  int index = 0;
  for (size_t i = 0; i < res.size(); i++) {
    for (size_t j = 0; j < res[i].size(); j++) {
      res[i][j]->to_buffer(buffer + index);
      index += res[i][j]->get_size();
    }
  }
  actual_size = size;
  return 0;
}
int32_t GraphTable::initialize() {
  _shards_task_pool.resize(task_pool_size_);
  for (size_t i = 0; i < _shards_task_pool.size(); ++i) {
    _shards_task_pool[i].reset(new ::ThreadPool(1));
  }
  server_num = _shard_num;
  // VLOG(0) << "in init graph table server num = " << server_num;
  /*
  _shard_num is actually server number here
  when a server initialize its tables, it sets tables' _shard_num to server_num,
  and _shard_idx to server
  rank
  */
  shard_num = _config.shard_num();
  VLOG(0) << "in init graph table shard num = " << shard_num << " shard_idx"
          << _shard_idx;
  shard_num_per_table = sparse_local_shard_num(shard_num, server_num);
  shard_start = _shard_idx * shard_num_per_table;
  shard_end = shard_start + shard_num_per_table;
  VLOG(0) << "in init graph table shard idx = " << _shard_idx << " shard_start "
          << shard_start << " shard_end " << shard_end;
  // shards.resize(shard_num_per_table);
  shards = std::vector<GraphShard>(shard_num_per_table, GraphShard(shard_num));
  return 0;
}
}
};


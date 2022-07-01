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
int32_t GraphTable::Load_to_ssd(const std::string &path,
                                const std::string &param) {
  bool load_edge = (param[0] == 'e');
  bool load_node = (param[0] == 'n');
  if (load_edge) {
    bool reverse_edge = (param[1] == '<');
    std::string edge_type = param.substr(2);
    return this->load_edges_to_ssd(path, reverse_edge, edge_type);
  }
  if (load_node) {
    std::string node_type = param.substr(1);
    return this->load_nodes(path, node_type);
  }
  return 0;
}

paddle::framework::GpuPsCommGraph GraphTable::make_gpu_ps_graph(
    int idx, std::vector<int64_t> ids) {
  std::vector<std::vector<int64_t>> bags(task_pool_size_);
  for (auto x : ids) {
    int location = x % shard_num % task_pool_size_;
    bags[location].push_back(x);
  }
  std::vector<std::future<int>> tasks;
  std::vector<int64_t> edge_array[task_pool_size_];
  std::vector<paddle::framework::GpuPsGraphNode> node_array[task_pool_size_];
  for (size_t i = 0; i < bags.size(); i++) {
    if (bags[i].size() > 0) {
      tasks.push_back(_shards_task_pool[i]->enqueue([&, i, this]() -> int {
        paddle::framework::GpuPsGraphNode x;
        for (size_t j = 0; j < bags[i].size(); j++) {
          Node *v = find_node(0, idx, bags[i][j]);
          x.node_id = bags[i][j];
          if (v == NULL) {
            x.neighbor_size = 0;
            x.neighbor_offset = 0;
            node_array[i].push_back(x);
          } else {
            x.neighbor_size = v->get_neighbor_size();
            x.neighbor_offset = edge_array[i].size();
            node_array[i].push_back(x);
            for (size_t k = 0; k < x.neighbor_size; k++) {
              edge_array[i].push_back(v->get_neighbor_id(k));
            }
          }
        }
        return 0;
      }));
    }
  }
  for (int i = 0; i < static_cast<int>(tasks.size()); i++) tasks[i].get();
  paddle::framework::GpuPsCommGraph res;
  int64_t tot_len = 0;
  for (int i = 0; i < task_pool_size_; i++) {
    tot_len += edge_array[i].size();
  }
  // res.neighbor_size = tot_len;
  // res.node_size = ids.size();
  // res.neighbor_list = new int64_t[tot_len];
  // res.node_list = new paddle::framework::GpuPsGraphNode[ids.size()];
  res.init_on_cpu(tot_len, ids.size());
  int64_t offset = 0, ind = 0;
  for (int i = 0; i < task_pool_size_; i++) {
    for (int j = 0; j < static_cast<int>(node_array[i].size()); j++) {
      res.node_list[ind] = node_array[i][j];
      res.node_list[ind++].neighbor_offset += offset;
    }
    for (size_t j = 0; j < edge_array[i].size(); j++) {
      res.neighbor_list[offset + j] = edge_array[i][j];
    }
    offset += edge_array[i].size();
  }
  return res;
}

int32_t GraphTable::add_node_to_ssd(
    int type_id, int idx, int64_t src_id, char *data, int len) {
  if (_db != NULL) {
    char ch[sizeof(int) * 2 + sizeof(int64_t)];
    memcpy(ch, &type_id, sizeof(int));
    memcpy(ch + sizeof(int), &idx, sizeof(int));
    memcpy(ch + sizeof(int) * 2, &src_id, sizeof(int64_t));
    std::string str;
    if (_db->get(src_id % shard_num % task_pool_size_,
                 ch,
                 sizeof(int) * 2 + sizeof(int64_t),
                 str) == 0) {
      int64_t *stored_data = (reinterpret_cast<const int64_t *>(str.c_str()));
      int n = str.size() / sizeof(int64_t);
      char *new_data = new char[n * sizeof(int64_t) + len];
      memcpy(new_data, stored_data, n * sizeof(int64_t));
      memcpy(new_data + n * sizeof(int64_t), data, len);
      _db->put(src_id % shard_num % task_pool_size_,
               ch,
               sizeof(int) * 2 + sizeof(int64_t),
               reinterpret_cast<char *>(new_data),
               n * sizeof(int64_t) + len);
      delete[] new_data;
    } else {
      _db->put(src_id % shard_num % task_pool_size_,
               ch,
               sizeof(int) * 2 + sizeof(int64_t),
               reinterpret_cast<char *>(data),
               len);
    }
    // _db->flush(src_id % shard_num % task_pool_size_);
    // std::string x;
    // if (_db->get(src_id % shard_num % task_pool_size_, ch, sizeof(int64_t) +
    // 2 * sizeof(int), x) ==0){
    // VLOG(0)<<"put result";
    // for(int i = 0;i < x.size();i+=8){
    //   VLOG(0)<<"get an id "<<*((int64_t *)(x.c_str() + i));
    // }
    //}
    // if(src_id == 429){
    //   str = "";
    //   _db->get(src_id % shard_num % task_pool_size_, ch,
    //            sizeof(int) * 2 + sizeof(int64_t), str);
    //   int64_t *stored_data = ((int64_t *)str.c_str());
    //   int n = str.size() / sizeof(int64_t);
    //   VLOG(0)<<"429 has "<<n<<"neighbors";
    //   for(int i =0;i< n;i++){
    //     VLOG(0)<<"get an id "<<*((int64_t *)(str.c_str() +
    //     i*sizeof(int64_t)));
    //   }
    // }
  }
  return 0;
}
char *GraphTable::random_sample_neighbor_from_ssd(
    int idx,
    int64_t id,
    int sample_size,
    const std::shared_ptr<std::mt19937_64> rng,
    int &actual_size) {
  if (_db == NULL) {
    actual_size = 0;
    return NULL;
  }
  std::string str;
  VLOG(2) << "sample ssd for key " << id;
  char ch[sizeof(int) * 2 + sizeof(int64_t)];
  memset(ch, 0, sizeof(int));
  memcpy(ch + sizeof(int), &idx, sizeof(int));
  memcpy(ch + sizeof(int) * 2, &id, sizeof(int64_t));
  if (_db->get(id % shard_num % task_pool_size_,
               ch,
               sizeof(int) * 2 + sizeof(int64_t),
               str) == 0) {
    int64_t *data = (reinterpret_cast<const int64_t *>(str.c_str()));
    int n = str.size() / sizeof(int64_t);
    std::unordered_map<int, int> m;
    // std::vector<int64_t> res;
    int sm_size = std::min(n, sample_size);
    actual_size = sm_size * Node::id_size;
    char *buff = new char[actual_size];
    for (int i = 0; i < sm_size; i++) {
      std::uniform_int_distribution<int> distrib(0, n - i - 1);
      int t = distrib(*rng);
      // int t = rand() % (n-i);
      int pos = 0;
      auto iter = m.find(t);
      if (iter != m.end()) {
        pos = iter->second;
      } else {
        pos = t;
      }
      auto iter2 = m.find(n - i - 1);

      int key2 = iter2 == m.end() ? n - i - 1 : iter2->second;
      m[t] = key2;
      m.erase(n - i - 1);
      memcpy(buff + i * Node::id_size, &data[pos], Node::id_size);
      // res.push_back(data[pos]);
    }
    for (int i = 0; i < actual_size; i += 8) {
      VLOG(2) << "sampled an neighbor "
              << *(reinterpret_cast<int64_t *>(&buff[i]));
    }
    return buff;
  }
  actual_size = 0;
  return NULL;
}

int64_t GraphTable::load_graph_to_memory_from_ssd(int idx,
                                                  std::vector<int64_t> &ids) {
  std::vector<std::vector<int64_t>> bags(task_pool_size_);
  for (auto x : ids) {
    int location = x % shard_num % task_pool_size_;
    bags[location].push_back(x);
  }
  std::vector<std::future<int>> tasks;
  std::vector<int64_t> count(task_pool_size_, 0);
  for (size_t i = 0; i < bags.size(); i++) {
    if (bags[i].size() > 0) {
      tasks.push_back(_shards_task_pool[i]->enqueue([&, i, idx, this]() -> int {
        char ch[sizeof(int) * 2 + sizeof(int64_t)];
        memset(ch, 0, sizeof(int));
        memcpy(ch + sizeof(int), &idx, sizeof(int));
        for (size_t k = 0; k < bags[i].size(); k++) {
          auto v = bags[i][k];
          memcpy(ch + sizeof(int) * 2, &v, sizeof(int64_t));
          std::string str;
          if (_db->get(i, ch, sizeof(int) * 2 + sizeof(int64_t), str) == 0) {
            count[i] += (int64_t)str.size();
            for (int j = 0; j < str.size(); j += sizeof(int64_t)) {
              int64_t id =
                  *(reinterpret_cast<const int64_t *>(str.c_str() + j));
              add_comm_edge(idx, v, id);
            }
          }
        }
        return 0;
      }));
    }
  }

  for (int i = 0; i < static_cast<int>(tasks.size()); i++) tasks[i].get();
  int64_t tot = 0;
  for (auto x : count) tot += x;
  return tot;
}

void GraphTable::make_partitions(int idx, int64_t byte_size, int device_len) {
  VLOG(2) << "start to make graph partitions , byte_size = " << byte_size
          << " total memory cost = " << total_memory_cost;
  if (total_memory_cost == 0) {
    VLOG(0) << "no edges are detected,make partitions exits";
    return;
  }
  auto &weight_map = node_weight[0][idx];
  const double a = 2.0, y = 1.25, weight_param = 1.0;
  int64_t gb_size_by_discount = byte_size * 0.8 * device_len;
  if (gb_size_by_discount <= 0) gb_size_by_discount = 1;
  int part_len = total_memory_cost / gb_size_by_discount;
  if (part_len == 0) part_len = 1;

  VLOG(2) << "part_len = " << part_len
          << " byte size = " << gb_size_by_discount;
  partitions[idx].clear();
  partitions[idx].resize(part_len);
  std::vector<double> weight_cost(part_len, 0);
  std::vector<int64_t> memory_remaining(part_len, gb_size_by_discount);
  std::vector<double> score(part_len, 0);
  std::unordered_map<int64_t, int> id_map;
  std::vector<rocksdb::Iterator *> iters;
  for (int i = 0; i < task_pool_size_; i++) {
    iters.push_back(_db->get_iterator(i));
    iters[i]->SeekToFirst();
  }
  int next = 0;
  while (iters.size()) {
    if (next >= iters.size()) {
      next = 0;
    }
    if (!iters[next]->Valid()) {
      iters.erase(iters.begin() + next);
      continue;
    }
    std::string key = iters[next]->key().ToString();
    int type_idx = *(reinterpret_cast<const int *>(key.c_str()));
    int temp_idx = *(reinterpret_cast<const int *>(key.c_str() + sizeof(int)));
    if (type_idx != 0 || temp_idx != idx) {
      iters[next]->Next();
      next++;
      continue;
    }
    std::string value = iters[next]->value().ToString();
    std::int64_t i_key =
        *(reinterpret_cast<const int64_t *>(key.c_str() + sizeof(int) * 2));
    for (int i = 0; i < part_len; i++) {
      if (memory_remaining[i] < (int64_t)value.size()) {
        score[i] = -100000.0;
      } else {
        score[i] = 0;
      }
    }
    for (int j = 0; j < value.size(); j += sizeof(int64_t)) {
      int64_t v = *(reinterpret_cast<const int64_t *>(value.c_str() + j));
      int index = -1;
      if (id_map.find(v) != id_map.end()) {
        index = id_map[v];
        score[index]++;
      }
    }
    double base, weight_base = 0;
    double w = 0;
    bool has_weight = false;
    if (weight_map.find(i_key) != weight_map.end()) {
      w = weight_map[i_key];
      has_weight = true;
    }
    int index = 0;
    for (int i = 0; i < part_len; i++) {
      base = gb_size_by_discount - memory_remaining[i] + value.size();
      if (has_weight) {
        weight_base = weight_cost[i] + w * weight_param;
      } else {
        weight_base = 0;
      }
      score[i] -= a * y * std::pow(1.0 * base, y - 1) + weight_base;
      if (score[i] > score[index]) index = i;
      VLOG(2) << "score" << i << " = " << score[i] << " memory left "
              << memory_remaining[i];
    }
    id_map[i_key] = index;
    partitions[idx][index].push_back(i_key);
    memory_remaining[index] -= (int64_t)value.size();
    if (has_weight) weight_cost[index] += w;
    iters[next]->Next();
    next++;
  }
  for (int i = 0; i < part_len; i++) {
    if (partitions[idx][i].size() == 0) {
      partitions[idx].erase(partitions[idx].begin() + i);
      i--;
      part_len--;
      continue;
    }
    VLOG(2) << " partition " << i << " size = " << partitions[idx][i].size();
    for (auto x : partitions[idx][i]) {
      VLOG(2) << "find a id " << x;
    }
  }
  next_partition = 0;
}

void GraphTable::export_partition_files(int idx, std::string file_path) {
  int part_len = partitions[idx].size();
  if (part_len == 0) return;
  if (file_path == "") file_path = ".";
  if (file_path[static_cast<int>(file_path.size()) - 1] != '/') {
    file_path += "/";
  }
  std::vector<std::future<int>> tasks;
  for (int i = 0; i < part_len; i++) {
    tasks.push_back(_shards_task_pool[i % task_pool_size_]->enqueue(
        [&, i, idx, this]() -> int {
          std::string output_path =
              file_path + "partition_" + std::to_string(i);

          std::ofstream ofs(output_path);
          if (ofs.fail()) {
            VLOG(0) << "creating " << output_path << " failed";
            return 0;
          }
          for (auto x : partitions[idx][i]) {
            auto str = std::to_string(x);
            ofs.write(str.c_str(), str.size());
            ofs.write("\n", 1);
          }
          ofs.close();
          return 0;
        }));
  }

  for (int i = 0; i < static_cast<int>(tasks.size()); i++) tasks[i].get();
}
void GraphTable::clear_graph(int idx) {
  for (auto p : edge_shards[idx]) {
    delete p;
  }

  edge_shards[idx].clear();
  for (size_t i = 0; i < shard_num_per_server; i++) {
    edge_shards[idx].push_back(new GraphShard());
  }
}
int32_t GraphTable::load_next_partition(int idx) {
  if (next_partition >= partitions[idx].size()) {
    VLOG(0) << "partition iteration is done";
    return -1;
  }
  clear_graph(idx);
  load_graph_to_memory_from_ssd(idx, partitions[idx][next_partition]);
  next_partition++;
  return 0;
}
int32_t GraphTable::load_edges_to_ssd(const std::string &path,
                                      bool reverse_edge,
                                      const std::string &edge_type) {
  int idx = 0;
  if (edge_type == "") {
    VLOG(0) << "edge_type not specified, loading edges to " << id_to_edge[0]
            << " part";
  } else {
    if (edge_to_id.find(edge_type) == edge_to_id.end()) {
      VLOG(0) << "edge_type " << edge_type
              << " is not defined, nothing will be loaded";
      return 0;
    }
    idx = edge_to_id[edge_type];
  }
  total_memory_cost = 0;
  auto paths = paddle::string::split_string<std::string>(path, ";");
  int64_t count = 0;
  std::string sample_type = "random";
  bool is_weighted = false;
  int valid_count = 0;
  for (auto path : paths) {
    std::ifstream file(path);
    std::string line;
    while (std::getline(file, line)) {
      VLOG(0) << "get a line from file " << line;
      auto values = paddle::string::split_string<std::string>(line, "\t");
      count++;
      if (values.size() < 2) continue;
      auto src_id = std::stoll(values[0]);
      auto dist_ids = paddle::string::split_string<std::string>(values[1], ";");
      std::vector<int64_t> dist_data;
      for (auto x : dist_ids) {
        dist_data.push_back(std::stoll(x));
        total_memory_cost += sizeof(int64_t);
      }
      add_node_to_ssd(0,
                      idx,
                      src_id,
                      reinterpret_cast<char *>(dist_data.data()),
                      static_cast<int>(dist_data.size() * sizeof(int64_t)));
    }
  }
  VLOG(0) << "total memory cost = " << total_memory_cost << " bytes";
  return 0;
}

int32_t GraphTable::dump_edges_to_ssd(int idx) {
  VLOG(2) << "calling dump edges to ssd";
  const int64_t fixed_size = 10000;
  // std::vector<int64_t> edge_array[task_pool_size_];
  std::vector<std::unordered_map<int64_t, int>> count(task_pool_size_);
  std::vector<std::future<int64_t>> tasks;
  auto &shards = edge_shards[idx];
  for (size_t i = 0; i < shards.size(); ++i) {
    tasks.push_back(_shards_task_pool[i % task_pool_size_]->enqueue(
        [&, i, this]() -> int64_t {
          int64_t cost = 0;
          std::vector<Node *> &v = shards[i]->get_bucket();
          size_t ind = i % this->task_pool_size_;
          for (size_t j = 0; j < v.size(); j++) {
            std::vector<int64_t> s;
            for (int k = 0; k < v[j]->get_neighbor_size(); k++) {
              s.push_back(v[j]->get_neighbor_id(k));
            }
            cost += v[j]->get_neighbor_size() * sizeof(int64_t);
            add_node_to_ssd(0,
                            idx,
                            v[j]->get_id(),
                            reinterpret_cast<char *>(s.data()),
                            s.size() * sizeof(int64_t));
          }
          return cost;
        }));
  }
  for (size_t i = 0; i < tasks.size(); i++) total_memory_cost += tasks[i].get();
  return 0;
}
int32_t GraphTable::make_complementary_graph(int idx, int64_t byte_size) {
  VLOG(0) << "make_complementary_graph";
  const int64_t fixed_size = byte_size / 8;
  // std::vector<int64_t> edge_array[task_pool_size_];
  std::vector<std::unordered_map<int64_t, int>> count(task_pool_size_);
  std::vector<std::future<int>> tasks;
  auto &shards = edge_shards[idx];
  for (size_t i = 0; i < shards.size(); ++i) {
    tasks.push_back(
        _shards_task_pool[i % task_pool_size_]->enqueue([&, i, this]() -> int {
          std::vector<Node *> &v = shards[i]->get_bucket();
          size_t ind = i % this->task_pool_size_;
          for (size_t j = 0; j < v.size(); j++) {
            // size_t location = v[j]->get_id();
            for (int k = 0; k < v[j]->get_neighbor_size(); k++) {
              count[ind][v[j]->get_neighbor_id(k)]++;
            }
          }
          return 0;
        }));
  }
  for (size_t i = 0; i < tasks.size(); i++) tasks[i].get();
  std::unordered_map<int64_t, int> final_count;
  std::map<int, std::vector<int64_t>> count_to_id;
  std::vector<int64_t> buffer;
  clear_graph(idx);

  for (int i = 0; i < task_pool_size_; i++) {
    for (auto &p : count[i]) {
      final_count[p.first] = final_count[p.first] + p.second;
    }
    count[i].clear();
  }
  for (auto &p : final_count) {
    count_to_id[p.second].push_back(p.first);
    VLOG(2) << p.first << " appear " << p.second << " times";
  }
  auto iter = count_to_id.rbegin();
  while (iter != count_to_id.rend() && byte_size > 0) {
    for (auto x : iter->second) {
      buffer.push_back(x);
      if (buffer.size() >= fixed_size) {
        int64_t res = load_graph_to_memory_from_ssd(idx, buffer);
        buffer.clear();
        byte_size -= res;
      }
      if (byte_size <= 0) break;
    }
    iter++;
  }
  if (byte_size > 0 && buffer.size() > 0) {
    int64_t res = load_graph_to_memory_from_ssd(idx, buffer);
    byte_size -= res;
  }
  std::string sample_type = "random";
  for (auto &shard : edge_shards[idx]) {
    auto bucket = shard->get_bucket();
    for (size_t i = 0; i < bucket.size(); i++) {
      bucket[i]->build_sampler(sample_type);
    }
  }
  return 0;
}
#endif

/*
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
    //__sync_fetch_and_add
    std::function<int(int, int64_t)> bfs = [&, this](int i, int id) -> int {
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
    VLOG(0)<<"bfs returning";
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
*/
std::vector<Node *> GraphShard::get_batch(int start, int end, int step) {
  if (start < 0) start = 0;
  std::vector<Node *> res;
  for (int pos = start; pos < std::min(end, static_cast<int>(bucket.size()));
       pos += step) {
    res.push_back(bucket[pos]);
  }
  return res;
}

size_t GraphShard::get_size() { return bucket.size(); }

int32_t GraphTable::add_comm_edge(int idx, int64_t src_id, int64_t dst_id) {
  size_t src_shard_id = src_id % shard_num;

  if (src_shard_id >= shard_end || src_shard_id < shard_start) {
    return -1;
  }
  size_t index = src_shard_id - shard_start;
  edge_shards[idx][index]->add_graph_node(src_id)->build_edges(false);
  edge_shards[idx][index]->add_neighbor(src_id, dst_id, 1.0);
  return 0;
}
int32_t GraphTable::add_graph_node(int idx,
                                   std::vector<int64_t> &id_list,
                                   std::vector<bool> &is_weight_list) {
  auto &shards = edge_shards[idx];
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
    tasks.push_back(
        _shards_task_pool[i]->enqueue([&shards, &batch, i, this]() -> int {
          for (auto &p : batch[i]) {
            size_t index = p.first % this->shard_num - this->shard_start;
            shards[index]->add_graph_node(p.first)->build_edges(p.second);
          }
          return 0;
        }));
  }
  for (size_t i = 0; i < tasks.size(); i++) tasks[i].get();
  return 0;
}

int32_t GraphTable::remove_graph_node(int idx, std::vector<int64_t> &id_list) {
  size_t node_size = id_list.size();
  std::vector<std::vector<int64_t>> batch(task_pool_size_);
  for (size_t i = 0; i < node_size; i++) {
    size_t shard_id = id_list[i] % shard_num;
    if (shard_id >= shard_end || shard_id < shard_start) continue;
    batch[get_thread_pool_index(id_list[i])].push_back(id_list[i]);
  }
  auto &shards = edge_shards[idx];
  std::vector<std::future<int>> tasks;
  for (size_t i = 0; i < batch.size(); ++i) {
    if (!batch[i].size()) continue;
    tasks.push_back(
        _shards_task_pool[i]->enqueue([&shards, &batch, i, this]() -> int {
          for (auto &p : batch[i]) {
            size_t index = p % this->shard_num - this->shard_start;
            shards[index]->delete_node(p);
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
  if (pos != static_cast<int>(bucket.size()) - 1) {
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
  return reinterpret_cast<GraphNode *>(bucket[node_location[id]]);
}

GraphNode *GraphShard::add_graph_node(Node *node) {
  auto id = node->get_id();
  if (node_location.find(id) == node_location.end()) {
    node_location[id] = bucket.size();
    bucket.push_back(node);
  }
  return reinterpret_cast<GraphNode *>(bucket[node_location[id]]);
}
FeatureNode *GraphShard::add_feature_node(int64_t id) {
  if (node_location.find(id) == node_location.end()) {
    node_location[id] = bucket.size();
    bucket.push_back(new FeatureNode(id));
  }
  return reinterpret_cast<FeatureNode *>(bucket[node_location[id]]);
}

void GraphShard::add_neighbor(int64_t id, int64_t dst_id, float weight) {
  find_node(id)->add_edge(dst_id, weight);
}

Node *GraphShard::find_node(int64_t id) {
  auto iter = node_location.find(id);
  return iter == node_location.end() ? nullptr : bucket[iter->second];
}

GraphTable::~GraphTable() {
  for (int i = 0; i < static_cast<int>(edge_shards.size()); i++) {
    for (auto p : edge_shards[i]) {
      delete p;
    }
    edge_shards[i].clear();
  }

  for (int i = 0; i < static_cast<int>(feature_shards.size()); i++) {
    for (auto p : feature_shards[i]) {
      delete p;
    }
    feature_shards[i].clear();
  }
}

int32_t GraphTable::Load(const std::string &path, const std::string &param) {
  bool load_edge = (param[0] == 'e');
  bool load_node = (param[0] == 'n');
  if (load_edge) {
    bool reverse_edge = (param[1] == '<');
    std::string edge_type = param.substr(2);
    return this->load_edges(path, reverse_edge, edge_type);
  }
  if (load_node) {
    std::string node_type = param.substr(1);
    return this->load_nodes(path, node_type);
  }
  return 0;
}

int32_t GraphTable::get_nodes_ids_by_ranges(
    int type_id,
    int idx,
    std::vector<std::pair<int, int>> ranges,
    std::vector<int64_t> &res) {
  int start = 0, end, index = 0, total_size = 0;
  res.clear();
  auto &shards = type_id == 0 ? edge_shards[idx] : feature_shards[idx];
  std::vector<std::future<std::vector<int64_t>>> tasks;
  for (size_t i = 0;
       i < shards.size() && index < static_cast<int>(ranges.size());
       i++) {
    end = total_size + shards[i]->get_size();
    start = total_size;
    while (start < end && index < static_cast<int>(ranges.size())) {
      if (ranges[index].second <= start) {
        index++;
      } else if (ranges[index].first >= end) {
        break;
      } else {
        int first = std::max(ranges[index].first, start);
        int second = std::min(ranges[index].second, end);
        start = second;
        first -= total_size;
        second -= total_size;
        tasks.push_back(_shards_task_pool[i % task_pool_size_]->enqueue(
            [&shards, this, first, second, i]() -> std::vector<int64_t> {
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
      std::swap(res[rand() % res.size()],
                res[static_cast<int>(res.size()) - 1]);  // NOLINT
    }
  }
  return 0;
}

int32_t GraphTable::load_nodes(const std::string &path, std::string node_type) {
  auto paths = paddle::string::split_string<std::string>(path, ";");
  int64_t count = 0;
  int64_t valid_count = 0;
  int idx = 0;
  if (node_type == "") {
    VLOG(0) << "node_type not specified, loading edges to " << id_to_feature[0]
            << " part";
  } else {
    if (feature_to_id.find(node_type) == feature_to_id.end()) {
      VLOG(0) << "node_type " << node_type
              << " is not defined, nothing will be loaded";
      return 0;
    }
    idx = feature_to_id[node_type];
  }
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

      // auto node = shards[index]->add_feature_node(id);
      auto node = feature_shards[idx][index]->add_feature_node(id);
      node->set_feature_size(feat_name[idx].size());

      for (size_t slice = 2; slice < values.size(); slice++) {
        auto feat = this->parse_feature(idx, values[slice]);
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

int32_t GraphTable::build_sampler(int idx, std::string sample_type) {
  for (auto &shard : edge_shards[idx]) {
    auto bucket = shard->get_bucket();
    for (size_t i = 0; i < bucket.size(); i++) {
      bucket[i]->build_sampler(sample_type);
    }
  }
  return 0;
}
int32_t GraphTable::load_edges(const std::string &path,
                               bool reverse_edge,
                               const std::string &edge_type) {
#ifdef PADDLE_WITH_HETERPS
  // if (gpups_mode) pthread_rwlock_rdlock(rw_lock.get());
  if (search_level == 2) total_memory_cost = 0;
  const int64_t fixed_load_edges = 1000000;
#endif
  int idx = 0;
  if (edge_type == "") {
    VLOG(0) << "edge_type not specified, loading edges to " << id_to_edge[0]
            << " part";
  } else {
    if (edge_to_id.find(edge_type) == edge_to_id.end()) {
      VLOG(0) << "edge_type " << edge_type
              << " is not defined, nothing will be loaded";
      return 0;
    }
    idx = edge_to_id[edge_type];
  }

  auto paths = paddle::string::split_string<std::string>(path, ";");
  int64_t count = 0;
  std::string sample_type = "random";
  bool is_weighted = false;
  int valid_count = 0;
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
        VLOG(4) << "will not load " << src_id << " from " << path
                << ", please check id distribution";
        continue;
      }

      if (count % 1000000 == 0) {
        VLOG(0) << count << " edges are loaded from filepath";
        VLOG(0) << line;
      }

      size_t index = src_shard_id - shard_start;
      edge_shards[idx][index]->add_graph_node(src_id)->build_edges(is_weighted);
      edge_shards[idx][index]->add_neighbor(src_id, dst_id, weight);
      valid_count++;
#ifdef PADDLE_WITH_HETERPS
      // if (gpups_mode) pthread_rwlock_rdlock(rw_lock.get());
      if (count > fixed_load_edges && search_level == 2) {
        dump_edges_to_ssd(idx);
        VLOG(0) << "dumping edges to ssd, edge count is reset to 0";
        clear_graph(idx);
        count = 0;
      }
#endif
    }
  }
  VLOG(0) << valid_count << "/" << count << " edges are loaded successfully in "
          << path;

// Build Sampler j
#ifdef PADDLE_WITH_HETERPS
  // if (gpups_mode) pthread_rwlock_rdlock(rw_lock.get());
  if (search_level == 2) {
    if (count > 0) {
      dump_edges_to_ssd(idx);
      VLOG(0) << "dumping edges to ssd, edge count is reset to 0";
      clear_graph(idx);
      count = 0;
    }
    return 0;
  }
#endif
  for (auto &shard : edge_shards[idx]) {
    auto bucket = shard->get_bucket();
    for (size_t i = 0; i < bucket.size(); i++) {
      bucket[i]->build_sampler(sample_type);
    }
  }

  return 0;
}

Node *GraphTable::find_node(int type_id, int idx, int64_t id) {
  size_t shard_id = id % shard_num;
  if (shard_id >= shard_end || shard_id < shard_start) {
    return nullptr;
  }
  size_t index = shard_id - shard_start;
  auto &search_shards = type_id == 0 ? edge_shards[idx] : feature_shards[idx];
  Node *node = search_shards[index]->find_node(id);
  return node;
}
uint32_t GraphTable::get_thread_pool_index(int64_t node_id) {
  return node_id % shard_num % shard_num_per_server % task_pool_size_;
}

uint32_t GraphTable::get_thread_pool_index_by_shard_index(int64_t shard_index) {
  return shard_index % shard_num_per_server % task_pool_size_;
}

int32_t GraphTable::clear_nodes(int type_id, int idx) {
  auto &search_shards = type_id == 0 ? edge_shards[idx] : feature_shards[idx];
  for (size_t i = 0; i < search_shards.size(); i++) {
    search_shards[i]->clear();
  }
  return 0;
}

int32_t GraphTable::random_sample_nodes(int type_id,
                                        int idx,
                                        int sample_size,
                                        std::unique_ptr<char[]> &buffer,
                                        int &actual_size) {
  int total_size = 0;
  auto &shards = type_id == 0 ? edge_shards[idx] : feature_shards[idx];
  for (int i = 0; i < static_cast<int>(shards.size()); i++) {
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
    while (separator_set.find(num = rand() % (sample_size - 1)) !=  // NOLINT
           separator_set.end()) {
    }
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
    while (separator_set.find(num = rand() % remain) !=  // NOLINT
           separator_set.end()) {
    }
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
  int start_index = rand() % total_size;  // NOLINT
  for (size_t i = 0; i < ranges_len.size() && i < ranges_pos.size(); i++) {
    if (ranges_pos[i] + ranges_len[i] - 1 + start_index < total_size) {
      first_half.push_back({ranges_pos[i] + start_index,
                            ranges_pos[i] + ranges_len[i] + start_index});
    } else if (ranges_pos[i] + start_index >= total_size) {
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
  get_nodes_ids_by_ranges(type_id, idx, second_half, res);
  actual_size = res.size() * sizeof(int64_t);
  buffer.reset(new char[actual_size]);
  char *pointer = buffer.get();
  memcpy(pointer, res.data(), actual_size);
  return 0;
}
int32_t GraphTable::random_sample_neighbors(
    int idx,
    int64_t *node_ids,
    int sample_size,
    std::vector<std::shared_ptr<char>> &buffers,
    std::vector<int> &actual_sizes,
    bool need_weight) {
  size_t node_num = buffers.size();
  std::function<void(char *)> char_del = [](char *c) { delete[] c; };
  std::vector<std::future<int>> tasks;
  std::vector<std::vector<uint32_t>> seq_id(task_pool_size_);
  std::vector<std::vector<SampleKey>> id_list(task_pool_size_);
  size_t index;
  for (size_t idy = 0; idy < node_num; ++idy) {
    index = get_thread_pool_index(node_ids[idy]);
    seq_id[index].emplace_back(idy);
    id_list[index].emplace_back(idx, node_ids[idy], sample_size, need_weight);
  }

  for (int i = 0; i < static_cast<int>(seq_id.size()); i++) {
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
      std::vector<SampleResult> sample_res;
      std::vector<SampleKey> sample_keys;
      auto &rng = _shards_task_rng_pool[i];
      for (size_t k = 0; k < id_list[i].size(); k++) {
        if (index < static_cast<int>(r.size()) &&
            r[index].first.node_key == id_list[i][k].node_key) {
          int idy = seq_id[i][k];
          actual_sizes[idy] = r[index].second.actual_size;
          buffers[idy] = r[index].second.buffer;
          index++;
        } else {
          node_id = id_list[i][k].node_key;
          Node *node = find_node(0, idx, node_id);
          int idy = seq_id[i][k];
          int &actual_size = actual_sizes[idy];
          if (node == nullptr) {
#ifdef PADDLE_WITH_HETERPS
            if (search_level == 2) {
              VLOG(2) << "enter sample from ssd for node_id " << node_id;
              char *buffer_addr = random_sample_neighbor_from_ssd(
                  idx, node_id, sample_size, rng, actual_size);
              if (actual_size != 0) {
                std::shared_ptr<char> &buffer = buffers[idy];
                buffer.reset(buffer_addr, char_del);
              }
              VLOG(2) << "actual sampled size from ssd = " << actual_sizes[idy];
              continue;
            }
#endif
            actual_size = 0;
            continue;
          }
          std::shared_ptr<char> &buffer = buffers[idy];
          std::vector<int> res = node->sample_k(sample_size, rng);
          actual_size =
              res.size() * (need_weight ? (Node::id_size + Node::weight_size)
                                        : Node::id_size);
          int offset = 0;
          int64_t id;
          float weight;
          char *buffer_addr = new char[actual_size];
          if (response == LRUResponse::ok) {
            sample_keys.emplace_back(idx, node_id, sample_size, need_weight);
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
        scaled_lru->insert(
            i, sample_keys.data(), sample_res.data(), sample_keys.size());
      }
      return 0;
    }));
  }
  for (auto &t : tasks) {
    t.get();
  }
  return 0;
}

int32_t GraphTable::get_node_feat(int idx,
                                  const std::vector<int64_t> &node_ids,
                                  const std::vector<std::string> &feature_names,
                                  std::vector<std::vector<std::string>> &res) {
  size_t node_num = node_ids.size();
  std::vector<std::future<int>> tasks;
  for (size_t idy = 0; idy < node_num; ++idy) {
    int64_t node_id = node_ids[idy];
    tasks.push_back(_shards_task_pool[get_thread_pool_index(node_id)]->enqueue(
        [&, idx, idy, node_id]() -> int {
          Node *node = find_node(1, idx, node_id);

          if (node == nullptr) {
            return 0;
          }
          for (int feat_idx = 0;
               feat_idx < static_cast<int>(feature_names.size());
               ++feat_idx) {
            const std::string &feature_name = feature_names[feat_idx];
            if (feat_id_map[idx].find(feature_name) != feat_id_map[idx].end()) {
              // res[feat_idx][idx] =
              // node->get_feature(feat_id_map[feature_name]);
              auto feat = node->get_feature(feat_id_map[idx][feature_name]);
              res[feat_idx][idy] = feat;
            }
          }
          return 0;
        }));
  }
  for (size_t idy = 0; idy < node_num; ++idy) {
    tasks[idy].get();
  }
  return 0;
}

int32_t GraphTable::set_node_feat(
    int idx,
    const std::vector<int64_t> &node_ids,
    const std::vector<std::string> &feature_names,
    const std::vector<std::vector<std::string>> &res) {
  size_t node_num = node_ids.size();
  std::vector<std::future<int>> tasks;
  for (size_t idy = 0; idy < node_num; ++idy) {
    int64_t node_id = node_ids[idy];
    tasks.push_back(_shards_task_pool[get_thread_pool_index(node_id)]->enqueue(
        [&, idx, idy, node_id]() -> int {
          size_t index = node_id % this->shard_num - this->shard_start;
          auto node = feature_shards[idx][index]->add_feature_node(node_id);
          node->set_feature_size(this->feat_name[idx].size());
          for (int feat_idx = 0;
               feat_idx < static_cast<int>(feature_names.size());
               ++feat_idx) {
            const std::string &feature_name = feature_names[feat_idx];
            if (feat_id_map[idx].find(feature_name) != feat_id_map[idx].end()) {
              node->set_feature(feat_id_map[idx][feature_name],
                                res[feat_idx][idy]);
            }
          }
          return 0;
        }));
  }
  for (size_t idy = 0; idy < node_num; ++idy) {
    tasks[idy].get();
  }
  return 0;
}

std::pair<int32_t, std::string> GraphTable::parse_feature(
    int idx, std::string feat_str) {
  // Return (feat_id, btyes) if name are in this->feat_name, else return (-1,
  // "")
  auto fields = paddle::string::split_string<std::string>(feat_str, " ");
  if (feat_id_map[idx].count(fields[0])) {
    // if (this->feat_id_map.count(fields[0])) {
    int32_t id = this->feat_id_map[idx][fields[0]];
    std::string dtype = this->feat_dtype[idx][id];
    std::vector<std::string> values(fields.begin() + 1, fields.end());
    if (dtype == "feasign") {
      return std::make_pair(int32_t(id),
                            paddle::string::join_strings(values, ' '));
    } else if (dtype == "string") {
      return std::make_pair(int32_t(id),
                            paddle::string::join_strings(values, ' '));
    } else if (dtype == "float32") {
      return std::make_pair(int32_t(id),
                            FeatureNode::parse_value_to_bytes<float>(values));
    } else if (dtype == "float64") {
      return std::make_pair(int32_t(id),
                            FeatureNode::parse_value_to_bytes<double>(values));
    } else if (dtype == "int32") {
      return std::make_pair(int32_t(id),
                            FeatureNode::parse_value_to_bytes<int32_t>(values));
    } else if (dtype == "int64") {
      return std::make_pair(int32_t(id),
                            FeatureNode::parse_value_to_bytes<int64_t>(values));
    }
  }
  return std::make_pair(-1, "");
}

std::vector<std::vector<int64_t>> GraphTable::get_all_id(int type_id,
                                                         int idx,
                                                         int slice_num) {
  std::vector<std::vector<int64_t>> res(slice_num);
  auto &search_shards = type_id == 0 ? edge_shards[idx] : feature_shards[idx];
  std::vector<std::future<std::vector<int64_t>>> tasks;
  for (size_t i = 0; i < search_shards.size(); i++) {
    tasks.push_back(_shards_task_pool[i % task_pool_size_]->enqueue(
        [&search_shards, i]() -> std::vector<int64_t> {
          return search_shards[i]->get_all_id();
        }));
  }
  for (size_t i = 0; i < tasks.size(); ++i) {
    tasks[i].wait();
  }
  for (size_t i = 0; i < tasks.size(); i++) {
    auto ids = tasks[i].get();
    for (auto &id : ids) res[(uint64_t)(id) % slice_num].push_back(id);
  }
  return res;
}
int32_t GraphTable::pull_graph_list(int type_id,
                                    int idx,
                                    int start,
                                    int total_size,
                                    std::unique_ptr<char[]> &buffer,
                                    int &actual_size,
                                    bool need_feature,
                                    int step) {
  if (start < 0) start = 0;
  int size = 0, cur_size;
  auto &search_shards = type_id == 0 ? edge_shards[idx] : feature_shards[idx];
  std::vector<std::future<std::vector<Node *>>> tasks;
  for (size_t i = 0; i < search_shards.size() && total_size > 0; i++) {
    cur_size = search_shards[i]->get_size();
    if (size + cur_size <= start) {
      size += cur_size;
      continue;
    }
    int count = std::min(1 + (size + cur_size - start - 1) / step, total_size);
    int end = start + (count - 1) * step + 1;
    tasks.push_back(_shards_task_pool[i % task_pool_size_]->enqueue(
        [&search_shards, this, i, start, end, step, size]()
            -> std::vector<Node *> {
          return search_shards[i]->get_batch(start - size, end - size, step);
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
int32_t GraphTable::Initialize(const TableParameter &config,
                               const FsClientParameter &fs_config) {
  LOG(INFO) << "in graphTable initialize";
  _config = config;
  if (InitializeAccessor() != 0) {
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
  return Initialize(graph);
}

void GraphTable::load_node_weight(int type_id, int idx, std::string path) {
  auto paths = paddle::string::split_string<std::string>(path, ";");
  int64_t count = 0;
  auto &weight_map = node_weight[type_id][idx];
  for (auto path : paths) {
    std::ifstream file(path);
    std::string line;
    while (std::getline(file, line)) {
      auto values = paddle::string::split_string<std::string>(line, "\t");
      count++;
      if (values.size() < 2) continue;
      auto src_id = std::stoull(values[0]);
      double weight = std::stod(values[1]);
      weight_map[src_id] = weight;
    }
  }
}
int32_t GraphTable::Initialize(const GraphParameter &graph) {
  task_pool_size_ = graph.task_pool_size();

#ifdef PADDLE_WITH_HETERPS
  _db = NULL;
  search_level = graph.search_level();
  if (search_level >= 2) {
    _db = paddle::distributed::RocksDBHandler::GetInstance();
    _db->initialize("./temp_gpups_db", task_pool_size_);
  }
// gpups_mode = true;
// auto *sampler =
//     CREATE_PSCORE_CLASS(GraphSampler, graph.gpups_graph_sample_class());
// auto slices =
//     string::split_string<std::string>(graph.gpups_graph_sample_args(), ",");
// std::cout << "slices" << std::endl;
// for (auto x : slices) std::cout << x << std::endl;
// sampler->init(graph.gpu_num(), this, slices);
// graph_sampler.reset(sampler);
#endif
  if (shard_num == 0) {
    server_num = 1;
    _shard_idx = 0;
    shard_num = graph.shard_num();
  }
  use_cache = graph.use_cache();
  if (use_cache) {
    cache_size_limit = graph.cache_size_limit();
    cache_ttl = graph.cache_ttl();
    make_neighbor_sample_cache(static_cast<size_t>(cache_size_limit),
                               static_cast<size_t>(cache_ttl));
  }
  _shards_task_pool.resize(task_pool_size_);
  for (size_t i = 0; i < _shards_task_pool.size(); ++i) {
    _shards_task_pool[i].reset(new ::ThreadPool(1));
    _shards_task_rng_pool.push_back(paddle::framework::GetCPURandomEngine(0));
  }
  auto graph_feature = graph.graph_feature();
  auto node_types = graph.node_types();
  auto edge_types = graph.edge_types();
  VLOG(0) << "got " << edge_types.size() << "edge types in total";
  feat_id_map.resize(node_types.size());
  for (int k = 0; k < edge_types.size(); k++) {
    VLOG(0) << "in initialize: get a edge_type " << edge_types[k];
    edge_to_id[edge_types[k]] = k;
    id_to_edge.push_back(edge_types[k]);
  }
  feat_name.resize(node_types.size());
  feat_shape.resize(node_types.size());
  feat_dtype.resize(node_types.size());
  VLOG(0) << "got " << node_types.size() << "node types in total";
  for (int k = 0; k < node_types.size(); k++) {
    feature_to_id[node_types[k]] = k;
    auto node_type = node_types[k];
    auto feature = graph_feature[k];
    id_to_feature.push_back(node_type);
    int feat_conf_size = static_cast<int>(feature.name().size());

    for (int i = 0; i < feat_conf_size; i++) {
      // auto &f_name = common.attributes()[i];
      // auto &f_shape = common.dims()[i];
      // auto &f_dtype = common.params()[i];
      auto &f_name = feature.name()[i];
      auto &f_shape = feature.shape()[i];
      auto &f_dtype = feature.dtype()[i];
      feat_name[k].push_back(f_name);
      feat_shape[k].push_back(f_shape);
      feat_dtype[k].push_back(f_dtype);
      feat_id_map[k][f_name] = i;
      VLOG(0) << "init graph table feat conf name:" << f_name
              << " shape:" << f_shape << " dtype:" << f_dtype;
    }
  }
  // this->table_name = common.table_name();
  // this->table_type = common.name();
  this->table_name = graph.table_name();
  this->table_type = graph.table_type();
  VLOG(0) << " init graph table type " << this->table_type << " table name "
          << this->table_name;
  // int feat_conf_size = static_cast<int>(common.attributes().size());
  // int feat_conf_size = static_cast<int>(graph_feature.name().size());
  VLOG(0) << "in init graph table shard num = " << shard_num << " shard_idx"
          << _shard_idx;
  shard_num_per_server = sparse_local_shard_num(shard_num, server_num);
  shard_start = _shard_idx * shard_num_per_server;
  shard_end = shard_start + shard_num_per_server;
  VLOG(0) << "in init graph table shard idx = " << _shard_idx << " shard_start "
          << shard_start << " shard_end " << shard_end;
  edge_shards.resize(id_to_edge.size());
  node_weight.resize(2);
  node_weight[0].resize(id_to_edge.size());
#ifdef PADDLE_WITH_HETERPS
  partitions.resize(id_to_edge.size());
#endif
  for (int k = 0; k < static_cast<int>(edge_shards.size()); k++) {
    for (size_t i = 0; i < shard_num_per_server; i++) {
      edge_shards[k].push_back(new GraphShard());
    }
  }
  node_weight[1].resize(id_to_feature.size());
  feature_shards.resize(id_to_feature.size());
  for (int k = 0; k < static_cast<int>(feature_shards.size()); k++) {
    for (size_t i = 0; i < shard_num_per_server; i++) {
      feature_shards[k].push_back(new GraphShard());
    }
  }

  return 0;
}

}  // namespace distributed
};  // namespace paddle

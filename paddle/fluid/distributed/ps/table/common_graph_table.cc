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

#include "gflags/gflags.h"
#include "paddle/fluid/distributed/common/utils.h"
#include "paddle/fluid/distributed/ps/table/graph/graph_node.h"
#include "paddle/fluid/framework/fleet/fleet_wrapper.h"
#include "paddle/fluid/framework/fleet/heter_ps/graph_gpu_wrapper.h"
#include "paddle/fluid/framework/io/fs.h"
#include "paddle/fluid/platform/timer.h"
#include "paddle/fluid/string/printf.h"
#include "paddle/fluid/string/string_helper.h"
#include "paddle/phi/core/generator.h"

DECLARE_bool(graph_load_in_parallel);
DECLARE_bool(graph_get_neighbor_id);
DECLARE_int32(gpugraph_storage_mode);
DECLARE_uint64(gpugraph_slot_feasign_max_num);
DECLARE_bool(graph_metapath_split_opt);

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

paddle::framework::GpuPsCommGraphFea GraphTable::make_gpu_ps_graph_fea(
    int gpu_id, std::vector<uint64_t> &node_ids, int slot_num) {
  size_t shard_num = 64;
  std::vector<std::vector<uint64_t>> bags(shard_num);
  std::vector<uint64_t> feature_array[shard_num];
  std::vector<uint8_t> slot_id_array[shard_num];
  std::vector<uint64_t> node_id_array[shard_num];
  std::vector<paddle::framework::GpuPsFeaInfo> node_fea_info_array[shard_num];
  for (size_t i = 0; i < shard_num; i++) {
    auto predsize = node_ids.size() / shard_num;
    bags[i].reserve(predsize * 1.2);
    feature_array[i].reserve(predsize * 1.2 * slot_num);
    slot_id_array[i].reserve(predsize * 1.2 * slot_num);
    node_id_array[i].reserve(predsize * 1.2);
    node_fea_info_array[i].reserve(predsize * 1.2);
  }

  for (auto x : node_ids) {
    int location = x % shard_num;
    bags[location].push_back(x);
  }

  std::vector<std::future<int>> tasks;
  if (slot_feature_num_map_.size() == 0) {
    slot_feature_num_map_.resize(slot_num);
    for (int k = 0; k < slot_num; ++k) {
      slot_feature_num_map_[k] = 0;
    }
  }

  for (size_t i = 0; i < bags.size(); i++) {
    if (bags[i].size() > 0) {
      tasks.push_back(_cpu_worker_pool[gpu_id]->enqueue([&, i, this]() -> int {
        uint64_t node_id;
        paddle::framework::GpuPsFeaInfo x;
        std::vector<uint64_t> feature_ids;
        for (size_t j = 0; j < bags[i].size(); j++) {
          Node *v = find_node(GraphTableType::FEATURE_TABLE, bags[i][j]);
          node_id = bags[i][j];
          if (v == NULL) {
            x.feature_size = 0;
            x.feature_offset = 0;
            node_fea_info_array[i].push_back(x);
          } else {
            // x <- v
            x.feature_offset = feature_array[i].size();
            int total_feature_size = 0;
            for (int k = 0; k < slot_num; ++k) {
              auto feature_ids_size =
                  v->get_feature_ids(k, feature_array[i], slot_id_array[i]);
              if (slot_feature_num_map_[k] < feature_ids_size) {
                slot_feature_num_map_[k] = feature_ids_size;
              }
              total_feature_size += feature_ids_size;
            }
            x.feature_size = total_feature_size;
            node_fea_info_array[i].push_back(x);
          }
          node_id_array[i].push_back(node_id);
        }
        return 0;
      }));
    }
  }
  for (size_t i = 0; i < tasks.size(); i++) tasks[i].get();

  if (FLAGS_v > 0) {
    std::stringstream ss;
    for (int k = 0; k < slot_num; ++k) {
      ss << slot_feature_num_map_[k] << " ";
    }
    VLOG(1) << "slot_feature_num_map: " << ss.str();
  }

  tasks.clear();

  paddle::framework::GpuPsCommGraphFea res;
  uint64_t tot_len = 0;
  for (size_t i = 0; i < shard_num; i++) {
    tot_len += feature_array[i].size();
  }
  VLOG(1) << "Loaded feature table on cpu, feature_list_size[" << tot_len
          << "] node_ids_size[" << node_ids.size() << "]";
  res.init_on_cpu(tot_len, (unsigned int)node_ids.size(), slot_num);
  unsigned int offset = 0, ind = 0;
  for (size_t i = 0; i < shard_num; i++) {
    tasks.push_back(
        _cpu_worker_pool[gpu_id]->enqueue([&, i, ind, offset, this]() -> int {
          auto start = ind;
          for (size_t j = 0; j < node_id_array[i].size(); j++) {
            res.node_list[start] = node_id_array[i][j];
            res.fea_info_list[start] = node_fea_info_array[i][j];
            res.fea_info_list[start++].feature_offset += offset;
          }
          for (size_t j = 0; j < feature_array[i].size(); j++) {
            res.feature_list[offset + j] = feature_array[i][j];
            res.slot_id_list[offset + j] = slot_id_array[i][j];
          }
          return 0;
        }));
    offset += feature_array[i].size();
    ind += node_id_array[i].size();
  }
  for (size_t i = 0; i < tasks.size(); i++) tasks[i].get();
  return res;
}

paddle::framework::GpuPsCommGraph GraphTable::make_gpu_ps_graph(
    int idx, const std::vector<uint64_t> &ids) {
  std::vector<std::vector<uint64_t>> bags(task_pool_size_);
  for (int i = 0; i < task_pool_size_; i++) {
    auto predsize = ids.size() / task_pool_size_;
    bags[i].reserve(predsize * 1.2);
  }
  for (auto x : ids) {
    int location = x % shard_num % task_pool_size_;
    bags[location].push_back(x);
  }

  std::vector<std::future<int>> tasks;
  std::vector<uint64_t> node_array[task_pool_size_];  // node id list
  std::vector<paddle::framework::GpuPsNodeInfo> info_array[task_pool_size_];
  std::vector<uint64_t> edge_array[task_pool_size_];  // edge id list

  for (size_t i = 0; i < bags.size(); i++) {
    if (bags[i].size() > 0) {
      tasks.push_back(_shards_task_pool[i]->enqueue([&, i, this]() -> int {
        node_array[i].resize(bags[i].size());
        info_array[i].resize(bags[i].size());
        edge_array[i].reserve(bags[i].size());

        for (size_t j = 0; j < bags[i].size(); j++) {
          auto node_id = bags[i][j];
          node_array[i][j] = node_id;
          Node *v = find_node(GraphTableType::EDGE_TABLE, idx, node_id);
          if (v != nullptr) {
            info_array[i][j].neighbor_offset = edge_array[i].size();
            info_array[i][j].neighbor_size = v->get_neighbor_size();
            for (size_t k = 0; k < v->get_neighbor_size(); k++) {
              edge_array[i].push_back(v->get_neighbor_id(k));
            }
          } else {
            info_array[i][j].neighbor_offset = 0;
            info_array[i][j].neighbor_size = 0;
          }
        }
        return 0;
      }));
    }
  }
  for (size_t i = 0; i < tasks.size(); i++) tasks[i].get();

  int64_t tot_len = 0;
  for (int i = 0; i < task_pool_size_; i++) {
    tot_len += edge_array[i].size();
  }

  paddle::framework::GpuPsCommGraph res;
  res.init_on_cpu(tot_len, ids.size());
  int64_t offset = 0, ind = 0;
  for (int i = 0; i < task_pool_size_; i++) {
    for (size_t j = 0; j < node_array[i].size(); j++) {
      res.node_list[ind] = node_array[i][j];
      res.node_info_list[ind] = info_array[i][j];
      res.node_info_list[ind++].neighbor_offset += offset;
    }
    for (size_t j = 0; j < edge_array[i].size(); j++) {
      res.neighbor_list[offset + j] = edge_array[i][j];
    }
    offset += edge_array[i].size();
  }
  return res;
}

int32_t GraphTable::add_node_to_ssd(
    int type_id, int idx, uint64_t src_id, char *data, int len) {
  if (_db != NULL) {
    char ch[sizeof(int) * 2 + sizeof(uint64_t)];
    memcpy(ch, &type_id, sizeof(int));
    memcpy(ch + sizeof(int), &idx, sizeof(int));
    memcpy(ch + sizeof(int) * 2, &src_id, sizeof(uint64_t));
    std::string str;
    if (_db->get(src_id % shard_num % task_pool_size_,
                 ch,
                 sizeof(int) * 2 + sizeof(uint64_t),
                 str) == 0) {
      const uint64_t *stored_data =
          reinterpret_cast<const uint64_t *>(str.c_str());  // NOLINT
      int n = str.size() / sizeof(uint64_t);
      char *new_data = new char[n * sizeof(uint64_t) + len];
      memcpy(new_data, stored_data, n * sizeof(uint64_t));
      memcpy(new_data + n * sizeof(uint64_t), data, len);
      _db->put(src_id % shard_num % task_pool_size_,
               ch,
               sizeof(int) * 2 + sizeof(uint64_t),
               reinterpret_cast<char *>(new_data),
               n * sizeof(uint64_t) + len);
      delete[] new_data;
    } else {
      _db->put(src_id % shard_num % task_pool_size_,
               ch,
               sizeof(int) * 2 + sizeof(uint64_t),
               reinterpret_cast<char *>(data),
               len);
    }
  }
  return 0;
}
char *GraphTable::random_sample_neighbor_from_ssd(
    int idx,
    uint64_t id,
    int sample_size,
    const std::shared_ptr<std::mt19937_64> rng,
    int &actual_size) {
  if (_db == NULL) {
    actual_size = 0;
    return NULL;
  }
  std::string str;
  VLOG(2) << "sample ssd for key " << id;
  char ch[sizeof(int) * 2 + sizeof(uint64_t)];
  memset(ch, 0, sizeof(int));
  memcpy(ch + sizeof(int), &idx, sizeof(int));
  memcpy(ch + sizeof(int) * 2, &id, sizeof(uint64_t));
  if (_db->get(id % shard_num % task_pool_size_,
               ch,
               sizeof(int) * 2 + sizeof(uint64_t),
               str) == 0) {
    const uint64_t *data = reinterpret_cast<const uint64_t *>(str.c_str());
    int n = str.size() / sizeof(uint64_t);
    std::unordered_map<int, int> m;
    // std::vector<uint64_t> res;
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
              << *reinterpret_cast<uint64_t *>(&buff[i]);
    }
    return buff;
  }
  actual_size = 0;
  return NULL;
}

int64_t GraphTable::load_graph_to_memory_from_ssd(int idx,
                                                  std::vector<uint64_t> &ids) {
  std::vector<std::vector<uint64_t>> bags(task_pool_size_);
  for (auto x : ids) {
    int location = x % shard_num % task_pool_size_;
    bags[location].push_back(x);
  }
  std::vector<std::future<int>> tasks;
  std::vector<int64_t> count(task_pool_size_, 0);
  for (size_t i = 0; i < bags.size(); i++) {
    if (bags[i].size() > 0) {
      tasks.push_back(_shards_task_pool[i]->enqueue([&, i, idx, this]() -> int {
        char ch[sizeof(int) * 2 + sizeof(uint64_t)];
        memset(ch, 0, sizeof(int));
        memcpy(ch + sizeof(int), &idx, sizeof(int));
        for (size_t k = 0; k < bags[i].size(); k++) {
          auto v = bags[i][k];
          memcpy(ch + sizeof(int) * 2, &v, sizeof(uint64_t));
          std::string str;
          if (_db->get(i, ch, sizeof(int) * 2 + sizeof(uint64_t), str) == 0) {
            count[i] += (int64_t)str.size();
            for (size_t j = 0; j < str.size(); j += sizeof(uint64_t)) {
              uint64_t id =
                  *reinterpret_cast<const uint64_t *>(str.c_str() + j);
              add_comm_edge(idx, v, id);
            }
          }
        }
        return 0;
      }));
    }
  }

  for (size_t i = 0; i < tasks.size(); i++) tasks[i].get();
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
  std::unordered_map<uint64_t, int> id_map;
  std::vector<rocksdb::Iterator *> iters;
  for (int i = 0; i < task_pool_size_; i++) {
    iters.push_back(_db->get_iterator(i));
    iters[i]->SeekToFirst();
  }
  size_t next = 0;
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
    std::uint64_t i_key =
        *reinterpret_cast<const uint64_t *>(key.c_str() + sizeof(int) * 2);
    for (int i = 0; i < part_len; i++) {
      if (memory_remaining[i] < (int64_t)value.size()) {
        score[i] = -100000.0;
      } else {
        score[i] = 0;
      }
    }
    for (size_t j = 0; j < value.size(); j += sizeof(uint64_t)) {
      uint64_t v = *(reinterpret_cast<const uint64_t *>(value.c_str() + j));
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
  if (file_path[file_path.size() - 1] != '/') {
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

  for (size_t i = 0; i < tasks.size(); i++) tasks[i].get();
}
#endif

void GraphTable::clear_graph(int idx) {
  for (auto p : edge_shards[idx]) {
    p->clear();
    delete p;
  }

  edge_shards[idx].clear();
  for (size_t i = 0; i < shard_num_per_server; i++) {
    edge_shards[idx].push_back(new GraphShard());
  }
}

#ifdef PADDLE_WITH_HETERPS
void GraphTable::release_graph() {
  // Before releasing graph, prepare for sampling ids and embedding keys.
  build_graph_type_keys();

  if (FLAGS_gpugraph_storage_mode ==
      paddle::framework::GpuGraphStorageMode::WHOLE_HBM) {
    build_graph_total_keys();
  }
  // clear graph
  if (FLAGS_gpugraph_storage_mode == paddle::framework::GpuGraphStorageMode::
                                         MEM_EMB_FEATURE_AND_GPU_GRAPH ||
      FLAGS_gpugraph_storage_mode == paddle::framework::GpuGraphStorageMode::
                                         SSD_EMB_AND_MEM_FEATURE_GPU_GRAPH) {
    clear_edge_shard();
  } else {
    clear_graph();
  }
}

void GraphTable::release_graph_edge() {
  if (FLAGS_gpugraph_storage_mode ==
      paddle::framework::GpuGraphStorageMode::WHOLE_HBM) {
    build_graph_total_keys();
  }
  clear_edge_shard();
}

void GraphTable::release_graph_node() {
  build_graph_type_keys();
  if (FLAGS_graph_metapath_split_opt) {
    clear_feature_shard();
  } else {
    if (FLAGS_gpugraph_storage_mode != paddle::framework::GpuGraphStorageMode::
                                           MEM_EMB_FEATURE_AND_GPU_GRAPH &&
        FLAGS_gpugraph_storage_mode != paddle::framework::GpuGraphStorageMode::
                                           SSD_EMB_AND_MEM_FEATURE_GPU_GRAPH) {
      clear_feature_shard();
    } else {
      merge_feature_shard();
      feature_shrink_to_fit();
    }
  }
}
#endif

void GraphTable::clear_edge_shard() {
  VLOG(0) << "begin clear edge shard";
  std::vector<std::future<int>> tasks;
  for (auto &type_shards : edge_shards) {
    for (auto &shard : type_shards) {
      tasks.push_back(
          load_node_edge_task_pool->enqueue([&shard, this]() -> int {
            delete shard;
            return 0;
          }));
    }
  }
  for (size_t i = 0; i < tasks.size(); i++) tasks[i].get();
  for (auto &shards : edge_shards) {
    shards.clear();
    for (size_t i = 0; i < shard_num_per_server; i++) {
      shards.push_back(new GraphShard());
    }
  }
  VLOG(0) << "finish clear edge shard";
}

void GraphTable::clear_feature_shard() {
  VLOG(0) << "begin clear feature shard";
  std::vector<std::future<int>> tasks;
  for (auto &type_shards : feature_shards) {
    for (auto &shard : type_shards) {
      tasks.push_back(
          load_node_edge_task_pool->enqueue([&shard, this]() -> int {
            delete shard;
            return 0;
          }));
    }
  }
  for (size_t i = 0; i < tasks.size(); i++) tasks[i].get();
  for (auto &shards : feature_shards) {
    shards.clear();
    for (size_t i = 0; i < shard_num_per_server; i++) {
      shards.push_back(new GraphShard());
    }
  }
  VLOG(0) << "finish clear feature shard";
}

#ifdef PADDLE_WITH_HETERPS
void GraphTable::feature_shrink_to_fit() {
  std::vector<std::future<int>> tasks;
  for (auto &type_shards : feature_shards) {
    for (auto &shard : type_shards) {
      tasks.push_back(
          load_node_edge_task_pool->enqueue([&shard, this]() -> int {
            shard->shrink_to_fit();
            return 0;
          }));
    }
  }
  for (size_t i = 0; i < tasks.size(); i++) tasks[i].get();
}

void GraphTable::merge_feature_shard() {
  VLOG(0) << "begin merge_feature_shard";
  std::vector<std::future<int>> tasks;
  for (size_t i = 0; i < feature_shards[0].size(); i++) {
    tasks.push_back(load_node_edge_task_pool->enqueue([i, this]() -> int {
      for (size_t j = 1; j < feature_shards.size(); j++) {
        feature_shards[0][i]->merge_shard(feature_shards[j][i]);
      }
      return 0;
    }));
  }
  for (size_t i = 0; i < tasks.size(); i++) tasks[i].get();
  feature_shards.resize(1);
}

#endif

void GraphTable::clear_graph() {
  VLOG(0) << "begin clear_graph";
  clear_edge_shard();
  clear_feature_shard();
  VLOG(0) << "finish clear_graph";
}

#ifdef PADDLE_WITH_HETERPS
int32_t GraphTable::load_next_partition(int idx) {
  if (next_partition >= static_cast<int>(partitions[idx].size())) {
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
      std::vector<uint64_t> dist_data;
      for (auto x : dist_ids) {
        dist_data.push_back(std::stoll(x));
        total_memory_cost += sizeof(uint64_t);
      }
      add_node_to_ssd(0,
                      idx,
                      src_id,
                      reinterpret_cast<char *>(dist_data.data()),
                      static_cast<int>(dist_data.size() * sizeof(uint64_t)));
    }
  }
  VLOG(0) << "total memory cost = " << total_memory_cost << " bytes";
  return 0;
}

int32_t GraphTable::dump_edges_to_ssd(int idx) {
  VLOG(2) << "calling dump edges to ssd";
  std::vector<std::future<int64_t>> tasks;
  auto &shards = edge_shards[idx];
  for (size_t i = 0; i < shards.size(); ++i) {
    tasks.push_back(_shards_task_pool[i % task_pool_size_]->enqueue(
        [&, i, this]() -> int64_t {
          int64_t cost = 0;
          std::vector<Node *> &v = shards[i]->get_bucket();
          for (size_t j = 0; j < v.size(); j++) {
            std::vector<uint64_t> s;
            for (size_t k = 0; k < v[j]->get_neighbor_size(); k++) {
              s.push_back(v[j]->get_neighbor_id(k));
            }
            cost += v[j]->get_neighbor_size() * sizeof(uint64_t);
            add_node_to_ssd(0,
                            idx,
                            v[j]->get_id(),
                            (char *)(s.data()),  // NOLINT
                            s.size() * sizeof(uint64_t));
          }
          return cost;
        }));
  }
  for (size_t i = 0; i < tasks.size(); i++) total_memory_cost += tasks[i].get();
  return 0;
}
int32_t GraphTable::make_complementary_graph(int idx, int64_t byte_size) {
  VLOG(0) << "make_complementary_graph";
  const size_t fixed_size = byte_size / 8;
  std::vector<std::unordered_map<uint64_t, int>> count(task_pool_size_);
  std::vector<std::future<int>> tasks;
  auto &shards = edge_shards[idx];
  for (size_t i = 0; i < shards.size(); ++i) {
    tasks.push_back(
        _shards_task_pool[i % task_pool_size_]->enqueue([&, i, this]() -> int {
          std::vector<Node *> &v = shards[i]->get_bucket();
          size_t ind = i % this->task_pool_size_;
          for (size_t j = 0; j < v.size(); j++) {
            // size_t location = v[j]->get_id();
            for (size_t k = 0; k < v[j]->get_neighbor_size(); k++) {
              count[ind][v[j]->get_neighbor_id(k)]++;
            }
          }
          return 0;
        }));
  }
  for (size_t i = 0; i < tasks.size(); i++) tasks[i].get();
  std::unordered_map<uint64_t, int> final_count;
  std::map<int, std::vector<uint64_t>> count_to_id;
  std::vector<uint64_t> buffer;
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
                    static_cast<int>sample_neighbors_ex[ind][location].size();
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
        int search_size = std::min(init_search_size, static_cast<int>v.size());
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
                  static_cast<int>sample_neighbors_ex[ind][location].size();
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

int32_t GraphTable::add_comm_edge(int idx, uint64_t src_id, uint64_t dst_id) {
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
                                   std::vector<uint64_t> &id_list,
                                   std::vector<bool> &is_weight_list) {
  auto &shards = edge_shards[idx];
  size_t node_size = id_list.size();
  std::vector<std::vector<std::pair<uint64_t, bool>>> batch(task_pool_size_);
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

int32_t GraphTable::remove_graph_node(int idx, std::vector<uint64_t> &id_list) {
  size_t node_size = id_list.size();
  std::vector<std::vector<uint64_t>> batch(task_pool_size_);
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

void GraphShard::delete_node(uint64_t id) {
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
GraphNode *GraphShard::add_graph_node(uint64_t id) {
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

FeatureNode *GraphShard::add_feature_node(uint64_t id, bool is_overlap) {
  if (node_location.find(id) == node_location.end()) {
    node_location[id] = bucket.size();
    bucket.push_back(new FeatureNode(id));
    return reinterpret_cast<FeatureNode *>(bucket[node_location[id]]);
  }
  if (is_overlap) {
    return reinterpret_cast<FeatureNode *>(bucket[node_location[id]]);
  }

  return NULL;
}

void GraphShard::add_neighbor(uint64_t id, uint64_t dst_id, float weight) {
  find_node(id)->add_edge(dst_id, weight);
}

Node *GraphShard::find_node(uint64_t id) {
  auto iter = node_location.find(id);
  return iter == node_location.end() ? nullptr : bucket[iter->second];
}

GraphTable::~GraphTable() {
#ifdef PADDLE_WITH_GPU_GRAPH
  clear_graph();
#endif
}

int32_t GraphTable::Load(const std::string &path, const std::string &param) {
  bool load_edge = (param[0] == 'e');
  bool load_node = (param[0] == 'n');
  if (load_edge) {
    bool reverse_edge = (param[1] == '<');
    std::string edge_type = param.substr(2);
    int ret = this->load_edges(path, reverse_edge, edge_type);
    if (ret != 0) {
      VLOG(0) << "Fail to load edges, path[" << path << "] edge_type["
              << edge_type << "]";
      return -1;
    }
  }
  if (load_node) {
    std::string node_type = param.substr(1);
    int ret = this->load_nodes(path, node_type);
    if (ret != 0) {
      VLOG(0) << "Fail to load nodes, path[" << path << "] node_type["
              << node_type << "]";
      return -1;
    }
  }
  return 0;
}

std::string GraphTable::get_inverse_etype(std::string &etype) {
  auto etype_split = paddle::string::split_string<std::string>(etype, "2");
  std::string res;
  if (etype_split.size() == 3) {
    res = etype_split[2] + "2" + etype_split[1] + "2" + etype_split[0];
  } else {
    res = etype_split[1] + "2" + etype_split[0];
  }
  return res;
}

int32_t GraphTable::parse_type_to_typepath(
    std::string &type2files,
    std::string graph_data_local_path,
    std::vector<std::string> &res_type,
    std::unordered_map<std::string, std::string> &res_type2path) {
  auto type2files_split =
      paddle::string::split_string<std::string>(type2files, ",");
  if (type2files_split.size() == 0) {
    return -1;
  }
  for (auto one_type2file : type2files_split) {
    auto one_type2file_split =
        paddle::string::split_string<std::string>(one_type2file, ":");
    auto type = one_type2file_split[0];
    auto type_dir = one_type2file_split[1];
    res_type.push_back(type);
    res_type2path[type] = graph_data_local_path + "/" + type_dir;
  }
  return 0;
}

int32_t GraphTable::parse_edge_and_load(
    std::string etype2files,
    std::string graph_data_local_path,
    int part_num,
    bool reverse,
    const std::vector<bool> &is_reverse_edge_map) {
  std::vector<std::string> etypes;
  std::unordered_map<std::string, std::string> edge_to_edgedir;
  int res = parse_type_to_typepath(
      etype2files, graph_data_local_path, etypes, edge_to_edgedir);
  if (res != 0) {
    VLOG(0) << "parse edge type and edgedir failed!";
    return -1;
  }
  VLOG(0) << "etypes size: " << etypes.size();
  VLOG(0) << "whether reverse: " << reverse;
  is_load_reverse_edge = reverse;
  std::string delim = ";";
  size_t total_len = etypes.size();

  std::vector<std::future<int>> tasks;
  for (size_t i = 0; i < total_len; i++) {
    tasks.push_back(
        _shards_task_pool[i % task_pool_size_]->enqueue([&, i, this]() -> int {
          std::string etype_path = edge_to_edgedir[etypes[i]];
          bool only_load_reverse_edge = false;
          if (!reverse) {
            only_load_reverse_edge = is_reverse_edge_map[i];
          }
          if (only_load_reverse_edge) {
            VLOG(1) << "only_load_reverse_edge is True, etype[" << etypes[i]
                    << "], file_path[" << etype_path << "]";
          } else {
            VLOG(1) << "only_load_reverse_edge is False, etype[" << etypes[i]
                    << "], file_path[" << etype_path << "]";
          }
          auto etype_path_list = paddle::framework::localfs_list(etype_path);
          std::string etype_path_str;
          if (part_num > 0 &&
              part_num < static_cast<int>(etype_path_list.size())) {
            std::vector<std::string> sub_etype_path_list(
                etype_path_list.begin(), etype_path_list.begin() + part_num);
            etype_path_str =
                paddle::string::join_strings(sub_etype_path_list, delim);
          } else {
            etype_path_str =
                paddle::string::join_strings(etype_path_list, delim);
          }
          if (!only_load_reverse_edge) {
            this->load_edges(etype_path_str, false, etypes[i]);
            if (reverse) {
              std::string r_etype = get_inverse_etype(etypes[i]);
              this->load_edges(etype_path_str, true, r_etype);
            }
          } else {
            this->load_edges(etype_path_str, true, etypes[i]);
          }
          return 0;
        }));
  }
  for (size_t i = 0; i < tasks.size(); i++) tasks[i].get();
  return 0;
}

int32_t GraphTable::parse_node_and_load(std::string ntype2files,
                                        std::string graph_data_local_path,
                                        int part_num) {
  std::vector<std::string> ntypes;
  std::unordered_map<std::string, std::string> node_to_nodedir;
  int res = parse_type_to_typepath(
      ntype2files, graph_data_local_path, ntypes, node_to_nodedir);
  if (res != 0) {
    VLOG(0) << "parse node type and nodedir failed!";
    return -1;
  }
  std::string delim = ";";
  std::string npath = node_to_nodedir[ntypes[0]];
  auto npath_list = paddle::framework::localfs_list(npath);
  std::string npath_str;
  if (part_num > 0 && part_num < static_cast<int>(npath_list.size())) {
    std::vector<std::string> sub_npath_list(npath_list.begin(),
                                            npath_list.begin() + part_num);
    npath_str = paddle::string::join_strings(sub_npath_list, delim);
  } else {
    npath_str = paddle::string::join_strings(npath_list, delim);
  }

  if (ntypes.size() == 0) {
    VLOG(0) << "node_type not specified, nothing will be loaded ";
    return 0;
  }
  if (FLAGS_graph_load_in_parallel) {
    int ret = this->load_nodes(npath_str, "");
    if (ret != 0) {
      VLOG(0) << "Fail to load nodes, path[" << npath << "]";
      return -1;
    }
  } else {
    for (size_t j = 0; j < ntypes.size(); j++) {
      int ret = this->load_nodes(npath_str, ntypes[j]);
      if (ret != 0) {
        VLOG(0) << "Fail to load nodes, path[" << npath << "], ntypes["
                << ntypes[j] << "]";
        return -1;
      }
    }
  }
  return 0;
}

int32_t GraphTable::load_node_and_edge_file(
    std::string etype2files,
    std::string ntype2files,
    std::string graph_data_local_path,
    int part_num,
    bool reverse,
    const std::vector<bool> &is_reverse_edge_map) {
  std::vector<std::string> etypes;
  std::unordered_map<std::string, std::string> edge_to_edgedir;
  int res = parse_type_to_typepath(
      etype2files, graph_data_local_path, etypes, edge_to_edgedir);
  if (res != 0) {
    VLOG(0) << "parse edge type and edgedir failed!";
    return -1;
  }
  std::vector<std::string> ntypes;
  std::unordered_map<std::string, std::string> node_to_nodedir;
  res = parse_type_to_typepath(
      ntype2files, graph_data_local_path, ntypes, node_to_nodedir);
  if (res != 0) {
    VLOG(0) << "parse node type and nodedir failed!";
    return -1;
  }

  VLOG(0) << "etypes size: " << etypes.size();
  VLOG(0) << "whether reverse: " << reverse;
  is_load_reverse_edge = reverse;
  std::string delim = ";";
  size_t total_len = etypes.size() + 1;  // 1 is for node

  std::vector<std::future<int>> tasks;
  for (size_t i = 0; i < total_len; i++) {
    tasks.push_back(
        _shards_task_pool[i % task_pool_size_]->enqueue([&, i, this]() -> int {
          if (i < etypes.size()) {
            std::string etype_path = edge_to_edgedir[etypes[i]];
            bool only_load_reverse_edge = false;
            if (!reverse) {
              only_load_reverse_edge = is_reverse_edge_map[i];
            }
            if (only_load_reverse_edge) {
              VLOG(1) << "only_load_reverse_edge is True, etype[" << etypes[i]
                      << "], file_path[" << etype_path << "]";
            } else {
              VLOG(1) << "only_load_reverse_edge is False, etype[" << etypes[i]
                      << "], file_path[" << etype_path << "]";
            }
            auto etype_path_list = paddle::framework::localfs_list(etype_path);
            std::string etype_path_str;
            if (part_num > 0 &&
                part_num < static_cast<int>(etype_path_list.size())) {
              std::vector<std::string> sub_etype_path_list(
                  etype_path_list.begin(), etype_path_list.begin() + part_num);
              etype_path_str =
                  paddle::string::join_strings(sub_etype_path_list, delim);
            } else {
              etype_path_str =
                  paddle::string::join_strings(etype_path_list, delim);
            }
            if (!only_load_reverse_edge) {
              this->load_edges(etype_path_str, false, etypes[i]);
              if (reverse) {
                std::string r_etype = get_inverse_etype(etypes[i]);
                this->load_edges(etype_path_str, true, r_etype);
              }
            } else {
              this->load_edges(etype_path_str, true, etypes[i]);
            }
          } else {
            std::string npath = node_to_nodedir[ntypes[0]];
            auto npath_list = paddle::framework::localfs_list(npath);
            std::string npath_str;
            if (part_num > 0 &&
                part_num < static_cast<int>(npath_list.size())) {
              std::vector<std::string> sub_npath_list(
                  npath_list.begin(), npath_list.begin() + part_num);
              npath_str = paddle::string::join_strings(sub_npath_list, delim);
            } else {
              npath_str = paddle::string::join_strings(npath_list, delim);
            }

            if (ntypes.size() == 0) {
              VLOG(0) << "node_type not specified, nothing will be loaded ";
              return 0;
            }
            if (FLAGS_graph_load_in_parallel) {
              int ret = this->load_nodes(npath_str, "");
              if (ret != 0) {
                VLOG(0) << "Fail to load nodes, path[" << npath_str << "]";
                return -1;
              }
            } else {
              for (size_t j = 0; j < ntypes.size(); j++) {
                int ret = this->load_nodes(npath_str, ntypes[j]);
                if (ret != 0) {
                  VLOG(0) << "Fail to load nodes, path[" << npath_str
                          << "], ntypes[" << ntypes[j] << "]";
                  return -1;
                }
              }
            }
          }
          return 0;
        }));
  }
  for (size_t i = 0; i < tasks.size(); i++) tasks[i].get();
  if (is_parse_node_fail_) {
    VLOG(0) << "Fail to load node_and_edge_file";
    return -1;
  }
  return 0;
}

int32_t GraphTable::get_nodes_ids_by_ranges(
    GraphTableType table_type,
    int idx,
    std::vector<std::pair<int, int>> ranges,
    std::vector<uint64_t> &res) {
  std::mutex mutex;
  int start = 0, end, index = 0, total_size = 0;
  res.clear();
  auto &shards = table_type == GraphTableType::EDGE_TABLE ? edge_shards[idx]
                                                          : feature_shards[idx];
  std::vector<std::future<size_t>> tasks;
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
            [&shards, this, first, second, i, &res, &mutex]() -> size_t {
              std::vector<uint64_t> keys;
              shards[i]->get_ids_by_range(first, second, &keys);

              size_t num = keys.size();
              mutex.lock();
              res.reserve(res.size() + num);
              for (auto &id : keys) {
                res.push_back(id);
                std::swap(res[rand() % res.size()],
                          res[static_cast<int>(res.size()) - 1]);
              }
              mutex.unlock();

              return num;
            }));
      }
    }
    total_size += shards[i]->get_size();
  }
  for (size_t i = 0; i < tasks.size(); i++) {
    tasks[i].get();
  }
  return 0;
}

std::pair<uint64_t, uint64_t> GraphTable::parse_node_file(
    const std::string &path, const std::string &node_type, int idx) {
  std::ifstream file(path);
  std::string line;
  uint64_t local_count = 0;
  uint64_t local_valid_count = 0;

  int num = 0;
  std::vector<paddle::string::str_ptr> vals;
  size_t n = node_type.length();
  while (std::getline(file, line)) {
    if (strncmp(line.c_str(), node_type.c_str(), n) != 0) {
      continue;
    }
    vals.clear();
    num = paddle::string::split_string_ptr(
        line.c_str() + n + 1, line.length() - n - 1, '\t', &vals);
    if (num == 0) {
      continue;
    }
    uint64_t id = std::strtoul(vals[0].ptr, NULL, 10);
    size_t shard_id = id % shard_num;
    if (shard_id >= shard_end || shard_id < shard_start) {
      VLOG(4) << "will not load " << id << " from " << path
              << ", please check id distribution";
      continue;
    }
    local_count++;

    size_t index = shard_id - shard_start;
    auto node = feature_shards[idx][index]->add_feature_node(id, false);
    if (node != NULL) {
      node->set_feature_size(feat_name[idx].size());
      for (int i = 1; i < num; ++i) {
        auto &v = vals[i];
        int ret = parse_feature(idx, v.ptr, v.len, node);
        if (ret != 0) {
          VLOG(0) << "Fail to parse feature, node_id[" << id << "]";
          is_parse_node_fail_ = true;
          return {0, 0};
        }
      }
    }
    local_valid_count++;
  }
  VLOG(2) << "node_type[" << node_type << "] loads " << local_count
          << " nodes from filepath->" << path;
  return {local_count, local_valid_count};
}

std::pair<uint64_t, uint64_t> GraphTable::parse_node_file(
    const std::string &path) {
  std::ifstream file(path);
  std::string line;
  uint64_t local_count = 0;
  uint64_t local_valid_count = 0;
  int idx = 0;

  auto path_split = paddle::string::split_string<std::string>(path, "/");
  auto path_name = path_split[path_split.size() - 1];

  int num = 0;
  std::vector<paddle::string::str_ptr> vals;

  while (std::getline(file, line)) {
    vals.clear();
    num = paddle::string::split_string_ptr(
        line.c_str(), line.length(), '\t', &vals);
    if (vals.empty()) {
      continue;
    }
    std::string parse_node_type = vals[0].to_string();
    auto it = feature_to_id.find(parse_node_type);
    if (it == feature_to_id.end()) {
      VLOG(0) << parse_node_type << "type error, please check";
      continue;
    }
    idx = it->second;
    uint64_t id = std::strtoul(vals[1].ptr, NULL, 10);
    size_t shard_id = id % shard_num;
    if (shard_id >= shard_end || shard_id < shard_start) {
      VLOG(4) << "will not load " << id << " from " << path
              << ", please check id distribution";
      continue;
    }
    local_count++;

    size_t index = shard_id - shard_start;
    auto node = feature_shards[idx][index]->add_feature_node(id, false);
    if (node != NULL) {
      for (int i = 2; i < num; ++i) {
        auto &v = vals[i];
        int ret = parse_feature(idx, v.ptr, v.len, node);
        if (ret != 0) {
          VLOG(0) << "Fail to parse feature, node_id[" << id << "]";
          is_parse_node_fail_ = true;
          return {0, 0};
        }
      }
    }
    local_valid_count++;
  }
  VLOG(2) << local_valid_count << "/" << local_count << " nodes from filepath->"
          << path;
  return {local_count, local_valid_count};
}

// // TODO(danleifeng): opt load all node_types in once reading
int32_t GraphTable::load_nodes(const std::string &path, std::string node_type) {
  auto paths = paddle::string::split_string<std::string>(path, ";");
  uint64_t count = 0;
  uint64_t valid_count = 0;
  int idx = 0;
  if (FLAGS_graph_load_in_parallel) {
    if (node_type == "") {
      VLOG(0) << "Begin GraphTable::load_nodes(), will load all node_type once";
    }
    std::vector<std::future<std::pair<uint64_t, uint64_t>>> tasks;
    for (size_t i = 0; i < paths.size(); i++) {
      tasks.push_back(load_node_edge_task_pool->enqueue(
          [&, i, this]() -> std::pair<uint64_t, uint64_t> {
            return parse_node_file(paths[i]);
          }));
    }
    for (size_t i = 0; i < tasks.size(); i++) {
      auto res = tasks[i].get();
      count += res.first;
      valid_count += res.second;
    }
  } else {
    VLOG(0) << "Begin GraphTable::load_nodes() node_type[" << node_type << "]";
    if (node_type == "") {
      VLOG(0) << "node_type not specified, loading edges to "
              << id_to_feature[0] << " part";
    } else {
      if (feature_to_id.find(node_type) == feature_to_id.end()) {
        VLOG(0) << "node_type " << node_type
                << " is not defined, nothing will be loaded";
        return 0;
      }
      idx = feature_to_id[node_type];
    }
    for (auto path : paths) {
      VLOG(2) << "Begin GraphTable::load_nodes(), path[" << path << "]";
      auto res = parse_node_file(path, node_type, idx);
      count += res.first;
      valid_count += res.second;
    }
  }
  if (is_parse_node_fail_) {
    VLOG(0) << "Fail to load nodes, path[" << paths[0] << ".."
            << paths[paths.size() - 1] << "] node_type[" << node_type << "]";
    return -1;
  }

  VLOG(0) << valid_count << "/" << count << " nodes in node_type[ " << node_type
          << "] are loaded successfully!";
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

std::pair<uint64_t, uint64_t> GraphTable::parse_edge_file(
    const std::string &path, int idx, bool reverse) {
  std::string sample_type = "random";
  bool is_weighted = false;
  std::ifstream file(path);
  std::string line;
  uint64_t local_count = 0;
  uint64_t local_valid_count = 0;
  uint64_t part_num = 0;
  if (FLAGS_graph_load_in_parallel) {
    auto path_split = paddle::string::split_string<std::string>(path, "/");
    auto part_name_split = paddle::string::split_string<std::string>(
        path_split[path_split.size() - 1], "-");
    part_num = std::stoull(part_name_split[part_name_split.size() - 1]);
  }

  while (std::getline(file, line)) {
    size_t start = line.find_first_of('\t');
    if (start == std::string::npos) continue;
    local_count++;
    uint64_t src_id = std::stoull(&line[0]);
    uint64_t dst_id = std::stoull(&line[start + 1]);
    if (reverse) {
      std::swap(src_id, dst_id);
    }
    size_t src_shard_id = src_id % shard_num;
    if (FLAGS_graph_load_in_parallel) {
      if (src_shard_id != (part_num % shard_num)) {
        continue;
      }
    }

    float weight = 1;
    size_t last = line.find_last_of('\t');
    if (start != last) {
      weight = std::stof(&line[last + 1]);
      sample_type = "weighted";
      is_weighted = true;
    }

    if (src_shard_id >= shard_end || src_shard_id < shard_start) {
      VLOG(4) << "will not load " << src_id << " from " << path
              << ", please check id distribution";
      continue;
    }
    size_t index = src_shard_id - shard_start;
    auto node = edge_shards[idx][index]->add_graph_node(src_id);
    if (node != NULL) {
      node->build_edges(is_weighted);
      node->add_edge(dst_id, weight);
    }

    local_valid_count++;
  }
  VLOG(2) << local_valid_count << "/" << local_count
          << " edges are loaded from filepath->" << path;
  return {local_count, local_valid_count};
}

int32_t GraphTable::load_edges(const std::string &path,
                               bool reverse_edge,
                               const std::string &edge_type) {
#ifdef PADDLE_WITH_HETERPS
  if (search_level == 2) total_memory_cost = 0;
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
  uint64_t count = 0;
  uint64_t valid_count = 0;

  VLOG(0) << "Begin GraphTable::load_edges() edge_type[" << edge_type << "]";
  if (FLAGS_graph_load_in_parallel) {
    std::vector<std::future<std::pair<uint64_t, uint64_t>>> tasks;
    for (size_t i = 0; i < paths.size(); i++) {
      tasks.push_back(load_node_edge_task_pool->enqueue(
          [&, i, idx, this]() -> std::pair<uint64_t, uint64_t> {
            return parse_edge_file(paths[i], idx, reverse_edge);
          }));
    }
    for (size_t j = 0; j < tasks.size(); j++) {
      auto res = tasks[j].get();
      count += res.first;
      valid_count += res.second;
    }
  } else {
    for (auto path : paths) {
      auto res = parse_edge_file(path, idx, reverse_edge);
      count += res.first;
      valid_count += res.second;
    }
  }
  VLOG(0) << valid_count << "/" << count << " edge_type[" << edge_type
          << "] edges are loaded successfully";
  std::string edge_size = edge_type + ":" + std::to_string(valid_count);
  edge_type_size.push_back(edge_size);

#ifdef PADDLE_WITH_HETERPS
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

  if (!build_sampler_on_cpu) {
    // To reduce memory overhead, CPU samplers won't be created in gpugraph.
    // In order not to affect the sampler function of other scenario,
    // this optimization is only performed in load_edges function.
    VLOG(0) << "run in gpugraph mode!";
  } else {
    std::string sample_type = "random";
    VLOG(0) << "build sampler ... ";
    for (auto &shard : edge_shards[idx]) {
      auto bucket = shard->get_bucket();
      for (size_t i = 0; i < bucket.size(); i++) {
        bucket[i]->build_sampler(sample_type);
      }
    }
  }

  return 0;
}

Node *GraphTable::find_node(GraphTableType table_type, uint64_t id) {
  size_t shard_id = id % shard_num;
  if (shard_id >= shard_end || shard_id < shard_start) {
    return nullptr;
  }
  Node *node = nullptr;
  size_t index = shard_id - shard_start;
  auto &search_shards =
      table_type == GraphTableType::EDGE_TABLE ? edge_shards : feature_shards;
  for (auto &search_shard : search_shards) {
    PADDLE_ENFORCE_NOT_NULL(search_shard[index],
                            paddle::platform::errors::InvalidArgument(
                                "search_shard[%d] should not be null.", index));
    node = search_shard[index]->find_node(id);
    if (node != nullptr) {
      break;
    }
  }
  return node;
}

Node *GraphTable::find_node(GraphTableType table_type, int idx, uint64_t id) {
  size_t shard_id = id % shard_num;
  if (shard_id >= shard_end || shard_id < shard_start) {
    return nullptr;
  }
  size_t index = shard_id - shard_start;
  auto &search_shards = table_type == GraphTableType::EDGE_TABLE
                            ? edge_shards[idx]
                            : feature_shards[idx];
  PADDLE_ENFORCE_NOT_NULL(search_shards[index],
                          paddle::platform::errors::InvalidArgument(
                              "search_shard[%d] should not be null.", index));
  Node *node = search_shards[index]->find_node(id);
  return node;
}
uint32_t GraphTable::get_thread_pool_index(uint64_t node_id) {
  return node_id % shard_num % shard_num_per_server % task_pool_size_;
}

uint32_t GraphTable::get_thread_pool_index_by_shard_index(
    uint64_t shard_index) {
  return shard_index % shard_num_per_server % task_pool_size_;
}

int32_t GraphTable::clear_nodes(GraphTableType table_type, int idx) {
  auto &search_shards = table_type == GraphTableType::EDGE_TABLE
                            ? edge_shards[idx]
                            : feature_shards[idx];
  for (size_t i = 0; i < search_shards.size(); i++) {
    search_shards[i]->clear();
  }
  return 0;
}

int32_t GraphTable::random_sample_nodes(GraphTableType table_type,
                                        int idx,
                                        int sample_size,
                                        std::unique_ptr<char[]> &buffer,
                                        int &actual_size) {
  int total_size = 0;
  auto &shards = table_type == GraphTableType::EDGE_TABLE ? edge_shards[idx]
                                                          : feature_shards[idx];
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
    } else if ((ranges_pos[i] + start_index) >= total_size) {
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
  std::vector<uint64_t> res;
  get_nodes_ids_by_ranges(table_type, idx, second_half, res);
  actual_size = res.size() * sizeof(uint64_t);
  buffer.reset(new char[actual_size]);
  char *pointer = buffer.get();
  memcpy(pointer, res.data(), actual_size);
  return 0;
}
int32_t GraphTable::random_sample_neighbors(
    int idx,
    uint64_t *node_ids,
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

  for (size_t i = 0; i < seq_id.size(); i++) {
    if (seq_id[i].size() == 0) continue;
    tasks.push_back(_shards_task_pool[i]->enqueue([&, i, this]() -> int {
      uint64_t node_id;
      std::vector<std::pair<SampleKey, SampleResult>> r;
      LRUResponse response = LRUResponse::blocked;
      if (use_cache) {
        response =
            scaled_lru->query(i, id_list[i].data(), id_list[i].size(), r);
      }
      size_t index = 0;
      std::vector<SampleResult> sample_res;
      std::vector<SampleKey> sample_keys;
      auto &rng = _shards_task_rng_pool[i];
      for (size_t k = 0; k < id_list[i].size(); k++) {
        if (index < r.size() &&
            r[index].first.node_key == id_list[i][k].node_key) {
          int idy = seq_id[i][k];
          actual_sizes[idy] = r[index].second.actual_size;
          buffers[idy] = r[index].second.buffer;
          index++;
        } else {
          node_id = id_list[i][k].node_key;
          Node *node = find_node(GraphTableType::EDGE_TABLE, idx, node_id);
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
          uint64_t id;
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
                                  const std::vector<uint64_t> &node_ids,
                                  const std::vector<std::string> &feature_names,
                                  std::vector<std::vector<std::string>> &res) {
  size_t node_num = node_ids.size();
  std::vector<std::future<int>> tasks;
  for (size_t idy = 0; idy < node_num; ++idy) {
    uint64_t node_id = node_ids[idy];
    tasks.push_back(_shards_task_pool[get_thread_pool_index(node_id)]->enqueue(
        [&, idx, idy, node_id]() -> int {
          Node *node = find_node(GraphTableType::FEATURE_TABLE, idx, node_id);

          if (node == nullptr) {
            return 0;
          }
          for (size_t feat_idx = 0; feat_idx < feature_names.size();
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
    const std::vector<uint64_t> &node_ids,
    const std::vector<std::string> &feature_names,
    const std::vector<std::vector<std::string>> &res) {
  size_t node_num = node_ids.size();
  std::vector<std::future<int>> tasks;
  for (size_t idy = 0; idy < node_num; ++idy) {
    uint64_t node_id = node_ids[idy];
    tasks.push_back(_shards_task_pool[get_thread_pool_index(node_id)]->enqueue(
        [&, idx, idy, node_id]() -> int {
          size_t index = node_id % this->shard_num - this->shard_start;
          auto node = feature_shards[idx][index]->add_feature_node(node_id);
          node->set_feature_size(this->feat_name[idx].size());
          for (size_t feat_idx = 0; feat_idx < feature_names.size();
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

void string_vector_2_string(std::vector<std::string>::iterator strs_begin,
                            std::vector<std::string>::iterator strs_end,
                            char delim,
                            std::string *output) {
  size_t i = 0;
  for (std::vector<std::string>::iterator iter = strs_begin; iter != strs_end;
       ++iter) {
    if (i > 0) {
      *output += delim;
    }

    *output += *iter;
    ++i;
  }
}

void string_vector_2_string(
    std::vector<paddle::string::str_ptr>::iterator strs_begin,
    std::vector<paddle::string::str_ptr>::iterator strs_end,
    char delim,
    std::string *output) {
  size_t i = 0;
  for (auto iter = strs_begin; iter != strs_end; ++iter) {
    if (i > 0) {
      output->append(&delim, 1);
    }
    output->append((*iter).ptr, (*iter).len);
    ++i;
  }
}

int GraphTable::parse_feature(int idx,
                              const char *feat_str,
                              size_t len,
                              FeatureNode *node) {
  // Return (feat_id, btyes) if name are in this->feat_name, else return (-1,
  // "")
  thread_local std::vector<paddle::string::str_ptr> fields;
  fields.clear();
  char c = slot_feature_separator_.at(0);
  paddle::string::split_string_ptr(feat_str, len, c, &fields);

  thread_local std::vector<paddle::string::str_ptr> fea_fields;
  fea_fields.clear();
  c = feature_separator_.at(0);
  paddle::string::split_string_ptr(fields[1].ptr,
                                   fields[1].len,
                                   c,
                                   &fea_fields,
                                   FLAGS_gpugraph_slot_feasign_max_num);
  std::string name = fields[0].to_string();
  auto it = feat_id_map[idx].find(name);
  if (it != feat_id_map[idx].end()) {
    int32_t id = it->second;
    std::string *fea_ptr = node->mutable_feature(id);
    std::string dtype = this->feat_dtype[idx][id];
    if (dtype == "feasign") {
      //      string_vector_2_string(fields.begin() + 1, fields.end(), ' ',
      //      fea_ptr);
      int ret = FeatureNode::parse_value_to_bytes<uint64_t>(
          fea_fields.begin(), fea_fields.end(), fea_ptr);
      if (ret != 0) {
        VLOG(0) << "Fail to parse value";
        return -1;
      }
      return 0;
    } else if (dtype == "string") {
      string_vector_2_string(
          fea_fields.begin(), fea_fields.end(), ' ', fea_ptr);
      return 0;
    } else if (dtype == "float32") {
      int ret = FeatureNode::parse_value_to_bytes<float>(
          fea_fields.begin(), fea_fields.end(), fea_ptr);
      if (ret != 0) {
        VLOG(0) << "Fail to parse value";
        return -1;
      }
      return 0;
    } else if (dtype == "float64") {
      int ret = FeatureNode::parse_value_to_bytes<double>(
          fea_fields.begin(), fea_fields.end(), fea_ptr);
      if (ret != 0) {
        VLOG(0) << "Fail to parse value";
        return -1;
      }
      return 0;
    } else if (dtype == "int32") {
      int ret = FeatureNode::parse_value_to_bytes<int32_t>(
          fea_fields.begin(), fea_fields.end(), fea_ptr);
      if (ret != 0) {
        VLOG(0) << "Fail to parse value";
        return -1;
      }
      return 0;
    } else if (dtype == "int64") {
      int ret = FeatureNode::parse_value_to_bytes<uint64_t>(
          fea_fields.begin(), fea_fields.end(), fea_ptr);
      if (ret != 0) {
        VLOG(0) << "Fail to parse value";
        return -1;
      }
      return 0;
    }
  } else {
    VLOG(4) << "feature_name[" << name << "] is not in feat_id_map, ntype_id["
            << idx << "] feat_id_map_size[" << feat_id_map.size() << "]";
  }

  return 0;
}
// thread safe shard vector merge
class MergeShardVector {
 public:
  MergeShardVector(std::vector<std::vector<uint64_t>> *output, int slice_num) {
    _slice_num = slice_num;
    _shard_keys = output;
    _shard_keys->resize(slice_num);
    _mutexs = new std::mutex[slice_num];
  }
  ~MergeShardVector() {
    if (_mutexs != nullptr) {
      delete[] _mutexs;
      _mutexs = nullptr;
    }
  }
  // merge shard keys
  void merge(const std::vector<std::vector<uint64_t>> &shard_keys) {
    // add to shard
    for (int shard_id = 0; shard_id < _slice_num; ++shard_id) {
      auto &dest = (*_shard_keys)[shard_id];
      auto &src = shard_keys[shard_id];

      _mutexs[shard_id].lock();
      dest.insert(dest.end(), src.begin(), src.end());
      _mutexs[shard_id].unlock();
    }
  }

 private:
  int _slice_num = 0;
  std::mutex *_mutexs = nullptr;
  std::vector<std::vector<uint64_t>> *_shard_keys;
};

int GraphTable::get_all_id(GraphTableType table_type,
                           int slice_num,
                           std::vector<std::vector<uint64_t>> *output) {
  MergeShardVector shard_merge(output, slice_num);
  auto &search_shards =
      table_type == GraphTableType::EDGE_TABLE ? edge_shards : feature_shards;
  std::vector<std::future<size_t>> tasks;
  for (size_t idx = 0; idx < search_shards.size(); idx++) {
    for (size_t j = 0; j < search_shards[idx].size(); j++) {
      tasks.push_back(_shards_task_pool[j % task_pool_size_]->enqueue(
          [&search_shards, idx, j, slice_num, &shard_merge]() -> size_t {
            std::vector<std::vector<uint64_t>> shard_keys;
            size_t num =
                search_shards[idx][j]->get_all_id(&shard_keys, slice_num);
            // add to shard
            shard_merge.merge(shard_keys);
            return num;
          }));
    }
  }
  for (size_t i = 0; i < tasks.size(); ++i) {
    tasks[i].wait();
  }
  return 0;
}

int GraphTable::get_all_neighbor_id(
    GraphTableType table_type,
    int slice_num,
    std::vector<std::vector<uint64_t>> *output) {
  MergeShardVector shard_merge(output, slice_num);
  auto &search_shards =
      table_type == GraphTableType::EDGE_TABLE ? edge_shards : feature_shards;
  std::vector<std::future<size_t>> tasks;
  for (size_t idx = 0; idx < search_shards.size(); idx++) {
    for (size_t j = 0; j < search_shards[idx].size(); j++) {
      tasks.push_back(_shards_task_pool[j % task_pool_size_]->enqueue(
          [&search_shards, idx, j, slice_num, &shard_merge]() -> size_t {
            std::vector<std::vector<uint64_t>> shard_keys;
            size_t num = search_shards[idx][j]->get_all_neighbor_id(&shard_keys,
                                                                    slice_num);
            // add to shard
            shard_merge.merge(shard_keys);
            return num;
          }));
    }
  }
  for (size_t i = 0; i < tasks.size(); ++i) {
    tasks[i].wait();
  }
  return 0;
}

int GraphTable::get_all_id(GraphTableType table_type,
                           int idx,
                           int slice_num,
                           std::vector<std::vector<uint64_t>> *output) {
  MergeShardVector shard_merge(output, slice_num);
  auto &search_shards = table_type == GraphTableType::EDGE_TABLE
                            ? edge_shards[idx]
                            : feature_shards[idx];
  std::vector<std::future<size_t>> tasks;
  VLOG(3) << "begin task, task_pool_size_[" << task_pool_size_ << "]";
  for (size_t i = 0; i < search_shards.size(); i++) {
    tasks.push_back(_shards_task_pool[i % task_pool_size_]->enqueue(
        [&search_shards, i, slice_num, &shard_merge]() -> size_t {
          std::vector<std::vector<uint64_t>> shard_keys;
          size_t num = search_shards[i]->get_all_id(&shard_keys, slice_num);
          // add to shard
          shard_merge.merge(shard_keys);
          return num;
        }));
  }
  for (size_t i = 0; i < tasks.size(); ++i) {
    tasks[i].wait();
  }
  VLOG(3) << "end task, task_pool_size_[" << task_pool_size_ << "]";
  return 0;
}

int GraphTable::get_all_neighbor_id(
    GraphTableType table_type,
    int idx,
    int slice_num,
    std::vector<std::vector<uint64_t>> *output) {
  MergeShardVector shard_merge(output, slice_num);
  auto &search_shards = table_type == GraphTableType::EDGE_TABLE
                            ? edge_shards[idx]
                            : feature_shards[idx];
  std::vector<std::future<size_t>> tasks;
  VLOG(3) << "begin task, task_pool_size_[" << task_pool_size_ << "]";
  for (size_t i = 0; i < search_shards.size(); i++) {
    tasks.push_back(_shards_task_pool[i % task_pool_size_]->enqueue(
        [&search_shards, i, slice_num, &shard_merge]() -> size_t {
          std::vector<std::vector<uint64_t>> shard_keys;
          size_t num =
              search_shards[i]->get_all_neighbor_id(&shard_keys, slice_num);
          // add to shard
          shard_merge.merge(shard_keys);
          return num;
        }));
  }
  for (size_t i = 0; i < tasks.size(); ++i) {
    tasks[i].wait();
  }
  VLOG(3) << "end task, task_pool_size_[" << task_pool_size_ << "]";
  return 0;
}

int GraphTable::get_all_feature_ids(
    GraphTableType table_type,
    int idx,
    int slice_num,
    std::vector<std::vector<uint64_t>> *output) {
  MergeShardVector shard_merge(output, slice_num);
  auto &search_shards = table_type == GraphTableType::EDGE_TABLE
                            ? edge_shards[idx]
                            : feature_shards[idx];
  std::vector<std::future<size_t>> tasks;
  for (size_t i = 0; i < search_shards.size(); i++) {
    tasks.push_back(_shards_task_pool[i % task_pool_size_]->enqueue(
        [&search_shards, i, slice_num, &shard_merge]() -> size_t {
          std::vector<std::vector<uint64_t>> shard_keys;
          size_t num =
              search_shards[i]->get_all_feature_ids(&shard_keys, slice_num);
          // add to shard
          shard_merge.merge(shard_keys);
          return num;
        }));
  }
  for (size_t i = 0; i < tasks.size(); ++i) {
    tasks[i].wait();
  }
  return 0;
}

int GraphTable::get_node_embedding_ids(
    int slice_num, std::vector<std::vector<uint64_t>> *output) {
  if (is_load_reverse_edge && !FLAGS_graph_get_neighbor_id) {
    return get_all_id(GraphTableType::EDGE_TABLE, slice_num, output);
  } else {
    get_all_id(GraphTableType::EDGE_TABLE, slice_num, output);
    return get_all_neighbor_id(GraphTableType::EDGE_TABLE, slice_num, output);
  }
}

int32_t GraphTable::pull_graph_list(GraphTableType table_type,
                                    int idx,
                                    int start,
                                    int total_size,
                                    std::unique_ptr<char[]> &buffer,
                                    int &actual_size,
                                    bool need_feature,
                                    int step) {
  if (start < 0) start = 0;
  int size = 0, cur_size;
  auto &search_shards = table_type == GraphTableType::EDGE_TABLE
                            ? edge_shards[idx]
                            : feature_shards[idx];
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

void GraphTable::set_feature_separator(const std::string &ch) {
  feature_separator_ = ch;
}

void GraphTable::set_slot_feature_separator(const std::string &ch) {
  slot_feature_separator_ = ch;
}

int32_t GraphTable::get_server_index_by_id(uint64_t id) {
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
  build_sampler_on_cpu = graph.build_sampler_on_cpu();

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
    make_neighbor_sample_cache(cache_size_limit, cache_ttl);
  }
  _shards_task_pool.resize(task_pool_size_);
  for (size_t i = 0; i < _shards_task_pool.size(); ++i) {
    _shards_task_pool[i].reset(new ::ThreadPool(1));
    _shards_task_rng_pool.push_back(phi::GetCPURandomEngine(0));
  }
  load_node_edge_task_pool.reset(new ::ThreadPool(load_thread_num));

  auto graph_feature = graph.graph_feature();
  auto node_types = graph.node_types();
  auto edge_types = graph.edge_types();
  VLOG(0) << "got " << edge_types.size() << " edge types in total";
  feat_id_map.resize(node_types.size());
  for (int k = 0; k < edge_types.size(); k++) {
    VLOG(0) << "in initialize: get a edge_type " << edge_types[k];
    edge_to_id[edge_types[k]] = k;
    id_to_edge.push_back(edge_types[k]);
  }
  feat_name.resize(node_types.size());
  feat_shape.resize(node_types.size());
  feat_dtype.resize(node_types.size());
  VLOG(0) << "got " << node_types.size() << " node types in total";
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
  for (size_t k = 0; k < edge_shards.size(); k++) {
    for (size_t i = 0; i < shard_num_per_server; i++) {
      edge_shards[k].push_back(new GraphShard());
    }
  }
  node_weight[1].resize(id_to_feature.size());
  feature_shards.resize(id_to_feature.size());
  for (size_t k = 0; k < feature_shards.size(); k++) {
    for (size_t i = 0; i < shard_num_per_server; i++) {
      feature_shards[k].push_back(new GraphShard());
    }
  }

  return 0;
}

void GraphTable::init_worker_poll(int gpu_num) {
  _cpu_worker_pool.resize(gpu_num);
  for (int i = 0; i < gpu_num; i++) {
    _cpu_worker_pool[i].reset(new ::ThreadPool(16));
  }
}

void GraphTable::build_graph_total_keys() {
  VLOG(0) << "begin insert edge to graph_total_keys";
  // build node embedding id
  std::vector<std::vector<uint64_t>> keys;
  this->get_node_embedding_ids(1, &keys);
  graph_total_keys_.insert(
      graph_total_keys_.end(), keys[0].begin(), keys[0].end());

  VLOG(0) << "finish insert edge to graph_total_keys";
}

void GraphTable::build_graph_type_keys() {
  VLOG(0) << "begin build_graph_type_keys";
  graph_type_keys_.clear();
  graph_type_keys_.resize(this->feature_to_id.size());

  int cnt = 0;
  for (auto &it : this->feature_to_id) {
    auto node_idx = it.second;
    std::vector<std::vector<uint64_t>> keys;
    this->get_all_id(GraphTableType::FEATURE_TABLE, node_idx, 1, &keys);
    type_to_index_[node_idx] = cnt;
    graph_type_keys_[cnt++] = std::move(keys[0]);
  }
  VLOG(0) << "finish build_graph_type_keys";

  VLOG(0) << "begin insert feature into graph_total_keys";
  // build feature embedding id
  for (auto &it : this->feature_to_id) {
    auto node_idx = it.second;
    std::vector<std::vector<uint64_t>> keys;
    this->get_all_feature_ids(
        GraphTableType::FEATURE_TABLE, node_idx, 1, &keys);
    graph_total_keys_.insert(
        graph_total_keys_.end(), keys[0].begin(), keys[0].end());
  }
  VLOG(0) << "finish insert feature into graph_total_keys";
}

}  // namespace distributed
};  // namespace paddle

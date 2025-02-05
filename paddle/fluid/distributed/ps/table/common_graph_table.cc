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

#include <ctime>

#include <algorithm>
#include <chrono>
#include <set>
#include <sstream>
#include <tuple>

#include "paddle/common/flags.h"
#include "paddle/fluid/distributed/common/utils.h"
#include "paddle/fluid/distributed/ps/table/graph/graph_node.h"
#include "paddle/fluid/framework/fleet/fleet_wrapper.h"
#include "paddle/fluid/framework/fleet/heter_ps/graph_gpu_wrapper.h"
#if defined(PADDLE_WITH_HETERPS) && defined(PADDLE_WITH_PSCORE)
#include "paddle/fluid/framework/fleet/ps_gpu_wrapper.h"
#endif
#include "paddle/fluid/framework/io/fs.h"
#include "paddle/phi/core/generator.h"
#include "paddle/phi/core/platform/timer.h"
#include "paddle/utils/string/printf.h"
#include "paddle/utils/string/string_helper.h"

COMMON_DECLARE_bool(graph_load_in_parallel);
COMMON_DECLARE_bool(graph_get_neighbor_id);
COMMON_DECLARE_int32(gpugraph_storage_mode);
COMMON_DECLARE_uint64(gpugraph_slot_feasign_max_num);
COMMON_DECLARE_bool(graph_metapath_split_opt);
COMMON_DECLARE_double(graph_neighbor_size_percent);

PHI_DEFINE_EXPORTED_bool(graph_edges_split_only_by_src_id,
                         false,
                         "multi-node split edges only by src id");
PHI_DEFINE_EXPORTED_string(
    graph_edges_split_mode,
    "hard",
    "graph split split, optional: [dbh,hard,fennel,none], default:hard");
PHI_DEFINE_EXPORTED_bool(graph_edges_split_debug,
                         false,
                         "graph split by debug");
PHI_DEFINE_EXPORTED_int32(graph_edges_debug_node_id, 0, "graph debug node id");
PHI_DEFINE_EXPORTED_int32(graph_edges_debug_node_num,
                          2,
                          "graph debug node num");

namespace paddle::distributed {

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

::paddle::framework::GpuPsCommGraphFea GraphTable::make_gpu_ps_graph_fea(
    int gpu_id, std::vector<uint64_t> &node_ids, int slot_num) {
  size_t shard_num = 64;
  std::vector<std::vector<uint64_t>> bags(shard_num);
  std::vector<uint64_t> feature_array[shard_num];
  std::vector<uint8_t> slot_id_array[shard_num];
  std::vector<uint64_t> node_id_array[shard_num];
  std::vector<::paddle::framework::GpuPsFeaInfo> node_fea_info_array[shard_num];
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
        ::paddle::framework::GpuPsFeaInfo x;
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

  ::paddle::framework::GpuPsCommGraphFea res;
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

::paddle::framework::GpuPsCommGraphFloatFea
GraphTable::make_gpu_ps_graph_float_fea(int gpu_id,
                                        std::vector<uint64_t> &node_ids,
                                        int float_slot_num) {
  size_t shard_num = 64;
  std::vector<std::vector<uint64_t>> bags(shard_num);
  std::vector<float> feature_array[shard_num];
  std::vector<uint8_t> slot_id_array[shard_num];
  std::vector<uint64_t> node_id_array[shard_num];
  std::vector<::paddle::framework::GpuPsFeaInfo> node_fea_info_array[shard_num];
  for (size_t i = 0; i < shard_num; i++) {
    auto predsize = node_ids.size() / shard_num;
    bags[i].reserve(predsize * 1.2);
    feature_array[i].reserve(predsize * 1.2 * float_slot_num);
    slot_id_array[i].reserve(predsize * 1.2 * float_slot_num);
    node_id_array[i].reserve(predsize * 1.2);
    node_fea_info_array[i].reserve(predsize * 1.2);
  }

  for (auto x : node_ids) {
    int location = x % shard_num;
    bags[location].push_back(x);
  }

  std::vector<std::future<int>> tasks;

  for (size_t i = 0; i < bags.size(); i++) {
    if (bags[i].size() > 0) {
      tasks.push_back(_cpu_worker_pool[gpu_id]->enqueue([&, i, this]() -> int {
        uint64_t node_id;
        ::paddle::framework::GpuPsFeaInfo x;
        // std::vector<uint64_t> feature_ids;
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
            for (int k = 0; k < float_slot_num; ++k) {
              auto float_feature_size =
                  v->get_float_feature(k, feature_array[i], slot_id_array[i]);
              total_feature_size += float_feature_size;
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

  tasks.clear();

  ::paddle::framework::GpuPsCommGraphFloatFea res;
  uint64_t tot_len = 0;
  for (size_t i = 0; i < shard_num; i++) {
    tot_len += feature_array[i].size();
  }
  VLOG(1) << "Loaded float feature table on cpu, float feature_list_size["
          << tot_len << "] node_ids_size[" << node_ids.size() << "]";
  res.init_on_cpu(tot_len, (unsigned int)node_ids.size(), float_slot_num);
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

::paddle::framework::GpuPsCommGraph GraphTable::make_gpu_ps_graph(
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
  std::vector<::paddle::framework::GpuPsNodeInfo> info_array[task_pool_size_];
  std::vector<uint64_t> edge_array[task_pool_size_];  // edge id list

  // get edge weight
  std::vector<half> weight_array[task_pool_size_];  // neighbor weight list

  for (size_t i = 0; i < bags.size(); i++) {
    if (bags[i].size() > 0) {
      tasks.push_back(_shards_task_pool[i]->enqueue([&, i, this]() -> int {
        node_array[i].resize(bags[i].size());
        info_array[i].resize(bags[i].size());
        edge_array[i].reserve(bags[i].size());

        if (is_weighted_) {
          weight_array[i].reserve(bags[i].size());
        }

        for (size_t j = 0; j < bags[i].size(); j++) {
          auto node_id = bags[i][j];
          node_array[i][j] = node_id;
          Node *v = find_node(GraphTableType::EDGE_TABLE, idx, node_id);
          if (v != nullptr) {
            info_array[i][j].neighbor_offset = edge_array[i].size();
            info_array[i][j].neighbor_size = v->get_neighbor_size();
            for (size_t k = 0; k < v->get_neighbor_size(); k++) {
              edge_array[i].push_back(v->get_neighbor_id(k));
              if (is_weighted_) {
                weight_array[i].push_back(v->get_neighbor_weight(k));
              }
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

  ::paddle::framework::GpuPsCommGraph res;
  res.init_on_cpu(tot_len, ids.size(), is_weighted_);
  int64_t offset = 0, ind = 0;
  for (int i = 0; i < task_pool_size_; i++) {
    for (size_t j = 0; j < node_array[i].size(); j++) {
      res.node_list[ind] = node_array[i][j];
      res.node_info_list[ind] = info_array[i][j];
      res.node_info_list[ind++].neighbor_offset += offset;
    }
    for (size_t j = 0; j < edge_array[i].size(); j++) {
      res.neighbor_list[offset + j] = edge_array[i][j];

      if (is_weighted_) {
        res.weight_list[offset + j] = weight_array[i][j];
      }
    }
    offset += edge_array[i].size();
  }
  return res;
}

paddle::framework::GpuPsCommRankFea GraphTable::make_gpu_ps_rank_fea(
    int gpu_id) {
  paddle::framework::GpuPsCommRankFea res;
  if (edge_node_rank_.empty()) {
    return res;
  }
  std::vector<size_t> node_num_vec(shard_num_per_server, 0);
  auto rank_nodes = edge_node_rank_.get_rank_nodes();
  // 遍历 rank_nodes[i][shard_num]，分8份，分配到 res
  std::vector<std::future<size_t>> tasks;

  auto mutexes = new std::mutex[shard_num_per_server];
  for (int i = 0; i < node_num_; i++) {
    for (size_t shard_id = 0; shard_id < shard_num_per_server; shard_id++) {
      tasks.push_back(_cpu_worker_pool[gpu_id]->enqueue(
          [i, gpu_id, shard_id, &rank_nodes, &node_num_vec, &mutexes]()
              -> size_t {
            auto &rank_node = rank_nodes[i][shard_id];
            size_t start = 0;
            for (auto it = rank_node.begin(); it != rank_node.end(); it++) {
              if (gpu_id == static_cast<int>(*it % 8)) {
                start++;
              }
            }
            mutexes[shard_id].lock();
            node_num_vec[shard_id] += start;
            mutexes[shard_id].unlock();
            return start;
          }));
    }
  }
  size_t all_size = 0;
  for (size_t i = 0; i < tasks.size(); i++) {
    all_size += tasks[i].get();
  }
  res.init_on_cpu(all_size);
  tasks.clear();
  size_t ind = 0;
  for (size_t shard_id = 0; shard_id < shard_num_per_server; shard_id++) {
    tasks.push_back(_cpu_worker_pool[gpu_id]->enqueue(
        [&, shard_id, ind, gpu_id, this]() -> size_t {
          auto start = ind;
          for (int i = 0; i < node_num_; i++) {
            auto &rank_node = rank_nodes[i][shard_id];
            for (auto it = rank_node.begin(); it != rank_node.end(); it++) {
              if (gpu_id == static_cast<int>((*it) % 8)) {
                res.node_list[start] = *it;
                res.rank_list[start++] = i;
              }
            }
          }
          return 0;
        }));
    ind += node_num_vec[shard_id];
  }
  for (size_t i = 0; i < tasks.size(); i++) tasks[i].get();
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

void GraphTable::release_graph() {
  // Before releasing graph, prepare for sampling ids and embedding keys.
  build_graph_type_keys();

  if (FLAGS_gpugraph_storage_mode ==
      ::paddle::framework::GpuGraphStorageMode::WHOLE_HBM) {
    build_graph_total_keys();
  }
  // clear graph
  if (FLAGS_gpugraph_storage_mode == ::paddle::framework::GpuGraphStorageMode::
                                         MEM_EMB_FEATURE_AND_GPU_GRAPH ||
      FLAGS_gpugraph_storage_mode == ::paddle::framework::GpuGraphStorageMode::
                                         SSD_EMB_AND_MEM_FEATURE_GPU_GRAPH) {
    clear_edge_shard();
  } else {
    clear_graph();
  }
}

void GraphTable::release_graph_edge() {
  if (FLAGS_gpugraph_storage_mode ==
      ::paddle::framework::GpuGraphStorageMode::WHOLE_HBM) {
    build_graph_total_keys();
  }
  clear_edge_shard();
}

void GraphTable::release_graph_node() {
  build_graph_type_keys();
  if (FLAGS_graph_metapath_split_opt) {
    clear_feature_shard();
  } else {
    if (FLAGS_gpugraph_storage_mode !=
            ::paddle::framework::GpuGraphStorageMode::
                MEM_EMB_FEATURE_AND_GPU_GRAPH &&
        FLAGS_gpugraph_storage_mode !=
            ::paddle::framework::GpuGraphStorageMode::
                SSD_EMB_AND_MEM_FEATURE_GPU_GRAPH) {
      clear_feature_shard();
    } else {
      merge_feature_shard();
      feature_shrink_to_fit();
    }
  }
}

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
  auto paths = ::paddle::string::split_string<std::string>(path, ";");
  int64_t count = 0;
  std::string sample_type = "random";
  for (auto path : paths) {
    std::ifstream file(path);
    std::string line;
    while (std::getline(file, line)) {
      VLOG(0) << "get a line from file " << line;
      auto values = ::paddle::string::split_string<std::string>(line, "\t");
      count++;
      if (values.size() < 2) continue;
      auto src_id = std::stoll(values[0]);
      auto dist_ids =
          ::paddle::string::split_string<std::string>(values[1], ";");
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

void GraphTable::graph_partition(bool is_edge) {
  std::string mode = FLAGS_graph_edges_split_mode;
  if (mode == "dbh" || mode == "DBH") {
    VLOG(0) << "Graph partitioning DBH";
    if (is_edge) {
      dbh_graph_edge_partition();
    } else {
      dbh_graph_feature_partition();
    }
    VLOG(0) << "Graph partitioning DBH Done";
  } else if (mode == "hard" || mode == "HARD") {
    VLOG(0) << "Graph partitioning Hard Hash Split";
    if (is_edge) {
      stat_graph_edge_info(0);
    }
  } else if (strncasecmp(mode.c_str(), "fennel", 6) == 0) {
    if (is_edge) {
      fennel_graph_edge_partition();
    } else {
      fennel_graph_feature_partition();
    }
  } else {
    // TODO(danleifeng): Graph partitioning other method.
    VLOG(0) << "Unknown graph partitioning mode " << mode;
  }
}

void GraphTable::dbh_graph_edge_partition() {
  VLOG(0) << "start to process dbh edge shard";
  std::vector<std::vector<GraphShard *>> tmp_edge_shards;
  tmp_edge_shards.resize(edge_shards.size());
  for (size_t k = 0; k < edge_shards.size(); k++) {
    for (size_t i = 0; i < shard_num_per_server; i++) {
      tmp_edge_shards[k].push_back(new GraphShard());
    }
  }

  // all edges
  std::vector<std::future<int>> tasks;
  for (size_t idx = 0; idx < id_to_edge.size(); idx++) {
    tasks.push_back(_shards_task_pool[idx % task_pool_size_]->enqueue(
        [&, idx, this]() -> int {
          auto node_type =
              paddle::string::split_string<std::string>(id_to_edge[idx], "2");
          std::vector<int> src_edge_ids;
          std::vector<int> dest_edge_ids;
          for (size_t k = 0; k < id_to_edge.size(); k++) {
            if (id_to_edge[k] == id_to_edge[idx]) {
              VLOG(2) << "continue, edge_type:" << id_to_edge[k];
              continue;
            }
            auto edge_type_list =
                paddle::string::split_string<std::string>(id_to_edge[k], "2");
            if (node_type[0] == edge_type_list[0]) {
              src_edge_ids.push_back(k);
            }
            if (node_type[1] == edge_type_list[0]) {
              dest_edge_ids.push_back(k);
            }
          }
          int src_fea_idx = node_type_str_to_node_types_idx[node_type[0]];
          std::vector<std::future<int>> shard_tasks;
          for (size_t part_id = 0; part_id < shard_num; ++part_id) {
            shard_tasks.push_back(
                load_node_edge_task_pool->enqueue([&,
                                                   part_id,
                                                   idx,
                                                   src_fea_idx,
                                                   node_type,
                                                   src_edge_ids,
                                                   dest_edge_ids,
                                                   this]() -> int {
                  auto &shards = edge_shards[idx][part_id]->get_bucket();
                  for (auto node : shards) {
                    uint64_t id = node->get_id();
                    // 由于节点id的度计算需要到不同类型的边表中去查找
                    size_t src_degree = node->get_neighbor_size();
                    for (size_t edge_type_id = 0;
                         edge_type_id < src_edge_ids.size();
                         edge_type_id++) {
                      Node *src_node = find_node(GraphTableType::EDGE_TABLE,
                                                 src_edge_ids[edge_type_id],
                                                 id);
                      if (src_node == nullptr) {
                        VLOG(3) << "src_node " << id << " from type"
                                << src_edge_ids[edge_type_id] << "not found";
                      } else {
                        src_degree += src_node->get_neighbor_size();
                      }
                    }
                    for (size_t n_i = 0; n_i < node->get_neighbor_size();
                         ++n_i) {
                      auto d_id = node->get_neighbor_id(n_i);
                      auto is_weighted = node->get_is_weighted();
                      auto weight = node->get_neighbor_weight(n_i);
                      size_t dest_degree = 0;
                      for (size_t dst_type_id = 0;
                           dst_type_id < dest_edge_ids.size();
                           dst_type_id++) {
                        Node *dst_node = find_node(GraphTableType::EDGE_TABLE,
                                                   dest_edge_ids[dst_type_id],
                                                   d_id);
                        if (dst_node == nullptr) {
                          VLOG(3) << "dst_node " << d_id << " from type"
                                  << dest_edge_ids[dst_type_id] << "not found";
                        } else {
                          dest_degree += dst_node->get_neighbor_size();
                        }
                      }
                      if (src_degree <= dest_degree) {
                        // 每台机器只保存%hash的一部分id到tmp_edge_shards
                        if (is_key_for_self_rank(id)) {
                          VLOG(5) << "Add src; neighbor id " << d_id
                                  << " degree: " << dest_degree << "; src id "
                                  << id << " degree: " << src_degree;
                          auto new_node =
                              tmp_edge_shards[idx][part_id]->add_graph_node(id);
                          if (new_node != NULL) {
                            new_node->build_edges(is_weighted);
                            new_node->add_edge(d_id, weight);
                          }
                        }
                      } else {
                        // hash到dest_id对应的机器上
                        if (is_key_for_self_rank(d_id)) {
                          VLOG(5) << "Add dest; neighbor id " << d_id
                                  << " degree: " << dest_degree << "; src id "
                                  << id << " degree: " << src_degree;
                          auto new_node =
                              tmp_edge_shards[idx][part_id]->add_graph_node(id);
                          if (new_node != NULL) {
                            new_node->build_edges(is_weighted);
                            new_node->add_edge(d_id, weight);
                          }
                        }
                      }
                    }
                  }
                  return 0;
                }));
          }
          for (size_t j = 0; j < shard_tasks.size(); j++) {
            shard_tasks[j].get();
          }
          return 0;
        }));
  }
  for (size_t j = 0; j < tasks.size(); j++) {
    tasks[j].get();
  }
  // 替换原来的shards
  clear_edge_shard();
  edge_shards = std::move(tmp_edge_shards);
  std::vector<std::vector<uint64_t>> all_edge_keys;
  unique_all_edge_keys_.clear();
  VLOG(1) << "begin to get all_edge_keys";
  this->get_all_id(GraphTableType::EDGE_TABLE, 1, &all_edge_keys);
  VLOG(1) << "end to get all_edge_keys, size:" << all_edge_keys[0].size();
  unique_all_edge_keys_.insert(all_edge_keys[0].begin(),
                               all_edge_keys[0].end());
  VLOG(1) << "insert unique_all_edge_keys_ done, size:"
          << unique_all_edge_keys_.size();
  VLOG(0) << "end to process dbh edge shard";
}

void GraphTable::dbh_graph_feature_partition() {
  VLOG(0) << "start to process dbh feature shard";
  std::vector<std::future<int>> tasks;
  for (auto &it : this->node_type_str_to_node_types_idx) {
    auto node_types_idx = it.second;
    for (size_t i = 0; i < shard_num_per_server; i++) {
      tasks.push_back(load_node_edge_task_pool->enqueue(
          [&, node_types_idx, i, this]() -> int {
            std::vector<uint64_t> remove_keys;
            auto &shards = feature_shards[node_types_idx][i]->get_bucket();
            for (auto node : shards) {
              uint64_t id = node->get_id();
              // 在边表里的key以及hash对应的key保留，其余删除
              if (!is_key_for_self_rank(id) &&
                  (unique_all_edge_keys_.find(id) ==
                   unique_all_edge_keys_.end())) {
                remove_keys.push_back(id);
              }
            }
            for (auto &key : remove_keys) {
              feature_shards[node_types_idx][i]->delete_node(key);
            }
            return 0;
          }));
    }
    for (size_t i = 0; i < tasks.size(); ++i) {
      tasks[i].wait();
    }
    unique_all_edge_keys_.clear();
    VLOG(0) << "end to process dbh feature shard";
  }
}
// query all ids rank
void GraphTable::query_all_ids_rank(const size_t &total,
                                    const uint64_t *ids,
                                    uint32_t *ranks) {
  std::vector<std::future<size_t>> wait_tasks;
  size_t step =
      static_cast<size_t>((total + load_thread_num_ - 1) / load_thread_num_);
  for (size_t start = 0; start < total; start = start + step) {
    size_t end = (start + step > total) ? total : (start + step);
    wait_tasks.push_back(
        load_node_edge_task_pool->enqueue([this, &ids, &ranks, start, end]() {
          size_t cnt = 0;
          for (size_t k = start; k < end; ++k) {
            int rank = edge_node_rank_.find(ids[k]);
            if (rank < 0) {
              rank = partition_key_for_rank(ids[k]);
              ++cnt;
            }
            ranks[k] = rank;
          }
          return cnt;
        }));
  }
  // all
  size_t hash_count = 0;
  for (auto &t : wait_tasks) {
    hash_count += t.get();
  }
  VLOG(0) << "query total keys=" << total << ", hash count=" << hash_count;
}
void GraphTable::fennel_graph_edge_partition() {
  VLOG(0) << "start to process fennel2 edge shard";
  std::vector<std::future<size_t>> wait_tasks;
  robin_hood::unordered_flat_map<uint64_t, std::vector<Node *>>
      neighbor_nodes[shard_num_per_server];
  // 聚合所有边表关系
  for (size_t part_id = 0; part_id < shard_num_per_server; ++part_id) {
    wait_tasks.push_back(
        load_node_edge_task_pool->enqueue([this, part_id, &neighbor_nodes]() {
          size_t cnt = 0;
          auto &n_nodes = neighbor_nodes[part_id];
          for (size_t idx = 0; idx < edge_shards.size(); ++idx) {
            auto &shard = edge_shards[idx][part_id];
            auto &nodes = shard->get_bucket();
            for (auto &node : nodes) {
              auto it = n_nodes.find(node->get_id());
              if (it != n_nodes.end()) {
                it->second.push_back(node);
              } else {
                std::vector<Node *> vec;
                vec.push_back(node);
                n_nodes.emplace(node->get_id(), vec);
                ++cnt;
              }
            }
          }
          return cnt;
        }));
  }
  // 等待聚合线程结束
  size_t total_node = 0;
  for (auto &t : wait_tasks) {
    total_node += t.get();
  }
  size_t load_limit =
      static_cast<size_t>((total_node + node_num_ - 1) / node_num_);

  const double gamma = 1.5;
  const double alpha =
      total_node * pow(node_num_, (gamma - 1)) / pow(total_node, gamma);
  VLOG(0) << "gather total edge node count=" << total_node
          << ", load_limit:" << load_limit << ", alpha: " << alpha;
  // init,每个子图插入一个节点, 记录每台机器子图已有的节点set
  edge_node_rank_.clear();
  edge_node_rank_.init(node_num_, shard_num_per_server);
  for (size_t part_id = 0; part_id < shard_num_per_server; ++part_id) {
    edge_node_rank_.rehash(part_id, neighbor_nodes[part_id].size());
  }
  // 获取与邻居的边交集数量
  auto get_inter_cost = [this, &neighbor_nodes](
                            const uint64_t &key,
                            const std::vector<Node *> &nodes,
                            std::vector<int> *inter_cost) {
    bool find = false;
    for (auto &node : nodes) {
      for (size_t n_i = 0; n_i < node->get_neighbor_size(); ++n_i) {
        uint64_t d_id = node->get_neighbor_id(n_i);
        int rank = edge_node_rank_.find(d_id);
        if (rank >= 0) {
          ++(*inter_cost)[rank];
          find = true;
        }
      }
    }
    return find;
  };
  // 根据score计算当前所属机器
  auto get_rank_by_score =
      [this, gamma, alpha, &load_limit, &neighbor_nodes, get_inter_cost](
          const uint64_t &key) -> int {
    thread_local std::vector<int> inter_cost(node_num_, 0);
    auto &shard = neighbor_nodes[key % shard_num_per_server];
    auto it = shard.find(key);
    if (it == shard.end()) {
      //      VLOG(0) << "get rank by score not found key=" << key;
      return -1;
    }
    if (!get_inter_cost(key, it->second, &inter_cost)) {
      return -1;
    }
    int index = -1;
    //    double max_score = INT_MIN;
    int max_score = INT_MIN;
    for (int i = 0; i < node_num_; ++i) {
      // 计算最大score所在的机器
      //      double score = 0.0;
      //      if (edge_node_rank_.nodes_num(i) < load_limit) {
      //        score = inter_cost[i] - alpha * gamma *
      //        pow(edge_node_rank_.nodes_num(i), gamma - 1);
      //      } else {
      //        score = - alpha * gamma * pow(edge_node_rank_.nodes_num(i),
      //        gamma - 1);
      //      }
      auto &score = inter_cost[i];
      if (score > max_score) {
        max_score = score;
        index = i;
      }
      inter_cost[i] = 0;
    }
    PADDLE_ENFORCE_GT(
        max_score,
        0,
        common::errors::InvalidArgument("max_score should be greater than 0"));
    return index;
  };
  // 查找关系最远点作为起点
  auto find_farthest_start_node = [this, &neighbor_nodes, get_inter_cost](
                                      const int &rank_id,
                                      const int &max_step) -> uint64_t {
    uint64_t key = 0xffffffffffffffffL;
    int min_inter_cost = INT_MAX;
    int step = 0;
    std::vector<int> inter_cost(node_num_, 0);
    for (size_t shard_id = 0; shard_id < shard_num_per_server; ++shard_id) {
      auto &n_nodes = neighbor_nodes[shard_id];
      if (n_nodes.empty()) {
        continue;
      }
      for (auto it = n_nodes.begin(); it != n_nodes.end(); ++it) {
        if (edge_node_rank_.find(it->first) >= 0) {
          continue;
        }
        get_inter_cost(it->first, it->second, &inter_cost);
        for (int k = 0; k < node_num_; ++k) {
          if (k == rank_id) {
            inter_cost[k] = 0;
            continue;
          }
          if (min_inter_cost > inter_cost[k]) {
            min_inter_cost = inter_cost[k];
            key = it->first;
          }
          inter_cost[k] = 0;
        }
        ++step;
        if (step >= max_step) {
          break;
        }
      }
      if (step >= max_step) {
        break;
      }
    }
    PADDLE_ENFORCE_NE(key,
                      0xffffffffffffffffL,
                      common::errors::InvalidArgument(
                          "key should not be 0xffffffffffffffffL"));
    return key;
  };
  // 其它结点都添加完成，剩余的点就直接放到这个机器上面
  auto add_all_left_node = [this,
                            &neighbor_nodes](const int &rank_id) -> size_t {
    size_t cnt = 0;
    for (size_t shard_id = 0; shard_id < shard_num_per_server; ++shard_id) {
      auto &n_nodes = neighbor_nodes[shard_id];
      if (n_nodes.empty()) {
        continue;
      }
      for (auto it = n_nodes.begin(); it != n_nodes.end(); ++it) {
        if (edge_node_rank_.find(it->first) >= 0) {
          continue;
        }
        edge_node_rank_.insert(it->first, rank_id);
        ++cnt;
      }
    }
    return cnt;
  };
  typedef robin_hood::unordered_flat_set<uint64_t> UniqueSet;
  typedef std::deque<uint64_t> NodeQueue;
  // 添加所有邻居入队列
  auto add_neighbor_to_queue = [this, &neighbor_nodes](const uint64_t &key,
                                                       NodeQueue *queue,
                                                       UniqueSet *unique) {
    auto &shard = neighbor_nodes[key % shard_num_per_server];
    if (shard.empty()) {
      return;
    }
    auto it = shard.find(key);
    if (it == shard.end()) {
      return;
    }
    unique->insert(key);
    for (auto &node : it->second) {
      for (size_t n_i = 0; n_i < node->get_neighbor_size(); ++n_i) {
        uint64_t d_id = node->get_neighbor_id(n_i);
        if (unique->find(d_id) != unique->end()) {
          continue;
        }
        unique->insert(d_id);
        queue->push_back(d_id);
      }
    }
    shard.erase(it);
  };

  UniqueSet keys_unique;
  keys_unique.rehash(total_node);
  NodeQueue node_queue[node_num_];  // NOLINT

  size_t cnt = 0;
  size_t start_print = 0;
  size_t max_node = 0;
  int max_step = 10;
  std::vector<size_t> rank_nums(node_num_, 0);
  while (cnt < total_node) {
    int need_proc_node = node_num_;
    for (int rank_id = 0; rank_id < node_num_; ++rank_id) {
      auto &num = edge_node_rank_.nodes_num(rank_id);
      if (num >= load_limit) {
        --need_proc_node;
        continue;
      }
      // 如果发现某一个机器上面点明显变多就让下一个节点处理
      if (num > max_node) {
        max_node = num;
        continue;
      }
      auto &queue = node_queue[rank_id];
      if (queue.empty()) {
        // 如果其它节点都处理完，这个节点就直接收集所有其它点就可以
        if (need_proc_node == 1) {
          cnt += add_all_left_node(rank_id);
        } else {
          uint64_t key = find_farthest_start_node(rank_id, max_step);
          if (max_step == 10 || max_step == 9) {
            VLOG(0) << "rank id=" << rank_id << ", add first node key=" << key;
          }
          if (max_step > 0) {
            --max_step;
          }
          ++rank_nums[rank_id];
          edge_node_rank_.insert(key, rank_id);
          ++cnt;
          add_neighbor_to_queue(key, &queue, &keys_unique);
        }
        continue;
      }

      uint64_t key = queue.front();
      queue.pop_front();

      int rank = get_rank_by_score(key);
      if (rank >= 0) {
        edge_node_rank_.insert(key, rank);
        ++cnt;
        add_neighbor_to_queue(key, &node_queue[rank], &keys_unique);
      }
    }
    if (cnt - start_print > 10000000) {
      for (int i = 0; i < node_num_; i++) {
        VLOG(0) << "current total nodes count=" << cnt << ", edge_node_ids["
                << i << "] :" << edge_node_rank_.nodes_num(i)
                << ", queue size: " << node_queue[i].size()
                << ", first size: " << rank_nums[i]
                << ", keys_unique: " << keys_unique.size();
      }
      start_print = cnt;
    }
  }
  // end split nodes
  for (int i = 0; i < node_num_; i++) {
    VLOG(0) << " edge_node_ids[" << i << "] :" << edge_node_rank_.nodes_num(i);
  }
  filter_graph_edge_nodes();
  VLOG(0) << "end to process fennel new edge shard";
}
void GraphTable::filter_graph_edge_nodes() {
  VLOG(0) << "begin filter graph edge nodes";
  // 过滤不属于自己边表信息
  std::vector<std::future<std::pair<size_t, size_t>>> shard_tasks;
  std::vector<size_t> total_edge_count(shard_num_per_server, 0);
  std::vector<size_t> edge_before_count(edge_shards.size(), 0);

  std::vector<std::vector<size_t>> rank_edge_count;
  std::vector<std::vector<size_t>> cross_edge_count;
  rank_edge_count.resize(node_num_);
  cross_edge_count.resize(node_num_);
  for (int i = 0; i < node_num_; ++i) {
    rank_edge_count[i].resize(shard_num_per_server, 0);
    cross_edge_count[i].resize(shard_num_per_server, 0);
  }
  // 获取边是否跨机统计
  auto get_cross_edge_count = [this](const int &rank_id, Node *node) {
    size_t cnt = 0;
    for (size_t i = 0; i < node->get_neighbor_size(); ++i) {
      uint64_t nid = node->get_neighbor_id(i);
      if (edge_node_rank_.find(nid) != rank_id) {
        ++cnt;
      }
    }
    return cnt;
  };

  for (size_t idx = 0; idx < edge_shards.size(); ++idx) {
    for (size_t part_id = 0; part_id < shard_num_per_server; ++part_id) {
      edge_before_count[idx] += edge_shards[idx][part_id]->get_size();
    }
    for (size_t part_id = 0; part_id < shard_num_per_server; ++part_id) {
      shard_tasks.push_back(load_node_edge_task_pool->enqueue(
          [this,
           part_id,
           idx,
           &total_edge_count,
           &rank_edge_count,
           &cross_edge_count,
           get_cross_edge_count]() -> std::pair<size_t, size_t> {
            size_t total_cnt = 0;
            size_t remove_cnt = 0;

            auto &shard = edge_shards[idx][part_id];
            auto &nodes = shard->get_bucket();
            std::vector<uint64_t> remove_ids;
            for (auto &node : nodes) {
              total_edge_count[part_id] += node->get_neighbor_size();
              auto nid = node->get_id();
              // 节点插入到对应的机器shard中
              int rank = edge_node_rank_.find(nid);
              if (rank != node_id_) {
                remove_ids.push_back(nid);
                ++remove_cnt;
              }
              // 统计各节点边分布情况
              cross_edge_count[rank][part_id] +=
                  get_cross_edge_count(rank, node);
              rank_edge_count[rank][part_id] += node->get_neighbor_size();

              ++total_cnt;
            }
            // delete
            for (auto &id : remove_ids) {
              shard->delete_node(id);
            }
            return {total_cnt, remove_cnt};
          }));
    }
  }
  size_t all_cut_size = 0;
  size_t all_node_size = 0;
  for (size_t j = 0; j < shard_tasks.size(); j++) {
    auto res = shard_tasks[j].get();
    all_node_size += res.first;
    all_cut_size += res.second;
  }
  // 分类型打印边的分布情况
  for (size_t idx = 0; idx < edge_shards.size(); ++idx) {
    size_t total = 0;
    for (size_t part_id = 0; part_id < shard_num_per_server; ++part_id) {
      total += edge_shards[idx][part_id]->get_size();
    }
    VLOG(0) << "filter edge idx=" << idx
            << ", total edge nodes=" << edge_before_count[idx]
            << ", left edge nodes count=" << total;
  }
  // 统计所有边以及跨节点情况
  size_t total_edge_cnt = 0;
  std::vector<size_t> rank_edge_cnt(node_num_, 0);
  std::vector<size_t> cross_edge_cnt(node_num_, 0);
  for (size_t part_id = 0; part_id < shard_num_per_server; ++part_id) {
    total_edge_cnt += total_edge_count[part_id];
    for (int rank = 0; rank < node_num_; ++rank) {
      rank_edge_cnt[rank] += rank_edge_count[rank][part_id];
      cross_edge_cnt[rank] += cross_edge_count[rank][part_id];
    }
  }
  VLOG(0) << "total edge count: " << total_edge_cnt;
  for (int rank = 0; rank < node_num_; ++rank) {
    VLOG(0) << "rank=" << rank << ", edge count: " << rank_edge_cnt[rank]
            << ", cross count: " << cross_edge_cnt[rank] << ", cross rate: "
            << double(cross_edge_cnt[rank]) / double(rank_edge_cnt[rank]);
  }
  VLOG(0) << "total edge start node size: " << all_node_size
          << ", cut start node size: " << all_cut_size << ", end filter edge";
}
void GraphTable::fennel_graph_feature_partition() {
  VLOG(0) << "start to process fennel feature shard";
  std::vector<std::future<std::pair<size_t, size_t>>> tasks;
  for (size_t node_idx = 0; node_idx < feature_shards.size(); ++node_idx) {
    for (size_t i = 0; i < shard_num_per_server; ++i) {
      tasks.push_back(load_node_edge_task_pool->enqueue(
          [&, node_idx, i, this]() -> std::pair<size_t, size_t> {
            std::vector<uint64_t> remove_keys;
            auto &shard = feature_shards[node_idx][i];
            size_t total = shard->get_size();
            auto &nodes = shard->get_bucket();
            for (auto node : nodes) {
              uint64_t id = node->get_id();
              int rank = edge_node_rank_.find(id);
              if (rank < 0) {  // single node
                if (!is_key_for_self_rank(id)) {
                  remove_keys.push_back(id);
                }
              } else if (rank != node_id_) {
                remove_keys.push_back(id);
              }
            }
            for (auto &key : remove_keys) {
              shard->delete_node(key);
            }
            return {total, remove_keys.size()};
          }));
    }
  }
  size_t total = 0;
  size_t del_cnt = 0;
  for (auto &task : tasks) {
    auto it = task.get();
    total += it.first;
    del_cnt += it.second;
  }
  for (size_t node_idx = 0; node_idx < feature_shards.size(); ++node_idx) {
    size_t total = 0;
    for (size_t i = 0; i < shard_num_per_server; ++i) {
      total += feature_shards[node_idx][i]->get_size();
    }
    VLOG(0) << "node idx=" << node_idx << ", filter left node count=" << total;
  }
  VLOG(0) << "total count=" << total << ", delete count=" << del_cnt
          << ", end to process fennel feature shard";
}
void GraphTable::stat_graph_edge_info(int type) {
  std::vector<std::future<std::pair<size_t, size_t>>> shard_tasks;
  // 获取边是否跨机统计
  std::function<size_t(Node *)> get_cross_edge_count = nullptr;
  if (type == 1) {
    // 贪心
    get_cross_edge_count = [this](Node *node) {
      size_t cnt = 0;
      for (size_t i = 0; i < node->get_neighbor_size(); ++i) {
        uint64_t nid = node->get_neighbor_id(i);
        if (edge_node_rank_.find(nid) != node_id_) {
          ++cnt;
        }
      }
      return cnt;
    };
  } else {
    // 硬拆
    get_cross_edge_count = [this](Node *node) {
      size_t cnt = 0;
      for (size_t i = 0; i < node->get_neighbor_size(); ++i) {
        uint64_t nid = node->get_neighbor_id(i);
        if (is_key_for_self_rank(nid)) {
          ++cnt;
        }
      }
      return cnt;
    };
  }
  for (size_t idx = 0; idx < edge_shards.size(); ++idx) {
    for (size_t part_id = 0; part_id < shard_num_per_server; ++part_id) {
      shard_tasks.push_back(load_node_edge_task_pool->enqueue(
          [this, part_id, idx, get_cross_edge_count]()
              -> std::pair<size_t, size_t> {
            size_t total_cnt = 0;
            size_t cross_cnt = 0;
            auto &nodes = edge_shards[idx][part_id]->get_bucket();
            for (auto &node : nodes) {
              // 统计各节点边分布情况
              total_cnt += node->get_neighbor_size();
              cross_cnt += get_cross_edge_count(node);
            }
            return {total_cnt, cross_cnt};
          }));
    }
  }
  size_t all_node_size = 0;
  size_t all_cross_size = 0;
  for (auto &t : shard_tasks) {
    auto res = t.get();
    all_node_size += res.first;
    all_cross_size += res.second;
  }
  VLOG(0) << "rank=" << node_id_ << ", edge count: " << all_node_size
          << ", cross count: " << all_cross_size
          << ", cross rate: " << double(all_cross_size) / double(all_node_size);
}

#endif  // PADDLE_WITH_HETERPS

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

void GraphTable::clear_edge_shard() {
  VLOG(0) << "begin clear edge shard";
  std::vector<std::future<int>> tasks;
  for (auto &type_shards : edge_shards) {
    for (auto &shard : type_shards) {
      tasks.push_back(load_node_edge_task_pool->enqueue([&shard]() -> int {
        delete shard;
        return 0;
      }));
    }
  }
  for (auto &task : tasks) task.get();
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
      tasks.push_back(load_node_edge_task_pool->enqueue([&shard]() -> int {
        delete shard;
        return 0;
      }));
    }
  }
  for (auto &task : tasks) task.get();
  for (auto &shards : feature_shards) {
    shards.clear();
    for (size_t i = 0; i < shard_num_per_server; i++) {
      shards.push_back(new GraphShard());
    }
  }
  VLOG(0) << "finish clear feature shard";
}

void GraphTable::clear_node_shard() {
  VLOG(0) << "begin clear node shard";
  std::vector<std::future<int>> tasks;
  for (auto &type_shards : node_shards) {
    for (auto &shard : type_shards) {
      tasks.push_back(load_node_edge_task_pool->enqueue([&shard]() -> int {
        delete shard;
        return 0;
      }));
    }
  }
  for (size_t i = 0; i < tasks.size(); i++) tasks[i].get();
  for (auto &shards : node_shards) {
    shards.clear();
    for (size_t i = 0; i < shard_num_per_server; i++) {
      shards.push_back(new GraphShard());
    }
  }
  VLOG(0) << "finish clear node shard";
}

void GraphTable::clear_graph() {
  VLOG(0) << "begin clear_graph";
  clear_edge_shard();
  clear_feature_shard();
  VLOG(0) << "finish clear_graph";
}

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
    if (batch[i].empty()) continue;
    tasks.push_back(
        _shards_task_pool[i]->enqueue([&shards, &batch, i, this]() -> int {
          for (auto &p : batch[i]) {
            size_t index = p.first % this->shard_num - this->shard_start;
            shards[index]->add_graph_node(p.first)->build_edges(p.second);
          }
          return 0;
        }));
  }
  for (auto &task : tasks) task.get();
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
    if (batch[i].empty()) continue;
    tasks.push_back(
        _shards_task_pool[i]->enqueue([&shards, &batch, i, this]() -> int {
          for (auto &p : batch[i]) {
            size_t index = p % this->shard_num - this->shard_start;
            shards[index]->delete_node(p);
          }
          return 0;
        }));
  }
  for (auto &task : tasks) task.get();
  return 0;
}

void GraphShard::clear() {
  for (auto &item : bucket) {
    delete item;
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

FeatureNode *GraphShard::add_feature_node(uint64_t id,
                                          bool is_overlap,
                                          int float_fea_num) {
  if (node_location.find(id) == node_location.end()) {
    node_location[id] = bucket.size();
    if (float_fea_num > 0) {
      bucket.push_back(new FloatFeatureNode(id));
    } else {
      bucket.push_back(new FeatureNode(id));
    }
    return reinterpret_cast<FeatureNode *>(bucket[node_location[id]]);
  }
  if (is_overlap) {
    return reinterpret_cast<FeatureNode *>(bucket[node_location[id]]);
  }
  return nullptr;
}

void GraphShard::add_neighbor(uint64_t id, uint64_t dst_id, float weight) {
  find_node(id)->add_edge(dst_id, weight);
}

Node *GraphShard::find_node(uint64_t id) {
  auto iter = node_location.find(id);
  return iter == node_location.end() ? nullptr : bucket[iter->second];
}

GraphTable::~GraphTable() {  // NOLINT
#if defined(PADDLE_WITH_HETERPS) && defined(PADDLE_WITH_PSCORE)
  clear_graph();
#endif
}

int32_t GraphTable::Load(const std::string &path, const std::string &param) {
  bool load_edge = (param[0] == 'e');
  bool load_node = (param[0] == 'n');
  if (load_edge) {
    bool reverse_edge = (param[1] == '<');
    std::string edge_type = param.substr(2);
    auto ret = this->load_edges(path, reverse_edge, edge_type);
    if (ret.first != 0) {
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
  auto etype_split = ::paddle::string::split_string<std::string>(etype, "2");
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
      ::paddle::string::split_string<std::string>(type2files, ",");
  if (type2files_split.empty()) {
    return -1;
  }
  for (auto one_type2file : type2files_split) {
    auto one_type2file_split =
        ::paddle::string::split_string<std::string>(one_type2file, ":");
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
    const std::vector<bool> &is_reverse_edge_map,
    bool use_weight) {
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

  std::vector<std::future<std::pair<uint64_t, uint64_t>>> tasks;
  for (size_t i = 0; i < total_len; i++) {
    tasks.push_back(_shards_task_pool[i % task_pool_size_]->enqueue(
        [&, i, this]() -> std::pair<uint64_t, uint64_t> {
          uint64_t edge_all_count = 0;
          uint64_t edge_valid_count = 0;
          std::string etype_path = edge_to_edgedir[etypes[i]];
          bool only_load_reverse_edge = false;
          if (!reverse) {
            only_load_reverse_edge = (i < is_reverse_edge_map.size())
                                         ? is_reverse_edge_map[i]
                                         : false;
          }
          if (only_load_reverse_edge) {
            VLOG(1) << "only_load_reverse_edge is True, etype[" << etypes[i]
                    << "], file_path[" << etype_path << "]";
          } else {
            VLOG(1) << "only_load_reverse_edge is False, etype[" << etypes[i]
                    << "], file_path[" << etype_path << "]";
          }
          auto etype_path_list = ::paddle::framework::localfs_list(etype_path);
          std::string etype_path_str;
          if (part_num > 0 &&
              part_num < static_cast<int>(etype_path_list.size())) {
            std::vector<std::string> sub_etype_path_list(
                etype_path_list.begin(), etype_path_list.begin() + part_num);
            etype_path_str =
                ::paddle::string::join_strings(sub_etype_path_list, delim);
          } else {
            etype_path_str =
                ::paddle::string::join_strings(etype_path_list, delim);
          }
          if (!only_load_reverse_edge) {
            auto pair_res =
                this->load_edges(etype_path_str, false, etypes[i], use_weight);
            edge_all_count += pair_res.first;
            edge_valid_count += pair_res.second;
            if (reverse) {
              std::string r_etype = get_inverse_etype(etypes[i]);
              pair_res =
                  this->load_edges(etype_path_str, true, r_etype, use_weight);
              edge_all_count += pair_res.first;
              edge_valid_count += pair_res.second;
            }
          } else {
            auto pair_res =
                this->load_edges(etype_path_str, true, etypes[i], use_weight);
            edge_all_count += pair_res.first;
            edge_valid_count += pair_res.second;
          }
          return {edge_all_count, edge_valid_count};
        }));
  }
  uint64_t edge_all_counts = 0;
  uint64_t edge_valid_counts = 0;
  for (size_t i = 0; i < tasks.size(); i++) {
    auto res = tasks[i].get();
    edge_all_counts += res.first;
    edge_valid_counts += res.second;
  }
  tasks.clear();
  VLOG(0) << " end load edges, edges now storage in CPU Memory";

#if defined(PADDLE_WITH_HETERPS) && defined(PADDLE_WITH_PSCORE)
  if (node_num_ > 1) {
    graph_partition(true);
  }
#endif
  VLOG(0) << "load this node edge total edge count: " << edge_valid_counts
          << " , all node edge count:" << edge_all_counts;

  // record all start node id
  std::vector<std::future<int>> tasks1;
  for (size_t idx = 0; idx < edge_shards.size(); ++idx) {
    for (size_t part_id = 0; part_id < shard_num_per_server; ++part_id) {
      tasks1.push_back(
          load_node_edge_task_pool->enqueue([this, idx, part_id]() {
            std::vector<std::vector<uint64_t>> all_keys;
            edge_shards[idx][part_id]->get_all_id(&all_keys, 1);
            int cnt = all_keys[0].size();
            edge_shards_keys_[idx][part_id] = std::move(all_keys[0]);
            all_keys[0].clear();
            return cnt;
          }));
    }
  }
  size_t total_cnt = 0;
  for (auto &t : tasks1) {
    total_cnt += t.get();
  }
  tasks.clear();
  tasks1.clear();
  VLOG(0) << "Node id= " << node_id_
          << " Parallel Partition the graph successfully!";
  VLOG(0) << "load all etypes total edge nodes count=" << total_cnt;

  return 0;
}

int32_t GraphTable::parse_node_and_load(std::string ntype2files,
                                        std::string graph_data_local_path,
                                        int part_num,
                                        bool load_slot) {
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
  auto npath_list = ::paddle::framework::localfs_list(npath);
  std::string npath_str;
  if (part_num > 0 && part_num < static_cast<int>(npath_list.size())) {
    std::vector<std::string> sub_npath_list(npath_list.begin(),
                                            npath_list.begin() + part_num);
    npath_str = ::paddle::string::join_strings(sub_npath_list, delim);
  } else {
    npath_str = ::paddle::string::join_strings(npath_list, delim);
  }

  if (ntypes.empty()) {
    VLOG(0) << "node_type not specified, nothing will be loaded ";
    return 0;
  }
  if (FLAGS_graph_load_in_parallel) {
    int ret = this->load_nodes(npath_str, "", load_slot);
    if (ret != 0) {
      VLOG(0) << "Fail to load nodes, path[" << npath << "]";
      return -1;
    }
  } else {
    for (size_t j = 0; j < ntypes.size(); j++) {
      int ret = this->load_nodes(npath_str, ntypes[j], load_slot);
      if (ret != 0) {
        VLOG(0) << "Fail to load nodes, path[" << npath << "], ntypes["
                << ntypes[j] << "]";
        return -1;
      }
    }
  }
  // fix node edge nodes
  fix_feature_node_shards(load_slot);
#if defined(PADDLE_WITH_HETERPS) && defined(PADDLE_WITH_PSCORE)
  if (node_num_ > 1) {
    graph_partition(false);
  }
#endif
  VLOG(0) << " end load nodes, nodes now storage in CPU Memory";
  return 0;
}
void GraphTable::fix_feature_node_shards(bool load_slot) {
  auto &shards = (load_slot) ? feature_shards : node_shards;
  VLOG(0) << "begin fix " << ((load_slot) ? "feature " : "")
          << "node type count=" << shards.size()
          << ", edge count=" << edge_shards.size();
  std::vector<std::future<std::tuple<size_t, size_t, size_t>>> tasks;
  for (size_t idx = 0; idx < shards.size(); ++idx) {
    for (size_t j = 0; j < shards[idx].size(); ++j) {
      tasks.push_back(load_node_edge_task_pool->enqueue(
          [this, idx, j, load_slot]() -> std::tuple<size_t, size_t, size_t> {
            size_t cnt = 0;
            size_t edge_node_cnt = 0;
            auto &features =
                (load_slot) ? feature_shards[idx][j] : node_shards[idx][j];
            for (auto edge_idx : nodeid_to_edgeids_[idx]) {
              auto &shard_keys = edge_shards_keys_[edge_idx][j];
              edge_node_cnt += shard_keys.size();
              if (shard_keys.empty()) {
                continue;
              }
              for (auto &key : shard_keys) {
                if (features->find_node(key) != nullptr) {
                  continue;
                }
                features->add_feature_node(key, false, 0);
                ++cnt;
              }
              // clear free memory
              shard_keys.clear();
              shard_keys.shrink_to_fit();
            }
            size_t total = features->get_size();
            VLOG(5) << "fix total edge node count=" << edge_node_cnt
                    << ", total feature node count=" << total
                    << ", node_types_idx=" << idx << ", shard id=" << j
                    << ", add_count=" << cnt;
            return std::make_tuple(total, cnt, idx);
          }));
    }
  }
  size_t total = 0;
  size_t add_cnt = 0;
  std::vector<size_t> add_cnt_vec;
  std::vector<size_t> total_vec;
  add_cnt_vec.assign(shards.size(), 0);
  total_vec.assign(shards.size(), 0);
  for (auto &t : tasks) {
    auto pair = t.get();
    total += std::get<0>(pair);
    add_cnt += std::get<1>(pair);
    int node_types_idx = std::get<2>(pair);
    total_vec[node_types_idx] += std::get<0>(pair);
    add_cnt_vec[node_types_idx] += std::get<1>(pair);
  }
  VLOG(0) << "fix node count=" << total << ", add count=" << add_cnt
          << ", with slot=" << load_slot;
  for (size_t i = 0; i < shards.size(); ++i) {
    VLOG(1) << "node_type[" << node_types_[i] << "] node_type_idx[" << i
            << "] orig[" << total_vec[i] - add_cnt_vec[i] << "] add_cnt["
            << add_cnt_vec[i] << "] total[" << total_vec[i] << "]";
  }
}
int32_t GraphTable::load_node_and_edge_file(
    std::string etype2files,
    std::string ntype2files,
    std::string graph_data_local_path,
    int part_num,
    bool reverse,
    const std::vector<bool> &is_reverse_edge_map,
    bool use_weight) {
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
            auto etype_path_list =
                ::paddle::framework::localfs_list(etype_path);
            std::string etype_path_str;
            if (part_num > 0 &&
                part_num < static_cast<int>(etype_path_list.size())) {
              std::vector<std::string> sub_etype_path_list(
                  etype_path_list.begin(), etype_path_list.begin() + part_num);
              etype_path_str =
                  ::paddle::string::join_strings(sub_etype_path_list, delim);
            } else {
              etype_path_str =
                  ::paddle::string::join_strings(etype_path_list, delim);
            }
            if (!only_load_reverse_edge) {
              this->load_edges(etype_path_str, false, etypes[i], use_weight);
              if (reverse) {
                std::string r_etype = get_inverse_etype(etypes[i]);
                this->load_edges(etype_path_str, true, r_etype, use_weight);
              }
            } else {
              this->load_edges(etype_path_str, true, etypes[i], use_weight);
            }
          } else {
            std::string npath = node_to_nodedir[ntypes[0]];
            auto npath_list = ::paddle::framework::localfs_list(npath);
            std::string npath_str;
            if (part_num > 0 &&
                part_num < static_cast<int>(npath_list.size())) {
              std::vector<std::string> sub_npath_list(
                  npath_list.begin(), npath_list.begin() + part_num);
              npath_str = ::paddle::string::join_strings(sub_npath_list, delim);
            } else {
              npath_str = ::paddle::string::join_strings(npath_list, delim);
            }

            if (ntypes.empty()) {
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
              for (auto &ntype : ntypes) {
                int ret = this->load_nodes(npath_str, ntype);
                if (ret != 0) {
                  VLOG(0) << "Fail to load nodes, path[" << npath_str
                          << "], ntypes[" << ntype << "]";
                  return -1;
                }
              }
            }
          }
          return 0;
        }));
  }
  for (auto &task : tasks) task.get();
  if (is_parse_node_fail_) {
    VLOG(0) << "Fail to load node_and_edge_file";
    return -1;
  }
  return 0;
}

bool GraphTable::is_key_for_self_rank(const uint64_t &id) {
#if defined(PADDLE_WITH_HETERPS) && defined(PADDLE_WITH_GLOO) && \
    defined(PADDLE_WITH_PSCORE)
  thread_local auto ps_wrapper =
      ::paddle::framework::PSGPUWrapper::GetInstance();
  if (FLAGS_graph_edges_split_debug && ps_wrapper->GetRankNum() == 1) {
    return (static_cast<int>((id / 8) % node_num_) == node_id_);
  }
  return ps_wrapper->IsKeyForSelfRank(id);
#else
  return true;
#endif
}
int GraphTable::partition_key_for_rank(const uint64_t &key) {
#if defined(PADDLE_WITH_HETERPS) && defined(PADDLE_WITH_GLOO) && \
    defined(PADDLE_WITH_PSCORE)
  thread_local auto ps_wrapper =
      ::paddle::framework::PSGPUWrapper::GetInstance();
  if (FLAGS_graph_edges_split_debug && ps_wrapper->GetRankNum() == 1) {
    return static_cast<int>((key / 8) % node_num_);
  }
  return ps_wrapper->PartitionKeyForRank(key);
#else
  return 0;
#endif
}
std::pair<uint64_t, uint64_t> GraphTable::parse_node_file(
    const std::string &path,
    const std::string &node_type,
    int idx,
    bool load_slot) {
  std::ifstream file(path);
  std::string line;
  uint64_t local_count = 0;
  uint64_t local_valid_count = 0;

  int num = 0;
  std::vector<::paddle::string::str_ptr> vals;
  size_t n = node_type.length();
  while (std::getline(file, line)) {
    if (strncmp(line.c_str(), node_type.c_str(), n) != 0) {
      continue;
    }
    vals.clear();
    num = ::paddle::string::split_string_ptr(
        line.c_str() + n + 1, line.length() - n - 1, '\t', &vals);
    if (num == 0) {
      continue;
    }
    uint64_t id = std::strtoul(vals[0].ptr, NULL, 10);
    if (FLAGS_graph_edges_split_mode == "hard" ||
        FLAGS_graph_edges_split_mode == "HARD") {
      if (!is_key_for_self_rank(id)) {
        VLOG(3) << "id " << id << " not matched, node_id: " << node_id_
                << " , node_num:" << node_num_;
        continue;
      }
    }
    size_t shard_id = id % shard_num;
    if (shard_id >= shard_end || shard_id < shard_start) {
      VLOG(4) << "will not load " << id << " from " << path
              << ", please check id distribution";
      continue;
    }
    local_count++;

    size_t index = shard_id - shard_start;
    int slot_fea_num = 0;
    if (feat_name.size() > 0) slot_fea_num = feat_name[idx].size();
    int float_fea_num = 0;
    if (float_feat_id_map.size() > 0) {
      float_fea_num = float_feat_id_map[idx].size();
    }
    if (load_slot) {
      auto node = feature_shards[idx][index]->add_feature_node(
          id, false, float_fea_num);
      if (node != NULL) {
        if (slot_fea_num > 0) node->set_feature_size(slot_fea_num);
        if (float_fea_num > 0) node->set_float_feature_size(float_fea_num);
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
    } else {
      node_shards[idx][index]->add_feature_node(id, false, float_fea_num);
    }
    local_valid_count++;
  }
  VLOG(2) << "node_type[" << node_type << "] loads " << local_count
          << " nodes from filepath->" << path;
  return {local_count, local_valid_count};
}

std::pair<uint64_t, uint64_t> GraphTable::parse_node_file_parallel(
    const std::string &path, bool load_slot) {
  std::ifstream file(path);
  std::string line;
  uint64_t local_count = 0;
  uint64_t local_valid_count = 0;
  int idx = 0;

  auto path_split = ::paddle::string::split_string<std::string>(path, "/");
  auto path_name = path_split[path_split.size() - 1];

  int num = 0;
  std::vector<::paddle::string::str_ptr> vals;
  size_t last_shard_id = 0;

  while (std::getline(file, line)) {
    vals.clear();
    num = ::paddle::string::split_string_ptr(
        line.c_str(), line.length(), '\t', &vals);
    if (vals.empty()) {
      continue;
    }
    std::string parse_node_type = vals[0].to_string();
    auto it = node_type_str_to_node_types_idx.find(parse_node_type);
    if (it == node_type_str_to_node_types_idx.end()) {
      VLOG(1) << parse_node_type << "type error, please check, line[" << line
              << "] file[" << path << "]";
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
    if (shard_id != last_shard_id && last_shard_id != 0) {
      VLOG(0) << "Maybe node file hasn't been sharded, file[" << path
              << "] shard_id[" << shard_id << "] last_shard_id["
              << last_shard_id << "], exit";
      VLOG(0) << "auto_shard in config should be set as True";
      is_parse_node_fail_ = true;
      return {0, 0};
    }

    local_count++;
    if (FLAGS_graph_edges_split_mode == "hard" ||
        FLAGS_graph_edges_split_mode == "HARD") {
      if (!is_key_for_self_rank(id)) {
        VLOG(3) << "id " << id << " not matched, node_id: " << node_id_
                << " , node_num:" << node_num_;
        continue;
      }
    }
    size_t index = shard_id - shard_start;
    int float_fea_num = 0;
    if (float_feat_id_map.size() > 0) {
      float_fea_num = float_feat_id_map[idx].size();
    }
    if (load_slot) {
      auto node = feature_shards[idx][index]->add_feature_node(
          id, false, float_fea_num);
      if (node != NULL) {
        for (int i = 2; i < num; ++i) {
          auto &v = vals[i];
          int ret = parse_feature(idx, v.ptr, v.len, node);
          if (ret != 0) {
            VLOG(0) << "Fail to parse feature, node_id[" << id << "] shard_idx["
                    << index << "] fea_type_id[" << idx << "]";
            is_parse_node_fail_ = true;
            return {0, 0};
          }
        }
      }
    } else {
      node_shards[idx][index]->add_feature_node(id, false, float_fea_num);
    }
    local_valid_count++;
    last_shard_id = shard_id;
  }
  VLOG(2) << local_valid_count << "/" << local_count << " nodes from filepath->"
          << path;
  return {local_count, local_valid_count};
}

// // TODO(danleifeng): opt load all node_types in once reading
int32_t GraphTable::load_nodes(const std::string &path,
                               std::string node_type,
                               bool load_slot) {
  auto paths = paddle::string::split_string<std::string>(path, ";");
  uint64_t count = 0;
  uint64_t valid_count = 0;
  int idx = 0;
  if (FLAGS_graph_load_in_parallel) {
    if (node_type.empty()) {
      VLOG(0) << "Begin GraphTable::load_nodes(), will load all node_type once";
    }
    std::vector<std::future<std::pair<uint64_t, uint64_t>>> tasks;
    for (size_t i = 0; i < paths.size(); i++) {
      tasks.push_back(load_node_edge_task_pool->enqueue(
          [&, i, this]() -> std::pair<uint64_t, uint64_t> {
            return parse_node_file_parallel(paths[i], load_slot);
          }));
    }
    for (size_t i = 0; i < tasks.size(); i++) {
      auto res = tasks[i].get();
      count += res.first;
      valid_count += res.second;
    }
  } else {
    VLOG(0) << "Begin GraphTable::load_nodes() node_type[" << node_type << "]";
    if (node_type.empty()) {
      VLOG(0) << "node_type not specified, loading edges to "
              << id_to_feature[0] << " part";
    } else {
      if (node_type_str_to_node_types_idx.find(node_type) ==
          node_type_str_to_node_types_idx.end()) {
        VLOG(0) << "node_type " << node_type
                << " is not defined, nothing will be loaded";
        return 0;
      }
      idx = node_type_str_to_node_types_idx[node_type];
    }
    for (auto path : paths) {
      VLOG(2) << "Begin GraphTable::load_nodes(), path[" << path << "]";
      auto res = parse_node_file(path, node_type, idx, load_slot);
      count += res.first;
      valid_count += res.second;
    }
  }
  if (is_parse_node_fail_) {
    VLOG(0) << "Fail to load nodes, path[" << paths[0] << ".."
            << paths[paths.size() - 1] << "] node_type[" << node_type << "]";
    return -1;
  }

  VLOG(0) << valid_count << "/" << count << " nodes in node_type[" << node_type
          << "] are loaded successfully!";
  return 0;
}

int32_t GraphTable::build_sampler(int idx, std::string sample_type) {
  for (auto &shard : edge_shards[idx]) {
    auto bucket = shard->get_bucket();
    for (auto item : bucket) {
      item->build_sampler(sample_type);
    }
  }
  return 0;
}

std::pair<uint64_t, uint64_t> GraphTable::parse_edge_file(
    const std::string &path, int idx, bool reverse, bool use_weight) {
  is_weighted_ = use_weight;
  std::ifstream file(path);
  std::string line;
  uint64_t local_count = 0;
  uint64_t local_valid_count = 0;
  uint64_t part_num = 0;
  if (FLAGS_graph_load_in_parallel) {
    auto path_split = ::paddle::string::split_string<std::string>(path, "/");
    auto part_name_split = ::paddle::string::split_string<std::string>(
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
    local_count++;
    if (src_shard_id >= shard_end || src_shard_id < shard_start) {
      VLOG(0) << "will not load " << src_id << " from " << path
              << ", please check id distribution";
      continue;
    }
    if (FLAGS_graph_edges_split_mode == "hard" ||
        FLAGS_graph_edges_split_mode == "HARD") {
      // only keep hash(src_id) = hash(dst_id) = node_id edges
      // src id
      if (!is_key_for_self_rank(src_id)) {
        VLOG(3) << " node num :" << src_id
                << " not split into node_id_:" << node_id_
                << " node_num:" << node_num_;
        continue;
      }
      // dst id
      if (!FLAGS_graph_edges_split_only_by_src_id &&
          !is_key_for_self_rank(dst_id)) {
        VLOG(3) << " dest node num :" << dst_id
                << " will not add edge, node_id_:" << node_id_
                << " node_num:" << node_num_;
        continue;
      }
    }

    float weight = 1;
    size_t last = line.find_last_of('\t');
    if (start != last) {
      weight = std::stof(&line[last + 1]);
    }

    if (src_shard_id >= shard_end || src_shard_id < shard_start) {
      VLOG(4) << "will not load " << src_id << " from " << path
              << ", please check id distribution";
      continue;
    }
    size_t index = src_shard_id - shard_start;
    auto node = edge_shards[idx][index]->add_graph_node(src_id);
    if (node != NULL) {
      node->build_edges(is_weighted_);
      node->add_edge(dst_id, weight);
    }

    local_valid_count++;
  }
  VLOG(2) << local_valid_count << "/" << local_count
          << " edges are loaded from filepath->" << path;
  return {local_count, local_valid_count};
}

std::pair<uint64_t, uint64_t> GraphTable::load_edges(
    const std::string &path,
    bool reverse_edge,
    const std::string &edge_type,
    bool use_weight) {
#ifdef PADDLE_WITH_HETERPS
  if (search_level == 2) total_memory_cost = 0;
#endif
  int idx = 0;
  if (edge_type.empty()) {
    VLOG(0) << "edge_type not specified, loading edges to " << id_to_edge[0]
            << " part";
  } else {
    if (edge_to_id.find(edge_type) == edge_to_id.end()) {
      VLOG(0) << "edge_type " << edge_type
              << " is not defined, nothing will be loaded";
      return {0, 0};
    }
    idx = edge_to_id[edge_type];
  }

  auto paths = ::paddle::string::split_string<std::string>(path, ";");
  uint64_t count = 0;
  uint64_t valid_count = 0;

  VLOG(0) << "Begin GraphTable::load_edges() edge_type[" << edge_type << "]";
  if (FLAGS_graph_load_in_parallel) {
    std::vector<std::future<std::pair<uint64_t, uint64_t>>> tasks;
    for (size_t i = 0; i < paths.size(); i++) {
      tasks.push_back(load_node_edge_task_pool->enqueue(
          [&, i, idx, this]() -> std::pair<uint64_t, uint64_t> {
            return parse_edge_file(paths[i], idx, reverse_edge, use_weight);
          }));
    }
    for (size_t j = 0; j < tasks.size(); j++) {
      auto res = tasks[j].get();
      count += res.first;
      valid_count += res.second;
    }
  } else {
    for (auto path : paths) {
      auto res = parse_edge_file(path, idx, reverse_edge, use_weight);
      count += res.first;
      valid_count += res.second;
    }
  }
  VLOG(0) << valid_count << "/" << count << " edge_type[" << edge_type
          << "] edges are loaded successfully";
  std::string edge_size = edge_type + ":" + std::to_string(valid_count);
  edge_type_size.push_back(edge_size);

#if defined(PADDLE_WITH_HETERPS) && defined(PADDLE_WITH_PSCORE)
  if (search_level == 2) {
    if (count > 0) {
      dump_edges_to_ssd(idx);
      VLOG(0) << "dumping edges to ssd, edge count is reset to 0";
      clear_graph(idx);
      count = 0;
    }
    return {0, 0};
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
      for (auto item : bucket) {
        item->build_sampler(sample_type);
      }
    }
  }

  return {count, valid_count};
}

Node *GraphTable::find_node(GraphTableType table_type, uint64_t id) {
  size_t shard_id = id % shard_num;
  if (shard_id >= shard_end || shard_id < shard_start) {
    return nullptr;
  }
  Node *node = nullptr;
  size_t index = shard_id - shard_start;
  auto &search_shards = table_type == GraphTableType::EDGE_TABLE ? edge_shards
                        : table_type == GraphTableType::FEATURE_TABLE
                            ? feature_shards
                            : node_shards;
  for (auto &search_shard : search_shards) {
    PADDLE_ENFORCE_NOT_NULL(search_shard[index],
                            common::errors::InvalidArgument(
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
  auto &search_shards =
      table_type == GraphTableType::EDGE_TABLE      ? edge_shards[idx]
      : table_type == GraphTableType::FEATURE_TABLE ? feature_shards[idx]
                                                    : node_shards[idx];
  PADDLE_ENFORCE_NOT_NULL(search_shards[index],
                          common::errors::InvalidArgument(
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
  auto &search_shards =
      table_type == GraphTableType::EDGE_TABLE      ? edge_shards[idx]
      : table_type == GraphTableType::FEATURE_TABLE ? feature_shards[idx]
                                                    : node_shards[idx];
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
  for (auto shard : shards) {
    total_size += shard->get_size();
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
    if (seq_id[i].empty()) continue;
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
#if defined(PADDLE_WITH_HETERPS) && defined(PADDLE_WITH_PSCORE)
              weight = node->get_neighbor_weight(x);
#else
              weight = 1.0;
#endif
              memcpy(buffer_addr + offset, &weight, Node::weight_size);
              offset += Node::weight_size;
            }
          }
        }
      }
      if (!sample_res.empty()) {
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
            [&shards, first, second, i, &res, &mutex]() -> size_t {
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
  for (auto &task : tasks) {
    task.get();
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
    std::vector<::paddle::string::str_ptr>::iterator strs_begin,
    std::vector<::paddle::string::str_ptr>::iterator strs_end,
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
  // Return (feat_id, bytes) if name are in this->feat_name, else return (-1,
  // "")
  thread_local std::vector<::paddle::string::str_ptr> fields;
  fields.clear();
  char c = slot_feature_separator_.at(0);
  ::paddle::string::split_string_ptr(feat_str, len, c, &fields);

  thread_local std::vector<::paddle::string::str_ptr> fea_fields;
  fea_fields.clear();
  c = feature_separator_.at(0);
  ::paddle::string::split_string_ptr(fields[1].ptr,
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
        VLOG(0) << "Fail to parse value, fea_type_id[" << idx << "] fea_str["
                << feat_str << "] len[" << len << "]";
        return -1;
      }
      return 0;
    } else if (dtype == "string") {
      string_vector_2_string(
          fea_fields.begin(), fea_fields.end(), ' ', fea_ptr);
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
    if (float_feat_id_map.size() > static_cast<size_t>(idx)) {
      auto float_it = float_feat_id_map[idx].find(name);
      if (float_it != float_feat_id_map[idx].end()) {
        int32_t id = float_it->second;
        std::string *fea_ptr = node->mutable_float_feature(id);
        std::string dtype = this->float_feat_dtype[idx][id];
        if (dtype == "float32") {
          int ret = FeatureNode::parse_value_to_bytes<float>(
              fea_fields.begin(), fea_fields.end(), fea_ptr);
          if (ret != 0) {
            VLOG(0) << "Fail to parse value";
            return -1;
          }
          return 0;
        }
        // else if (dtype == "float64") { // not used
        //  int ret = FeatureNode::parse_value_to_bytes<double>(
        //      fea_fields.begin(), fea_fields.end(), fea_ptr);
        //  if (ret != 0) {
        //    VLOG(0) << "Fail to parse value";
        //    return -1;
        //  }
        //  return 0;
        // }
      } else {
        VLOG(4) << "feature_name[" << name
                << "] is not in feat_id_map, ntype_id[" << idx
                << "] feat_id_map_size[" << feat_id_map.size() << "]";
      }
    }
  }
  return 0;
}
// thread safe shard vector merge
class MergeShardVector {
 public:
  MergeShardVector(std::vector<std::vector<uint64_t>> *output, int slice_num)
      : _shard_keys() {
    _slice_num = slice_num;
    _shard_keys = output;
    _shard_keys->resize(slice_num);
    _mutexes = new std::mutex[slice_num];
  }
  ~MergeShardVector() {
    if (_mutexes != nullptr) {
      delete[] _mutexes;
      _mutexes = nullptr;
    }
  }
  // merge shard keys
  void merge(const std::vector<std::vector<uint64_t>> &shard_keys) {
    // add to shard
    for (int shard_id = 0; shard_id < _slice_num; ++shard_id) {
      auto &dest = (*_shard_keys)[shard_id];
      auto &src = shard_keys[shard_id];

      _mutexes[shard_id].lock();
      dest.insert(dest.end(), src.begin(), src.end());
      _mutexes[shard_id].unlock();
    }
  }

 private:
  int _slice_num = 0;
  std::mutex *_mutexes = nullptr;
  std::vector<std::vector<uint64_t>> *_shard_keys;
};

int GraphTable::get_all_id(GraphTableType table_type,
                           int slice_num,
                           std::vector<std::vector<uint64_t>> *output) {
  MergeShardVector shard_merge(output, slice_num);
  auto &search_shards = table_type == GraphTableType::EDGE_TABLE ? edge_shards
                        : table_type == GraphTableType::FEATURE_TABLE
                            ? feature_shards
                            : node_shards;

  std::vector<std::future<size_t>> tasks;
  for (auto &search_shard : search_shards) {
    for (size_t j = 0; j < search_shard.size(); j++) {
      tasks.push_back(_shards_task_pool[j % task_pool_size_]->enqueue(
          [search_shard, j, slice_num, &shard_merge]() -> size_t {
            std::vector<std::vector<uint64_t>> shard_keys;
            size_t num = search_shard[j]->get_all_id(&shard_keys, slice_num);
            // add to shard
            shard_merge.merge(shard_keys);
            return num;
          }));
    }
  }
  for (auto &task : tasks) {
    task.wait();
  }
  return 0;
}

int GraphTable::get_all_neighbor_id(
    GraphTableType table_type,
    int slice_num,
    std::vector<std::vector<uint64_t>> *output) {
  MergeShardVector shard_merge(output, slice_num);
  auto &search_shards = table_type == GraphTableType::EDGE_TABLE ? edge_shards
                        : table_type == GraphTableType::FEATURE_TABLE
                            ? feature_shards
                            : node_shards;
  std::vector<std::future<size_t>> tasks;
  for (auto &search_shard : search_shards) {
    for (size_t j = 0; j < search_shard.size(); j++) {
      tasks.push_back(_shards_task_pool[j % task_pool_size_]->enqueue(
          [search_shard, j, slice_num, &shard_merge]() -> size_t {
            std::vector<std::vector<uint64_t>> shard_keys;
            size_t num =
                search_shard[j]->get_all_neighbor_id(&shard_keys, slice_num);
            // add to shard
            shard_merge.merge(shard_keys);
            return num;
          }));
    }
  }
  for (auto &task : tasks) {
    task.wait();
  }
  return 0;
}

int GraphTable::get_all_id(GraphTableType table_type,
                           int idx,
                           int slice_num,
                           std::vector<std::vector<uint64_t>> *output) {
  MergeShardVector shard_merge(output, slice_num);
  auto &search_shards =
      table_type == GraphTableType::EDGE_TABLE      ? edge_shards[idx]
      : table_type == GraphTableType::FEATURE_TABLE ? feature_shards[idx]
                                                    : node_shards[idx];
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
  for (auto &task : tasks) {
    task.wait();
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
  auto &search_shards =
      table_type == GraphTableType::EDGE_TABLE      ? edge_shards[idx]
      : table_type == GraphTableType::FEATURE_TABLE ? feature_shards[idx]
                                                    : node_shards[idx];
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
  for (auto &task : tasks) {
    task.wait();
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
  auto &search_shards =
      table_type == GraphTableType::EDGE_TABLE      ? edge_shards[idx]
      : table_type == GraphTableType::FEATURE_TABLE ? feature_shards[idx]
                                                    : node_shards[idx];
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
  for (auto &task : tasks) {
    task.wait();
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
  auto &search_shards =
      table_type == GraphTableType::EDGE_TABLE      ? edge_shards[idx]
      : table_type == GraphTableType::FEATURE_TABLE ? feature_shards[idx]
                                                    : node_shards[idx];
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
        [&search_shards, i, start, end, step, size]() -> std::vector<Node *> {
          return search_shards[i]->get_batch(start - size, end - size, step);
        }));
    start += count * step;
    total_size -= count;
    size += cur_size;
  }
  for (auto &task : tasks) {
    task.wait();
  }
  size = 0;
  std::vector<std::vector<Node *>> res;
  for (auto &task : tasks) {
    res.push_back(task.get());
    for (auto &item : res.back()) {
      size += item->get_size(need_feature);
    }
  }
  char *buffer_addr = new char[size];
  buffer.reset(buffer_addr);
  int index = 0;
  for (auto &items : res) {
    for (size_t j = 0; j < items.size(); j++) {
      items[j]->to_buffer(buffer_addr + index, need_feature);
      index += items[j]->get_size(need_feature);
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

std::string GraphTable::node_types_idx_to_node_type_str(int node_types_idx) {
  return node_types_[node_types_idx];
}

std::string GraphTable::index_to_node_type_str(int index) {
  int node_types_idx = index_to_type_[index];
  return node_types_idx_to_node_type_str(node_types_idx);
}

void GraphTable::load_node_weight(int type_id, int idx, std::string path) {
  auto paths = ::paddle::string::split_string<std::string>(path, ";");
  int64_t count = 0;
  auto &weight_map = node_weight[type_id][idx];
  for (auto path : paths) {
    std::ifstream file(path);
    std::string line;
    while (std::getline(file, line)) {
      auto values = ::paddle::string::split_string<std::string>(line, "\t");
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
    _db = ::paddle::distributed::RocksDBHandler::GetInstance();
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
  load_node_edge_task_pool.reset(new ::ThreadPool(load_thread_num_));

  auto graph_feature = graph.graph_feature();
  auto node_types = graph.node_types();
  node_types_.assign(node_types.begin(), node_types.end());
  auto edge_types = graph.edge_types();

#if defined(PADDLE_WITH_HETERPS) && defined(PADDLE_WITH_GLOO) && \
    defined(PADDLE_WITH_PSCORE)
  auto ps_wrapper = ::paddle::framework::PSGPUWrapper::GetInstance();
  node_id_ = ps_wrapper->GetRankId();
  node_num_ = ps_wrapper->GetRankNum();
  if (FLAGS_graph_edges_split_debug && node_num_ == 1) {
    node_num_ = FLAGS_graph_edges_debug_node_num;
    node_id_ = FLAGS_graph_edges_debug_node_id;
  }
#endif

  VLOG(0) << "got " << edge_types.size()
          << " edge types in total, rank id=" << node_id_
          << ", rank size=" << node_num_
          << ", graph_edges_split_only_by_src_id="
          << FLAGS_graph_edges_split_only_by_src_id;
  feat_id_map.resize(node_types.size());
  for (int k = 0; k < edge_types.size(); k++) {  // NOLINT
    VLOG(0) << "in initialize: get a edge_type " << edge_types[k];
    edge_to_id[edge_types[k]] = k;
    id_to_edge.push_back(edge_types[k]);
  }
  feat_name.resize(node_types.size());
  feat_shape.resize(node_types.size());
  feat_dtype.resize(node_types.size());
  VLOG(0) << "got " << node_types.size() << " node types in total";
  for (int k = 0; k < node_types.size(); k++) {
    node_type_str_to_node_types_idx[node_types[k]] = k;
    auto node_type = node_types[k];
    auto feature = graph_feature[k];
    id_to_feature.push_back(node_type);
    int feat_conf_size = static_cast<int>(feature.name().size());
    int feasign_idx = 0, float_idx = 0;
    for (int i = 0; i < feat_conf_size; i++) {
      // auto &f_name = common.attributes()[i];
      // auto &f_shape = common.dims()[i];
      // auto &f_dtype = common.params()[i];
      auto &f_name = feature.name()[i];
      auto &f_shape = feature.shape()[i];
      auto &f_dtype = feature.dtype()[i];
      if (f_dtype == "feasign" || f_dtype == "int64") {
        feat_name[k].push_back(f_name);
        feat_shape[k].push_back(f_shape);
        feat_dtype[k].push_back(f_dtype);
        feat_id_map[k][f_name] = feasign_idx++;
      } else if (f_dtype == "float32") {
        if (float_feat_id_map.size() < static_cast<size_t>(node_types.size())) {
          float_feat_name.resize(node_types.size());
          float_feat_shape.resize(node_types.size());
          float_feat_dtype.resize(node_types.size());
          float_feat_id_map.resize(node_types.size());
        }
        float_feat_name[k].push_back(f_name);
        float_feat_shape[k].push_back(f_shape);
        float_feat_dtype[k].push_back(f_dtype);
        float_feat_id_map[k][f_name] = float_idx++;
      }
      VLOG(0) << "init graph table feat conf name:" << f_name
              << " shape:" << f_shape << " dtype:" << f_dtype;
    }
  }
  nodeid_to_edgeids_.resize(node_type_str_to_node_types_idx.size());
  for (auto &obj : edge_to_id) {
    size_t pos = obj.first.find("2");
    PADDLE_ENFORCE_EQ((pos != std::string::npos),
                      true,
                      common::errors::InvalidArgument(
                          "The string does not contain the character '2'."));
    std::string nodetype = obj.first.substr(0, pos);
    auto it = node_type_str_to_node_types_idx.find(nodetype);
    PADDLE_ENFORCE_EQ(
        (it != node_type_str_to_node_types_idx.end()),
        true,
        common::errors::InvalidArgument(
            "Node type not found in node_type_str_to_node_types_idx."));
    nodeid_to_edgeids_[it->second].push_back(obj.second);
    VLOG(0) << "add edge [" << obj.first << "=" << obj.second << "] to ["
            << nodetype << "=" << it->second << "]";
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
#if defined(PADDLE_WITH_HETERPS) && defined(PADDLE_WITH_PSCORE)
  partitions.resize(id_to_edge.size());
#endif
  edge_shards_keys_.resize(id_to_edge.size());
  for (size_t k = 0; k < edge_shards.size(); k++) {
    edge_shards_keys_[k].resize(shard_num_per_server);
    for (size_t i = 0; i < shard_num_per_server; i++) {
      edge_shards[k].push_back(new GraphShard());
    }
  }
  node_weight[1].resize(id_to_feature.size());
  feature_shards.resize(id_to_feature.size());
  node_shards.resize(id_to_feature.size());
  for (size_t k = 0; k < feature_shards.size(); k++) {
    for (size_t i = 0; i < shard_num_per_server; i++) {
      feature_shards[k].push_back(new GraphShard());
      node_shards[k].push_back(new GraphShard());
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

void GraphTable::calc_edge_type_limit() {
  std::vector<uint64_t> graph_type_keys_;
  std::vector<int> graph_type_keys_neighbor_size_;
  std::vector<std::vector<int>> neighbor_size_array;
  neighbor_size_array.resize(task_pool_size_);

  int max_neighbor_size = 0;
  int neighbor_size_limit;
  size_t size_limit;
  double neighbor_size_percent = FLAGS_graph_neighbor_size_percent;
  for (auto &it : this->edge_to_id) {
    graph_type_keys_.clear();
    graph_type_keys_neighbor_size_.clear();
    for (int i = 0; i < task_pool_size_; i++) {
      neighbor_size_array[i].clear();
    }
    auto edge_type = it.first;
    auto edge_idx = it.second;
    std::vector<std::vector<uint64_t>> keys;
    this->get_all_id(GraphTableType::EDGE_TABLE, edge_idx, 1, &keys);
    graph_type_keys_ = std::move(keys[0]);

    std::vector<std::vector<uint64_t>> bags(task_pool_size_);
    for (int i = 0; i < task_pool_size_; i++) {
      auto predsize = graph_type_keys_.size() / task_pool_size_;
      bags[i].reserve(predsize * 1.2);
    }
    for (auto x : graph_type_keys_) {
      int location = x % task_pool_size_;
      bags[location].push_back(x);
    }

    std::vector<std::future<int>> tasks;
    for (size_t i = 0; i < bags.size(); i++) {
      if (bags[i].size() > 0) {
        tasks.push_back(
            _shards_task_pool[i]->enqueue([&, i, edge_idx, this]() -> int {
              neighbor_size_array[i].reserve(bags[i].size());
              for (size_t j = 0; j < bags[i].size(); j++) {
                auto node_id = bags[i][j];
                Node *v =
                    find_node(GraphTableType::EDGE_TABLE, edge_idx, node_id);
                if (v != nullptr) {
                  int neighbor_size = v->get_neighbor_size();
                  neighbor_size_array[i].push_back(neighbor_size);
                } else {
                  VLOG(0) << "node id:" << node_id
                          << ", not find in type: " << edge_idx;
                }
              }
              return 0;
            }));
      }
    }
    for (size_t i = 0; i < tasks.size(); i++) tasks[i].get();
    for (int i = 0; i < task_pool_size_; i++) {
      graph_type_keys_neighbor_size_.insert(
          graph_type_keys_neighbor_size_.end(),
          neighbor_size_array[i].begin(),
          neighbor_size_array[i].end());
    }
    std::sort(graph_type_keys_neighbor_size_.begin(),
              graph_type_keys_neighbor_size_.end());
    if (graph_type_keys_neighbor_size_.size() > 0) {
      max_neighbor_size =
          graph_type_keys_neighbor_size_[graph_type_keys_neighbor_size_.size() -
                                         1];
      size_limit =
          graph_type_keys_neighbor_size_.size() * neighbor_size_percent;
      if (size_limit < (graph_type_keys_neighbor_size_.size() - 1)) {
        neighbor_size_limit = graph_type_keys_neighbor_size_[size_limit];
      } else {
        neighbor_size_limit = max_neighbor_size;
      }
    } else {
      neighbor_size_limit = 0;
    }
    type_to_neighbor_limit_[edge_idx] = neighbor_size_limit;
    VLOG(0) << "edge_type: " << edge_type << ", edge_idx[" << edge_idx
            << "] max neighbor_size: " << max_neighbor_size
            << ", neighbor_size_limit: " << neighbor_size_limit;
  }
}

void GraphTable::build_graph_type_keys() {
  VLOG(0) << "begin build_graph_type_keys, feature size="
          << this->node_type_str_to_node_types_idx.size();
  graph_type_keys_.clear();
  graph_type_keys_.resize(this->node_type_str_to_node_types_idx.size());

  int cnt = 0;
  uint64_t total_key = 0;
  for (auto &it : this->node_type_str_to_node_types_idx) {
    auto node_types_idx = it.second;
    std::vector<std::vector<uint64_t>> keys;
    this->get_all_id(GraphTableType::FEATURE_TABLE, node_types_idx, 1, &keys);
    type_to_index_[node_types_idx] = cnt;
    index_to_type_[cnt] = node_types_idx;
    total_key += keys[0].size();
    VLOG(1) << "node_type[" << node_types_[node_types_idx]
            << "] node_types_idx[" << node_types_idx << "] index["
            << type_to_index_[node_types_idx] << "] graph_type_keys_[" << cnt
            << "]_size=" << keys[0].size() << " total_key[" << total_key << "]";
    graph_type_keys_[cnt++] = std::move(keys[0]);
  }
  VLOG(0) << "finish build_graph_type_keys";

  VLOG(0) << "begin insert feature into graph_total_keys, feature size="
          << this->node_type_str_to_node_types_idx.size();
  // build feature embedding id
  for (auto &it : this->node_type_str_to_node_types_idx) {
    auto node_types_idx = it.second;
    std::vector<std::vector<uint64_t>> keys;
    this->get_all_feature_ids(
        GraphTableType::FEATURE_TABLE, node_types_idx, 1, &keys);
    graph_total_keys_.insert(
        graph_total_keys_.end(), keys[0].begin(), keys[0].end());
  }
  VLOG(0)
      << "finish insert feature into graph_total_keys, feature embedding keys="
      << graph_total_keys_.size();
}

void GraphTable::build_node_iter_type_keys() {
  VLOG(0) << "enter build_node_iter_type_keys";
  graph_type_keys_.clear();
  graph_type_keys_.resize(this->node_type_str_to_node_types_idx.size());

  int cnt = 0;
  for (auto &it : this->node_type_str_to_node_types_idx) {
    auto node_types_idx = it.second;
    std::vector<std::vector<uint64_t>> keys;
    this->get_all_id(GraphTableType::NODE_TABLE, node_types_idx, 1, &keys);
    graph_type_keys_[cnt++] = std::move(keys[0]);
    VLOG(1) << "node_type[" << node_types_[node_types_idx]
            << "] node_types_idx[" << node_types_idx << "] index["
            << type_to_index_[node_types_idx] << "] graph_type_keys_num["
            << keys[0].size() << "]";
  }
  VLOG(0) << "finish build_node_iter_type_keys";
}

}  // namespace paddle::distributed

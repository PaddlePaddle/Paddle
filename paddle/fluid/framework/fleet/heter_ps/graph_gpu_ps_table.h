// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include <thrust/host_vector.h>

#include <chrono>

<<<<<<< HEAD
#include "paddle/fluid/distributed/ps/table/common_graph_table.h"
#include "paddle/fluid/framework/fleet/heter_ps/gpu_graph_node.h"
#include "paddle/fluid/framework/fleet/heter_ps/heter_comm.h"
=======
#include "heter_comm.h"
#include "paddle/fluid/distributed/ps/table/common_graph_table.h"
#include "paddle/fluid/framework/fleet/heter_ps/gpu_graph_node.h"
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
#include "paddle/fluid/framework/fleet/heter_ps/heter_comm_kernel.h"
#include "paddle/fluid/platform/enforce.h"
#ifdef PADDLE_WITH_HETERPS

DECLARE_double(gpugraph_hbm_table_load_factor);

namespace paddle {
namespace framework {
enum GraphTableType { EDGE_TABLE, FEATURE_TABLE };
class GpuPsGraphTable
    : public HeterComm<uint64_t, uint64_t, int, CommonFeatureValueAccessor> {
 public:
  int get_table_offset(int gpu_id, GraphTableType type, int idx) const {
    int type_id = type;
    return gpu_id * (graph_table_num_ + feature_table_num_) +
           type_id * graph_table_num_ + idx;
  }
  GpuPsGraphTable(std::shared_ptr<HeterPsResource> resource,
<<<<<<< HEAD
                  int graph_table_num)
      : HeterComm<uint64_t, uint64_t, int, CommonFeatureValueAccessor>(
            0, resource) {
    load_factor_ = FLAGS_gpugraph_hbm_table_load_factor;
    VLOG(0) << "load_factor = " << load_factor_
            << ", graph_table_num = " << graph_table_num;
=======
                  int topo_aware,
                  int graph_table_num)
      : HeterComm<uint64_t, uint64_t, int, CommonFeatureValueAccessor>(
            1, resource) {
    load_factor_ = FLAGS_gpugraph_hbm_table_load_factor;
    VLOG(0) << "load_factor = " << load_factor_;
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    rw_lock.reset(new pthread_rwlock_t());
    this->graph_table_num_ = graph_table_num;
    this->feature_table_num_ = 1;
    gpu_num = resource_->total_device();
    memset(global_device_map, -1, sizeof(global_device_map));
<<<<<<< HEAD

    tables_ = std::vector<Table *>(
        gpu_num * (graph_table_num_ + feature_table_num_), NULL);
    for (int i = 0; i < gpu_num; i++) {
      global_device_map[resource_->dev_id(i)] = i;
      for (int j = 0; j < graph_table_num_; j++) {
        gpu_graph_list_.push_back(GpuPsCommGraph());
      }
      for (int j = 0; j < feature_table_num_; j++) {
=======
    for (auto &table : tables_) {
      delete table;
      table = NULL;
    }
    int feature_table_num = 1;
    tables_ = std::vector<Table *>(
        gpu_num * (graph_table_num + feature_table_num), NULL);
    for (int i = 0; i < gpu_num; i++) {
      global_device_map[resource_->dev_id(i)] = i;
      for (int j = 0; j < graph_table_num; j++) {
        gpu_graph_list_.push_back(GpuPsCommGraph());
      }
      for (int j = 0; j < feature_table_num; j++) {
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        gpu_graph_fea_list_.push_back(GpuPsCommGraphFea());
      }
    }
    cpu_table_status = -1;
<<<<<<< HEAD
    device_mutex_.resize(gpu_num);
    for (int i = 0; i < gpu_num; i++) {
      device_mutex_[i] = new std::mutex();
    }
  }
  ~GpuPsGraphTable() {
    for (size_t i = 0; i < device_mutex_.size(); ++i) {
      delete device_mutex_[i];
    }
    device_mutex_.clear();
  }
=======
    if (topo_aware) {
      int total_gpu = resource_->total_device();
      std::map<int, int> device_map;
      for (int i = 0; i < total_gpu; i++) {
        device_map[resource_->dev_id(i)] = i;
        VLOG(1) << " device " << resource_->dev_id(i) << " is stored on " << i;
      }
      path_.clear();
      path_.resize(total_gpu);
      VLOG(1) << "topo aware overide";
      for (int i = 0; i < total_gpu; ++i) {
        path_[i].resize(total_gpu);
        for (int j = 0; j < total_gpu; ++j) {
          auto &nodes = path_[i][j].nodes_;
          nodes.clear();
          int from = resource_->dev_id(i);
          int to = resource_->dev_id(j);
          int transfer_id = i;
          if (need_transfer(from, to) &&
              (device_map.find((from + 4) % 8) != device_map.end() ||
               device_map.find((to + 4) % 8) != device_map.end())) {
            transfer_id = (device_map.find((from + 4) % 8) != device_map.end())
                              ? ((from + 4) % 8)
                              : ((to + 4) % 8);
            transfer_id = device_map[transfer_id];
            nodes.push_back(Node());
            Node &node = nodes.back();
            node.in_stream = resource_->comm_stream(i, transfer_id);
            node.out_stream = resource_->comm_stream(transfer_id, i);
            node.key_storage = NULL;
            node.val_storage = NULL;
            node.sync = 0;
            node.dev_num = transfer_id;
          }
          nodes.push_back(Node());
          Node &node = nodes.back();
          node.in_stream = resource_->comm_stream(i, transfer_id);
          node.out_stream = resource_->comm_stream(transfer_id, i);
          node.key_storage = NULL;
          node.val_storage = NULL;
          node.sync = 0;
          node.dev_num = j;
        }
      }
    }
  }
  ~GpuPsGraphTable() {}
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  void build_graph_on_single_gpu(const GpuPsCommGraph &g, int gpu_id, int idx);
  void build_graph_fea_on_single_gpu(const GpuPsCommGraphFea &g, int gpu_id);
  void clear_graph_info(int gpu_id, int index);
  void clear_graph_info(int index);
<<<<<<< HEAD
  void reset_feature_info(int gpu_id, size_t capacity, size_t feature_size);
=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  void clear_feature_info(int gpu_id, int index);
  void clear_feature_info(int index);
  void build_graph_from_cpu(const std::vector<GpuPsCommGraph> &cpu_node_list,
                            int idx);
  void build_graph_fea_from_cpu(
      const std::vector<GpuPsCommGraphFea> &cpu_node_list, int idx);
  NodeQueryResult graph_node_sample(int gpu_id, int sample_size);
  NeighborSampleResult graph_neighbor_sample_v3(NeighborSampleQuery q,
<<<<<<< HEAD
                                                bool cpu_switch,
                                                bool compress);
=======
                                                bool cpu_switch);
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  NeighborSampleResult graph_neighbor_sample(int gpu_id,
                                             uint64_t *key,
                                             int sample_size,
                                             int len);
  NeighborSampleResult graph_neighbor_sample_v2(int gpu_id,
                                                int idx,
                                                uint64_t *key,
                                                int sample_size,
                                                int len,
<<<<<<< HEAD
                                                bool cpu_query_switch,
                                                bool compress);
  NeighborSampleResultV2 graph_neighbor_sample_all_edge_type(
      int gpu_id,
      int edge_type_len,
      uint64_t *key,
      int sample_size,
      int len,
      std::vector<std::shared_ptr<phi::Allocation>> edge_type_graphs);
  std::vector<std::shared_ptr<phi::Allocation>> get_edge_type_graph(
      int gpu_id, int edge_type_len);
  void get_node_degree(int gpu_id,
                       int edge_idx,
                       uint64_t *key,
                       int len,
                       std::shared_ptr<phi::Allocation> node_degree);
  int get_feature_of_nodes(int gpu_id,
                           uint64_t *d_walk,
                           uint64_t *d_offset,
                           int size,
                           int slot_num,
                           int *d_slot_feature_num_map,
                           int fea_num_per_node);
  int get_feature_info_of_nodes(
      int gpu_id,
      uint64_t *d_nodes,
      int node_num,
      uint32_t *size_list,
      uint32_t *size_list_prefix_sum,
      std::shared_ptr<phi::Allocation> &feature_list,  // NOLINT
      std::shared_ptr<phi::Allocation> &slot_list);    // NOLINT
=======
                                                bool cpu_query_switch);

  int get_feature_of_nodes(
      int gpu_id, uint64_t *d_walk, uint64_t *d_offset, int size, int slot_num);
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

  NodeQueryResult query_node_list(int gpu_id,
                                  int idx,
                                  int start,
                                  int query_size);
  void display_sample_res(void *key, void *val, int len, int sample_len);
  void move_result_to_source_gpu(int gpu_id,
                                 int gpu_num,
                                 int sample_size,
                                 int *h_left,
                                 int *h_right,
                                 uint64_t *src_sample_res,
                                 int *actual_sample_size);
<<<<<<< HEAD
  void move_result_to_source_gpu(int start_index,
                                 int gpu_num,
                                 int *h_left,
                                 int *h_right,
                                 int *fea_left,
                                 uint32_t *fea_num_list,
                                 uint32_t *actual_feature_size,
                                 uint64_t *feature_list,
                                 uint8_t *slot_list);
  void move_degree_to_source_gpu(
      int gpu_id, int gpu_num, int *h_left, int *h_right, int *node_degree);
  void move_result_to_source_gpu_all_edge_type(int gpu_id,
                                               int gpu_num,
                                               int sample_size,
                                               int *h_left,
                                               int *h_right,
                                               uint64_t *src_sample_res,
                                               int *actual_sample_size,
                                               int edge_type_len,
                                               int len);
  int init_cpu_table(const paddle::distributed::GraphParameter &graph,
                     int gpu_num = 8);
  gpuStream_t get_local_stream(int gpu_id) {
    return resource_->local_stream(gpu_id, 0);
  }
=======
  int init_cpu_table(const paddle::distributed::GraphParameter &graph);
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

  int gpu_num;
  int graph_table_num_, feature_table_num_;
  std::vector<GpuPsCommGraph> gpu_graph_list_;
  std::vector<GpuPsCommGraphFea> gpu_graph_fea_list_;
  int global_device_map[32];
  const int parallel_sample_size = 1;
  const int dim_y = 256;
  std::shared_ptr<paddle::distributed::GraphTable> cpu_graph_table_;
  std::shared_ptr<pthread_rwlock_t> rw_lock;
  mutable std::mutex mutex_;
<<<<<<< HEAD
  std::vector<std::mutex *> device_mutex_;
  std::condition_variable cv_;
  int cpu_table_status;
};

};  // namespace framework
};  // namespace paddle
=======
  std::condition_variable cv_;
  int cpu_table_status;
};
}  // namespace framework
};  // namespace paddle
//#include "paddle/fluid/framework/fleet/heter_ps/graph_gpu_ps_table_inl.h"
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
#endif

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

#include "heter_comm.h"
#include "paddle/fluid/distributed/ps/table/common_graph_table.h"
#include "paddle/fluid/framework/fleet/heter_ps/gpu_graph_node.h"
#include "paddle/fluid/framework/fleet/heter_ps/heter_comm_kernel.h"
#include "paddle/fluid/platform/enforce.h"
#ifdef PADDLE_WITH_HETERPS
namespace paddle {
namespace framework {
class GpuPsGraphTable
    : public HeterComm<uint64_t, int64_t, int, CommonFeatureValueAccessor> {
 public:
  GpuPsGraphTable(std::shared_ptr<HeterPsResource> resource, int topo_aware)
      : HeterComm<uint64_t, int64_t, int, CommonFeatureValueAccessor>(
            1, resource) {
    load_factor_ = 0.25;
    rw_lock.reset(new pthread_rwlock_t());
    gpu_num = resource_->total_device();
    memset(global_device_map, -1, sizeof(global_device_map));
    for (int i = 0; i < gpu_num; i++) {
      gpu_graph_list.push_back(GpuPsCommGraph());
      global_device_map[resource_->dev_id(i)] = i;
      sample_status.push_back(NULL);
      tables_.push_back(NULL);
    }
    cpu_table_status = -1;
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
  ~GpuPsGraphTable() {
    // if (cpu_table_status != -1) {
    //   end_graph_sampling();
    // }
  }
  void build_graph_on_single_gpu(GpuPsCommGraph &g, int gpu_id);
  void clear_graph_info(int gpu_id);
  void build_graph_from_cpu(std::vector<GpuPsCommGraph> &cpu_node_list);
  NodeQueryResult graph_node_sample(int gpu_id, int sample_size);
  NeighborSampleResult graph_neighbor_sample_v3(NeighborSampleQuery q,
                                                bool cpu_switch);
  NeighborSampleResult graph_neighbor_sample(int gpu_id,
                                             int64_t *key,
                                             int sample_size,
                                             int len);
  NeighborSampleResult graph_neighbor_sample_v2(int gpu_id,
                                                int64_t *key,
                                                int sample_size,
                                                int len,
                                                bool cpu_query_switch);
  void init_sample_status();
  void free_sample_status();
  NodeQueryResult query_node_list(int gpu_id, int start, int query_size);
  void clear_graph_info();
  void display_sample_res(void *key, void *val, int len, int sample_len);
  void move_neighbor_sample_result_to_source_gpu(int gpu_id,
                                                 int gpu_num,
                                                 int sample_size,
                                                 int *h_left,
                                                 int *h_right,
                                                 int64_t *src_sample_res,
                                                 int *actual_sample_size);
  int init_cpu_table(const paddle::distributed::GraphParameter &graph);
  int gpu_num;
  std::vector<GpuPsCommGraph> gpu_graph_list;
  int global_device_map[32];
  std::vector<int *> sample_status;
  const int parallel_sample_size = 1;
  const int dim_y = 256;
  std::shared_ptr<paddle::distributed::GraphTable> cpu_graph_table;
  std::shared_ptr<pthread_rwlock_t> rw_lock;
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  int cpu_table_status;
};
}  // namespace framework
};  // namespace paddle
//#include "paddle/fluid/framework/fleet/heter_ps/graph_gpu_ps_table_inl.h"
#endif

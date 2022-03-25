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
#include "heter_comm.h"
#include "paddle/fluid/distributed/ps/table/common_graph_table.h"
#include "paddle/fluid/framework/fleet/heter_ps/gpu_graph_node.h"
#include "paddle/fluid/platform/enforce.h"
#ifdef PADDLE_WITH_HETERPS
namespace paddle {
namespace framework {

class GpuPsGraphTable : public HeterComm<int64_t, int, int> {
 public:
  GpuPsGraphTable(std::shared_ptr<HeterPsResource> resource)
      : HeterComm<int64_t, int, int>(1, resource) {
    load_factor_ = 0.25;
    rw_lock.reset(new pthread_rwlock_t());
    cpu_table_status = -1;
  }
  ~GpuPsGraphTable() {
    if (cpu_table_status != -1) {
      end_graph_sampling();
    }
  }
  void build_graph_from_cpu(std::vector<GpuPsCommGraph> &cpu_node_list);
  NodeQueryResult *graph_node_sample(int gpu_id, int sample_size);
  NeighborSampleResult *graph_neighbor_sample(int gpu_id, int64_t *key,
                                              int sample_size, int len);
  NodeQueryResult *query_node_list(int gpu_id, int start, int query_size);
  void clear_graph_info();
  void move_neighbor_sample_result_to_source_gpu(
      int gpu_id, int gpu_num, int *h_left, int *h_right,
      int64_t *src_sample_res, thrust::host_vector<int> &total_sample_size);
  void move_neighbor_sample_size_to_source_gpu(int gpu_id, int gpu_num,
                                               int *h_left, int *h_right,
                                               int *actual_sample_size,
                                               int *total_sample_size);
  int init_cpu_table(const paddle::distributed::GraphParameter &graph);
  int load(const std::string &path, const std::string &param);
  virtual int32_t end_graph_sampling() {
    return cpu_graph_table->end_graph_sampling();
  }

 private:
  std::vector<GpuPsCommGraph> gpu_graph_list;
  std::shared_ptr<paddle::distributed::GraphTable> cpu_graph_table;
  std::shared_ptr<pthread_rwlock_t> rw_lock;
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  int cpu_table_status;
};
}
};
#include "paddle/fluid/framework/fleet/heter_ps/graph_gpu_ps_table_inl.h"
#endif

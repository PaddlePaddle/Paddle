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

#include <gtest/gtest.h>
#include <algorithm>
#include <vector>
#include "paddle/fluid/framework/fleet/heter_ps/feature_value.h"
#include "paddle/fluid/framework/fleet/heter_ps/graph_gpu_ps_table.h"
#include "paddle/fluid/framework/fleet/heter_ps/heter_comm.h"
#include "paddle/fluid/framework/fleet/heter_ps/heter_resource.h"
#include "paddle/fluid/framework/fleet/heter_ps/optimizer.cuh.h"
#include "paddle/fluid/platform/cuda_device_guard.h"

using namespace paddle::framework;
namespace platform = paddle::platform;
// paddle::framework::GpuPsCommGraph GraphTable::make_gpu_ps_graph
// paddle::framework::GpuPsCommGraph GraphTable::make_gpu_ps_graph(
//     std::vector<int64_t> ids)
TEST(TEST_FLEET, test_cpu_cache) {
  int gpu_num = 0;
  int st = 0, u = 0;
  std::vector<int> device_id_mapping;
  for (int i = 0; i < 2; i++) device_id_mapping.push_back(i);
  gpu_num = device_id_mapping.size();
  ::paddle::distributed::GraphParameter table_proto;
  table_proto.set_shard_num(24);
  std::shared_ptr<HeterPsResource> resource =
      std::make_shared<HeterPsResource>(device_id_mapping);
  resource->enable_p2p();
  int use_nv = 1;
  GpuPsGraphTable g(resource, use_nv);
  g.init_cpu_table(table_proto);
  std::vector<paddle::framework::GpuPsCommGraph> vec;
  int n = 10;
  std::vector<int64_t> ids0, ids1;
  for (int i = 0; i < n; i++) {
    g.cpu_graph_table->add_comm_edge(i, (i + 1) % n);
    g.cpu_graph_table->add_comm_edge(i, (i - 1 + n) % n);
    if (i % 2 == 0) ids0.push_back(i);
  }
  ids1.push_back(5);
  vec.push_back(g.cpu_graph_table->make_gpu_ps_graph(ids0));
  vec.push_back(g.cpu_graph_table->make_gpu_ps_graph(ids1));
  vec[0].display_on_cpu();
  vec[1].display_on_cpu();
  g.build_graph_from_cpu(vec);
  int64_t cpu_key[3] = {0, 1, 2};
  void *key;
  platform::CUDADeviceGuard guard(0);
  cudaMalloc((void **)&key, 3 * sizeof(int64_t));
  cudaMemcpy(key, cpu_key, 3 * sizeof(int64_t), cudaMemcpyHostToDevice);
  auto neighbor_sample_res = g.graph_neighbor_sample(0, (int64_t *)key, 2, 3);
  int64_t *res = new int64_t[7];
  cudaMemcpy(res, neighbor_sample_res->val, 3 * 2 * sizeof(int64_t),
             cudaMemcpyDeviceToHost);
  int *actual_sample_size = new int[3];
  cudaMemcpy(actual_sample_size, neighbor_sample_res->actual_sample_size,
             3 * sizeof(int),
             cudaMemcpyDeviceToHost);  // 3, 1, 3

  //{0,9} or {9,0} is expected for key 0
  //{0,2} or {2,0} is expected for key 1
  //{1,3} or {3,1} is expected for key 2
  for (int i = 0; i < 3; i++) {
    VLOG(0) << "actual sample size for " << i << " is "
            << actual_sample_size[i];
    for (int j = 0; j < actual_sample_size[i]; j++) {
      VLOG(0) << "sampled an neighbor for node" << i << " : " << res[i * 2 + j];
    }
  }
}

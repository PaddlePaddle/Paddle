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

#include "paddle/fluid/distributed/ps/table/common_graph_table.h"
#include "paddle/fluid/framework/fleet/heter_ps/gpu_graph_node.h"
namespace paddle {
namespace framework {
class GraphGpuWrapper {
 public:
  char* graph_table;
  void initialize();
  void test();
  NeighborSampleResult* graph_neighbor_sample(int gpu_id, int64_t* key,
                                              int sample_size, int len);
  ::paddle::distributed::GraphParameter table_proto;
};
}
};

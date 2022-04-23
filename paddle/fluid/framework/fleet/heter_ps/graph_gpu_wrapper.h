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

#include <string>
#include <unordered_map>
#include <vector>
#include "paddle/fluid/distributed/ps/table/common_graph_table.h"
#include "paddle/fluid/framework/fleet/heter_ps/gpu_graph_node.h"
namespace paddle {
namespace framework {
#ifdef PADDLE_WITH_HETERPS
class GraphGpuWrapper {
 public:
  char* graph_table;
  void initialize();
  void test();
  void set_device(std::vector<int> ids);
  void init_service();
  void set_up_types(std::vector<std::string>& edge_type,
                    std::vector<std::string>& node_type);
  void upload_batch(std::vector<std::vector<int64_t>>& ids);
  void add_table_feat_conf(std::string table_name, std::string feat_name,
                           std::string feat_dtype, int feat_shape);
  void load_edge_file(std::string name, std::string filepath, bool reverse);
  void load_node_file(std::string name, std::string filepath);
  NeighborSampleResult* graph_neighbor_sample(int gpu_id, int64_t* key,
                                              int sample_size, int len);
  std::unordered_map<std::string, int> edge_to_id, feature_to_id;
  std::vector<std::string> id_to_feature, id_to_edge;
  std::vector<std::unordered_map<std::string, int>> table_feat_mapping;
  std::vector<std::vector<std::string>> table_feat_conf_feat_name;
  std::vector<std::vector<std::string>> table_feat_conf_feat_dtype;
  std::vector<std::vector<int>> table_feat_conf_feat_shape;
  ::paddle::distributed::GraphParameter table_proto;
  std::vector<int> device_id_mapping;
};
#endif
}
};

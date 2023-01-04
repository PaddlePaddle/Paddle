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
#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/distributed/ps/table/common_graph_table.h"
#include "paddle/fluid/framework/fleet/heter_ps/gpu_graph_node.h"
namespace paddle {
namespace framework {
#ifdef PADDLE_WITH_HETERPS

enum GpuGraphStorageMode {
  WHOLE_HBM = 1,
  MEM_EMB_AND_GPU_GRAPH,
  MEM_EMB_FEATURE_AND_GPU_GRAPH,
  SSD_EMB_AND_MEM_FEATURE_GPU_GRAPH
};

class GraphGpuWrapper {
 public:
  static std::shared_ptr<GraphGpuWrapper> GetInstance() {
    if (NULL == s_instance_) {
      s_instance_.reset(new paddle::framework::GraphGpuWrapper());
    }
    return s_instance_;
  }
  static std::shared_ptr<GraphGpuWrapper> s_instance_;
  void init_conf(const std::string& first_node_type,
                 const std::string& meta_path);
  void initialize();
  void finalize();
  void set_device(std::vector<int> ids);
  void init_service();
  void set_up_types(const std::vector<std::string>& edge_type,
                    const std::vector<std::string>& node_type);
  void upload_batch(int type,
                    int idx,
                    int slice_num,
                    const std::string& edge_type);
  void upload_batch(int type, int slice_num, int slot_num);
  std::vector<GpuPsCommGraphFea> get_sub_graph_fea(
      std::vector<std::vector<uint64_t>>& node_ids, int slot_num);    // NOLINT
  void build_gpu_graph_fea(GpuPsCommGraphFea& sub_graph_fea, int i);  // NOLINT
  void add_table_feat_conf(std::string table_name,
                           std::string feat_name,
                           std::string feat_dtype,
                           int feat_shape);
  void load_edge_file(std::string name, std::string filepath, bool reverse);
  void load_edge_file(std::string etype2files,
                      std::string graph_data_local_path,
                      int part_num,
                      bool reverse);

  void load_node_file(std::string name, std::string filepath);
  void load_node_file(std::string ntype2files,
                      std::string graph_data_local_path,
                      int part_num);
  void load_node_and_edge(std::string etype2files,
                          std::string ntype2files,
                          std::string graph_data_local_path,
                          int part_num,
                          bool reverse);
  int32_t load_next_partition(int idx);
  int32_t get_partition_num(int idx);
  void load_node_weight(int type_id, int idx, std::string path);
  void export_partition_files(int idx, std::string file_path);
  std::vector<uint64_t> get_partition(int idx, int num);
  void make_partitions(int idx, int64_t byte_size, int device_len);
  void make_complementary_graph(int idx, int64_t byte_size);
  void set_search_level(int level);
  void init_search_level(int level);
  int get_all_id(int type,
                 int slice_num,
                 std::vector<std::vector<uint64_t>>* output);
  int get_all_neighbor_id(int type,
                          int slice_num,
                          std::vector<std::vector<uint64_t>>* output);
  int get_all_id(int type,
                 int idx,
                 int slice_num,
                 std::vector<std::vector<uint64_t>>* output);
  int get_all_neighbor_id(int type,
                          int idx,
                          int slice_num,
                          std::vector<std::vector<uint64_t>>* output);
  int get_all_feature_ids(int type,
                          int idx,
                          int slice_num,
                          std::vector<std::vector<uint64_t>>* output);
  NodeQueryResult query_node_list(int gpu_id,
                                  int idx,
                                  int start,
                                  int query_size);
  NeighborSampleResult graph_neighbor_sample_v3(NeighborSampleQuery q,
                                                bool cpu_switch,
                                                bool compress);
  NeighborSampleResult graph_neighbor_sample(int gpu_id,
                                             uint64_t* device_keys,
                                             int walk_degree,
                                             int len);
  NeighborSampleResultV2 graph_neighbor_sample_all_edge_type(
      int gpu_id,
      int edge_type_len,
      uint64_t* key,
      int sample_size,
      int len,
      std::vector<std::shared_ptr<phi::Allocation>> edge_type_graphs);
  gpuStream_t get_local_stream(int gpuid);
  std::vector<uint64_t> graph_neighbor_sample(
      int gpu_id,
      int idx,
      std::vector<uint64_t>& key,  // NOLINT
      int sample_size);
  std::vector<std::shared_ptr<phi::Allocation>> get_edge_type_graph(
      int gpu_id, int edge_type_len);
  std::vector<int> slot_feature_num_map() const;
  void set_feature_separator(std::string ch);
  void set_slot_feature_separator(std::string ch);
  int get_feature_of_nodes(int gpu_id,
                           uint64_t* d_walk,
                           uint64_t* d_offset,
                           uint32_t size,
                           int slot_num,
                           int* d_slot_feature_num_map,
                           int fea_num_per_node);
  int get_feature_info_of_nodes(
      int gpu_id,
      uint64_t* d_nodes,
      int node_num,
      uint32_t* size_list,
      uint32_t* size_list_prefix_sum,
      std::shared_ptr<phi::Allocation>& feature_list,  // NOLINT
      std::shared_ptr<phi::Allocation>& slot_list);    // NOLINT
  void init_metapath(std::string cur_metapath,
                     int cur_metapath_index,
                     int cur_metapath_len);
  void clear_metapath_state();
  void release_graph();
  void release_graph_edge();
  void release_graph_node();
  void init_type_keys();
  std::vector<uint64_t>& get_graph_total_keys();
  std::vector<std::vector<uint64_t>>& get_graph_type_keys();
  std::unordered_map<int, int>& get_graph_type_to_index();
  std::string& get_node_type_size(std::string first_node_type);
  std::string& get_edge_type_size();

  std::unordered_map<std::string, int> edge_to_id, feature_to_id;
  std::vector<std::string> id_to_feature, id_to_edge;
  std::vector<std::unordered_map<std::string, int>> table_feat_mapping;
  std::vector<std::vector<std::string>> table_feat_conf_feat_name;
  std::vector<std::vector<std::string>> table_feat_conf_feat_dtype;
  std::vector<std::vector<int>> table_feat_conf_feat_shape;
  ::paddle::distributed::GraphParameter table_proto;
  std::vector<int> device_id_mapping;
  int search_level = 1;
  void* graph_table;
  int upload_num = 8;
  std::shared_ptr<::ThreadPool> upload_task_pool;
  std::string feature_separator_ = std::string(" ");
  bool conf_initialized_ = false;
  std::vector<int> first_node_type_;
  std::vector<std::vector<int>> meta_path_;

  std::vector<std::set<int>> finish_node_type_;
  std::vector<std::unordered_map<int, size_t>> node_type_start_;
  std::vector<size_t> cur_metapath_start_;
  std::vector<std::unordered_map<int, size_t>> global_infer_node_type_start_;
  std::vector<size_t> infer_cursor_;
  std::vector<size_t> cursor_;
  std::vector<std::shared_ptr<phi::Allocation>> d_graph_train_total_keys_;
  std::vector<size_t> h_graph_train_keys_len_;
  std::vector<std::vector<std::shared_ptr<phi::Allocation>>>
      d_graph_all_type_total_keys_;
  std::vector<std::vector<uint64_t>> h_graph_all_type_keys_len_;
  std::string slot_feature_separator_ = std::string(" ");

  std::string cur_metapath_;
  std::vector<int> cur_parse_metapath_;
  std::vector<int> cur_parse_reverse_metapath_;
  int cur_metapath_index_;
  int cur_metapath_len_;
  std::set<std::string> uniq_first_node_;
  std::string node_type_size_str_;
  std::string edge_type_size_str_;
};
#endif
}  // namespace framework
};  // namespace paddle

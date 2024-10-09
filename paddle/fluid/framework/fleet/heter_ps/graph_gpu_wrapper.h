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

#ifdef PADDLE_WITH_HETERPS
#if defined(__NVCC__) || defined(__HIPCC__)
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>
#endif
#include "paddle/fluid/framework/fleet/heter_ps/mem_pool.h"
#include "paddle/phi/backends/dynload/nccl.h"
#include "paddle/phi/core/platform/device/gpu/gpu_info.h"
#endif

#ifdef PADDLE_WITH_GLOO
#include <gloo/broadcast.h>
#include "paddle/fluid/framework/data_set.h"
#include "paddle/fluid/framework/fleet/gloo_wrapper.h"
#endif

namespace paddle {
namespace framework {

#ifdef PADDLE_WITH_HETERPS
typedef paddle::distributed::GraphTableType GraphTableType;

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
                 const std::string& meta_path,
                 const std::string& excluded_train_pair,
                 const std::string& pair_label);
  void initialize();
  void finalize();
  void set_device(std::vector<int> ids);
  void init_service();
  std::string get_reverse_etype(std::string etype);
  std::vector<std::string> get_ntype_from_etype(std::string etype);
  void set_up_types(const std::vector<std::string>& edge_type,
                    const std::vector<std::string>& node_type);
  void upload_batch(int table_type,
                    int slice_num,
                    const std::string& edge_type);
  void upload_batch(int table_type,
                    int slice_num,
                    int slot_num,
                    int float_slot_num);
  std::vector<GpuPsCommGraphFea> get_sub_graph_fea(
      std::vector<std::vector<uint64_t>>& node_ids, int slot_num);  // NOLINT
  std::vector<GpuPsCommGraphFloatFea> get_sub_graph_float_fea(
      std::vector<std::vector<uint64_t>>& node_ids,                   // NOLINT
      int float_slot_num);                                            // NOLINT
  void build_gpu_graph_fea(GpuPsCommGraphFea& sub_graph_fea, int i);  // NOLINT
  void build_gpu_graph_float_fea(
      GpuPsCommGraphFloatFea& sub_graph_float_fea,  // NOLINT
      int i);                                       // NOLINT
  void add_table_feat_conf(std::string table_name,
                           std::string feat_name,
                           std::string feat_dtype,
                           int feat_shape);
  void load_edge_file(std::string name, std::string filepath, bool reverse);
  void load_edge_file(std::string etype2files,
                      std::string graph_data_local_path,
                      int part_num,
                      bool reverse,
                      const std::vector<bool>& is_reverse_edge_map,
                      bool use_weight);

  int load_node_file(std::string name, std::string filepath);
  int load_node_file(std::string ntype2files,
                     std::string graph_data_local_path,
                     int part_num);
  void load_node_and_edge(std::string etype2files,
                          std::string ntype2files,
                          std::string graph_data_local_path,
                          int part_num,
                          bool reverse,
                          const std::vector<bool>& is_reverse_edge_map);
  void calc_edge_type_limit();
  int set_node_iter_from_file(std::string ntype2files,
                              std::string nodes_file_path,
                              int part_num,
                              bool training,
                              bool shuffle);
  int set_node_iter_from_graph(bool training, bool shuffle);
  void shuffle_start_nodes_for_training();
  int32_t load_next_partition(int idx);
  int32_t get_partition_num(int idx);
  void load_node_weight(int type_id, int idx, std::string path);
  void export_partition_files(int idx, std::string file_path);
  std::vector<uint64_t> get_partition(int idx, int num);
  void make_partitions(int idx, int64_t byte_size, int device_len);
  void make_complementary_graph(int idx, int64_t byte_size);
  void set_search_level(int level);
  void init_search_level(int level);
  std::string node_types_idx_to_node_type_str(int node_types_idx);
  std::string index_to_node_type_str(int index);
  int get_all_id(int table_type,
                 int slice_num,
                 std::vector<std::vector<uint64_t>>* output);
  int get_all_neighbor_id(GraphTableType table_type,
                          int slice_num,
                          std::vector<std::vector<uint64_t>>* output);
  int get_all_id(int table_type,
                 int idx,
                 int slice_num,
                 std::vector<std::vector<uint64_t>>* output);
  int get_all_neighbor_id(GraphTableType table_type,
                          int idx,
                          int slice_num,
                          std::vector<std::vector<uint64_t>>* output);
  int get_all_feature_ids(GraphTableType table_type,
                          int idx,
                          int slice_num,
                          std::vector<std::vector<uint64_t>>* output);
  int get_node_embedding_ids(int slice_num,
                             std::vector<std::vector<uint64_t>>* output);
  NodeQueryResult query_node_list(int gpu_id,
                                  int idx,
                                  int start,
                                  int query_size);
  NeighborSampleResult graph_neighbor_sample_v3(NeighborSampleQuery q,
                                                bool cpu_switch,
                                                bool compress,
                                                bool weighted);
  void seek_keys_rank(int gpu_id,
                      const uint64_t* d_in_keys,
                      int len,
                      uint32_t* d_out_ranks);
  NeighborSampleResult graph_neighbor_sample(int gpu_id,
                                             uint64_t* device_keys,
                                             int walk_degree,
                                             int len);
  NeighborSampleResultV2 graph_neighbor_sample_sage(
      int gpu_id,
      int edge_type_len,
      const uint64_t* d_keys,
      int sample_size,
      int len,
      std::vector<std::shared_ptr<phi::Allocation>> edge_type_graphs,
      bool weighted,
      bool return_weight);
  std::shared_ptr<phi::Allocation> get_node_degree(int gpu_id,
                                                   int edge_idx,
                                                   uint64_t* key,
                                                   int len);
  gpuStream_t get_local_stream(int gpuid);
  std::vector<uint64_t> graph_neighbor_sample(
      int gpu_id,
      int idx,
      std::vector<uint64_t>& key,  // NOLINT
      int sample_size);
  std::vector<std::shared_ptr<phi::Allocation>> get_edge_type_graph(
      int gpu_id, int edge_type_len);
  std::vector<int> slot_feature_num_map() const;
  void set_feature_info(int slot_num_for_pull_feature, int float_slot_num);
  void set_feature_separator(std::string ch);
  void set_slot_feature_separator(std::string ch);
  void set_infer_mode(bool infer_mode);
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
      std::shared_ptr<phi::Allocation>& size_list,             // NOLINT
      std::shared_ptr<phi::Allocation>& size_list_prefix_sum,  // NOLINT
      std::shared_ptr<phi::Allocation>& feature_list,          // NOLINT
      std::shared_ptr<phi::Allocation>& slot_list,             // NOLINT
      bool sage_mode = false);
  int get_float_feature_info_of_nodes(
      int gpu_id,
      uint64_t* d_nodes,
      int node_num,
      std::shared_ptr<phi::Allocation>& size_list,             // NOLINT
      std::shared_ptr<phi::Allocation>& size_list_prefix_sum,  // NOLINT
      std::shared_ptr<phi::Allocation>& feature_list,          // NOLINT
      std::shared_ptr<phi::Allocation>& slot_list,             // NOLINT
      bool sage_mode = false);
  void init_metapath(std::string cur_metapath,
                     int cur_metapath_index,
                     int cur_metapath_len);
  void init_metapath_total_keys();
  void clear_metapath_state();
  void release_graph();
  void release_graph_edge();
  void release_graph_node();
  void init_type_keys(
      std::vector<std::vector<std::shared_ptr<phi::Allocation>>>&
          keys,                                   // NOLINT
      std::vector<std::vector<uint64_t>>& lens);  // NOLINT
  std::vector<uint64_t>& get_graph_total_keys();
  std::vector<std::vector<uint64_t>>& get_graph_type_keys();
  std::unordered_map<int, int>& get_type_to_neighbor_limit();
  std::unordered_map<int, int>& get_graph_type_to_index();
  std::string& get_node_type_size(std::string first_node_type);
  std::string& get_edge_type_size();
  void set_keys2rank(int gpu_id,
                     std::shared_ptr<HashTable<uint64_t, uint32_t>> keys2rank);
  void show_mem(const char* msg);
  void debug(const char* desc) const;

  std::unordered_map<std::string, int> edge_to_id, node_to_id;
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
  int slot_num_for_pull_feature_ = 0;
  int float_slot_num_ = 0;
  std::shared_ptr<::ThreadPool> upload_task_pool;
  std::string feature_separator_ = std::string(" ");
  bool conf_initialized_ = false;
  bool type_keys_initialized_ = false;
  std::vector<std::vector<int>> first_node_type_;
  std::vector<int> all_node_type_;
  std::vector<uint8_t> excluded_train_pair_;
  std::vector<int32_t> pair_label_conf_;
  std::vector<std::vector<std::vector<int>>> meta_path_;

  std::vector<std::vector<std::set<int>>> finish_node_type_;
  std::vector<std::vector<std::unordered_map<int, size_t>>> node_type_start_;
  std::vector<size_t> cur_metapath_start_;
  std::vector<std::vector<std::unordered_map<int, size_t>>>
      global_infer_node_type_start_;
  std::vector<std::vector<size_t>> infer_cursor_;
  std::vector<size_t> cursor_;
  int tensor_pair_num_;

  std::vector<std::vector<std::shared_ptr<phi::Allocation>>>
      d_graph_all_type_total_keys_;
  std::vector<std::vector<uint64_t>> h_graph_all_type_keys_len_;
  std::vector<std::vector<std::shared_ptr<phi::Allocation>>>
      d_node_iter_graph_all_type_keys_;
  std::vector<std::vector<uint64_t>> h_node_iter_graph_all_type_keys_len_;
  std::vector<std::shared_ptr<phi::Allocation>>
      d_node_iter_graph_metapath_keys_;
  std::vector<size_t> h_node_iter_graph_metapath_keys_len_;

  std::map<uint64_t,  // edge_id
           uint64_t   // src_node_id << 32 | dst_node_id
           >
      edge_to_node_map_;

  std::string slot_feature_separator_ = std::string(" ");
  std::string cur_metapath_;
  std::vector<int> cur_parse_metapath_;
  std::vector<int> cur_parse_reverse_metapath_;
  int cur_metapath_index_;
  int cur_metapath_len_;
  std::set<std::string> uniq_first_node_;
  std::string node_type_size_str_;
  std::string edge_type_size_str_;

  // add for multi-node
  int rank_id_ = 0;
  int node_size_ = 1;
  int multi_node_ = 0;
#ifdef PADDLE_WITH_CUDA
  std::vector<ncclComm_t> inner_comms_;
  std::vector<ncclComm_t> inter_comms_;
  std::vector<ncclUniqueId> inter_ncclids_;
#endif
};  // class GraphGpuWrapper
#endif

};  // namespace framework
};  // namespace paddle

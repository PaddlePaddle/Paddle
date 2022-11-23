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

#include "paddle/fluid/framework/fleet/heter_ps/graph_gpu_wrapper.h"
#include <sstream>
#include "paddle/fluid/framework/fleet/heter_ps/gpu_graph_utils.h"
#include "paddle/fluid/framework/fleet/heter_ps/graph_gpu_ps_table.h"
#include "paddle/fluid/framework/fleet/heter_ps/heter_resource.h"
DECLARE_int32(gpugraph_storage_mode);
namespace paddle {
namespace framework {
#ifdef PADDLE_WITH_HETERPS

std::shared_ptr<GraphGpuWrapper> GraphGpuWrapper::s_instance_(nullptr);
void GraphGpuWrapper::set_device(std::vector<int> ids) {
  for (auto device_id : ids) {
    device_id_mapping.push_back(device_id);
  }
}

<<<<<<< HEAD
void GraphGpuWrapper::init_conf(const std::string &first_node_type,
                                const std::string &meta_path) {
  static std::mutex mutex;
  {
    std::lock_guard<std::mutex> lock(mutex);
    if (conf_initialized_) {
      return;
    }
    VLOG(2) << "init path config";
    conf_initialized_ = true;
    auto node_types =
        paddle::string::split_string<std::string>(first_node_type, ";");
    VLOG(2) << "node_types: " << first_node_type;
    for (auto &type : node_types) {
      auto iter = feature_to_id.find(type);
      PADDLE_ENFORCE_NE(iter,
                        feature_to_id.end(),
                        platform::errors::NotFound(
                            "(%s) is not found in feature_to_id.", type));
      VLOG(2) << "feature_to_id[" << type << "] = " << iter->second;
      first_node_type_.push_back(iter->second);
    }
    meta_path_.resize(first_node_type_.size());
    auto meta_paths = paddle::string::split_string<std::string>(meta_path, ";");

    for (size_t i = 0; i < meta_paths.size(); i++) {
      auto path = meta_paths[i];
      auto nodes = paddle::string::split_string<std::string>(path, "-");
      for (auto &node : nodes) {
        auto iter = edge_to_id.find(node);
        PADDLE_ENFORCE_NE(iter,
                          edge_to_id.end(),
                          platform::errors::NotFound(
                              "(%s) is not found in edge_to_id.", node));
        VLOG(2) << "edge_to_id[" << node << "] = " << iter->second;
        meta_path_[i].push_back(iter->second);
      }
    }
    int max_dev_id = 0;
    for (size_t i = 0; i < device_id_mapping.size(); i++) {
      if (device_id_mapping[i] > max_dev_id) {
        max_dev_id = device_id_mapping[i];
      }
    }
    finish_node_type_.resize(max_dev_id + 1);
    node_type_start_.resize(max_dev_id + 1);
    global_infer_node_type_start_.resize(max_dev_id + 1);
    for (size_t i = 0; i < device_id_mapping.size(); i++) {
      int dev_id = device_id_mapping[i];
      auto &node_type_start = node_type_start_[i];
      auto &infer_node_type_start = global_infer_node_type_start_[i];
      auto &finish_node_type = finish_node_type_[i];
      finish_node_type.clear();

      for (size_t idx = 0; idx < feature_to_id.size(); idx++) {
        infer_node_type_start[idx] = 0;
      }
      for (auto &type : node_types) {
        auto iter = feature_to_id.find(type);
        node_type_start[iter->second] = 0;
        infer_node_type_start[iter->second] = 0;
      }
      infer_cursor_.push_back(0);
      cursor_.push_back(0);
    }
    init_type_keys();
  }
}

void GraphGpuWrapper::init_type_keys() {
  size_t thread_num = device_id_mapping.size();
  int cnt = 0;

  auto &graph_all_type_total_keys = get_graph_type_keys();
  auto &type_to_index = get_graph_type_to_index();
  std::vector<std::vector<uint64_t>> tmp_keys;
  tmp_keys.resize(thread_num);
  d_graph_all_type_total_keys_.resize(graph_all_type_total_keys.size());
  h_graph_all_type_keys_len_.resize(graph_all_type_total_keys.size());
  for (size_t f_idx = 0; f_idx < graph_all_type_total_keys.size(); f_idx++) {
    for (size_t j = 0; j < tmp_keys.size(); j++) {
      tmp_keys[j].clear();
    }
    d_graph_all_type_total_keys_[f_idx].resize(thread_num);
    auto &type_total_key = graph_all_type_total_keys[f_idx];
    for (size_t j = 0; j < type_total_key.size(); j++) {
      uint64_t shard = type_total_key[j] % thread_num;
      tmp_keys[shard].push_back(type_total_key[j]);
    }
    for (size_t j = 0; j < thread_num; j++) {
      h_graph_all_type_keys_len_[f_idx].push_back(tmp_keys[j].size());
      VLOG(1) << "node type: " << type_to_index[f_idx]
              << ", gpu_graph_device_keys[" << j
              << "] = " << tmp_keys[j].size();
    }
    for (size_t j = 0; j < thread_num; j++) {
      auto stream = get_local_stream(j);
      int gpuid = device_id_mapping[j];
      auto place = platform::CUDAPlace(gpuid);
      platform::CUDADeviceGuard guard(gpuid);
      d_graph_all_type_total_keys_[f_idx][j] =
          memory::AllocShared(place, tmp_keys[j].size() * sizeof(uint64_t));
      cudaMemcpyAsync(d_graph_all_type_total_keys_[f_idx][j]->ptr(),
                      tmp_keys[j].data(),
                      sizeof(uint64_t) * tmp_keys[j].size(),
                      cudaMemcpyHostToDevice,
                      stream);
    }
  }
  for (int i = 0; i < thread_num; i++) {
    auto stream = get_local_stream(i);
    cudaStreamSynchronize(stream);
  }
}

=======
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
int GraphGpuWrapper::get_all_id(int type,
                                int slice_num,
                                std::vector<std::vector<uint64_t>> *output) {
  return ((GpuPsGraphTable *)graph_table)
      ->cpu_graph_table_->get_all_id(type, slice_num, output);
}

int GraphGpuWrapper::get_all_neighbor_id(
    int type, int slice_num, std::vector<std::vector<uint64_t>> *output) {
  return ((GpuPsGraphTable *)graph_table)
      ->cpu_graph_table_->get_all_neighbor_id(type, slice_num, output);
}

int GraphGpuWrapper::get_all_id(int type,
                                int idx,
                                int slice_num,
                                std::vector<std::vector<uint64_t>> *output) {
  return ((GpuPsGraphTable *)graph_table)
      ->cpu_graph_table_->get_all_id(type, idx, slice_num, output);
}

int GraphGpuWrapper::get_all_neighbor_id(
    int type,
    int idx,
    int slice_num,
    std::vector<std::vector<uint64_t>> *output) {
  return ((GpuPsGraphTable *)graph_table)
      ->cpu_graph_table_->get_all_neighbor_id(type, idx, slice_num, output);
}

int GraphGpuWrapper::get_all_feature_ids(
    int type,
    int idx,
    int slice_num,
    std::vector<std::vector<uint64_t>> *output) {
  return ((GpuPsGraphTable *)graph_table)
      ->cpu_graph_table_->get_all_feature_ids(type, idx, slice_num, output);
}

<<<<<<< HEAD
int GraphGpuWrapper::get_node_embedding_ids(
    int slice_num, std::vector<std::vector<uint64_t>> *output) {
  return ((GpuPsGraphTable *)graph_table)
      ->cpu_graph_table_->get_node_embedding_ids(slice_num, output);
}

=======
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
void GraphGpuWrapper::set_up_types(std::vector<std::string> &edge_types,
                                   std::vector<std::string> &node_types) {
  id_to_edge = edge_types;
  for (size_t table_id = 0; table_id < edge_types.size(); table_id++) {
    int res = edge_to_id.size();
    edge_to_id[edge_types[table_id]] = res;
  }
  id_to_feature = node_types;
  for (size_t table_id = 0; table_id < node_types.size(); table_id++) {
    int res = feature_to_id.size();
    feature_to_id[node_types[table_id]] = res;
  }
  table_feat_mapping.resize(node_types.size());
  this->table_feat_conf_feat_name.resize(node_types.size());
  this->table_feat_conf_feat_dtype.resize(node_types.size());
  this->table_feat_conf_feat_shape.resize(node_types.size());
}

void GraphGpuWrapper::set_feature_separator(std::string ch) {
  feature_separator_ = ch;
  if (graph_table != nullptr) {
    ((GpuPsGraphTable *)graph_table)
        ->cpu_graph_table_->set_feature_separator(feature_separator_);
  }
}

<<<<<<< HEAD
void GraphGpuWrapper::set_slot_feature_separator(std::string ch) {
  slot_feature_separator_ = ch;
  if (graph_table != nullptr) {
    ((GpuPsGraphTable *)graph_table)
        ->cpu_graph_table_->set_slot_feature_separator(slot_feature_separator_);
  }
}

=======
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
void GraphGpuWrapper::make_partitions(int idx,
                                      int64_t byte_size,
                                      int device_len) {
  ((GpuPsGraphTable *)graph_table)
      ->cpu_graph_table_->make_partitions(idx, byte_size, device_len);
}
int32_t GraphGpuWrapper::load_next_partition(int idx) {
  return ((GpuPsGraphTable *)graph_table)
      ->cpu_graph_table_->load_next_partition(idx);
}

void GraphGpuWrapper::set_search_level(int level) {
  ((GpuPsGraphTable *)graph_table)->cpu_graph_table_->set_search_level(level);
}

std::vector<uint64_t> GraphGpuWrapper::get_partition(int idx, int num) {
  return ((GpuPsGraphTable *)graph_table)
      ->cpu_graph_table_->get_partition(idx, num);
}
int32_t GraphGpuWrapper::get_partition_num(int idx) {
  return ((GpuPsGraphTable *)graph_table)
      ->cpu_graph_table_->get_partition_num(idx);
}
void GraphGpuWrapper::make_complementary_graph(int idx, int64_t byte_size) {
  ((GpuPsGraphTable *)graph_table)
      ->cpu_graph_table_->make_complementary_graph(idx, byte_size);
}
void GraphGpuWrapper::load_edge_file(std::string name,
                                     std::string filepath,
                                     bool reverse) {
  // 'e' means load edge
  std::string params = "e";
  if (reverse) {
    // 'e<' means load edges from $2 to $1
    params += "<" + name;
  } else {
    // 'e>' means load edges from $1 to $2
    params += ">" + name;
  }
  if (edge_to_id.find(name) != edge_to_id.end()) {
    ((GpuPsGraphTable *)graph_table)
        ->cpu_graph_table_->Load(std::string(filepath), params);
  }
}

void GraphGpuWrapper::load_edge_file(std::string etype2files,
                                     std::string graph_data_local_path,
                                     int part_num,
                                     bool reverse) {
  ((GpuPsGraphTable *)graph_table)
      ->cpu_graph_table_->parse_edge_and_load(
          etype2files, graph_data_local_path, part_num, reverse);
}

void GraphGpuWrapper::load_node_file(std::string name, std::string filepath) {
  // 'n' means load nodes and 'node_type' follows

  std::string params = "n" + name;

  if (feature_to_id.find(name) != feature_to_id.end()) {
    ((GpuPsGraphTable *)graph_table)
        ->cpu_graph_table_->Load(std::string(filepath), params);
  }
}

<<<<<<< HEAD
void GraphGpuWrapper::load_node_file(std::string ntype2files,
                                     std::string graph_data_local_path,
                                     int part_num) {
  ((GpuPsGraphTable *)graph_table)
      ->cpu_graph_table_->parse_node_and_load(
          ntype2files, graph_data_local_path, part_num);
}

void GraphGpuWrapper::load_node_and_edge(std::string etype2files,
                                         std::string ntype2files,
                                         std::string graph_data_local_path,
=======
void GraphGpuWrapper::load_node_and_edge(std::string etype,
                                         std::string ntype,
                                         std::string epath,
                                         std::string npath,
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
                                         int part_num,
                                         bool reverse) {
  ((GpuPsGraphTable *)graph_table)
      ->cpu_graph_table_->load_node_and_edge_file(
<<<<<<< HEAD
          etype2files, ntype2files, graph_data_local_path, part_num, reverse);
=======
          etype, ntype, epath, npath, part_num, reverse);
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
}

void GraphGpuWrapper::add_table_feat_conf(std::string table_name,
                                          std::string feat_name,
                                          std::string feat_dtype,
                                          int feat_shape) {
  if (feature_to_id.find(table_name) != feature_to_id.end()) {
    int idx = feature_to_id[table_name];
    if (table_feat_mapping[idx].find(feat_name) ==
        table_feat_mapping[idx].end()) {
      int res = (int)table_feat_mapping[idx].size();
      table_feat_mapping[idx][feat_name] = res;
    }
    int feat_idx = table_feat_mapping[idx][feat_name];
    VLOG(0) << "table_name " << table_name << " mapping id " << idx;
    VLOG(0) << " feat name " << feat_name << " feat id" << feat_idx;
    if (feat_idx < table_feat_conf_feat_name[idx].size()) {
      // overide
      table_feat_conf_feat_name[idx][feat_idx] = feat_name;
      table_feat_conf_feat_dtype[idx][feat_idx] = feat_dtype;
      table_feat_conf_feat_shape[idx][feat_idx] = feat_shape;
    } else {
      // new
      table_feat_conf_feat_name[idx].push_back(feat_name);
      table_feat_conf_feat_dtype[idx].push_back(feat_dtype);
      table_feat_conf_feat_shape[idx].push_back(feat_shape);
    }
  }
  VLOG(0) << "add conf over";
}
void GraphGpuWrapper::init_search_level(int level) { search_level = level; }

gpuStream_t GraphGpuWrapper::get_local_stream(int gpuid) {
  return ((GpuPsGraphTable *)graph_table)->get_local_stream(gpuid);
}

void GraphGpuWrapper::init_service() {
<<<<<<< HEAD
  table_proto.set_task_pool_size(64);
=======
  table_proto.set_task_pool_size(24);
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
  table_proto.set_shard_num(1000);
  table_proto.set_build_sampler_on_cpu(false);
  table_proto.set_search_level(search_level);
  table_proto.set_table_name("cpu_graph_table_");
  table_proto.set_use_cache(false);
  for (int i = 0; i < id_to_edge.size(); i++)
    table_proto.add_edge_types(id_to_edge[i]);
  for (int i = 0; i < id_to_feature.size(); i++) {
    table_proto.add_node_types(id_to_feature[i]);
    auto feat_node = id_to_feature[i];
    ::paddle::distributed::GraphFeature *g_f = table_proto.add_graph_feature();
    for (int x = 0; x < table_feat_conf_feat_name[i].size(); x++) {
      g_f->add_name(table_feat_conf_feat_name[i][x]);
      g_f->add_dtype(table_feat_conf_feat_dtype[i][x]);
      g_f->add_shape(table_feat_conf_feat_shape[i][x]);
    }
  }
  std::shared_ptr<HeterPsResource> resource =
      std::make_shared<HeterPsResource>(device_id_mapping);
  resource->enable_p2p();
<<<<<<< HEAD
  GpuPsGraphTable *g = new GpuPsGraphTable(resource, id_to_edge.size());
  size_t gpu_num = device_id_mapping.size();
  g->init_cpu_table(table_proto, gpu_num);
  g->cpu_graph_table_->set_feature_separator(feature_separator_);
  g->cpu_graph_table_->set_slot_feature_separator(slot_feature_separator_);
  graph_table = (char *)g;
  upload_num = gpu_num;
=======
  GpuPsGraphTable *g = new GpuPsGraphTable(resource, 1, id_to_edge.size());
  g->init_cpu_table(table_proto);
  g->cpu_graph_table_->set_feature_separator(feature_separator_);
  graph_table = (char *)g;
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
  upload_task_pool.reset(new ::ThreadPool(upload_num));
}

void GraphGpuWrapper::finalize() {
  ((GpuPsGraphTable *)graph_table)->show_table_collisions();
}

void GraphGpuWrapper::upload_batch(int type,
                                   int idx,
                                   int slice_num,
                                   const std::string &edge_type) {
  VLOG(0) << "begin upload edge, type[" << edge_type << "]";
  std::vector<std::vector<uint64_t>> ids;
  ((GpuPsGraphTable *)graph_table)
      ->cpu_graph_table_->get_all_id(type, idx, slice_num, &ids);
  debug_gpu_memory_info("upload_batch node start");
  GpuPsGraphTable *g = (GpuPsGraphTable *)graph_table;
  std::vector<std::future<int>> tasks;

  for (int i = 0; i < ids.size(); i++) {
    tasks.push_back(upload_task_pool->enqueue([&, i, idx, this]() -> int {
      VLOG(0) << "begin make_gpu_ps_graph, node_id[" << i << "]_size["
              << ids[i].size() << "]";
      GpuPsCommGraph sub_graph =
          g->cpu_graph_table_->make_gpu_ps_graph(idx, ids[i]);
      g->build_graph_on_single_gpu(sub_graph, i, idx);
      sub_graph.release_on_cpu();
      VLOG(0) << "sub graph on gpu " << i << " is built";
      return 0;
    }));
  }
  for (size_t i = 0; i < tasks.size(); i++) tasks[i].get();
  debug_gpu_memory_info("upload_batch node end");
}

// feature table
void GraphGpuWrapper::upload_batch(int type, int slice_num, int slot_num) {
<<<<<<< HEAD
  if (type == 1 &&
      (FLAGS_gpugraph_storage_mode == paddle::framework::GpuGraphStorageMode::
                                          MEM_EMB_FEATURE_AND_GPU_GRAPH ||
       FLAGS_gpugraph_storage_mode == paddle::framework::GpuGraphStorageMode::
                                          SSD_EMB_AND_MEM_FEATURE_GPU_GRAPH)) {
    return;
  }
=======
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
  std::vector<std::vector<uint64_t>> node_ids;
  ((GpuPsGraphTable *)graph_table)
      ->cpu_graph_table_->get_all_id(type, slice_num, &node_ids);
  debug_gpu_memory_info("upload_batch feature start");
  GpuPsGraphTable *g = (GpuPsGraphTable *)graph_table;
  std::vector<std::future<int>> tasks;
  for (int i = 0; i < node_ids.size(); i++) {
    tasks.push_back(upload_task_pool->enqueue([&, i, this]() -> int {
      VLOG(0) << "begin make_gpu_ps_graph_fea, node_ids[" << i << "]_size["
              << node_ids[i].size() << "]";
      GpuPsCommGraphFea sub_graph =
<<<<<<< HEAD
          g->cpu_graph_table_->make_gpu_ps_graph_fea(i, node_ids[i], slot_num);
=======
          g->cpu_graph_table_->make_gpu_ps_graph_fea(node_ids[i], slot_num);
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
      // sub_graph.display_on_cpu();
      VLOG(0) << "begin build_graph_fea_on_single_gpu, node_ids[" << i
              << "]_size[" << node_ids[i].size() << "]";
      g->build_graph_fea_on_single_gpu(sub_graph, i);
      sub_graph.release_on_cpu();
      VLOG(0) << "sub graph fea on gpu " << i << " is built";
      return 0;
    }));
  }
  for (size_t i = 0; i < tasks.size(); i++) tasks[i].get();
  // g->build_graph_from_cpu(vec);
  debug_gpu_memory_info("upload_batch feature end");
<<<<<<< HEAD
}

// get sub_graph_fea
std::vector<GpuPsCommGraphFea> GraphGpuWrapper::get_sub_graph_fea(
    std::vector<std::vector<uint64_t>> &node_ids, int slot_num) {
  GpuPsGraphTable *g = (GpuPsGraphTable *)graph_table;
  std::vector<std::future<int>> tasks;
  std::vector<GpuPsCommGraphFea> sub_graph_feas(node_ids.size());
  for (int i = 0; i < node_ids.size(); i++) {
    tasks.push_back(upload_task_pool->enqueue([&, i, this]() -> int {
      GpuPsGraphTable *g = (GpuPsGraphTable *)graph_table;
      sub_graph_feas[i] =
          g->cpu_graph_table_->make_gpu_ps_graph_fea(i, node_ids[i], slot_num);
      return 0;
    }));
  }
  for (size_t i = 0; i < tasks.size(); i++) tasks[i].get();
  return sub_graph_feas;
}

// build_gpu_graph_fea
void GraphGpuWrapper::build_gpu_graph_fea(GpuPsCommGraphFea &sub_graph_fea,
                                          int i) {
  GpuPsGraphTable *g = (GpuPsGraphTable *)graph_table;
  g->build_graph_fea_on_single_gpu(sub_graph_fea, i);
  sub_graph_fea.release_on_cpu();
  VLOG(0) << "sub graph fea on gpu " << i << " is built";
  return;
=======
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
}

NeighborSampleResult GraphGpuWrapper::graph_neighbor_sample_v3(
    NeighborSampleQuery q, bool cpu_switch, bool compress = true) {
  return ((GpuPsGraphTable *)graph_table)
      ->graph_neighbor_sample_v3(q, cpu_switch, compress);
}

NeighborSampleResultV2 GraphGpuWrapper::graph_neighbor_sample_all_edge_type(
    int gpu_id,
    int edge_type_len,
    uint64_t *key,
    int sample_size,
    int len,
    std::vector<std::shared_ptr<phi::Allocation>> edge_type_graphs) {
  return ((GpuPsGraphTable *)graph_table)
      ->graph_neighbor_sample_all_edge_type(
          gpu_id, edge_type_len, key, sample_size, len, edge_type_graphs);
}

std::vector<std::shared_ptr<phi::Allocation>>
GraphGpuWrapper::get_edge_type_graph(int gpu_id, int edge_type_len) {
  return ((GpuPsGraphTable *)graph_table)
      ->get_edge_type_graph(gpu_id, edge_type_len);
}

int GraphGpuWrapper::get_feature_info_of_nodes(
    int gpu_id,
    uint64_t *d_nodes,
    int node_num,
    uint32_t *size_list,
    uint32_t *size_list_prefix_sum,
    std::shared_ptr<phi::Allocation> &feature_list,
    std::shared_ptr<phi::Allocation> &slot_list) {
  platform::CUDADeviceGuard guard(gpu_id);
  PADDLE_ENFORCE_NOT_NULL(graph_table,
                          paddle::platform::errors::InvalidArgument(
                              "graph_table should not be null"));
  return ((GpuPsGraphTable *)graph_table)
      ->get_feature_info_of_nodes(gpu_id,
                                  d_nodes,
                                  node_num,
                                  size_list,
                                  size_list_prefix_sum,
                                  feature_list,
                                  slot_list);
}

int GraphGpuWrapper::get_feature_of_nodes(int gpu_id,
                                          uint64_t *d_walk,
                                          uint64_t *d_offset,
                                          uint32_t size,
                                          int slot_num,
                                          int *d_slot_feature_num_map,
                                          int fea_num_per_node) {
  platform::CUDADeviceGuard guard(gpu_id);
  PADDLE_ENFORCE_NOT_NULL(graph_table,
                          paddle::platform::errors::InvalidArgument(
                              "graph_table should not be null"));
  return ((GpuPsGraphTable *)graph_table)
      ->get_feature_of_nodes(gpu_id,
                             d_walk,
                             d_offset,
                             size,
                             slot_num,
                             d_slot_feature_num_map,
                             fea_num_per_node);
}

NeighborSampleResult GraphGpuWrapper::graph_neighbor_sample(
    int gpu_id, uint64_t *device_keys, int walk_degree, int len) {
  platform::CUDADeviceGuard guard(gpu_id);
  auto neighbor_sample_res =
      ((GpuPsGraphTable *)graph_table)
          ->graph_neighbor_sample(gpu_id, device_keys, walk_degree, len);

  return neighbor_sample_res;
}

int GraphGpuWrapper::get_feature_of_nodes(int gpu_id,
                                          uint64_t *d_walk,
                                          uint64_t *d_offset,
                                          uint32_t size,
                                          int slot_num) {
  platform::CUDADeviceGuard guard(gpu_id);
  PADDLE_ENFORCE_NOT_NULL(graph_table,
                          paddle::platform::errors::InvalidArgument(
                              "graph_table should not be null"));
  return ((GpuPsGraphTable *)graph_table)
      ->get_feature_of_nodes(gpu_id, d_walk, d_offset, size, slot_num);
}

NeighborSampleResult GraphGpuWrapper::graph_neighbor_sample(
    int gpu_id, uint64_t *device_keys, int walk_degree, int len) {
  platform::CUDADeviceGuard guard(gpu_id);
  auto neighbor_sample_res =
      ((GpuPsGraphTable *)graph_table)
          ->graph_neighbor_sample(gpu_id, device_keys, walk_degree, len);

  return neighbor_sample_res;
}

// this function is contributed by Liwb5
std::vector<uint64_t> GraphGpuWrapper::graph_neighbor_sample(
    int gpu_id, int idx, std::vector<uint64_t> &key, int sample_size) {
  std::vector<uint64_t> res;
  if (key.size() == 0) {
    return res;
  }
  uint64_t *cuda_key;
  platform::CUDADeviceGuard guard(gpu_id);

  cudaMalloc(&cuda_key, key.size() * sizeof(uint64_t));
  cudaMemcpy(cuda_key,
             key.data(),
             key.size() * sizeof(uint64_t),
             cudaMemcpyHostToDevice);
  VLOG(0) << "key_size: " << key.size();
  auto neighbor_sample_res =
      ((GpuPsGraphTable *)graph_table)
          ->graph_neighbor_sample_v2(
<<<<<<< HEAD
              gpu_id, idx, cuda_key, sample_size, key.size(), false, true);
=======
              gpu_id, idx, cuda_key, sample_size, key.size(), false);
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
  int *actual_sample_size = new int[key.size()];
  cudaMemcpy(actual_sample_size,
             neighbor_sample_res.actual_sample_size,
             key.size() * sizeof(int),
             cudaMemcpyDeviceToHost);  // 3, 1, 3
  int cumsum = 0;
  for (int i = 0; i < key.size(); i++) {
    cumsum += actual_sample_size[i];
  }

  std::vector<uint64_t> cpu_key;
  cpu_key.resize(key.size() * sample_size);

  cudaMemcpy(cpu_key.data(),
             neighbor_sample_res.val,
             key.size() * sample_size * sizeof(uint64_t),
             cudaMemcpyDeviceToHost);
  for (int i = 0; i < key.size(); i++) {
    for (int j = 0; j < actual_sample_size[i]; j++) {
      res.push_back(key[i]);
      res.push_back(cpu_key[i * sample_size + j]);
    }
  }
  delete[] actual_sample_size;
  cudaFree(cuda_key);
  return res;
}

NodeQueryResult GraphGpuWrapper::query_node_list(int gpu_id,
                                                 int idx,
                                                 int start,
                                                 int query_size) {
  PADDLE_ENFORCE_EQ(FLAGS_gpugraph_load_node_list_into_hbm,
                    true,
                    paddle::platform::errors::PreconditionNotMet(
                        "when use query_node_list should set "
                        "gpugraph_load_node_list_into_hbm true"));
  return ((GpuPsGraphTable *)graph_table)
      ->query_node_list(gpu_id, idx, start, query_size);
}
void GraphGpuWrapper::load_node_weight(int type_id, int idx, std::string path) {
  return ((GpuPsGraphTable *)graph_table)
      ->cpu_graph_table_->load_node_weight(type_id, idx, path);
<<<<<<< HEAD
}

std::vector<int> GraphGpuWrapper::slot_feature_num_map() const {
  return ((GpuPsGraphTable *)graph_table)
      ->cpu_graph_table_->slot_feature_num_map();
=======
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
}

void GraphGpuWrapper::export_partition_files(int idx, std::string file_path) {
  return ((GpuPsGraphTable *)graph_table)
      ->cpu_graph_table_->export_partition_files(idx, file_path);
}
<<<<<<< HEAD

void GraphGpuWrapper::release_graph() {
  return ((GpuPsGraphTable *)graph_table)->cpu_graph_table_->release_graph();
}

void GraphGpuWrapper::release_graph_edge() {
  return ((GpuPsGraphTable *)graph_table)
      ->cpu_graph_table_->release_graph_edge();
}

void GraphGpuWrapper::release_graph_node() {
  return ((GpuPsGraphTable *)graph_table)
      ->cpu_graph_table_->release_graph_node();
}

std::vector<uint64_t> &GraphGpuWrapper::get_graph_total_keys() {
  return ((GpuPsGraphTable *)graph_table)->cpu_graph_table_->graph_total_keys_;
}

std::vector<std::vector<uint64_t>> &GraphGpuWrapper::get_graph_type_keys() {
  return ((GpuPsGraphTable *)graph_table)->cpu_graph_table_->graph_type_keys_;
}

std::unordered_map<int, int> &GraphGpuWrapper::get_graph_type_to_index() {
  return ((GpuPsGraphTable *)graph_table)->cpu_graph_table_->type_to_index_;
}

=======
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
#endif
}  // namespace framework
};  // namespace paddle

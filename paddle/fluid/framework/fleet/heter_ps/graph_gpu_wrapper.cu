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

#include "paddle/fluid/framework/fleet/heter_ps/graph_gpu_ps_table.h"
#include "paddle/fluid/framework/fleet/heter_ps/graph_gpu_wrapper.h"
#include "paddle/fluid/framework/fleet/heter_ps/heter_resource.h"
namespace paddle {
namespace framework {
#ifdef PADDLE_WITH_HETERPS
std::string nodes[] = {
    std::string("user\t37\ta 0.34\tb 13 14\tc hello\td abc"),
    std::string("user\t96\ta 0.31\tb 15 10\tc 96hello\td abcd"),
    std::string("user\t59\ta 0.11\tb 11 14"),
    std::string("user\t97\ta 0.11\tb 12 11"),
    std::string("item\t45\ta 0.21"),
    std::string("item\t145\ta 0.21"),
    std::string("item\t112\ta 0.21"),
    std::string("item\t48\ta 0.21"),
    std::string("item\t247\ta 0.21"),
    std::string("item\t111\ta 0.21"),
    std::string("item\t46\ta 0.21"),
    std::string("item\t146\ta 0.21"),
    std::string("item\t122\ta 0.21"),
    std::string("item\t49\ta 0.21"),
    std::string("item\t248\ta 0.21"),
    std::string("item\t113\ta 0.21")};
char node_file_name[] = "nodes.txt";
std::vector<std::string> user_feature_name = {"a", "b", "c", "d"};
std::vector<std::string> item_feature_name = {"a"};
std::vector<std::string> user_feature_dtype = {"float32", "int32", "string",
                                               "string"};
std::vector<std::string> item_feature_dtype = {"float32"};
std::vector<int> user_feature_shape = {1, 2, 1, 1};
std::vector<int> item_feature_shape = {1};
void prepare_file(char file_name[]) {
  std::ofstream ofile;
  ofile.open(file_name);

  for (auto x : nodes) {
    ofile << x << std::endl;
  }
  ofile.close();
}

void GraphGpuWrapper::set_device(std::vector<int> ids) {
  for (auto device_id : ids) {
    device_id_mapping.push_back(device_id);
  }
}
std::vector<std::vector<int64_t>> GraphGpuWrapper::get_all_id(int type, int idx,
                                                              int slice_num) {
  return ((GpuPsGraphTable *)graph_table)
      ->cpu_graph_table->get_all_id(type, idx, slice_num);
}
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

void GraphGpuWrapper::make_partitions(int idx, int64_t byte_size,
                                      int device_len) {
  ((GpuPsGraphTable *)graph_table)
      ->cpu_graph_table->make_partitions(idx, byte_size, device_len);
}
int32_t GraphGpuWrapper::load_next_partition(int idx) {
  return ((GpuPsGraphTable *)graph_table)
      ->cpu_graph_table->load_next_partition(idx);
}

void GraphGpuWrapper::set_search_level(int level) {
  ((GpuPsGraphTable *)graph_table)->cpu_graph_table->set_search_level(level);
}

std::vector<int64_t> GraphGpuWrapper::get_partition(int idx, int num) {
  return ((GpuPsGraphTable *)graph_table)
      ->cpu_graph_table->get_partition(idx, num);
}
int32_t GraphGpuWrapper::get_partition_num(int idx) {
  return ((GpuPsGraphTable *)graph_table)
      ->cpu_graph_table->get_partition_num(idx);
}
void GraphGpuWrapper::make_complementary_graph(int idx, int64_t byte_size) {
  ((GpuPsGraphTable *)graph_table)
      ->cpu_graph_table->make_complementary_graph(idx, byte_size);
}
void GraphGpuWrapper::load_edge_file(std::string name, std::string filepath,
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
        ->cpu_graph_table->Load(std::string(filepath), params);
  }
}

void GraphGpuWrapper::load_node_file(std::string name, std::string filepath) {
  // 'n' means load nodes and 'node_type' follows

  std::string params = "n" + name;

  if (feature_to_id.find(name) != feature_to_id.end()) {
    ((GpuPsGraphTable *)graph_table)
        ->cpu_graph_table->Load(std::string(filepath), params);
  }
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

void GraphGpuWrapper::init_service() {
  table_proto.set_task_pool_size(24);
  table_proto.set_search_level(search_level);
  table_proto.set_table_name("cpu_graph_table");
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
  GpuPsGraphTable *g = new GpuPsGraphTable(resource, 1);
  g->init_cpu_table(table_proto);
  graph_table = (char *)g;
}

void GraphGpuWrapper::upload_batch(int idx,
                                   std::vector<std::vector<int64_t>> &ids) {
  GpuPsGraphTable *g = (GpuPsGraphTable *)graph_table;
  // std::vector<paddle::framework::GpuPsCommGraph> vec;
  for (int i = 0; i < ids.size(); i++) {
    // vec.push_back(g->cpu_graph_table->make_gpu_ps_graph(idx, ids[i]));
    GpuPsCommGraph sub_graph =
        g->cpu_graph_table->make_gpu_ps_graph(idx, ids[i]);
    g->build_graph_on_single_gpu(sub_graph, i);
    sub_graph.release_on_cpu();
    VLOG(0) << "sub graph on gpu " << i << " is built";
  }
  // g->build_graph_from_cpu(vec);
}

void GraphGpuWrapper::initialize() {
  std::vector<int> device_id_mapping;
  for (int i = 0; i < 2; i++) device_id_mapping.push_back(i);
  int gpu_num = device_id_mapping.size();
  ::paddle::distributed::GraphParameter table_proto;
  table_proto.add_edge_types("u2u");
  table_proto.add_node_types("user");
  table_proto.add_node_types("item");
  ::paddle::distributed::GraphFeature *g_f = table_proto.add_graph_feature();

  for (int i = 0; i < user_feature_name.size(); i++) {
    g_f->add_name(user_feature_name[i]);
    g_f->add_dtype(user_feature_dtype[i]);
    g_f->add_shape(user_feature_shape[i]);
  }
  ::paddle::distributed::GraphFeature *g_f1 = table_proto.add_graph_feature();
  for (int i = 0; i < item_feature_name.size(); i++) {
    g_f1->add_name(item_feature_name[i]);
    g_f1->add_dtype(item_feature_dtype[i]);
    g_f1->add_shape(item_feature_shape[i]);
  }
  prepare_file(node_file_name);
  table_proto.set_shard_num(24);

  std::shared_ptr<HeterPsResource> resource =
      std::make_shared<HeterPsResource>(device_id_mapping);
  resource->enable_p2p();
  GpuPsGraphTable *g = new GpuPsGraphTable(resource, 1);
  g->init_cpu_table(table_proto);
  graph_table = (char *)g;
  g->cpu_graph_table->Load(node_file_name, "nuser");
  g->cpu_graph_table->Load(node_file_name, "nitem");
  std::remove(node_file_name);
  std::vector<paddle::framework::GpuPsCommGraph> vec;
  std::vector<int64_t> node_ids;
  node_ids.push_back(37);
  node_ids.push_back(96);
  std::vector<std::vector<std::string>> node_feat(2,
                                                  std::vector<std::string>(2));
  std::vector<std::string> feature_names;
  feature_names.push_back(std::string("c"));
  feature_names.push_back(std::string("d"));
  g->cpu_graph_table->get_node_feat(0, node_ids, feature_names, node_feat);
  VLOG(0) << "get_node_feat: " << node_feat[0][0];
  VLOG(0) << "get_node_feat: " << node_feat[0][1];
  VLOG(0) << "get_node_feat: " << node_feat[1][0];
  VLOG(0) << "get_node_feat: " << node_feat[1][1];
  int n = 10;
  std::vector<int64_t> ids0, ids1;
  for (int i = 0; i < n; i++) {
    g->cpu_graph_table->add_comm_edge(0, i, (i + 1) % n);
    g->cpu_graph_table->add_comm_edge(0, i, (i - 1 + n) % n);
    if (i % 2 == 0) ids0.push_back(i);
  }
  g->cpu_graph_table->build_sampler(0);
  ids1.push_back(5);
  vec.push_back(g->cpu_graph_table->make_gpu_ps_graph(0, ids0));
  vec.push_back(g->cpu_graph_table->make_gpu_ps_graph(0, ids1));
  vec[0].display_on_cpu();
  vec[1].display_on_cpu();
  g->build_graph_from_cpu(vec);
}
void GraphGpuWrapper::test() {
  int64_t cpu_key[3] = {0, 1, 2};
  void *key;
  platform::CUDADeviceGuard guard(0);
  cudaMalloc((void **)&key, 3 * sizeof(int64_t));
  cudaMemcpy(key, cpu_key, 3 * sizeof(int64_t), cudaMemcpyHostToDevice);
  auto neighbor_sample_res =
      ((GpuPsGraphTable *)graph_table)
          ->graph_neighbor_sample(0, (int64_t *)key, 2, 3);
  int64_t *res = new int64_t[7];
  cudaMemcpy(res, neighbor_sample_res.val, 3 * 2 * sizeof(int64_t),
             cudaMemcpyDeviceToHost);
  int *actual_sample_size = new int[3];
  cudaMemcpy(actual_sample_size, neighbor_sample_res.actual_sample_size,
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
NeighborSampleResult GraphGpuWrapper::graph_neighbor_sample_v3(
    NeighborSampleQuery q, bool cpu_switch) {
  return ((GpuPsGraphTable *)graph_table)
      ->graph_neighbor_sample_v3(q, cpu_switch);
}

// this function is contributed by Liwb5
std::vector<int64_t> GraphGpuWrapper::graph_neighbor_sample(
    int gpu_id, std::vector<int64_t> &key, int sample_size) {
  int64_t *cuda_key;
  platform::CUDADeviceGuard guard(gpu_id);

  cudaMalloc(&cuda_key, key.size() * sizeof(int64_t));
  cudaMemcpy(cuda_key, key.data(), key.size() * sizeof(int64_t),
             cudaMemcpyHostToDevice);

  auto neighbor_sample_res =
      ((GpuPsGraphTable *)graph_table)
          ->graph_neighbor_sample(gpu_id, cuda_key, sample_size, key.size());

  int *actual_sample_size = new int[key.size()];
  cudaMemcpy(actual_sample_size, neighbor_sample_res.actual_sample_size,
             key.size() * sizeof(int),
             cudaMemcpyDeviceToHost);  // 3, 1, 3
  int cumsum = 0;
  for (int i = 0; i < key.size(); i++) {
    cumsum += actual_sample_size[i];
  }
  /* VLOG(0) << "cumsum " << cumsum; */

  std::vector<int64_t> cpu_key, res;
  cpu_key.resize(key.size() * sample_size);

  cudaMemcpy(cpu_key.data(), neighbor_sample_res.val,
             key.size() * sample_size * sizeof(int64_t),
             cudaMemcpyDeviceToHost);
  for (int i = 0; i < key.size(); i++) {
    for (int j = 0; j < actual_sample_size[i]; j++) {
      res.push_back(key[i]);
      res.push_back(cpu_key[i * sample_size + j]);
    }
  }
  /* for(int i = 0;i < res.size();i ++) { */
  /*     VLOG(0) << i << " " << res[i]; */
  /* } */

  cudaFree(cuda_key);
  return res;
}

NodeQueryResult GraphGpuWrapper::query_node_list(int gpu_id, int start,
                                                 int query_size) {
  return ((GpuPsGraphTable *)graph_table)
      ->query_node_list(gpu_id, start, query_size);
}
#endif
}
};

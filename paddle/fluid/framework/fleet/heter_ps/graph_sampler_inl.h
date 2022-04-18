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

#ifdef PADDLE_WITH_HETERPS
namespace paddle {
namespace framework {
int CommonGraphSampler::load_from_ssd(std::string path) {
  std::ifstream file(path);
  auto _db = table->_db;
  std::string line;
  while (std::getline(file, line)) {
    auto values = paddle::string::split_string<std::string>(line, "\t");
    std::cout << values.size();
    if (values.size() < 2) continue;
    auto neighbors = paddle::string::split_string<std::string>(values[1], ";");
    std::vector<int64_t> neighbor_data;
    for (auto x : neighbors) {
      neighbor_data.push_back(std::stoll(x));
    }
    auto src_id = std::stoll(values[0]);
    _db->put(0, (char *)&src_id, sizeof(uint64_t), (char *)neighbor_data.data(),
             sizeof(int64_t) * neighbor_data.size());
    int gpu_shard = src_id % gpu_num;
    if (gpu_edges_count[gpu_shard] + neighbor_data.size() <=
        gpu_edges_each_limit) {
      gpu_edges_count[gpu_shard] += neighbor_data.size();
      gpu_set[gpu_shard].insert(src_id);
    }
    if (cpu_edges_count + neighbor_data.size() <= cpu_edges_limit) {
      cpu_edges_count += neighbor_data.size();
      for (auto x : neighbor_data) {
        // table->add_neighbor(src_id, x);
        table->shards[src_id % table->shard_num]
            ->add_graph_node(src_id)
            ->build_edges(false);
        table->shards[src_id % table->shard_num]->add_neighbor(src_id, x, 1.0);
      }
    }
    std::vector<paddle::framework::GpuPsCommGraph> graph_list;
    for (int i = 0; i < gpu_num; i++) {
      std::vector<int64_t> ids(gpu_set[i].begin(), gpu_set[i].end());
      graph_list.push_back(table->make_gpu_ps_graph(ids));
    }
    gpu_table->build_graph_from_cpu(graph_list);
    for (int i = 0; i < graph_list.size(); i++) {
      delete[] graph_list[i].node_list;
      delete[] graph_list[i].neighbor_list;
    }
  }
}
int CommonGraphSampler::run_graph_sampling() { return 0; }
void CommonGraphSampler::init(GpuPsGraphTable *g,
                              std::vector<std::string> args) {
  this->gpu_table = g;
  gpu_num = g->gpu_num;
  gpu_edges_limit = args.size() > 0 ? std::stoll(args[0]) : 1000000000LL;
  cpu_edges_limit = args.size() > 1 ? std::stoll(args[1]) : 1000000000LL;
  gpu_edges_each_limit = gpu_edges_limit / gpu_num;
  if (gpu_edges_each_limit > INT_MAX) gpu_edges_each_limit = INT_MAX;
  table = g->cpu_graph_table.get();
  gpu_edges_count = std::vector<int64_t>(gpu_num, 0);
  cpu_edges_count = 0;
  gpu_set = std::vector<std::unordered_set<int64_t>>(gpu_num);
}

int AllInGpuGraphSampler::run_graph_sampling() { return 0; }
int AllInGpuGraphSampler::load_from_ssd(std::string path) {
  graph_table->load_edges(path, false);
  sample_nodes.clear();
  sample_neighbors.clear();
  sample_res.clear();
  sample_nodes.resize(gpu_num);
  sample_neighbors.resize(gpu_num);
  sample_res.resize(gpu_num);
  std::vector<std::vector<std::vector<paddle::framework::GpuPsGraphNode>>>
      sample_nodes_ex(graph_table->task_pool_size_);
  std::vector<std::vector<std::vector<int64_t>>> sample_neighbors_ex(
      graph_table->task_pool_size_);
  for (int i = 0; i < graph_table->task_pool_size_; i++) {
    sample_nodes_ex[i].resize(gpu_num);
    sample_neighbors_ex[i].resize(gpu_num);
  }
  std::vector<std::future<int>> tasks;
  for (size_t i = 0; i < graph_table->shards.size(); ++i) {
    tasks.push_back(
        graph_table->_shards_task_pool[i % graph_table->task_pool_size_]
            ->enqueue([&, i, this]() -> int {
              if (this->status == GraphSamplerStatus::terminating) return 0;
              paddle::framework::GpuPsGraphNode node;
              std::vector<paddle::distributed::Node *> &v =
                  this->graph_table->shards[i]->get_bucket();
              size_t ind = i % this->graph_table->task_pool_size_;
              for (size_t j = 0; j < v.size(); j++) {
                size_t location = v[j]->get_id() % this->gpu_num;
                node.node_id = v[j]->get_id();
                node.neighbor_size = v[j]->get_neighbor_size();
                node.neighbor_offset =
                    (int)sample_neighbors_ex[ind][location].size();
                sample_nodes_ex[ind][location].emplace_back(node);
                for (int k = 0; k < node.neighbor_size; k++)
                  sample_neighbors_ex[ind][location].push_back(
                      v[j]->get_neighbor_id(k));
              }
              return 0;
            }));
  }
  for (size_t i = 0; i < tasks.size(); i++) tasks[i].get();
  tasks.clear();
  for (size_t i = 0; i < gpu_num; i++) {
    tasks.push_back(
        graph_table->_shards_task_pool[i % graph_table->task_pool_size_]
            ->enqueue([&, i, this]() -> int {
              if (this->status == GraphSamplerStatus::terminating) return 0;
              int total_offset = 0;
              size_t ind = i;
              for (int j = 0; j < this->graph_table->task_pool_size_; j++) {
                for (size_t k = 0; k < sample_nodes_ex[j][ind].size(); k++) {
                  sample_nodes[ind].push_back(sample_nodes_ex[j][ind][k]);
                  sample_nodes[ind].back().neighbor_offset += total_offset;
                }
                size_t neighbor_size = sample_neighbors_ex[j][ind].size();
                total_offset += neighbor_size;
                for (size_t k = 0; k < neighbor_size; k++) {
                  sample_neighbors[ind].push_back(
                      sample_neighbors_ex[j][ind][k]);
                }
              }
              return 0;
            }));
  }
  for (size_t i = 0; i < tasks.size(); i++) tasks[i].get();
  for (size_t i = 0; i < gpu_num; i++) {
    sample_res[i].node_list = sample_nodes[i].data();
    sample_res[i].neighbor_list = sample_neighbors[i].data();
    sample_res[i].node_size = sample_nodes[i].size();
    sample_res[i].neighbor_size = sample_neighbors[i].size();
  }

  gpu_table->build_graph_from_cpu(sample_res);
  return 0;
}
void AllInGpuGraphSampler::init(GpuPsGraphTable *g,
                                std::vector<std::string> args_) {
  this->gpu_table = g;
  this->gpu_num = g->gpu_num;
  graph_table = g->cpu_graph_table.get();
}
}
};
#endif

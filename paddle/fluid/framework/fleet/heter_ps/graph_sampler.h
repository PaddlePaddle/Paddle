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
#include <time.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include "paddle/fluid/distributed/ps/table/common_graph_table.h"
#include "paddle/fluid/framework/fleet/heter_ps/gpu_graph_node.h"
#include "paddle/fluid/framework/fleet/heter_ps/graph_gpu_ps_table.h"
#include "paddle/fluid/string/printf.h"
#include "paddle/fluid/string/string_helper.h"
#ifdef PADDLE_WITH_HETERPS
namespace paddle {
namespace framework {
enum GraphSamplerStatus { waiting = 0, running = 1, terminating = 2 };
class GraphSampler {
 public:
  GraphSampler() {
    status = GraphSamplerStatus::waiting;
    thread_pool.reset(new ::ThreadPool(1));
  }
  virtual int start_service(std::string path) {
    load_from_ssd(path);
    VLOG(0) << "load from ssd over";
    std::promise<int> prom;
    std::future<int> fut = prom.get_future();
    graph_sample_task_over = thread_pool->enqueue([&prom, this]() {
      VLOG(0) << " promise set ";
      prom.set_value(0);
      status = GraphSamplerStatus::running;
      return run_graph_sampling();
    });
    return fut.get();
    return 0;
  }
  virtual int end_graph_sampling() {
    if (status == GraphSamplerStatus::running) {
      status = GraphSamplerStatus::terminating;
      return graph_sample_task_over.get();
    }
    return -1;
  }
  ~GraphSampler() { end_graph_sampling(); }
  virtual int load_from_ssd(std::string path) = 0;
  ;
  virtual int run_graph_sampling() = 0;
  ;
  virtual void init(GpuPsGraphTable *gpu_table,
                    std::vector<std::string> args_) = 0;
  std::shared_ptr<::ThreadPool> thread_pool;
  GraphSamplerStatus status;
  std::future<int> graph_sample_task_over;
};

class CommonGraphSampler : public GraphSampler {
 public:
  CommonGraphSampler() {}
  virtual ~CommonGraphSampler() {}
  GpuPsGraphTable *g_table;
  virtual int load_from_ssd(std::string path);
  virtual int run_graph_sampling();
  virtual void init(GpuPsGraphTable *g, std::vector<std::string> args);
  GpuPsGraphTable *gpu_table;
  paddle::distributed::GraphTable *table;
  std::vector<int64_t> gpu_edges_count;
  int64_t cpu_edges_count;
  int64_t gpu_edges_limit, cpu_edges_limit, gpu_edges_each_limit;
  std::vector<std::unordered_set<int64_t>> gpu_set;
  int gpu_num;
};

class AllInGpuGraphSampler : public GraphSampler {
 public:
  AllInGpuGraphSampler() {}
  virtual ~AllInGpuGraphSampler() {}
  // virtual pthread_rwlock_t *export_rw_lock();
  virtual int run_graph_sampling();
  virtual int load_from_ssd(std::string path);
  virtual void init(GpuPsGraphTable *g, std::vector<std::string> args_);

 protected:
  paddle::distributed::GraphTable *graph_table;
  GpuPsGraphTable *gpu_table;
  std::vector<std::vector<paddle::framework::GpuPsGraphNode>> sample_nodes;
  std::vector<std::vector<int64_t>> sample_neighbors;
  std::vector<GpuPsCommGraph> sample_res;
  // std::shared_ptr<std::mt19937_64> random;
  int gpu_num;
};
}
};
#include "paddle/fluid/framework/fleet/heter_ps/graph_sampler_inl.h"
#endif

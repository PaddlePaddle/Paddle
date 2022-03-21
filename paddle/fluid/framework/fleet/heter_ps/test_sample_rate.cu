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

#include <unistd.h>
#include <condition_variable>  // NOLINT
#include <fstream>
#include <iomanip>
#include <string>
#include <thread>  // NOLINT
#include <unordered_set>
#include <vector>
#include "google/protobuf/text_format.h"

#include <chrono>
#include "gtest/gtest.h"
#include "paddle/fluid/distributed/ps.pb.h"
#include "paddle/fluid/distributed/ps/service/env.h"
#include "paddle/fluid/distributed/ps/service/sendrecv.pb.h"
#include "paddle/fluid/distributed/ps/table/common_graph_table.h"
#include "paddle/fluid/distributed/ps/table/graph/graph_node.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/string/printf.h"
#include "paddle/phi/kernels/funcs/math_function.h"

#include "paddle/fluid/framework/fleet/heter_ps/feature_value.h"
#include "paddle/fluid/framework/fleet/heter_ps/graph_gpu_ps_table.h"
#include "paddle/fluid/framework/fleet/heter_ps/heter_comm.h"
#include "paddle/fluid/framework/fleet/heter_ps/heter_resource.h"
#include "paddle/fluid/framework/fleet/heter_ps/optimizer.cuh.h"
#include "paddle/fluid/platform/cuda_device_guard.h"

using namespace paddle::framework;
namespace platform = paddle::platform;
namespace operators = paddle::operators;
namespace memory = paddle::memory;
namespace distributed = paddle::distributed;

std::vector<std::string> edges = {
    std::string("37\t45\t0.34"),  std::string("37\t145\t0.31"),
    std::string("37\t112\t0.21"), std::string("96\t48\t1.4"),
    std::string("96\t247\t0.31"), std::string("96\t111\t1.21"),
    std::string("59\t45\t0.34"),  std::string("59\t145\t0.31"),
    std::string("59\t122\t0.21"), std::string("97\t48\t0.34"),
    std::string("97\t247\t0.31"), std::string("97\t111\t0.21")};
// odd id:96 48 122 112
char edge_file_name[] = "edges.txt";

void prepare_file(char file_name[], std::vector<std::string> data) {
  std::ofstream ofile;
  ofile.open(file_name);
  for (auto x : data) {
    ofile << x << std::endl;
  }

  ofile.close();
}

void testSampleRate() {
#ifdef PADDLE_WITH_HETERPS
  std::vector<int64_t> ids;
  int start = 0;
  pthread_rwlock_t rwlock;
  pthread_rwlock_init(&rwlock, NULL);
  int fixed_key_size = 100, sample_size = 100;
  {
    ::paddle::distributed::GraphParameter table_proto;
    table_proto.set_gpups_mode(false);
    table_proto.set_shard_num(127);
    table_proto.set_task_pool_size(24);
    std::cerr << "initializing begin";
    distributed::GraphTable graph_table;
    graph_table.initialize(table_proto);
    std::cerr << "initializing done";
    prepare_file(edge_file_name, edges);
    graph_table.load(std::string(edge_file_name), std::string("e>"));
    int sample_actual_size = -1;
    int step = 4, cur = 0;
    while (sample_actual_size != 0) {
      std::unique_ptr<char[]> buffer;
      graph_table.pull_graph_list(cur, step, buffer, sample_actual_size, false,
                                  1);
      if (sample_actual_size == 0) break;
      char *buff = buffer.get();
      for (int i = 0; i < sample_actual_size; i++) {
        ids.push_back(*(int64_t *)(buff + i * sizeof(int64_t)));
        int swap_pos = rand() % ids.size();
        std::swap(ids[swap_pos], ids[(int)ids.size() - 1]);
      }
      cur += sample_actual_size;
    }
    std::cerr << "load ids done" << std::endl;
    std::vector<int64_t> sample_id[10], sample_neighbors[10];
    std::vector<int> actual_size[10];
    auto func = [&rwlock, &graph_table, &ids, &sample_id, &actual_size,
                 &sample_neighbors, &start, &fixed_key_size,
                 &sample_size](int i) {
      while (true) {
        int s, sn;
        bool exit = false;
        pthread_rwlock_wrlock(&rwlock);
        if (start < ids.size()) {
          s = start;
          sn = ids.size() - start;
          sn = min(sn, fixed_key_size);
          start += sn;
        } else {
          exit = true;
        }
        pthread_rwlock_unlock(&rwlock);
        if (exit) break;
        std::vector<std::shared_ptr<char>> buffers(sn);
        std::vector<int> ac(sn);
        auto status = graph_table.random_sample_neighbors(
            ids.data() + s, sample_size, buffers, ac, false);
        for (int j = s; j < s + sn; j++) {
          sample_id[i].push_back(ids[j]);
          actual_size[i].push_back(ac[j - s] / sizeof(int64_t));
          int ss = ac[j - s] / sizeof(int64_t);
          for (int k = 0; k < ss; k++) {
            sample_neighbors[i].push_back(
                *((int64_t *)(buffers[j - s].get() + k * sizeof(int64_t))));
          }
        }
      }
      VLOG(0) << "func " << i << " returns ";
    };
    auto start1 = std::chrono::steady_clock::now();
    std::thread thr[10];
    for (int i = 0; i < 10; i++) {
      thr[i] = std::thread(func, i);
    }
    for (int i = 0; i < 10; i++) thr[i].join();
    auto end1 = std::chrono::steady_clock::now();
    auto tt =
        std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);
    std::cerr << "total time cost without cache is " << tt.count() << " us"
              << std::endl;
  }
/*
  const int gpu_num = 8;
  ::paddle::distributed::GraphParameter table_proto;
  table_proto.set_gpups_mode(true);
  table_proto.set_shard_num(127);
  table_proto.set_gpu_num(gpu_num);
  table_proto.set_gpups_graph_sample_class("BasicBfsGraphSampler");
  table_proto.set_gpups_graph_sample_args("10000000,10000000,1,1");
  // prepare_file(edge_file_name, edges);
  std::vector<int> dev_ids;
  for (int i = 0; i < gpu_num; i++) {
    dev_ids.push_back(i);
  }
  std::shared_ptr<HeterPsResource> resource =
      std::make_shared<HeterPsResource>(dev_ids);
  resource->enable_p2p();
  GpuPsGraphTable g(resource);
  g.init_cpu_table(table_proto);
  g.load(std::string(edge_file_name), std::string("e>"));
  void *key;
  cudaMalloc((void **)&key, ids.size() * sizeof(int64_t));
  cudaMemcpy(key, ids.data(), ids.size() * sizeof(int64_t),
             cudaMemcpyHostToDevice);
  std::vector<NeighborSampleResult *> res[gpu_num];
  start = 0;
  auto func = [&rwlock, &g, &res, &start, &fixed_key_size, &sample_size,
               &gpu_num, &ids, &key](int i) {
    while (true) {
      int s, sn;
      bool exit = false;
      pthread_rwlock_wrlock(&rwlock);
      if (start < ids.size()) {
        s = start;
        sn = ids.size() - start;
        sn = min(sn, fixed_key_size);
        start += sn;
      } else {
        exit = true;
      }
      pthread_rwlock_unlock(&rwlock);
      if (exit) break;
      auto r =
          g.graph_neighbor_sample(i, (int64_t *)(key + s), sample_size, sn);
      res[i].push_back(r);
    }
  };
  auto start1 = std::chrono::steady_clock::now();
  std::thread thr[gpu_num];
  for (int i = 0; i < gpu_num; i++) {
    thr[i] = std::thread(func, i);
  }
  for (int i = 0; i < gpu_num; i++) thr[i].join();
  auto end1 = std::chrono::steady_clock::now();
  auto tt =
      std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);
  std::cerr << "total time cost without cache is " << tt.count() << " us"
            << std::endl;
*/
#endif
}

TEST(testSampleRate, Run) { testSampleRate(); }

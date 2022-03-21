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
namespace framework = paddle::framework;
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
  int fixed_sample_size = 100, sample_size = 100;
  {
    ::paddle::distributed::GraphParameter table_proto;
    table_proto.set_gpups_mode(false);
    table_proto.set_gpups_mode_shard_num(127);
    table_proto.set_task_pool_size(24);

    distributed::GraphTable graph_table;
    graph_table.initialize(table_proto);
    prepare_file(edge_file_name, edges);
    graph_table.load(std::string(edge_file_name), std::string("e>"));
    int actual_size = -1;
    int step = 4, cur = 0;
    while (actual_size != 0) {
      std::unique_ptr<char[]> buffer;
      graph_table.pull_graph_list(cur, step, buffer, actual_size, false, 1);
      if (actual_size == 0) break;
      char *buff = buffer.get();
      for (int i = 0; i < actual_size; i++) {
        ids.push_back((int64_t *)(buff + i * sizeof(int64_t)));
        int swap_pos = rand() % ids.size();
        swap(ids[swap_pos], ids[(int)ids.size() - 1]);
      }
      cur += actual_size;
    }
    std::vector<int64_t> sample_id[10], actual_size[10], sample_neighbors[10];
    auto func = [](int i) {
      while (true) {
        int s, sn;
        bool exit = false;
        pthread_rwlock_wrlock(&rwlock);
        if (start < ids.size()) {
          s = start;
          sn = ids.size() - start;
          sn = min(sn, fixed_sample_size);
          start += sn;
        } else {
          exit = true;
        }
        pthread_rwlock_unlock(&rwlock);
        if (exit) break;
        std::vector<std::shared_ptr<char>> buffers(sn);
        std::vector<int> ac(sn);
        graph_table.random_sample_neighbors(ids.begin() + start, sample_size,
                                            bufffers, ac, false);
        for (int j = start; j < start + sn; j++) {
          sample_id[i].push_back(ids[j]);
          actual_size[i].push_back(ac[j]);
          for (int k = 0; k < ac[j]; k++) {
            sample_neighbors[i].push_back(
                *((int64_t *)(buffers[j].get() + k * sizeof(int64_t))));
          }
        }
      }
    };
    auto start1 = std::chrono::steady_clock::now();
    std::thread thr[10];
    for (int i = 0; i < 10; i++) {
      thr[i] = std::thread(i);
    }
    for (int i = 0; i < 10; i++) thr[i].join();
    auto end1 = std::chrono::steady_clock::now();
    std::cerr << "total time cost without cache is " << tt.count() << " us"
              << std::endl;
  }
  // pthread_rwlock_wrlock(&rwlock);
  // pthread_rwlock_unlock(&rwlock);
  // std::vector<paddle::framework::GpuPsCommGraph> res;
  // std::promise<int> prom;
  // std::future<int> fut = prom.get_future();
  // graph_table.set_graph_sample_callback(
  //     [&res, &prom](std::vector<paddle::framework::GpuPsCommGraph> &res0) {
  //       res = res0;
  //       prom.set_value(0);
  //     });
  // graph_table.start_graph_sampling();
  // fut.get();
  // graph_table.end_graph_sampling();
  // ASSERT_EQ(2, res.size());
  // // 37 59 97
  // for (int i = 0; i < (int)res[1].node_size; i++) {
  //   std::cout << res[1].node_list[i].node_id << std::endl;
  // }
  // ASSERT_EQ(3, res[1].node_size);

  ::paddle::distributed::GraphParameter table_proto;
  table_proto.set_gpups_mode(true);
  table_proto.set_gpups_mode_shard_num(127);
  table_proto.set_gpu_num(3);
  table_proto.set_gpups_graph_sample_class("BasicBfsGraphSampler");
  table_proto.set_gpups_graph_sample_args("10000000,10000000,1,1");
  prepare_file(edge_file_name, edges);
  g.init_cpu_table(table_proto);
  g.load(std::string(edge_file_name), std::string("e>"));
  void *key;
  cudaMalloc((void **)&key, ids.size() * sizeof(int64_t));
  cudaMemcpy(key, ids.data(), ids.size() * sizeof(int64_t),
             cudaMemcpyHostToDevice);
  std::vector<NeighborSampleResult *> res[8];
  start = 0;
  auto func = [](int i) {
    while (true) {
      int s, sn;
      bool exit = false;
      pthread_rwlock_wrlock(&rwlock);
      if (start < ids.size()) {
        s = start;
        sn = ids.size() - start;
        sn = min(sn, fixed_sample_size);
        start += sn;
      } else {
        exit = true;
      }
      pthread_rwlock_unlock(&rwlock);
      if (exit) break;
      auto r = graph_table.graph_neighbor_sample(i, (int64_t *)(key + s),
                                                 fixed_sample_size, sn);
      res[i].push_back(r);
    }
  };
  auto start1 = std::chrono::steady_clock::now();
  std::thread thr[8];
  for (int i = 0; i < 8; i++) {
    thr[i] = std::thread(i);
  }
  for (int i = 0; i < 8; i++) thr[i].join();
  auto end1 = std::chrono::steady_clock::now();
  std::cerr << "total time cost without cache is " << tt.count() << " us"
            << std::endl;

#endif
}

TEST(testSampleRate, Run) { testSampleRate(); }

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

#include <chrono>
#include <condition_variable>  // NOLINT
#include <fstream>
#include <iomanip>
#include <string>
#include <thread>  // NOLINT
#include <unordered_set>
#include <vector>

#include "google/protobuf/text_format.h"
#include "gtest/gtest.h"
#include "paddle/fluid/distributed/ps/service/env.h"
#include "paddle/fluid/distributed/ps/service/sendrecv.pb.h"
#include "paddle/fluid/distributed/ps/table/common_graph_table.h"
#include "paddle/fluid/distributed/ps/table/graph/graph_node.h"
#include "paddle/fluid/distributed/the_one_ps.pb.h"
#include "paddle/fluid/framework/fleet/heter_ps/feature_value.h"
#include "paddle/fluid/framework/fleet/heter_ps/graph_gpu_ps_table.h"
#include "paddle/fluid/framework/fleet/heter_ps/graph_sampler.h"
#include "paddle/fluid/framework/fleet/heter_ps/heter_comm.h"
#include "paddle/fluid/framework/fleet/heter_ps/heter_resource.h"
#include "paddle/fluid/framework/fleet/heter_ps/optimizer.cuh.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/platform/cuda_device_guard.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/utils/string/printf.h"

using paddle::framework;
namespace platform = paddle::platform;
namespace operators = paddle::operators;
namespace memory = paddle::memory;
namespace distributed = paddle::distributed;

const char *input_file;
int exe_count = 100;
int use_nv = 1;
int fixed_key_size = 50000, sample_size = 32,
    bfs_sample_nodes_in_each_shard = 10000, init_search_size = 1,
    bfs_sample_edges = 20, gpu_num1 = 8, gpu_num = 8;
const char *gpu_str = "0,1,2,3,4,5,6,7";
int64_t *key[8];
std::vector<std::string> edges = {std::string("37\t45\t0.34"),
                                  std::string("37\t145\t0.31"),
                                  std::string("37\t112\t0.21"),
                                  std::string("96\t48\t1.4"),
                                  std::string("96\t247\t0.31"),
                                  std::string("96\t111\t1.21"),
                                  std::string("59\t45\t0.34"),
                                  std::string("59\t145\t0.31"),
                                  std::string("59\t122\t0.21"),
                                  std::string("97\t48\t0.34"),
                                  std::string("97\t247\t0.31"),
                                  std::string("97\t111\t0.21")};
// odd id:96 48 122 112
char edge_file_name[] = "test_edges.txt";

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

  {
    ::paddle::distributed::GraphParameter table_proto;
    // table_proto.set_gpups_mode(false);
    table_proto.set_shard_num(127);
    table_proto.set_task_pool_size(24);
    std::cerr << "initializing begin";
    distributed::GraphTable graph_table;
    graph_table.Initialize(table_proto);
    std::cerr << "initializing done";
    graph_table.Load(input_file, std::string("e>"));
    int sample_actual_size = -1;
    int step = fixed_key_size, cur = 0;
    while (sample_actual_size != 0) {
      std::unique_ptr<char[]> buffer;
      graph_table.pull_graph_list(
          cur, step, buffer, sample_actual_size, false, 1);
      int index = 0;
      while (index < sample_actual_size) {
        paddle::distributed::FeatureNode node;
        node.recover_from_buffer(buffer.get() + index);
        index += node.get_size(false);
        // res.push_back(node);
        ids.push_back(node.get_id());
        int swap_pos = rand_r() % ids.size();
        std::swap(ids[swap_pos], ids[static_cast<int>(ids.size()) - 1]);
      }
      cur = ids.size();
      // if (sample_actual_size == 0) break;
      // char *buff = buffer.get();
      // for (int i = 0; i < sample_actual_size/sizeof(int64_t); i++) {
      //   ids.push_back(*((int64_t *)buff + i));
      //   int swap_pos = rand() % ids.size();
      //   std::swap(ids[swap_pos], ids[(int)ids.size() - 1]);
      // }
      // cur += sample_actual_size/sizeof(int64_t);
    }
    std::cerr << "load ids done" << std::endl;
    std::vector<int64_t> sample_id[10], sample_neighbors[10];
    std::vector<int> actual_size[10];
    auto func = [&rwlock,
                 &graph_table,
                 &ids,
                 &sample_id,
                 &actual_size,
                 &sample_neighbors,
                 &start](int i) {
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
            sample_neighbors[i].push_back(*(reinterpret_cast<int64_t *>(
                buffers[j - s].get() + k * sizeof(int64_t))));
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
    int64_t tot = 0;
    for (int i = 0; i < 10; i++) {
      for (auto x : sample_id[i]) tot += x;
    }
    VLOG(0) << "sum = " << tot;
  }
  gpu_num = 0;
  int st = 0, u = 0;
  std::vector<int> device_id_mapping;
  while (u < gpu_str.size()) {
    VLOG(0) << u << " " << gpu_str[u];
    if (gpu_str[u] == ',') {
      auto p = gpu_str.substr(st, u - st);
      int id = std::stoi(p);
      VLOG(0) << "got a new device id" << id;
      device_id_mapping.push_back(id);
      st = u + 1;
    }
    u++;
  }
  auto p = gpu_str.substr(st, gpu_str.size() - st);
  int id = std::stoi(p);
  VLOG(0) << "got a new device id" << id;
  device_id_mapping.push_back(id);
  gpu_num = device_id_mapping.size();
  ::paddle::distributed::GraphParameter table_proto;
  table_proto.set_shard_num(24);
  // table_proto.set_gpups_graph_sample_class("CompleteGraphSampler");

  std::shared_ptr<HeterPsResource> resource =
      std::make_shared<HeterPsResource>(device_id_mapping);
  resource->enable_p2p();
  GpuPsGraphTable g(resource, use_nv);
  g.init_cpu_table(table_proto);
  std::vector<std::string> arg;
  AllInGpuGraphSampler sampler;
  sampler.init(&g, arg);
  // g.load(std::string(input_file), std::string("e>"));
  // sampler.start(std::string(input_file));
  // sampler.load_from_ssd(std::string(input_file));
  sampler.start_service(input_file);
  /*
  NodeQueryResult *query_node_res;
  query_node_res = g.query_node_list(0, 0, ids.size() + 10000);

  VLOG(0) << "gpu got " << query_node_res->actual_sample_size << " nodes ";
  VLOG(0) << "cpu got " << ids.size() << " nodes";
  ASSERT_EQ((int)query_node_res->actual_sample_size, (int)ids.size());

  int64_t *gpu_node_res = new int64_t[ids.size()];
  cudaMemcpy(gpu_node_res, query_node_res->val, ids.size() * sizeof(int64_t),
             cudaMemcpyDeviceToHost);
  std::unordered_set<int64_t> cpu_node_set, gpu_node_set;
  for (auto x : ids) {
    cpu_node_set.insert(x);
  }
  for (int i = 0; i < (int)query_node_res->actual_sample_size; i++) {
    auto x = gpu_node_res[i];
    ASSERT_EQ(cpu_node_set.find(x) != cpu_node_set.end(), true);
    gpu_node_set.insert(x);
  }
  VLOG(0) << " cpu_node_size = " << cpu_node_set.size();
  VLOG(0) << " gpu_node_size = " << gpu_node_set.size();
  ASSERT_EQ(cpu_node_set.size(), gpu_node_set.size());
  for (int i = 0; i < 20; i++) {
    int st = ids.size() / 20 * i;
    auto q = g.query_node_list(0, st, ids.size() / 20);
    VLOG(0) << " the " << i << "th iteration size = " << q->actual_sample_size;
  }
  // NodeQueryResult *query_node_list(int gpu_id, int start, int query_size);
*/
  for (int i = 0; i < gpu_num1; i++) {
    platform::CUDADeviceGuard guard(device_id_mapping[i]);
    cudaMalloc(reinterpret_cast<void **>(&key[i]),
               ids.size() * sizeof(int64_t));
    cudaMemcpy(key[i],
               ids.data(),
               ids.size() * sizeof(int64_t),
               cudaMemcpyHostToDevice);
  }
  /*
  cudaMalloc((void **)&key, ids.size() * sizeof(int64_t));
  cudaMemcpy(key, ids.data(), ids.size() * sizeof(int64_t),
             cudaMemcpyHostToDevice);
             */
  /*
  std::vector<std::vector<NeighborSampleResult *>> res(gpu_num1);
  for (int i = 0; i < gpu_num1; i++) {
    int st = 0;
    int size = ids.size();
    NeighborSampleResult *result = new NeighborSampleResult(sample_size, size);
    phi::GPUPlace place = phi::GPUPlace(device_id_mapping[i]);
    platform::CUDADeviceGuard guard(device_id_mapping[i]);
    cudaMalloc((void **)&result->val, size * sample_size * sizeof(int64_t));
    cudaMalloc((void **)&result->actual_sample_size, size * sizeof(int));
    res[i].push_back(result);
  }
  */

  // g.graph_neighbor_sample
  start = 0;
  auto func = [&rwlock, &g, &start, &ids](int i) {
    int st = 0;
    int size = ids.size();
    for (int k = 0; k < exe_count; k++) {
      st = 0;
      while (st < size) {
        int len = std::min(fixed_key_size, static_cast<int>(ids.size()) - st);
        auto r = g.graph_neighbor_sample(
            i, reinterpret_cast<int64_t *>(key[i] + st), sample_size, len);
        st += len;
        delete r;
      }
    }
  };
  auto start1 = std::chrono::steady_clock::now();
  std::thread thr[gpu_num1];  // NOLINT
  for (int i = 0; i < gpu_num1; i++) {
    thr[i] = std::thread(func, i);
  }
  for (int i = 0; i < gpu_num1; i++) thr[i].join();
  auto end1 = std::chrono::steady_clock::now();
  auto tt =
      std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);
  std::cerr << "total time cost without cache for v1 is "
            << tt.count() / exe_count / gpu_num1 << " us" << std::endl;

  // g.graph_neighbor_sample_v2
  start = 0;
  auto func2 = [&rwlock, &g, &start, &ids](int i) {
    int st = 0;
    int size = ids.size();
    for (int k = 0; k < exe_count; k++) {
      st = 0;
      while (st < size) {
        int len = std::min(fixed_key_size, static_cast<int>(ids.size()) - st);
        auto r =
            g.graph_neighbor_sample_v2(i,
                                       reinterpret_cast<int64_t *>(key[i] + st),
                                       sample_size,
                                       len,
                                       false);
        st += len;
        delete r;
      }
    }
  };
  auto start2 = std::chrono::steady_clock::now();
  std::thread thr2[gpu_num1];  // NOLINT
  for (int i = 0; i < gpu_num1; i++) {
    thr2[i] = std::thread(func2, i);
  }
  for (int i = 0; i < gpu_num1; i++) thr2[i].join();
  auto end2 = std::chrono::steady_clock::now();
  auto tt2 =
      std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2);
  std::cerr << "total time cost without cache for v2 is "
            << tt2.count() / exe_count / gpu_num1 << " us" << std::endl;

  for (int i = 0; i < gpu_num1; i++) {
    cudaFree(key[i]);
  }
#endif
}

TEST(TEST_FLEET, sample_rate) { testSampleRate(); }

int main(int argc, char *argv[]) {
  for (int i = 0; i < argc; i++)
    VLOG(0) << "Argument " << i << " is " << std::string(argv[i]);
  if (argc > 1) {
    input_file = argv[1];
  } else {
    prepare_file(edge_file_name, edges);
    input_file = edge_file_name;
  }
  VLOG(0) << "input_file is " << input_file;
  if (argc > 2) {
    fixed_key_size = std::stoi(argv[2]);
  }
  VLOG(0) << "sample_node_size for every batch is " << fixed_key_size;
  if (argc > 3) {
    sample_size = std::stoi(argv[3]);
  }
  VLOG(0) << "sample_size neighbor_size is " << sample_size;
  if (argc > 4) init_search_size = std::stoi(argv[4]);
  VLOG(0) << " init_search_size " << init_search_size;
  if (argc > 5) {
    gpu_str = argv[5];
  }
  VLOG(0) << " gpu_str= " << gpu_str;
  gpu_num = 0;
  if (argc > 6) gpu_num1 = std::stoi(argv[6]);
  VLOG(0) << " gpu_thread_num= " << gpu_num1;
  if (argc > 7) use_nv = std::stoi(argv[7]);
  VLOG(0) << " use_nv " << use_nv;
  testSampleRate();
}

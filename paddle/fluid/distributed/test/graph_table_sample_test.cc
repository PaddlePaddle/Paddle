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

std::vector<std::string> nodes = {
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

void prepare_file(char file_name[], std::vector<std::string> data) {
  std::ofstream ofile;
  ofile.open(file_name);
  for (auto x : data) {
    ofile << x << std::endl;
  }

  ofile.close();
}

void testGraphSample() {
#ifdef PADDLE_WITH_HETERPS
  ::paddle::distributed::GraphParameter table_proto;
  table_proto.set_gpups_mode(true);
  table_proto.set_shard_num(127);
  table_proto.set_gpu_num(2);

  distributed::GraphTable graph_table, graph_table1;
  graph_table.initialize(table_proto);
  prepare_file(edge_file_name, edges);
  graph_table.load(std::string(edge_file_name), std::string("e>"));
  std::vector<paddle::framework::GpuPsCommGraph> res;
  std::promise<int> prom;
  std::future<int> fut = prom.get_future();
  graph_table.set_graph_sample_callback(
      [&res, &prom](std::vector<paddle::framework::GpuPsCommGraph> &res0) {
        res = res0;
        prom.set_value(0);
      });
  graph_table.start_graph_sampling();
  fut.get();
  graph_table.end_graph_sampling();
  ASSERT_EQ(2, res.size());
  // 37 59 97
  for (int i = 0; i < (int)res[1].node_size; i++) {
    std::cout << res[1].node_list[i].node_id << std::endl;
  }
  ASSERT_EQ(3, res[1].node_size);

  ::paddle::distributed::GraphParameter table_proto1;
  table_proto1.set_gpups_mode(true);
  table_proto1.set_shard_num(127);
  table_proto1.set_gpu_num(2);
  table_proto1.set_gpups_graph_sample_class("BasicBfsGraphSampler");
  table_proto1.set_gpups_graph_sample_args("5,5,1,1");
  graph_table1.initialize(table_proto1);
  graph_table1.load(std::string(edge_file_name), std::string("e>"));
  std::vector<paddle::framework::GpuPsCommGraph> res1;
  std::promise<int> prom1;
  std::future<int> fut1 = prom1.get_future();
  graph_table1.set_graph_sample_callback(
      [&res1, &prom1](std::vector<paddle::framework::GpuPsCommGraph> &res0) {
        res1 = res0;
        prom1.set_value(0);
      });
  graph_table1.start_graph_sampling();
  fut1.get();
  graph_table1.end_graph_sampling();
  // distributed::BasicBfsGraphSampler *sampler1 =
  //     (distributed::BasicBfsGraphSampler *)graph_table1.get_graph_sampler();
  //     sampler1->start_graph_sampling();
  //     std::this_thread::sleep_for (std::chrono::seconds(1));
  // std::vector<paddle::framework::GpuPsCommGraph> res1;// =
  // sampler1->fetch_sample_res();
  ASSERT_EQ(2, res1.size());
  // odd id:96 48 122 112
  for (int i = 0; i < (int)res1[0].node_size; i++) {
    std::cout << res1[0].node_list[i].node_id << std::endl;
  }
  ASSERT_EQ(4, res1[0].node_size);
#endif
}

TEST(testGraphSample, Run) { testGraphSample(); }

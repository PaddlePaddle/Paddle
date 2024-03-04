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
#include "paddle/fluid/distributed/ps/table/common_graph_table.h"
#include "paddle/fluid/distributed/the_one_ps.pb.h"

namespace distributed = paddle::distributed;

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
char edge_file_name[] = "edges.txt";  // NOLINT

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
char node_file_name[] = "nodes.txt";  // NOLINT

void prepare_file(char file_name[], std::vector<std::string> data) {  // NOLINT
  std::ofstream ofile;
  ofile.open(file_name);
  for (auto x : data) {
    ofile << x << std::endl;
  }

  ofile.close();
}

void testGraphSample() {
  ::paddle::distributed::GraphParameter table_proto;
  // table_proto.set_gpu_num(2);

  distributed::GraphTable graph_table;
  graph_table.Initialize(table_proto);
}

TEST(testGraphSample, Run) { testGraphSample(); }

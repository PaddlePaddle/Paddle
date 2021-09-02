// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/node.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/platform/variant.h"

namespace paddle {

namespace framework {

class Graph;

class CostData {
  public:
    std::map<int, int64> GetTimeMap();
    std::map<int, int64> GetMemMap();
    int64 GetWholeTime();
    int64 GetWholeMem();

  private:
    Graph* graph_;
    Program* program_;
    std::map<int, int64> time_; // from Op Node id to time
    std::map<int, int64> memory_; // from Op Node id to total memory bytes
    std::map<int, int64> comm_; // from Op Node id to communicate cost
    int64 whole_time_; // time cost of the whole program or graph
    int64 whole_memory; // memory cost of the whole program or graph
    int64 whole_comm_;// communication cost of the whole program or graph
    const static int NOT_MEASURED = -1;

};
class CostModel {
 public:
  CostData profile_measure(Program* program, std::string device, fetch_cost_list);

 private:
  // Number of times Node has been executed
  std::vector<int> count_;
  // Total execution time
  std::vector<int>
  // Total Bytes output on each channel
}

}  // namespace framework
}  // namespace paddle

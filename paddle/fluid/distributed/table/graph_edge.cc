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

#include "paddle/fluid/distributed/table/graph_edge.h"
#include <cstring>
namespace paddle {
namespace distributed {

void GraphEdgeBlob::add_edge(uint64_t id, float weight = 1) {
  id_arr.push_back(id);
}

void WeightedGraphEdgeBlob::add_edge(uint64_t id, float weight = 1) {
  id_arr.push_back(id);
  weight_arr.push_back(weight);
}
}
}

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
#ifdef PADDLE_WITH_HETERPS
namespace paddle {
namespace framework {
struct GpuPsGraphNode {
  int64_t node_id;
  int neighbor_size, neighbor_offset;
  // this node's neighbor is stored on [neighbor_offset,neighbor_offset +
  // neighbor_size) of int64_t *neighbor_list;
};

struct GpuPsCommGraph {
  int64_t *neighbor_list;
  GpuPsGraphNode *node_list;
  int neighbor_size, node_size;
  // the size of neighbor array and graph_node_list array
  GpuPsCommGraph()
      : neighbor_list(NULL), node_list(NULL), neighbor_size(0), node_size(0) {}
  GpuPsCommGraph(int64_t *neighbor_list_, GpuPsGraphNode *node_list_,
                 int neighbor_size_, int node_size_)
      : neighbor_list(neighbor_list_),
        node_list(node_list_),
        neighbor_size(neighbor_size_),
        node_size(node_size_) {}
};

/*
suppose we have a graph like this

0----3-----5----7
 \   |\         |\
 17  8 9        1 2

we save the nodes in arbitrary order,
in this example,the order is
[0,5,1,2,7,3,8,9,17]
let us name this array u_id;
we record each node's neighbors:
0:3,17
5:3,7
1:7
2:7
7:1,2,5
3:0,5,8,9
8:3
9:3
17:0

by concatenating each node's neighbor_list in the order we save the node id.
we get [3,17,3,7,7,7,1,2,5,0,5,8,9,3,3,0]
this is the neighbor_list of GpuPsCommGraph
given this neighbor_list and the order to save node id,
we know,
node 0's neighbors are in the range [0,1] of neighbor_list
node 5's neighbors are in the range [2,3] of neighbor_list
node 1's neighbors are in the range [4,4] of neighbor_list
node 2:[5,5]
node 7:[6,6]
node 3:[9,12]
node 8:[13,13]
node 9:[14,14]
node 17:[15,15]
...
by the above information,
we generate a node_list:GpuPsGraphNode *graph_node_list in GpuPsCommGraph
of size 9,
where node_list[i].id = u_id[i]
then we have:
node_list[0]-> node_id:0, neighbor_size:2, neighbor_offset:0
node_list[1]-> node_id:5, neighbor_size:2, neighbor_offset:2
node_list[2]-> node_id:1, neighbor_size:1, neighbor_offset:4
node_list[3]-> node_id:2, neighbor_size:1, neighbor_offset:5
node_list[4]-> node_id:7, neighbor_size:3, neighbor_offset:6
node_list[5]-> node_id:3, neighbor_size:4, neighbor_offset:9
node_list[6]-> node_id:8, neighbor_size:1, neighbor_offset:13
node_list[7]-> node_id:9, neighbor_size:1, neighbor_offset:14
node_list[8]-> node_id:17, neighbor_size:1, neighbor_offset:15
*/
struct NeighborSampleResult {
  int64_t *val;
  int *actual_sample_size, sample_size, key_size;
  int *offset;
  NeighborSampleResult(int _sample_size, int _key_size)
      : sample_size(_sample_size), key_size(_key_size) {
    actual_sample_size = NULL;
    val = NULL;
    offset = NULL;
  };
  ~NeighborSampleResult() {
    if (val != NULL) cudaFree(val);
    if (actual_sample_size != NULL) cudaFree(actual_sample_size);
    if (offset != NULL) cudaFree(offset);
  }
};

struct NodeQueryResult {
  int64_t *val;
  int actual_sample_size;
  NodeQueryResult() {
    val = NULL;
    actual_sample_size = 0;
  };
  ~NodeQueryResult() {
    if (val != NULL) cudaFree(val);
  }
};
}
};
#endif

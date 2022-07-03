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
#include <iostream>
#include <memory>
#include <string>

#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/memory/memory.h"
#include "paddle/fluid/platform/cuda_device_guard.h"
namespace paddle {
namespace framework {
struct GpuPsGraphNode {
  int64_t node_id;
  int64_t neighbor_size, neighbor_offset;
  // this node's neighbor is stored on [neighbor_offset,neighbor_offset +
  // neighbor_size) of int64_t *neighbor_list;
};

struct GpuPsCommGraph {
  int64_t *neighbor_list;
  GpuPsGraphNode *node_list;
  int64_t neighbor_size, node_size;
  // the size of neighbor array and graph_node_list array
  GpuPsCommGraph()
      : neighbor_list(NULL), node_list(NULL), neighbor_size(0), node_size(0) {}
  GpuPsCommGraph(int64_t *neighbor_list_,
                 GpuPsGraphNode *node_list_,
                 int64_t neighbor_size_,
                 int64_t node_size_)
      : neighbor_list(neighbor_list_),
        node_list(node_list_),
        neighbor_size(neighbor_size_),
        node_size(node_size_) {}
  void init_on_cpu(int64_t neighbor_size, int64_t node_size) {
    this->neighbor_size = neighbor_size;
    this->node_size = node_size;
    this->neighbor_list = new int64_t[neighbor_size];
    this->node_list = new paddle::framework::GpuPsGraphNode[node_size];
  }
  void release_on_cpu() {
    delete[] neighbor_list;
    delete[] node_list;
  }
  void display_on_cpu() {
    VLOG(0) << "neighbor_size = " << neighbor_size;
    VLOG(0) << "node_size = " << node_size;
    for (size_t i = 0; i < neighbor_size; i++) {
      VLOG(0) << "neighbor " << i << " " << neighbor_list[i];
    }
    for (size_t i = 0; i < node_size; i++) {
      VLOG(0) << "node i " << node_list[i].node_id
              << " neighbor_size = " << node_list[i].neighbor_size;
      std::string str;
      int offset = node_list[i].neighbor_offset;
      for (size_t j = 0; j < node_list[i].neighbor_size; j++) {
        if (j > 0) str += ",";
        str += std::to_string(neighbor_list[j + offset]);
      }
      VLOG(0) << str;
    }
  }
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
struct NeighborSampleQuery {
  int gpu_id;
  int64_t *key;
  int sample_size;
  int len;
  void initialize(int gpu_id, int64_t key, int sample_size, int len) {
    this->gpu_id = gpu_id;
    this->key = (int64_t *)key;
    this->sample_size = sample_size;
    this->len = len;
  }
  void display() {
    int64_t *sample_keys = new int64_t[len];
    VLOG(0) << "device_id " << gpu_id << " sample_size = " << sample_size;
    VLOG(0) << "there are " << len << " keys ";
    std::string key_str;
    cudaMemcpy(sample_keys, key, len * sizeof(int64_t), cudaMemcpyDeviceToHost);

    for (int i = 0; i < len; i++) {
      if (key_str.size() > 0) key_str += ";";
      key_str += std::to_string(sample_keys[i]);
    }
    VLOG(0) << key_str;
    delete[] sample_keys;
  }
};
struct NeighborSampleResult {
  int64_t *val;
  int64_t *actual_val;
  int *actual_sample_size, sample_size, key_size;
  int total_sample_size;
  std::shared_ptr<memory::Allocation> val_mem, actual_sample_size_mem;
  std::shared_ptr<memory::Allocation> actual_val_mem;
  int64_t *get_val() { return val; }
  int64_t get_actual_val() { return (int64_t)actual_val; }
  int *get_actual_sample_size() { return actual_sample_size; }
  int get_sample_size() { return sample_size; }
  int get_key_size() { return key_size; }
  void set_total_sample_size(int s) { total_sample_size = s; }
  int get_len() { return total_sample_size; }
  void initialize(int _sample_size, int _key_size, int dev_id) {
    sample_size = _sample_size;
    key_size = _key_size;
    platform::CUDADeviceGuard guard(dev_id);
    platform::CUDAPlace place = platform::CUDAPlace(dev_id);
    val_mem =
        memory::AllocShared(place, _sample_size * _key_size * sizeof(int64_t));
    val = (int64_t *)val_mem->ptr();
    actual_sample_size_mem =
        memory::AllocShared(place, _key_size * sizeof(int));
    actual_sample_size = (int *)actual_sample_size_mem->ptr();
  }
  void display() {
    VLOG(0) << "in node sample result display ------------------";
    int64_t *res = new int64_t[sample_size * key_size];
    cudaMemcpy(res,
               val,
               sample_size * key_size * sizeof(int64_t),
               cudaMemcpyDeviceToHost);
    int *ac_size = new int[key_size];
    cudaMemcpy(ac_size,
               actual_sample_size,
               key_size * sizeof(int),
               cudaMemcpyDeviceToHost);  // 3, 1, 3
    int total_sample_size = 0;
    for (int i = 0; i < key_size; i++) {
      total_sample_size += ac_size[i];
    }
    int64_t *res2 = new int64_t[total_sample_size];  // r
    cudaMemcpy(res2,
               actual_val,
               total_sample_size * sizeof(int64_t),
               cudaMemcpyDeviceToHost);  // r

    int start = 0;
    for (int i = 0; i < key_size; i++) {
      VLOG(0) << "actual sample size for " << i << "th key is " << ac_size[i];
      VLOG(0) << "sampled neighbors are ";
      std::string neighbor, neighbor2;
      for (int j = 0; j < ac_size[i]; j++) {
        // if (neighbor.size() > 0) neighbor += ";";
        if (neighbor2.size() > 0) neighbor2 += ";";  // r
        // neighbor += std::to_string(res[i * sample_size + j]);
        neighbor2 += std::to_string(res2[start + j]);  // r
      }
      VLOG(0) << neighbor << " " << neighbor2;
      start += ac_size[i];  // r
    }
    delete[] res;
    delete[] res2;  // r
    delete[] ac_size;
    VLOG(0) << " ------------------";
  }
  std::vector<int64_t> get_sampled_graph(NeighborSampleQuery q) {
    std::vector<int64_t> graph;
    int64_t *sample_keys = new int64_t[q.len];
    std::string key_str;
    cudaMemcpy(
        sample_keys, q.key, q.len * sizeof(int64_t), cudaMemcpyDeviceToHost);
    int64_t *res = new int64_t[sample_size * key_size];
    cudaMemcpy(res,
               val,
               sample_size * key_size * sizeof(int64_t),
               cudaMemcpyDeviceToHost);
    int *ac_size = new int[key_size];
    cudaMemcpy(ac_size,
               actual_sample_size,
               key_size * sizeof(int),
               cudaMemcpyDeviceToHost);  // 3, 1, 3
    int total_sample_size = 0;
    for (int i = 0; i < key_size; i++) {
      total_sample_size += ac_size[i];
    }
    int64_t *res2 = new int64_t[total_sample_size];  // r
    cudaMemcpy(res2,
               actual_val,
               total_sample_size * sizeof(int64_t),
               cudaMemcpyDeviceToHost);  // r

    int start = 0;
    for (int i = 0; i < key_size; i++) {
      graph.push_back(sample_keys[i]);
      graph.push_back(ac_size[i]);
      for (int j = 0; j < ac_size[i]; j++) {
        graph.push_back(res2[start + j]);
      }
      start += ac_size[i];  // r
    }
    delete[] res;
    delete[] res2;  // r
    delete[] ac_size;
    delete[] sample_keys;
    return graph;
  }
  NeighborSampleResult(){};
  ~NeighborSampleResult() {}
};

struct NodeQueryResult {
  int64_t *val;
  int actual_sample_size;
  int64_t get_val() { return (int64_t)val; }
  int get_len() { return actual_sample_size; }
  std::shared_ptr<memory::Allocation> val_mem;
  void initialize(int query_size, int dev_id) {
    platform::CUDADeviceGuard guard(dev_id);
    platform::CUDAPlace place = platform::CUDAPlace(dev_id);
    val_mem = memory::AllocShared(place, query_size * sizeof(int64_t));
    val = (int64_t *)val_mem->ptr();

    // cudaMalloc((void **)&val, query_size * sizeof(int64_t));
    actual_sample_size = 0;
  }
  void display() {
    VLOG(0) << "in node query result display ------------------";
    int64_t *res = new int64_t[actual_sample_size];
    cudaMemcpy(
        res, val, actual_sample_size * sizeof(int64_t), cudaMemcpyDeviceToHost);

    VLOG(0) << "actual_sample_size =" << actual_sample_size;
    std::string str;
    for (int i = 0; i < actual_sample_size; i++) {
      if (str.size() > 0) str += ";";
      str += std::to_string(res[i]);
    }
    VLOG(0) << str;
    delete[] res;
    VLOG(0) << " ------------------";
  }
  NodeQueryResult() {
    val = NULL;
    actual_sample_size = 0;
  };
  ~NodeQueryResult() {}
};
}  // namespace framework
};  // namespace paddle
#endif

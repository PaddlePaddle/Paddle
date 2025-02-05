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
#include <cuda_fp16.h>
#include <iostream>
#include <memory>
#include <string>

#include "paddle/common/flags.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/memory/allocation/allocator.h"
#include "paddle/phi/core/memory/memory.h"
#include "paddle/phi/core/platform/cuda_device_guard.h"

COMMON_DECLARE_bool(gpugraph_load_node_list_into_hbm);
namespace paddle {
namespace framework {
struct GpuPsNodeInfo {
  uint32_t neighbor_size, neighbor_offset;
  GpuPsNodeInfo() : neighbor_size(0), neighbor_offset(0) {}
  // this node's neighbor is stored on [neighbor_offset,neighbor_offset +
  // neighbor_size) of int64_t *neighbor_list;
};

struct GpuPsCommGraph {
  uint64_t *node_list;
  // when FLAGS_gpugraph_load_node_list_into_hbm is true locate on both side
  // else only locate on host side
  int64_t node_size;              //  the size of node_list
  GpuPsNodeInfo *node_info_list;  // only locate on host side
  uint64_t *neighbor_list;        // locate on both side
  int64_t neighbor_size;          // the size of neighbor_list
  half *weight_list;  // locate on both side, which length is the same as
                      // neighbor_list
  bool is_weighted;
  GpuPsCommGraph()
      : node_list(nullptr),
        node_size(0),
        node_info_list(nullptr),
        neighbor_list(nullptr),
        neighbor_size(0),
        weight_list(nullptr),
        is_weighted(false) {}
  GpuPsCommGraph(uint64_t *node_list_,
                 int64_t node_size_,
                 GpuPsNodeInfo *node_info_list_,
                 uint64_t *neighbor_list_,
                 int64_t neighbor_size_,
                 half *weight_list_,
                 bool is_weighted_)
      : node_list(node_list_),
        node_size(node_size_),
        node_info_list(node_info_list_),
        neighbor_list(neighbor_list_),
        neighbor_size(neighbor_size_),
        weight_list(weight_list_),
        is_weighted(is_weighted_) {}
  void init_on_cpu(int64_t neighbor_size_,
                   int64_t node_size_,
                   bool is_weighted_) {
    if (node_size_ > 0) {
      this->node_size = node_size_;
      this->node_list = new uint64_t[node_size_];
      this->node_info_list = new paddle::framework::GpuPsNodeInfo[node_size_];
    }
    if (neighbor_size_) {
      this->is_weighted = is_weighted_;
      this->neighbor_size = neighbor_size_;
      this->neighbor_list = new uint64_t[neighbor_size_];
      if (is_weighted_) {
        this->weight_list = new half[neighbor_size_];
      }
    }
  }
  void release_on_cpu() {
#define DEL_PTR_ARRAY(p) \
  if (p != nullptr) {    \
    delete[] p;          \
    p = nullptr;         \
  }
    DEL_PTR_ARRAY(node_list);
    DEL_PTR_ARRAY(neighbor_list);
    DEL_PTR_ARRAY(node_info_list);
    DEL_PTR_ARRAY(weight_list);
    node_size = 0;
    neighbor_size = 0;
  }
  void display_on_cpu(int edge_idx = -1) const {
    for (int64_t i = 0; i < node_size; i++) {
      auto id = node_list[i];
      auto val = node_info_list[i];
      std::string neighbor_str;
      for (size_t j = 0; j < val.neighbor_size; ++j) {
        if (j > 0) neighbor_str += ";";
        neighbor_str += std::to_string(neighbor_list[val.neighbor_offset + j]);
      }
      VLOG(0) << "node id " << id << ", edge_idx:" << edge_idx << ", "
              << val.neighbor_offset << ":" << val.neighbor_size << " "
              << neighbor_str;
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
we generate a node_list and node_info_list in GpuPsCommGraph,
node_list: [0,5,1,2,7,3,8,9,17]
node_info_list: [(2,0),(2,2),(1,4),(1,5),(3,6),(4,9),(1,13),(1,14),(1,15)]
Here, we design the data in this format to better
adapt to gpu and avoid to convert again.
*/
struct NeighborSampleQuery {
  int gpu_id;
  int table_idx;
  uint64_t *src_nodes;
  int len;
  int sample_size;
  int sample_step;
  int neighbor_size_limit;
  void initialize(int gpu_id,
                  int table_idx,
                  uint64_t src_nodes,
                  int sample_size,
                  int len,
                  int neighbor_size_limit,
                  int sample_step = 1) {
    this->table_idx = table_idx;
    this->gpu_id = gpu_id;
    this->src_nodes = reinterpret_cast<uint64_t *>(src_nodes);
    this->len = len;
    this->sample_size = sample_size;
    this->neighbor_size_limit = neighbor_size_limit;
    this->sample_step = sample_step;
  }
  void display() {
    uint64_t *sample_keys = new uint64_t[len];
    VLOG(0) << "device_id " << gpu_id << " sample_size = " << sample_size;
    VLOG(0) << "there are " << len << " keys to sample for graph " << table_idx;
    std::string key_str;
    cudaMemcpy(
        sample_keys, src_nodes, len * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    for (int i = 0; i < len; i++) {
      if (key_str.size() > 0) key_str += ";";
      key_str += std::to_string(sample_keys[i]);
    }
    VLOG(0) << key_str;
    delete[] sample_keys;
  }
};
struct NeighborSampleResult {
  // Used in deepwalk.
  uint64_t *val;
  uint64_t *actual_val;
  int *actual_sample_size, sample_size, key_size;
  int total_sample_size = 0;
  cudaStream_t stream = 0;
  std::shared_ptr<memory::Allocation> val_mem, actual_sample_size_mem;
  std::shared_ptr<memory::Allocation> actual_val_mem;
  NeighborSampleQuery query;
  uint64_t *get_val() { return val; }
  uint64_t get_actual_val() { return (uint64_t)actual_val; }
  int *get_actual_sample_size() { return actual_sample_size; }
  int get_sample_size() { return sample_size; }
  int get_key_size() { return key_size; }
  void set_total_sample_size(int s) { total_sample_size = s; }
  int get_len() { return total_sample_size; }
  void set_stream(cudaStream_t stream_t) { stream = stream_t; }
  void initialize(int _sample_size, int _key_size, int dev_id) {
    sample_size = _sample_size;
    key_size = _key_size;
    platform::CUDADeviceGuard guard(dev_id);
    phi::GPUPlace place = phi::GPUPlace(dev_id);
    if (stream != 0) {
      val_mem = memory::AllocShared(
          place,
          _sample_size * _key_size * sizeof(uint64_t),
          phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
      actual_sample_size_mem = memory::AllocShared(
          place,
          _key_size * sizeof(int),
          phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
      cudaStreamSynchronize(stream);
    } else {
      val_mem = memory::AllocShared(
          place, _sample_size * _key_size * sizeof(uint64_t));
      actual_sample_size_mem =
          memory::AllocShared(place, _key_size * sizeof(int));
    }
    val = reinterpret_cast<uint64_t *>(val_mem->ptr());
    actual_sample_size = reinterpret_cast<int *>(actual_sample_size_mem->ptr());
  }

  void display() {
    VLOG(0) << "in node sample result display ------------------";
    uint64_t *sample_keys = new uint64_t[query.len];
    cudaMemcpy(sample_keys,
               query.src_nodes,
               query.len * sizeof(uint64_t),
               cudaMemcpyDeviceToHost);

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
    uint64_t *res2 = new uint64_t[total_sample_size];  // r
    cudaMemcpy(res2,
               actual_val,
               total_sample_size * sizeof(uint64_t),
               cudaMemcpyDeviceToHost);  // r

    int start = 0;
    for (int i = 0; i < key_size; i++) {
      auto key = sample_keys[i];
      std::string neighbor;
      for (int j = 0; j < ac_size[i]; j++) {
        if (neighbor.size() > 0) neighbor += ";";     // r
        neighbor += std::to_string(res2[start + j]);  // r
      }
      VLOG(0) << "sample key:" << key << " gpu_id:" << query.gpu_id
              << " sample_size:" << query.sample_size
              << " sample_step:" << query.sample_step
              << " table_index:" << query.table_idx << " ac_size:" << ac_size[i]
              << " neighbor:" << neighbor;
      start += ac_size[i];  // r
    }
    delete[] res;
    delete[] res2;  // r
    delete[] ac_size;
  }
  void display2() {
    uint64_t *res = new uint64_t[total_sample_size];
    cudaMemcpy(res,
               actual_val,
               total_sample_size * sizeof(uint64_t),
               cudaMemcpyDeviceToHost);
    std::string sample_str;
    for (int i = 0; i < total_sample_size; i++) {
      if (sample_str.size() > 0) sample_str += ";";
      sample_str += std::to_string(res[i]);
    }
    VLOG(0) << "sample result: " << sample_str;
    delete[] res;
  }

  std::vector<uint64_t> get_sampled_graph(NeighborSampleQuery q) {
    std::vector<uint64_t> graph;
    int64_t *sample_keys = new int64_t[q.len];
    std::string key_str;
    cudaMemcpy(sample_keys,
               q.src_nodes,
               q.len * sizeof(uint64_t),
               cudaMemcpyDeviceToHost);
    uint64_t *res = new uint64_t[sample_size * key_size];
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
  NeighborSampleResult() {}
  ~NeighborSampleResult() {}
};

struct NeighborSampleResultV2 {
  // Used in graphsage.
  uint64_t *val;
  int *actual_sample_size;
  float *weight;
  int sample_size, key_size, edge_to_id_len;
  std::shared_ptr<memory::Allocation> val_mem, actual_sample_size_mem,
      weight_mem;
  cudaStream_t stream = 0;

  void set_stream(cudaStream_t stream_t) { stream = stream_t; }
  void initialize(int _sample_size,
                  int _key_size,
                  int _edge_to_id_len,
                  bool _return_weight,
                  int dev_id) {
    sample_size = _sample_size;
    key_size = _key_size;
    edge_to_id_len = _edge_to_id_len;
    platform::CUDADeviceGuard guard(dev_id);
    phi::GPUPlace place = phi::GPUPlace(dev_id);
    if (stream != 0) {
      val_mem = memory::AllocShared(
          place,
          _sample_size * _key_size * _edge_to_id_len * sizeof(uint64_t),
          phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
      actual_sample_size_mem = memory::AllocShared(
          place,
          _key_size * _edge_to_id_len * sizeof(int),
          phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
      if (_return_weight) {
        weight_mem = memory::AllocShared(
            place,
            _sample_size * _key_size * _edge_to_id_len * sizeof(float),
            phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
      }
    } else {
      val_mem = memory::AllocShared(
          place, _sample_size * _key_size * _edge_to_id_len * sizeof(uint64_t));
      actual_sample_size_mem =
          memory::AllocShared(place, _key_size * _edge_to_id_len * sizeof(int));
      if (_return_weight) {
        weight_mem = memory::AllocShared(
            place, _sample_size * _key_size * _edge_to_id_len * sizeof(float));
      }
    }
    val = reinterpret_cast<uint64_t *>(val_mem->ptr());
    actual_sample_size = reinterpret_cast<int *>(actual_sample_size_mem->ptr());
    if (_return_weight) {
      weight = reinterpret_cast<float *>(weight_mem->ptr());
    } else {
      weight = nullptr;
    }
  }
  void display() {
    int *ac_size = new int[key_size * edge_to_id_len];
    cudaMemcpy(ac_size,
               actual_sample_size,
               key_size * edge_to_id_len * sizeof(int),
               cudaMemcpyDeviceToHost);  // 0, 0, 0...
    std::string print_ac;
    for (int i = 0; i < key_size * edge_to_id_len; i++) {
      print_ac += std::to_string(ac_size[i]);
      print_ac += ";";
    }
    VLOG(0) << "actual_sample_size for all keys are: " << print_ac;
    delete[] ac_size;
  }
  NeighborSampleResultV2() {}
  ~NeighborSampleResultV2() {}
};

struct NodeQueryResult {
  uint64_t *val;
  int actual_sample_size;
  cudaStream_t stream = 0;
  void set_stream(cudaStream_t stream_t) { stream = stream_t; }
  uint64_t get_val() { return (uint64_t)val; }
  int get_len() { return actual_sample_size; }
  std::shared_ptr<memory::Allocation> val_mem;
  void initialize(int query_size, int dev_id) {
    platform::CUDADeviceGuard guard(dev_id);
    phi::GPUPlace place = phi::GPUPlace(dev_id);
    if (stream != 0) {
      val_mem = memory::AllocShared(
          place,
          query_size * sizeof(uint64_t),
          phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
    } else {
      val_mem = memory::AllocShared(place, query_size * sizeof(uint64_t));
    }
    val = reinterpret_cast<uint64_t *>(val_mem->ptr());
    actual_sample_size = 0;
  }
  void display() {
    VLOG(0) << "in node query result display ------------------";
    uint64_t *res = new uint64_t[actual_sample_size];
    cudaMemcpy(res,
               val,
               actual_sample_size * sizeof(uint64_t),
               cudaMemcpyDeviceToHost);

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
  }
  ~NodeQueryResult() {}
};  // end of struct NodeQueryResult

struct GpuPsFeaInfo {
  uint32_t feature_size, feature_offset;
  // this node's feature is stored on [feature_offset,feature_offset +
  // feature_size) of int64_t *feature_list;
};

struct GpuPsCommGraphFea {
  uint64_t *node_list;     // only locate on host side, the list of node id
  uint64_t *feature_list;  // locate on both side
  uint8_t *slot_id_list;   // locate on both side
  GpuPsFeaInfo
      *fea_info_list;  // only locate on host side, the list of fea_info
  uint64_t feature_size, node_size, feature_capacity;
  // the size of feature array and graph_node_list array
  GpuPsCommGraphFea()
      : node_list(NULL),
        feature_list(NULL),
        slot_id_list(NULL),
        fea_info_list(NULL),
        feature_size(0),
        node_size(0),
        feature_capacity(0) {}
  GpuPsCommGraphFea(uint64_t *node_list_,
                    uint64_t *feature_list_,
                    uint8_t *slot_id_list_,
                    GpuPsFeaInfo *fea_info_list_,
                    uint64_t feature_size_,
                    uint64_t node_size_)
      : node_list(node_list_),
        feature_list(feature_list_),
        slot_id_list(slot_id_list_),
        fea_info_list(fea_info_list_),
        feature_size(feature_size_),
        node_size(node_size_) {}
  void init_on_cpu(uint64_t feature_size,
                   uint64_t node_size,
                   uint32_t slot_num) {
    PADDLE_ENFORCE_LE(
        slot_num,
        255,
        common::errors::InvalidArgument(
            "The number of slot_num should not be greater than 255 "
            ", but the slot_num is %d ",
            slot_num));
    this->feature_size = feature_size;
    this->node_size = node_size;
    this->node_list = new uint64_t[node_size];
    this->feature_list = new uint64_t[feature_size];
    this->slot_id_list = new uint8_t[feature_size];
    this->fea_info_list = new GpuPsFeaInfo[node_size];
  }
  void release_on_cpu() {
#define DEL_PTR_ARRAY(p) \
  if (p != nullptr) {    \
    delete[] p;          \
    p = nullptr;         \
  }
    DEL_PTR_ARRAY(node_list);
    DEL_PTR_ARRAY(feature_list);
    DEL_PTR_ARRAY(slot_id_list);
    DEL_PTR_ARRAY(fea_info_list);
  }
  void display_on_cpu() const {
    VLOG(1) << "feature_size = " << feature_size;
    VLOG(1) << "node_size = " << node_size;
    for (uint64_t i = 0; i < feature_size; i++) {
      VLOG(1) << "feature_list[" << i << "] = " << feature_list[i];
    }
    for (uint64_t i = 0; i < node_size; i++) {
      VLOG(1) << "node_id[" << node_list[i]
              << "] feature_size = " << fea_info_list[i].feature_size;
      std::string str;
      uint32_t offset = fea_info_list[i].feature_offset;
      for (uint64_t j = 0; j < fea_info_list[i].feature_size; j++) {
        if (j > 0) str += ",";
        str += std::to_string(slot_id_list[j + offset]);
        str += ":";
        str += std::to_string(feature_list[j + offset]);
      }
      VLOG(1) << str;
    }
  }
};  // end of struct GpuPsCommGraphFea

struct GpuPsCommGraphFloatFea {
  uint64_t *node_list;    // only locate on host side, the list of node id
  float *feature_list;    // locate on both side
  uint8_t *slot_id_list;  // locate on both side
  GpuPsFeaInfo
      *fea_info_list;  // only locate on host side, the list of fea_info
  uint64_t feature_size, node_size, feature_capacity;

  // the size of feature array and graph_node_list array
  GpuPsCommGraphFloatFea()
      : node_list(NULL),
        feature_list(NULL),
        slot_id_list(NULL),
        fea_info_list(NULL),
        feature_size(0),
        node_size(0) {}
  GpuPsCommGraphFloatFea(uint64_t *node_list_,
                         float *feature_list_,
                         uint8_t *slot_id_list_,
                         GpuPsFeaInfo *fea_info_list_,
                         uint64_t feature_size_,
                         uint64_t node_size_)
      : node_list(node_list_),
        feature_list(feature_list_),
        slot_id_list(slot_id_list_),
        fea_info_list(fea_info_list_),
        feature_size(feature_size_),
        node_size(node_size_) {}
  void init_on_cpu(uint64_t feature_size,
                   uint64_t node_size,
                   uint32_t slot_num) {
    PADDLE_ENFORCE_LE(
        slot_num,
        255,
        common::errors::InvalidArgument(
            "The number of slot_num should not be greater than 255 "
            ", but the slot_num is %d ",
            slot_num));
    this->feature_size = feature_size;
    this->node_size = node_size;
    this->node_list = new uint64_t[node_size];
    this->feature_list = new float[feature_size];
    this->slot_id_list = new uint8_t[feature_size];
    this->fea_info_list = new GpuPsFeaInfo[node_size];
  }
  void release_on_cpu() {
#define DEL_PTR_ARRAY(p) \
  if (p != nullptr) {    \
    delete[] p;          \
    p = nullptr;         \
  }
    DEL_PTR_ARRAY(node_list);
    DEL_PTR_ARRAY(feature_list);
    DEL_PTR_ARRAY(slot_id_list);
    DEL_PTR_ARRAY(fea_info_list);
  }
  void display_on_cpu() const {
    VLOG(1) << "feature_size = " << feature_size;
    VLOG(1) << "node_size = " << node_size;
    for (uint64_t i = 0; i < feature_size; i++) {
      VLOG(1) << "feature_list[" << i << "] = " << feature_list[i];
    }
    for (uint64_t i = 0; i < node_size; i++) {
      VLOG(1) << "node_id[" << node_list[i]
              << "] feature_size = " << fea_info_list[i].feature_size;
      std::string str;
      uint32_t offset = fea_info_list[i].feature_offset;
      for (uint64_t j = 0; j < fea_info_list[i].feature_size; j++) {
        if (j > 0) str += ",";
        str += std::to_string(slot_id_list[j + offset]);
        str += ":";
        str += std::to_string(feature_list[j + offset]);
      }
      VLOG(1) << str;
    }
  }
};  // end of struct GpuPsCommGraphFloatFea

struct GpuPsCommRankFea {
  uint64_t *node_list;  // only locate on host side, the list of node id
  uint32_t *rank_list;  // locate on both side
  uint64_t feature_size, feature_capacity;
  // the size of feature array and graph_node_list array
  GpuPsCommRankFea()
      : node_list(NULL),
        rank_list(NULL),
        feature_size(0),
        feature_capacity(0) {}
  GpuPsCommRankFea(uint64_t *node_list_,
                   uint32_t *rank_list_,
                   uint64_t feature_size_)
      : node_list(node_list_),
        rank_list(rank_list_),
        feature_size(feature_size_) {}
  void init_on_cpu(uint64_t feature_size) {
    this->feature_size = feature_size;
    this->node_list = new uint64_t[feature_size];
    this->rank_list = new uint32_t[feature_size];
  }
  void release_on_cpu() {
#define DEL_PTR_ARRAY(p) \
  if (p != nullptr) {    \
    delete[] p;          \
    p = nullptr;         \
  }
    DEL_PTR_ARRAY(node_list);
    DEL_PTR_ARRAY(rank_list);
  }
  void display_on_cpu() const {
    VLOG(1) << "GpuPsCommRankFea feature_size = " << feature_size;
    for (uint64_t i = 0; i < feature_size; i++) {
      VLOG(1) << "GpuPsCommRankFea rank_list[" << i << "] = " << rank_list[i];
    }
  }
};  // end of struct GpuPsCommRankFea

}  // end of namespace framework
}  // end of namespace paddle
#endif

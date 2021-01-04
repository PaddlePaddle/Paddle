/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include <vector>
#include "cub/cub.cuh"
#include "hashtable.h"
#include "heter_resource.h"
#include "paddle/fluid/framework/fleet/heter_ps/optimizer.cuh"
#include "paddle/fluid/memory/memory.h"
#include "paddle/fluid/platform/cuda_device_guard.h"
#include "paddle/fluid/platform/place.h"
#include "thrust/pair.h"

#ifdef PADDLE_WITH_PSLIB

namespace paddle {
namespace framework {

struct CustomGradMerger {
  template <typename T>
  CUB_RUNTIME_FUNCTION __forceinline__ __device__ T
  operator()(const T& a, const T& b) const {
    T out;
    out.slot = a.slot;
    out.show = a.show + b.show;
    out.clk = a.clk + b.clk;
    out.lr_g = a.lr_g + b.lr_g;
    for (int i = 0; i < MF_DIM; ++i) {
      out.mf_g[i] = a.mf_g[i] + b.mf_g[i];
    }
    return out;
  }
};

template <typename KeyType, typename ValType, typename GradType>
class HeterComm {
 public:
  HeterComm(size_t capacity, std::shared_ptr<HeterPsResource> resource);
  virtual ~HeterComm();
  HeterComm(const HeterComm&) = delete;
  HeterComm& operator=(const HeterComm&) = delete;

  void split_input_to_shard(KeyType* d_keys, int* d_idx_ptr, size_t len,
                            int* left, int* right, int gpu_num);
  void merge_grad(int gpu_num, KeyType* d_keys, GradType* d_grads, size_t len,
                  int& uniq_len);
  void pull_sparse(int num, KeyType* d_keys, ValType* d_vals, size_t len);
  void build_ps(int num, KeyType* h_keys, ValType* h_vals, size_t len,
                size_t chunk_size, int stream_num);
  void dump();
  void show_one_table(int gpu_num);
  int get_index_by_devid(int devid);

  template <typename Sgd>
  void push_sparse(int num, KeyType* d_keys, GradType* d_grads, size_t len,
                   Sgd& sgd);

  int log2i(int x);
  bool need_transfer(int send_id, int receive_id) {
    return ((send_id / 4 != receive_id / 4) && (send_id + 4) % 8 != receive_id);
  }

  int get_transfer_devid(int send_id) { return (send_id + 4) % 8; }

  struct Node {
    cudaStream_t in_stream;
    cudaStream_t out_stream;
    char* key_storage;
    char* val_storage;
    int sync;
    int key_bytes_len;
    int val_bytes_len;
    int gpu_num;
  };

  struct Path {
    std::vector<Node> nodes_;
  };

  void init_path();
  void create_storage(
      int start_index, int end_index, int keylen, int vallen,
      std::vector<std::shared_ptr<memory::Allocation>>& local_strorage);
  void walk_to_src(int start_index, int end_index, char* src_val);
  void walk_to_dest(int start_index, int end_index, char* src_key,
                    char* src_val);

 private:
  using Table = HashTable<KeyType, ValType>;
  int block_size_{256};
  float load_factor_{0.75};
  std::vector<Table*> tables_;
  std::shared_ptr<HeterPsResource> resource_;
  CustomGradMerger merger_;
  int topo_aware_{1};
  std::vector<std::vector<Path>> path_;
};

}  // end namespace framework
}  // end namespace paddle
#include "paddle/fluid/framework/fleet/heter_ps/heter_comm.tpp"
#endif

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
#ifdef PADDLE_WITH_HETERPS
#include <queue>

namespace paddle {
namespace framework {

template <typename T>
__global__ void fill_idx(T* idx, size_t len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    idx[i] = i;
  }
}

template <typename T>
void show_tensor(T* input, size_t len, gpuStream_t stream, std::string name) {
  T tmp[len];  // NOLINT
  cudaMemcpyAsync(&tmp, input, sizeof(T) * len, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  std::cout << name;
  for (int i = 0; i < len; ++i) {
    std::cout << ":" << tmp[i];
  }
  std::cout << std::endl;
}

template <typename T>
__global__ void calc_shard_offset(T* idx, T* left, T* right, size_t len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len - 1) {
    if (idx[i] != idx[i + 1]) {
      right[idx[i]] = i;
      left[idx[i + 1]] = i + 1;
    }
  }
  if (i == 0) {
    left[idx[i]] = i;
  }
  if (i == (len - 1)) {
    right[idx[i]] = i;
  }
}

template <typename KeyType, typename T>
__global__ void calc_shard_index(KeyType* d_keys, size_t len, T* shard_index,
                                 int total_gpu) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    shard_index[i] = d_keys[i] % total_gpu;
  }
}

template <typename KeyType, typename T>
__global__ void fill_shard_key(KeyType* d_shard_keys, KeyType* d_keys, T* idx,
                               size_t len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    d_shard_keys[i] = d_keys[idx[i]];
  }
}

template <typename KeyType, typename GradType, typename T>
__global__ void fill_shard_grads(KeyType* d_shard_keys, KeyType* d_keys,
                                 GradType* d_shard_grads, GradType* d_grads,
                                 T* idx, size_t len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    d_shard_keys[i] = d_keys[idx[i]];
    d_shard_grads[i] = d_grads[idx[i]];
  }
}

template <typename KeyType, typename GradType, typename T>
__global__ void dy_mf_fill_shard_grads(KeyType* d_shard_keys, KeyType* d_keys,
                                       GradType* d_shard_grads,
                                       GradType* d_grads, T* idx, size_t len,
                                       size_t grad_value_size) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    d_shard_keys[i] = d_keys[idx[i]];
    *(GradType*)((char*)d_shard_grads + i * grad_value_size) =
        *(GradType*)((char*)d_grads + uint64_t(idx[i]) * grad_value_size);
  }
}

__global__ void merge_gradient_kernel(const uint32_t* offset,
                                      const uint32_t* fea_num,
                                      const uint32_t* index, const char* input,
                                      char* output, int n,
                                      size_t grad_value_size,
                                      CustomGradMerger& merger_) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (i < n) {
    uint32_t start = offset[i];
    uint32_t num = fea_num[i];
    int ori_index = index[start];

    FeaturePushValue& lhs = *(FeaturePushValue*)(output + i * grad_value_size);
    FeaturePushValue& in =
        *(FeaturePushValue*)(input + size_t(ori_index) * grad_value_size);
    merger_.copy_basic_field(lhs, in);
    
    for (int j = 1; j < num; ++j) {
      ori_index = index[start + j];
      in = *(FeaturePushValue*)(input + size_t(ori_index) * grad_value_size);
      merger_.add_basic_field(lhs, in);
    }
  }
  
}

template <typename ValType, typename T>
__global__ void fill_dvals(ValType* d_shard_vals, ValType* d_vals, T* idx,
                           size_t len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    d_vals[idx[i]] = d_shard_vals[i];
  }
}

template <typename ValType, typename T>
__global__ void dy_mf_fill_dvals(ValType* d_shard_vals, ValType* d_vals, T* idx,
                                 size_t len, size_t val_size) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    uint64_t new_offset = uint64_t(idx[i]) * val_size;
    *(ValType*)((char*)d_vals + new_offset) =
        *(ValType*)((char*)d_shard_vals + i * val_size);
  }
}

template <typename KeyType, typename ValType, typename GradType>
HeterComm<KeyType, ValType, GradType>::HeterComm(
    size_t capacity, std::shared_ptr<HeterPsResource> resource) {
  VLOG(1) << "Construct new HeterComm";
  resource_ = resource;
  storage_.resize(resource_->total_gpu());
  multi_mf_dim_ = resource->multi_mf();
  for (int i = 0; i < resource_->total_gpu(); ++i) {
    platform::CUDADeviceGuard guard(resource_->dev_id(i));
    allocators_.push_back(std::make_shared<cub::CachingDeviceAllocator>(
        2, 1, 20, (size_t)-1, false, false));  // NOLINT
    if (!multi_mf_dim_) {
      VLOG(0) << "yxf:: using table";
      auto table = new Table(capacity / load_factor_);
      tables_.push_back(table);
    } else {
      // VLOG(0) << "yxf:: using ptrtable";
      max_mf_dim_ = resource->max_mf_dim();
      size_t val_type_size =
      TYPEALIGN(8, sizeof(FeatureValue) + sizeof(float) * (max_mf_dim_ + 1));
      // VLOG(0) << "yxf:: using ptrtable val size: " << val_type_size << " max mf dim: " << max_mf_dim_;
      size_t grad_type_size =
      TYPEALIGN(8, sizeof(FeaturePushValue) + (max_mf_dim_ * sizeof(float)));
      // VLOG(0) << "yxf:: using ptrtable grad size: " << grad_type_size << " max mf dim: " << max_mf_dim_;
      auto ptr_table = new PtrTable(capacity / load_factor_);
      ptr_table->set_feature_value_size(val_type_size, grad_type_size);
      ptr_tables_.push_back(ptr_table);

    }
    if (multi_node_) {
      storage_[i].init(feanum_, resource_->dev_id(i));
    }
  }
  mg_time_1 = std::vector<double>(resource_->total_gpu(), 0.0);
  mg_time_2 = std::vector<double>(resource_->total_gpu(), 0.0);
  mg_time_3 = std::vector<double>(resource_->total_gpu(), 0.0);
  mg_time_4 = std::vector<double>(resource_->total_gpu(), 0.0);
  mg_time_5 = std::vector<double>(resource_->total_gpu(), 0.0);
  mg_time_6 = std::vector<double>(resource_->total_gpu(), 0.0);
  mg_time_7 = std::vector<double>(resource_->total_gpu(), 0.0);
  mg_time_8 = std::vector<double>(resource_->total_gpu(), 0.0);
  init_path();
}

template <typename KeyType, typename ValType, typename GradType>
void HeterComm<KeyType, ValType, GradType>::init_path() {
  int total_gpu = resource_->total_gpu();
  path_.resize(total_gpu);

  if (!topo_aware_) {
    VLOG(0) << "init path without topo aware";
    for (int i = 0; i < total_gpu; ++i) {
      path_[i].resize(total_gpu);
      for (int j = 0; j < total_gpu; ++j) {
        auto& nodes = path_[i][j].nodes_;
        nodes.resize(1);
        nodes[0].in_stream = resource_->comm_stream(i, j);
        nodes[0].out_stream = resource_->comm_stream(i, j);
        nodes[0].key_storage = NULL;
        nodes[0].val_storage = NULL;
        nodes[0].sync = 0;
        nodes[0].gpu_num = j;
      }
    }
  } else {
    VLOG(0) << "init path with topo aware";
    for (int i = 0; i < total_gpu; ++i) {
      path_[i].resize(total_gpu);
      for (int j = 0; j < total_gpu; ++j) {
        auto& nodes = path_[i][j].nodes_;
        int from = resource_->dev_id(i);
        int to = resource_->dev_id(j);
        int transfer_id = i;
        if (need_transfer(from, to)) {
          transfer_id = resource_->get_index_by_devid(get_transfer_devid(from));
          nodes.push_back(Node());
          Node& node = nodes.back();
          node.in_stream = resource_->comm_stream(i, transfer_id);
          node.out_stream = resource_->comm_stream(transfer_id, i);
          node.key_storage = NULL;
          node.val_storage = NULL;
          node.sync = 1;
          node.gpu_num = transfer_id;
        }
        nodes.push_back(Node());
        Node& node = nodes.back();
        node.in_stream = resource_->comm_stream(i, transfer_id);
        node.out_stream = resource_->comm_stream(transfer_id, i);
        node.key_storage = NULL;
        node.val_storage = NULL;
        node.sync = 0;
        node.gpu_num = j;
      }
    }
  }
}

template <typename KeyType, typename ValType, typename GradType>
void HeterComm<KeyType, ValType, GradType>::create_storage(int start_index,
                                                           int end_index,
                                                           int keylen,
                                                           int vallen) {
  auto& allocator = allocators_[start_index];
  auto& nodes = path_[start_index][end_index].nodes_;
  for (size_t i = 0; i < nodes.size(); ++i) {
    platform::CUDADeviceGuard guard(resource_->dev_id(nodes[i].gpu_num));
    allocator->DeviceAllocate(
        resource_->dev_id(nodes[i].gpu_num),
        (void**)&(nodes[i].key_storage),  // NOLINT
        keylen, resource_->remote_stream(nodes[i].gpu_num, start_index));
    allocator->DeviceAllocate(
        resource_->dev_id(nodes[i].gpu_num),
        (void**)&(nodes[i].val_storage),  // NOLINT
        vallen, resource_->remote_stream(nodes[i].gpu_num, start_index));

    nodes[i].key_bytes_len = keylen;
    nodes[i].val_bytes_len = vallen;
  }
}

template <typename KeyType, typename ValType, typename GradType>
void HeterComm<KeyType, ValType, GradType>::destroy_storage(int start_index,
                                                            int end_index) {
  auto& allocator = allocators_[start_index];
  auto& nodes = path_[start_index][end_index].nodes_;
  for (size_t i = 0; i < nodes.size(); ++i) {
    platform::CUDADeviceGuard guard(resource_->dev_id(nodes[i].gpu_num));

    allocator->DeviceFree(resource_->dev_id(nodes[i].gpu_num),
                          nodes[i].key_storage);
    allocator->DeviceFree(resource_->dev_id(nodes[i].gpu_num),
                          nodes[i].val_storage);
  }
}

template <typename KeyType, typename ValType, typename GradType>
void HeterComm<KeyType, ValType, GradType>::walk_to_dest(
    int start_index, int gpu_num, int* h_left, int* h_right, KeyType* src_key,
    GradType* src_val) {
  int need_copy_val = 0;
  if (src_val) {
    need_copy_val = 1;
  }
  std::queue<CopyTask> que;
  for (int i = 0; i < gpu_num; i++) {
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    int size = path_[start_index][i].nodes_.size();
    auto& node = path_[start_index][i].nodes_[0];
    CopyTask t(&path_[start_index][i], 0);
    que.push(t);
    cudaMemcpyAsync(node.key_storage,
                    reinterpret_cast<char*>(src_key + h_left[i]),
                    node.key_bytes_len, cudaMemcpyDefault, node.in_stream);
    if (need_copy_val) {
      cudaMemcpyAsync(node.val_storage,
                      reinterpret_cast<char*>(src_val + h_left[i]),
                      node.val_bytes_len, cudaMemcpyDefault, node.in_stream);
    }
  }
  while (!que.empty()) {
    CopyTask& cur_task = que.front();
    que.pop();
    if (cur_task.path->nodes_[cur_task.step].sync) {
      cudaStreamSynchronize(cur_task.path->nodes_[cur_task.step].in_stream);
    }
    if (cur_task.step != cur_task.path->nodes_.size() - 1) {
      int cur_step = cur_task.step;
      CopyTask c(cur_task.path, cur_step + 1);
      que.push(c);
      cudaMemcpyAsync(cur_task.path->nodes_[cur_step + 1].key_storage,
                      cur_task.path->nodes_[cur_step].key_storage,
                      cur_task.path->nodes_[cur_step + 1].key_bytes_len,
                      cudaMemcpyDefault,
                      cur_task.path->nodes_[cur_step + 1].in_stream);
      if (need_copy_val) {
        cudaMemcpyAsync(cur_task.path->nodes_[cur_step + 1].val_storage,
                        cur_task.path->nodes_[cur_step].val_storage,
                        cur_task.path->nodes_[cur_step + 1].val_bytes_len,
                        cudaMemcpyDefault,
                        cur_task.path->nodes_[cur_step + 1].in_stream);
      }
    }
  }
}

template <typename KeyType, typename ValType, typename GradType>
void HeterComm<KeyType, ValType, GradType>::walk_to_dest(
    int start_index, int gpu_num, int* h_left, int* h_right, KeyType* src_key,
    char* src_val, size_t val_size) {
  int need_copy_val = 0;
  if (src_val) {
    need_copy_val = 1;
  }
  std::queue<CopyTask> que;
  for (int i = 0; i < gpu_num; i++) {
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    int size = path_[start_index][i].nodes_.size();
    auto& node = path_[start_index][i].nodes_[0];
    CopyTask t(&path_[start_index][i], 0);
    que.push(t);
    cudaMemcpyAsync(node.key_storage,
                    reinterpret_cast<char*>(src_key + h_left[i]),
                    node.key_bytes_len, cudaMemcpyDefault, node.in_stream);
    if (need_copy_val) {
      // yxf tmp
      // cudaMemcpyAsync(node.val_storage,
      //                 src_val + uint64_t(h_left[i]) * uint64_t(80),
      //                 node.val_bytes_len, cudaMemcpyDefault, node.in_stream);
      cudaMemcpyAsync(node.val_storage,
                      src_val + uint64_t(h_left[i]) * uint64_t(val_size),
                      node.val_bytes_len, cudaMemcpyDefault, node.in_stream);
    }
  }
  while (!que.empty()) {
    CopyTask& cur_task = que.front();
    que.pop();
    if (cur_task.path->nodes_[cur_task.step].sync) {
      // VLOG(0) << "yxf:: node cpy syn";
      cudaStreamSynchronize(cur_task.path->nodes_[cur_task.step].in_stream);
    }
    if (cur_task.step != cur_task.path->nodes_.size() - 1) {
      // VLOG(0) << "yxf:: node cpy s2222";
      int cur_step = cur_task.step;
      CopyTask c(cur_task.path, cur_step + 1);
      que.push(c);
      cudaMemcpyAsync(cur_task.path->nodes_[cur_step + 1].key_storage,
                      cur_task.path->nodes_[cur_step].key_storage,
                      cur_task.path->nodes_[cur_step + 1].key_bytes_len,
                      cudaMemcpyDefault,
                      cur_task.path->nodes_[cur_step + 1].in_stream);
      if (need_copy_val) {
        cudaMemcpyAsync(cur_task.path->nodes_[cur_step + 1].val_storage,
                        cur_task.path->nodes_[cur_step].val_storage,
                        cur_task.path->nodes_[cur_step + 1].val_bytes_len,
                        cudaMemcpyDefault,
                        cur_task.path->nodes_[cur_step + 1].in_stream);
      }
    }
  }
}

template <typename KeyType, typename ValType, typename GradType>
void HeterComm<KeyType, ValType, GradType>::walk_to_src(
    int start_index, int gpu_num, int* h_left, int* h_right, ValType* src_val) {
  std::queue<CopyTask> que;
  for (int i = 0; i < gpu_num; i++) {
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    int cur_step = path_[start_index][i].nodes_.size() - 1;
    auto& node = path_[start_index][i].nodes_[cur_step];
    if (cur_step == 0) {
      cudaMemcpyAsync(reinterpret_cast<char*>(src_val + h_left[i]),
                      node.val_storage, node.val_bytes_len, cudaMemcpyDefault,
                      node.out_stream);
    } else {
      CopyTask t(&path_[start_index][i], cur_step - 1);
      que.push(t);
      cudaMemcpyAsync(path_[start_index][i].nodes_[cur_step - 1].val_storage,
                      node.val_storage,
                      path_[start_index][i].nodes_[cur_step - 1].val_bytes_len,
                      cudaMemcpyDefault,
                      path_[start_index][i].nodes_[cur_step - 1].out_stream);
    }
  }
  while (!que.empty()) {
    CopyTask& cur_task = que.front();
    que.pop();
    int cur_step = cur_task.step;
    if (cur_task.path->nodes_[cur_step].sync) {
      cudaStreamSynchronize(cur_task.path->nodes_[cur_step].out_stream);
    }
    if (cur_step > 0) {
      CopyTask c(cur_task.path, cur_step - 1);
      que.push(c);
      cudaMemcpyAsync(cur_task.path->nodes_[cur_step - 1].val_storage,
                      cur_task.path->nodes_[cur_step].val_storage,
                      cur_task.path->nodes_[cur_step - 1].val_bytes_len,
                      cudaMemcpyDefault,
                      cur_task.path->nodes_[cur_step - 1].out_stream);
    } else if (cur_step == 0) {
      int end_index = cur_task.path->nodes_.back().gpu_num;
      cudaMemcpyAsync(reinterpret_cast<char*>(src_val + h_left[end_index]),
                      cur_task.path->nodes_[cur_step].val_storage,
                      cur_task.path->nodes_[cur_step].val_bytes_len,
                      cudaMemcpyDefault,
                      cur_task.path->nodes_[cur_step].out_stream);
    }
  }
}

template <typename KeyType, typename ValType, typename GradType>
void HeterComm<KeyType, ValType, GradType>::walk_to_src(
    int start_index, int gpu_num, int* h_left, int* h_right, char* src_val, size_t val_size) {
  std::queue<CopyTask> que;
  for (int i = 0; i < gpu_num; i++) {
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    int cur_step = path_[start_index][i].nodes_.size() - 1;
    auto& node = path_[start_index][i].nodes_[cur_step];
    if (cur_step == 0) {
      // if (h_left[i] * val_size < 0) {
      //   VLOG(0) << "yxf:::offset wrong:: " << h_left[i] * val_size;
      // } 
      // yxf tmp
      // cudaMemcpyAsync(src_val + uint64_t(h_left[i]) * 80,
      //                 node.val_storage, node.val_bytes_len, cudaMemcpyDefault,
      //                 node.out_stream);
      cudaMemcpyAsync(src_val + uint64_t(h_left[i]) * val_size,
                      node.val_storage, node.val_bytes_len, cudaMemcpyDefault,
                      node.out_stream);
    } else {
      CopyTask t(&path_[start_index][i], cur_step - 1);
      que.push(t);
      cudaMemcpyAsync(path_[start_index][i].nodes_[cur_step - 1].val_storage,
                      node.val_storage,
                      path_[start_index][i].nodes_[cur_step - 1].val_bytes_len,
                      cudaMemcpyDefault,
                      path_[start_index][i].nodes_[cur_step - 1].out_stream);
    }
  }
  while (!que.empty()) {
    CopyTask& cur_task = que.front();
    que.pop();
    int cur_step = cur_task.step;
    if (cur_task.path->nodes_[cur_step].sync) {
      cudaStreamSynchronize(cur_task.path->nodes_[cur_step].out_stream);
    }
    if (cur_step > 0) {
      CopyTask c(cur_task.path, cur_step - 1);
      que.push(c);
      cudaMemcpyAsync(cur_task.path->nodes_[cur_step - 1].val_storage,
                      cur_task.path->nodes_[cur_step].val_storage,
                      cur_task.path->nodes_[cur_step - 1].val_bytes_len,
                      cudaMemcpyDefault,
                      cur_task.path->nodes_[cur_step - 1].out_stream);
    } else if (cur_step == 0) {
      int end_index = cur_task.path->nodes_.back().gpu_num;
      // if (h_left[end_index] * val_size < 0) {
      //   VLOG(0) << "yxf:::offset wrong11:: " << h_left[end_index] * val_size;
      // } 
      // yxf tmp 
      // cudaMemcpyAsync(src_val + uint64_t(h_left[end_index]) * 80,
      //                 cur_task.path->nodes_[cur_step].val_storage,
      //                 cur_task.path->nodes_[cur_step].val_bytes_len,
      //                 cudaMemcpyDefault,
      //                 cur_task.path->nodes_[cur_step].out_stream);
      cudaMemcpyAsync(src_val + uint64_t(h_left[end_index]) * val_size,
                      cur_task.path->nodes_[cur_step].val_storage,
                      cur_task.path->nodes_[cur_step].val_bytes_len,
                      cudaMemcpyDefault,
                      cur_task.path->nodes_[cur_step].out_stream);
    }
  }
}

template <typename KeyType, typename ValType, typename GradType>
HeterComm<KeyType, ValType, GradType>::~HeterComm() {
  if (!multi_mf_dim_) {
    for (auto& table : tables_) {
      delete table;
      table = nullptr;
    }
  } else {
    //VLOG(0) << "yxf:: heter comm ~hetercomm";
    for (auto& table : ptr_tables_) {
      delete table;
      table = nullptr;
    }
    for (auto& table : tables_) {
      delete table;
      table = nullptr;
    }
    for (size_t i = 1; i < mg_time_1.size(); i++) {
      mg_time_1[0] += mg_time_1[i];
      mg_time_2[0] += mg_time_2[i];
      mg_time_3[0] += mg_time_3[i];
      mg_time_4[0] += mg_time_4[i];
      mg_time_5[0] += mg_time_5[i];
      mg_time_6[0] += mg_time_6[i];
      mg_time_7[0] += mg_time_7[i];
      mg_time_8[0] += mg_time_8[i];
    }
    VLOG(0) << "yxf::mg_1::merge: " << mg_time_1[0];
    VLOG(0) << "yxf::mg_2:pull: " << mg_time_2[0];
    VLOG(0) << "yxf::mg_3:push: " << mg_time_3[0];
    VLOG(0) << "yxf::mg_4:sort: " << mg_time_4[0];
    VLOG(0) << "yxf::mg_5:encode: " << mg_time_5[0];
    VLOG(0) << "yxf::mg_6:sum: " << mg_time_6[0];
    VLOG(0) << "yxf::mg_7:merge_kernel: " << mg_time_7[0];
    VLOG(0) << "yxf::mg_8:merge_kernel_1: " << mg_time_8[0];
   }
}

template <typename KeyType, typename ValType, typename GradType>
void HeterComm<KeyType, ValType, GradType>::show_one_table(int gpu_num) {
  if (!multi_mf_dim_) {
    tables_[gpu_num]->show();
  } else {
    ptr_tables_[gpu_num]->show();
  }
}

template <typename KeyType, typename ValType, typename GradType>
int HeterComm<KeyType, ValType, GradType>::log2i(int x) {
  unsigned res = 0;
  while (x >>= 1) {
    ++res;
  }
  return res;
}

template <typename KeyType, typename ValType, typename GradType>
int HeterComm<KeyType, ValType, GradType>::get_index_by_devid(int devid) {
  return resource_->get_index_by_devid(devid);
}

template <typename KeyType, typename ValType, typename GradType>
void HeterComm<KeyType, ValType, GradType>::build_ps(int num, KeyType* h_keys,
                                                     ValType* h_vals,
                                                     size_t len,
                                                     size_t chunk_size,
                                                     int stream_num) {
  if (len <= 0) {
    return;
  }
  int dev_id = resource_->dev_id(num);
  platform::CUDAPlace place = platform::CUDAPlace(dev_id);
  platform::CUDADeviceGuard guard(dev_id);

  std::vector<memory::allocation::AllocationPtr> d_key_bufs;
  std::vector<memory::allocation::AllocationPtr> d_val_bufs;

  gpuStream_t streams[stream_num];  // NOLINT
  for (int i = 0; i < stream_num; ++i) {
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamCreate(&(streams[i])));
    auto d_k_buf = memory::Alloc(place, chunk_size * sizeof(KeyType));
    auto d_v_buf = memory::Alloc(place, chunk_size * sizeof(ValType));
    d_key_bufs.push_back(std::move(d_k_buf));
    d_val_bufs.push_back(std::move(d_v_buf));
  }

  int cur_len = 0;
  int cur_stream = 0;

  while (cur_len < len) {
    cur_stream = cur_stream % stream_num;
    int tmp_len = cur_len + chunk_size > len ? len - cur_len : chunk_size;
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaMemcpyAsync(d_key_bufs[cur_stream]->ptr(), h_keys + cur_len,
                        sizeof(KeyType) * tmp_len, cudaMemcpyHostToDevice,
                        streams[cur_stream]));
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaMemcpyAsync(d_val_bufs[cur_stream]->ptr(), h_vals + cur_len,
                        sizeof(ValType) * tmp_len, cudaMemcpyHostToDevice,
                        streams[cur_stream]));
    tables_[num]->insert(
        reinterpret_cast<KeyType*>(d_key_bufs[cur_stream]->ptr()),
        reinterpret_cast<ValType*>(d_val_bufs[cur_stream]->ptr()), tmp_len,
        streams[cur_stream]);
    cur_stream += 1;
    cur_len += tmp_len;
  }

  for (int i = 0; i < stream_num; ++i) {
    cudaStreamSynchronize(streams[i]);
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamDestroy(streams[i]));
  }
}

template <typename KeyType, typename ValType, typename GradType>
void HeterComm<KeyType, ValType, GradType>::build_ps(int num, KeyType* h_keys,
                                                     char* pool,
                                                     size_t len,
                                                     size_t feature_value_size,
                                                     size_t chunk_size,
                                                     int stream_num) {
  if (len <= 0) {
    return;
  }
  int dev_id = resource_->dev_id(num);
  platform::CUDAPlace place = platform::CUDAPlace(dev_id);
  platform::CUDADeviceGuard guard(dev_id);

  // use hbm pool
  std::vector<memory::allocation::AllocationPtr> d_key_bufs;

  gpuStream_t streams[stream_num];
  for (int i = 0; i < stream_num; ++i) {
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamCreate(&(streams[i])));
    auto d_k_buf = memory::Alloc(place, chunk_size * sizeof(KeyType));
    d_key_bufs.push_back(std::move(d_k_buf));
  }

  int cur_len = 0;
  int cur_stream = 0;

  while (cur_len < len) {
    cur_stream = cur_stream % stream_num;
    int tmp_len = cur_len + chunk_size > len ? len - cur_len : chunk_size;
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaMemcpyAsync(d_key_bufs[cur_stream]->ptr(), h_keys + cur_len,
                        sizeof(KeyType) * tmp_len, cudaMemcpyHostToDevice,
                        streams[cur_stream]));
    ptr_tables_[num]->insert(
        reinterpret_cast<KeyType*>(d_key_bufs[cur_stream]->ptr()), tmp_len,
        pool, feature_value_size, cur_len, streams[cur_stream]);
    cur_stream += 1;
    cur_len += tmp_len;
  }

  for (int i = 0; i < stream_num; ++i) {
    cudaStreamSynchronize(streams[i]);
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamDestroy(streams[i]));
  }
}

template <typename KeyType, typename ValType, typename GradType>
void HeterComm<KeyType, ValType, GradType>::merge_grad(
    int gpu_num, KeyType* d_keys, GradType* d_grads, size_t len,
    int& uniq_len) {  // NOLINT
  int dev_id = resource_->dev_id(gpu_num);
  platform::CUDAPlace place = platform::CUDAPlace(dev_id);
  platform::CUDADeviceGuard guard(dev_id);
  auto stream = resource_->local_stream(gpu_num, 0);

  size_t temp_storage_bytes;

  auto d_merge_keys = memory::Alloc(place, len * sizeof(KeyType));
  KeyType* d_merge_keys_ptr = reinterpret_cast<KeyType*>(d_merge_keys->ptr());

  auto d_merge_grads = memory::Alloc(place, len * sizeof(GradType));
  GradType* d_merge_grads_ptr =
      reinterpret_cast<GradType*>(d_merge_grads->ptr());

  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceRadixSort::SortPairs(
      NULL, temp_storage_bytes, d_keys, d_merge_keys_ptr, d_grads,
      d_merge_grads_ptr, len, 0, 8 * sizeof(KeyType), stream, false));

  void* d_buff = NULL;
  auto d_temp_storage = memory::Alloc(place, temp_storage_bytes);

  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceRadixSort::SortPairs(
      d_temp_storage->ptr(), temp_storage_bytes, d_keys, d_merge_keys_ptr,
      d_grads, d_merge_grads_ptr, len, 0, 8 * sizeof(KeyType), stream, false));
  temp_storage_bytes = 0;

  auto d_num_runs_out_mem = memory::Alloc(place, sizeof(int));
  int* d_num_runs_out = reinterpret_cast<int*>(d_num_runs_out_mem->ptr());

  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceReduce::ReduceByKey(
      NULL, temp_storage_bytes, d_merge_keys_ptr, d_keys, d_merge_grads_ptr,
      d_grads, d_num_runs_out, merger_, len, stream, false));

  if (d_temp_storage->size() < temp_storage_bytes) {
    d_temp_storage = NULL;
    d_temp_storage = memory::Alloc(place, temp_storage_bytes);
  }

  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceReduce::ReduceByKey(
      d_temp_storage->ptr(), temp_storage_bytes, d_merge_keys_ptr, d_keys,
      d_merge_grads_ptr, d_grads, d_num_runs_out, merger_, len, stream, false));

  cudaMemcpyAsync(&uniq_len, d_num_runs_out, sizeof(int),
                  cudaMemcpyDeviceToHost, stream);
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
}

template <typename KeyType, typename ValType, typename GradType>
void HeterComm<KeyType, ValType, GradType>::merge_grad(int gpu_num,
                                                       KeyType* d_keys,
                                                       GradType* d_grads,
                                                       float* mf, size_t len,
                                                       int& uniq_len) {
  platform::Timer timeline;
  timeline.Start();
  int dev_id = resource_->dev_id(gpu_num);
  platform::CUDAPlace place = platform::CUDAPlace(dev_id);
  platform::CUDADeviceGuard guard(dev_id);
  auto stream = resource_->local_stream(gpu_num, 0);

  size_t temp_storage_bytes;

  //VLOG(1) << "hetercomm merge_grad: max_mf_dim: " << max_mf_dim_;
  size_t grad_value_size =
      TYPEALIGN(8, sizeof(FeaturePushValue) + (max_mf_dim_ * sizeof(float)));

  auto d_merge_keys = memory::Alloc(place, len * sizeof(KeyType));
  KeyType* d_merge_keys_ptr = reinterpret_cast<KeyType*>(d_merge_keys->ptr());

  auto d_merge_grads = memory::Alloc(place, len * grad_value_size);
  GradType* d_merge_grads_ptr =
      reinterpret_cast<GradType*>(d_merge_grads->ptr());

  auto d_fea_num_info =
      memory::Alloc(place, sizeof(uint32_t) * (len * 3 + 1));
  uint32_t* d_fea_num_info_ptr =
      reinterpret_cast<uint32_t*>(d_fea_num_info->ptr());
  uint32_t* d_index = (uint32_t*)&d_fea_num_info_ptr[len];
  uint32_t* d_idx = (uint32_t*)&d_index[len];
  int* d_merged_size = (int*)&d_idx[len];
  int grid_size = (len - 1) / block_size_ + 1;
  fill_idx<<<grid_size, block_size_, 0, stream>>>(d_idx, len);
  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceRadixSort::SortPairs(
      NULL, temp_storage_bytes, d_keys, d_merge_keys_ptr, d_idx, d_index, len,
      0, 8 * sizeof(KeyType), stream));

  void* d_buff = NULL;
  auto d_temp_storage = memory::Alloc(place, temp_storage_bytes);
  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceRadixSort::SortPairs(
      d_temp_storage->ptr(), temp_storage_bytes, d_keys, d_merge_keys_ptr,
      d_idx, d_index, len, 0, 8 * sizeof(KeyType), stream));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
  timeline.Pause();
  mg_time_4[gpu_num] += timeline.ElapsedSec();
  timeline.Start();

  /*
  char* test_grad_values_1 =
      (char*)malloc(sizeof(KeyType) * len);
  char* test_grad_keys_1 =
      (char*)malloc(sizeof(KeyType) * len);
  cudaMemcpy(test_grad_values_1, d_merge_keys_ptr,
            sizeof(KeyType) * len, cudaMemcpyDeviceToHost);
  cudaMemcpy(test_grad_keys_1, d_index,
            sizeof(uint32_t) * len, cudaMemcpyDeviceToHost);
  for (int i = 0; i < len; i++) {
    VLOG(0) << "yxfpush22222:: i: " << i << " key: " << ((uint64_t*)test_grad_values_1)[i] << " cur index: " << ((uint32_t*)test_grad_keys_1)[i];
  }
  */

  temp_storage_bytes = 0;
  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceRunLengthEncode::Encode(
      NULL, temp_storage_bytes, d_merge_keys_ptr, d_keys, d_fea_num_info_ptr,
      d_merged_size, len, stream));
  if (d_temp_storage->size() < temp_storage_bytes) {
    d_temp_storage = NULL;
    d_temp_storage = memory::Alloc(place, temp_storage_bytes);
  }
  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceRunLengthEncode::Encode(
      d_temp_storage->ptr(), temp_storage_bytes, d_merge_keys_ptr, d_keys,
      d_fea_num_info_ptr, d_merged_size, len, stream));

  cudaMemcpyAsync((void*)&uniq_len, d_merged_size, sizeof(int),
                  cudaMemcpyDeviceToHost, stream);
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
  timeline.Pause();
  mg_time_5[gpu_num] += timeline.ElapsedSec();
  timeline.Start();

  assert(d_merged_size > 0);
  uint32_t* d_offset = (uint32_t*)&d_index[len];

  temp_storage_bytes = 0;
  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceScan::ExclusiveSum(
      NULL, temp_storage_bytes, d_fea_num_info_ptr, d_offset, uniq_len,
      stream));
  if (d_temp_storage->size() < temp_storage_bytes) {
    d_temp_storage = NULL;
    d_temp_storage = memory::Alloc(place, temp_storage_bytes);
  }
  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceScan::ExclusiveSum(
      d_temp_storage->ptr(), temp_storage_bytes, d_fea_num_info_ptr, d_offset,
      uniq_len, stream));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
  timeline.Pause();
  mg_time_6[gpu_num] += timeline.ElapsedSec();
  timeline.Start();

  /*
  char* h_offset =
      (char*)malloc(sizeof(uint32_t) * len);
  char* h_fea_num_info_ptr =
      (char*)malloc(sizeof(uint32_t) * len);
  char* h_index =
      (char*)malloc(sizeof(uint32_t) * len);
  char* h_merge_grads_ptr =
      (char*)malloc(grad_value_size * len);
  char* h_grads =
      (char*)malloc(grad_value_size * len);
  char* h_keys =
      (char*)malloc(sizeof(KeyType) * len);
  cudaMemcpy(h_merge_grads_ptr, (char*)d_merge_grads_ptr,
            grad_value_size * len, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_grads, (char*)d_grads,
            grad_value_size * len, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_offset, d_offset,
            sizeof(uint32_t) * len, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_fea_num_info_ptr, d_fea_num_info_ptr,
            sizeof(uint32_t) * len, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_index, d_index,
            sizeof(uint32_t) * len, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_keys, d_keys,
            sizeof(KeyType) * len, cudaMemcpyDeviceToHost);

  uint32_t test_index = 0;
  for (int i = 0; i < uniq_len; i++) {
    if (((uint64_t*)h_keys)[i] == 1306488441314719671) {
      test_index = i;
      break;
    }
  }
  VLOG(0) << "yxf::test_index: " << test_index;
  uint32_t start = ((uint32_t*)h_offset)[test_index];
  uint32_t num = ((uint32_t*)h_fea_num_info_ptr)[test_index];
  int ori_index = ((uint32_t*)h_index)[start];

  VLOG(0) << "yxf:: start: " << start << " num: " << num << " index: " << ori_index;
  FeaturePushValue* cur =
        (FeaturePushValue*)((char*)h_grads + ori_index * grad_value_size);
  VLOG(0) << "yxfpush33333::cur->slot: " << cur->slot << " mf_dim: " << cur->mf_dim
            << " show: " << cur->show << " clk: " << cur->clk << " : " << cur->mf_g[0] << " : " << cur->mf_g[1] << " : " << cur->mf_g[2] << " : " << cur->mf_g[3] << " : " << cur->mf_g[4] << " : " << cur->mf_g[5] << " : " << cur->mf_g[6] << " : " << cur->mf_g[7];
    
  for (int j = 1; j < num; ++j) {
    ori_index = ((uint32_t*)h_index)[start + j];
    VLOG(0) << "yxf:: ori_index: " << ori_index;
    cur = (FeaturePushValue*)((char*)h_grads + ori_index * grad_value_size);
    VLOG(0) << "yxfpush33333::cur->slot: " << cur->slot << " mf_dim: " << cur->mf_dim
            << " show: " << cur->show << " clk: " << cur->clk << " : " << cur->mf_g[0] << " : " << cur->mf_g[1] << " : " << cur->mf_g[2] << " : " << cur->mf_g[3] << " : " << cur->mf_g[4] << " : " << cur->mf_g[5] << " : " << cur->mf_g[6] << " : " << cur->mf_g[7];
  }
  */


  //VLOG(0) << "yxf111";

  grid_size = (uniq_len - 1) / block_size_ + 1;
  merge_gradient_kernel<<<grid_size, block_size_, 0, stream>>>(
      d_offset, d_fea_num_info_ptr, d_index, (char*)d_grads,
      (char*)d_merge_grads_ptr, uniq_len, grad_value_size, merger_);
  // grid_size = (len - 1) / block_size_ + 1;
  // shared_merge_gradient_kernel<<<grid_size, block_size_, len * grad_value_size, stream>>>(
  //     d_offset, d_fea_num_info_ptr, d_index, (char*)d_grads,
  //     (char*)d_merge_grads_ptr, uniq_len, grad_value_size, merger_, len);
  // // merge kernel use cuda shared memory
  // grid_size = (len - 1) / block_size_ + 1;
  // auto d_test = memory::Alloc(place, len * grad_value_size);
  // GradType* d_test_ptr =
  //     reinterpret_cast<GradType*>(d_test->ptr());
  // merge_gradient_kernel<<<grid_size, block_size_, len * 5 * sizeof(float), stream>>>(
  //     d_offset, d_fea_num_info_ptr, d_index, (char*)d_grads,
  //     (char*)d_merge_grads_ptr, uniq_len, grad_value_size, merger_, len, (char*)d_test_ptr);
  // PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
  // VLOG(0) << "yxf:: mergelen: " << len << " uniq_len: " << uniq_len;
  // char* test_grad_values =
  //     (char*)malloc(grad_value_size * len);
  // char* test_grad_keys =
  //     (char*)malloc(sizeof(KeyType) * len);
  // cudaMemcpy(test_grad_values, d_test_ptr,
  //           grad_value_size * len, cudaMemcpyDeviceToHost);
  // cudaMemcpy(test_grad_keys, d_keys,
  //           sizeof(KeyType) * len, cudaMemcpyDeviceToHost);
  // for (int i = 0; i < len; i++) {
  //   FeaturePushValue* cur =
  //       (FeaturePushValue*)(test_grad_values + i * grad_value_size);
  //   VLOG(0) << "yxfpush merge:: i: " << i << " key: " << ((uint64_t*)test_grad_keys)[i] << " cur->slot: " << cur->slot << " mf_dim: " << cur->mf_dim
  //           << " show: " << cur->show << " clk: " << cur->clk << " : " << cur->mf_g[0] << " : " << cur->mf_g[1] << " : " << cur->mf_g[2] << " : " << cur->mf_g[3] << " : " << cur->mf_g[4] << " : " << cur->mf_g[5] << " : " << cur->mf_g[6] << " : " << cur->mf_g[7];
  // }
  // VLOG(0) << "yxf::merge len: enddd";
  // VLOG(0) << "yxf222";
  
  // timeline.Pause();
  // mg_time_8[gpu_num] += timeline.ElapsedSec();
  // timeline.Start();
  // grid_size = (uniq_len * max_mf_dim_  - 1) / block_size_ + 1;
  // kernel_merge_embedx_value<<<grid_size, block_size_, 0, stream>>>(
  //               uniq_len * max_mf_dim_, max_mf_dim_, d_offset, d_fea_num_info_ptr, d_index, (char*)d_grads, (char*)d_merge_grads_ptr, grad_value_size);

  // kernel_merge_embedx_value<<<grid_size, block_size_, len * max_mf_dim_ * sizeof(float), stream>>>(
  // kernel_merge_embedx_value<<<grid_size, block_size_, len * max_mf_dim_ * sizeof(float), stream>>>(
  //               uniq_len * max_mf_dim_, max_mf_dim_, d_offset, d_fea_num_info_ptr, d_index, (char*)d_grads, (char*)d_merge_grads_ptr, grad_value_size, len * max_mf_dim_);
  // cudaMemcpyAsync(&uniq_len, d_merged_size, sizeof(int),
  //                cudaMemcpyDeviceToHost, stream);
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
  //VLOG(0) << "yxf333";
  timeline.Pause();
  mg_time_7[gpu_num] += timeline.ElapsedSec();
  timeline.Start();

  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaMemcpyAsync(d_grads, d_merge_grads_ptr, grad_value_size * uniq_len,
                      cudaMemcpyDeviceToDevice, stream));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
  //VLOG(0) << "yxf444";
  timeline.Pause();
  mg_time_1[gpu_num] += timeline.ElapsedSec();
  timeline.Start();
}

template <typename KeyType, typename ValType, typename GradType>
void HeterComm<KeyType, ValType, GradType>::split_input_to_shard(
    KeyType* d_keys, int* d_idx_ptr, size_t len, int* left, int* right,
    int gpu_num) {
  int total_gpu = resource_->total_gpu();
  int dev_id = resource_->dev_id(gpu_num);
  platform::CUDAPlace place = platform::CUDAPlace(dev_id);
  platform::CUDADeviceGuard guard(dev_id);
  auto stream = resource_->local_stream(gpu_num, 0);

  auto d_idx_tmp = memory::Alloc(place, len * sizeof(int));
  int* d_idx_tmp_ptr = reinterpret_cast<int*>(d_idx_tmp->ptr());

  auto d_shard_index = memory::Alloc(place, len * sizeof(int));
  int* d_shard_index_ptr = reinterpret_cast<int*>(d_shard_index->ptr());

  auto d_shard_index_tmp = memory::Alloc(place, len * sizeof(int));
  int* d_shard_index_tmp_ptr = reinterpret_cast<int*>(d_shard_index_tmp->ptr());

  int grid_size = (len - 1) / block_size_ + 1;
  fill_idx<<<grid_size, block_size_, 0, stream>>>(d_idx_tmp_ptr, len);
  calc_shard_index<<<grid_size, block_size_, 0, stream>>>(
      d_keys, len, d_shard_index_tmp_ptr, total_gpu);

  size_t temp_storage_bytes;
  const int num_bits = 1 + log2i(total_gpu);
  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceRadixSort::SortPairs(
      NULL, temp_storage_bytes, d_shard_index_tmp_ptr, d_shard_index_ptr,
      d_idx_tmp_ptr, d_idx_ptr, len, 0, num_bits, stream));

  auto d_temp_storage = memory::Alloc(place, temp_storage_bytes);
  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceRadixSort::SortPairs(
      d_temp_storage->ptr(), temp_storage_bytes, d_shard_index_tmp_ptr,
      d_shard_index_ptr, d_idx_tmp_ptr, d_idx_ptr, len, 0, num_bits, stream));
  calc_shard_offset<<<grid_size, block_size_, 0, stream>>>(d_shard_index_ptr,
                                                           left, right, len);
  cudaStreamSynchronize(stream);
}

template <typename KeyType, typename ValType, typename GradType>
void HeterComm<KeyType, ValType, GradType>::pull_sparse(int num,
                                                        KeyType* d_keys,
                                                        ValType* d_vals,
                                                        size_t len) {
  if (len == 0) {
    return;
  }

  int total_gpu = resource_->total_gpu();
  int dev_id = resource_->dev_id(num);
  platform::CUDAPlace place = platform::CUDAPlace(dev_id);
  platform::CUDADeviceGuard guard(dev_id);
  auto stream = resource_->local_stream(num, 0);

  int grid_size = (len - 1) / block_size_ + 1;

  int h_left[total_gpu];   // NOLINT
  int h_right[total_gpu];  // NOLINT

  auto d_left = memory::Alloc(place, total_gpu * sizeof(int));
  auto d_right = memory::Alloc(place, total_gpu * sizeof(int));
  int* d_left_ptr = reinterpret_cast<int*>(d_left->ptr());
  int* d_right_ptr = reinterpret_cast<int*>(d_right->ptr());

  cudaMemsetAsync(d_left_ptr, -1, total_gpu * sizeof(int), stream);
  cudaMemsetAsync(d_right_ptr, -1, total_gpu * sizeof(int), stream);
  //
  auto d_idx = memory::Alloc(place, len * sizeof(int));
  int* d_idx_ptr = reinterpret_cast<int*>(d_idx->ptr());

  size_t val_type_size = 0;
  if (!multi_mf_dim_) {
    val_type_size = sizeof(ValType);
  } else {
    val_type_size =
        TYPEALIGN(8, sizeof(FeatureValue) + sizeof(float) * (max_mf_dim_ + 1));
  }

  auto d_shard_keys = memory::Alloc(place, len * sizeof(KeyType));
  KeyType* d_shard_keys_ptr = reinterpret_cast<KeyType*>(d_shard_keys->ptr());
  // auto d_shard_vals = memory::Alloc(place, len * sizeof(ValType));
  auto d_shard_vals = memory::Alloc(place, len * val_type_size);
  ValType* d_shard_vals_ptr = reinterpret_cast<ValType*>(d_shard_vals->ptr());

  split_input_to_shard(d_keys, d_idx_ptr, len, d_left_ptr, d_right_ptr, num);

  fill_shard_key<<<grid_size, block_size_, 0, stream>>>(d_shard_keys_ptr,
                                                        d_keys, d_idx_ptr, len);

  cudaStreamSynchronize(stream);

  cudaMemcpy(h_left, d_left_ptr, total_gpu * sizeof(int),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_right, d_right_ptr, total_gpu * sizeof(int),
             cudaMemcpyDeviceToHost);

  for (int i = 0; i < total_gpu; ++i) {
    int shard_len = h_right[i] - h_left[i] + 1;
    if (shard_len == 0) {
      continue;
    }
    create_storage(num, i, shard_len * sizeof(KeyType),
                   shard_len * val_type_size);
  }

  walk_to_dest(num, total_gpu, h_left, h_right, d_shard_keys_ptr, NULL);

  std::vector<platform::Timer> time_lines;
  time_lines.resize(total_gpu);

  for (int i = 0; i < total_gpu; ++i) {
    time_lines[i].Start();
    if (h_left[i] == -1) {
      continue;
    }
    auto& node = path_[num][i].nodes_.back();
    cudaStreamSynchronize(node.in_stream);
    platform::CUDADeviceGuard guard(resource_->dev_id(i));
    if (!multi_mf_dim_) {
      tables_[i]->rwlock_->RDLock();
      tables_[i]->get(reinterpret_cast<KeyType*>(node.key_storage),
                      reinterpret_cast<ValType*>(node.val_storage),
                      h_right[i] - h_left[i] + 1,
                      resource_->remote_stream(i, num));
    } else {
      // VLOG(0) << "yxf:: start table get in device: " << num << " remote device: " << i;
      ptr_tables_[i]->rwlock_->RDLock();
      // VLOG(0) << "after lock yxf:: start table get in device: " << num << " remote device: " << i;
      ptr_tables_[i]->get(reinterpret_cast<KeyType*>(node.key_storage),
                          node.val_storage, h_right[i] - h_left[i] + 1,
                          resource_->remote_stream(i, num));
    }
  }
  for (int i = 0; i < total_gpu; ++i) {
    cudaStreamSynchronize(resource_->remote_stream(i, num));
    if (h_left[i] == -1) {
      continue;
    }
    if (!multi_mf_dim_) {
      tables_[i]->rwlock_->UNLock();
    } else {
      ptr_tables_[i]->rwlock_->UNLock();
    }
    time_lines[i].Pause();
    mg_time_2[i] += time_lines[i].ElapsedSec();
  }

  if (!multi_mf_dim_) {
    walk_to_src(num, total_gpu, h_left, h_right, d_shard_vals_ptr);
  } else {
    walk_to_src(num, total_gpu, h_left, h_right, reinterpret_cast<char*>(d_shard_vals_ptr), val_type_size);
  }

  for (int i = 0; i < total_gpu; ++i) {
    auto& node = path_[num][i].nodes_.front();
    cudaStreamSynchronize(node.out_stream);
  }

  VLOG(0) << "yxf::finish walk to src: " << num;

  if (!multi_mf_dim_) {
    fill_dvals<<<grid_size, block_size_, 0, stream>>>(d_shard_vals_ptr, d_vals,
                                                      d_idx_ptr, len);
  } else {
    // grid_size = (len - 1) / 2048 + 1;
    dy_mf_fill_dvals<<<grid_size, block_size_, 0, stream>>>(
        d_shard_vals_ptr, d_vals, d_idx_ptr, len, val_type_size);
  }
  cudaStreamSynchronize(stream);
  // VLOG(0) << "yxf::finish walk to fill: " << num;
  for (int i = 0; i < total_gpu; ++i) {
    destroy_storage(num, i);
    VLOG(0) << "yxf::end get device: " << num << "from device: " << i;
  }
}

template <typename KeyType, typename ValType, typename GradType>
template <typename Sgd>
void HeterComm<KeyType, ValType, GradType>::push_sparse(int gpu_num,
                                                        KeyType* d_keys,
                                                        GradType* d_grads,
                                                        size_t len,
                                                        Sgd& sgd) {  // NOLINT
  if (len == 0) {
    return;
  }

  int total_gpu = resource_->total_gpu();
  int dev_id = resource_->dev_id(gpu_num);
  platform::CUDAPlace place = platform::CUDAPlace(dev_id);
  platform::CUDADeviceGuard guard(dev_id);
  auto stream = resource_->local_stream(gpu_num, 0);

  size_t grad_value_size =
      TYPEALIGN(8, sizeof(FeaturePushValue) + (max_mf_dim_ * sizeof(float)));

  int h_left[total_gpu];   // NOLINT
  int h_right[total_gpu];  // NOLINT

  auto d_left = memory::Alloc(place, total_gpu * sizeof(int));
  auto d_right = memory::Alloc(place, total_gpu * sizeof(int));
  int* d_left_ptr = reinterpret_cast<int*>(d_left->ptr());
  int* d_right_ptr = reinterpret_cast<int*>(d_right->ptr());

  cudaMemsetAsync(d_left_ptr, -1, total_gpu * sizeof(int), stream);
  cudaMemsetAsync(d_right_ptr, -1, total_gpu * sizeof(int), stream);
  //
  auto d_idx = memory::Alloc(place, len * sizeof(int));
  int* d_idx_ptr = reinterpret_cast<int*>(d_idx->ptr());

  auto d_shard_keys = memory::Alloc(place, len * sizeof(KeyType));
  KeyType* d_shard_keys_ptr = reinterpret_cast<KeyType*>(d_shard_keys->ptr());

  GradType* d_shard_grads_ptr;
  if (!multi_mf_dim_) {
    auto d_shard_grads = memory::Alloc(place, len * sizeof(GradType));
    d_shard_grads_ptr = reinterpret_cast<GradType*>(d_shard_grads->ptr());
  } else {
    auto d_shard_grads = memory::Alloc(place, len * grad_value_size);
    d_shard_grads_ptr = reinterpret_cast<GradType*>(d_shard_grads->ptr());
  }

  int uniq_len = len;
  merge_grad(gpu_num, d_keys, d_grads, NULL, len, uniq_len);

  int grid_size = (uniq_len - 1) / block_size_ + 1;

  split_input_to_shard(d_keys, d_idx_ptr, uniq_len, d_left_ptr, d_right_ptr,
                       gpu_num);

  if (!multi_mf_dim_) {
    fill_shard_grads<<<grid_size, block_size_, 0, stream>>>(
        d_shard_keys_ptr, d_keys, d_shard_grads_ptr, d_grads, d_idx_ptr,
        uniq_len);
  } else {
    dy_mf_fill_shard_grads<<<grid_size, block_size_, 0, stream>>>(
        d_shard_keys_ptr, d_keys, d_shard_grads_ptr, d_grads, d_idx_ptr,
        uniq_len, grad_value_size);
  }

  cudaStreamSynchronize(stream);

  cudaMemcpy(h_left, d_left_ptr, total_gpu * sizeof(int),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_right, d_right_ptr, total_gpu * sizeof(int),
             cudaMemcpyDeviceToHost);

  for (int i = 0; i < total_gpu; ++i) {
    int shard_len = h_right[i] - h_left[i] + 1;
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    if (!multi_mf_dim_) {
      create_storage(gpu_num, i, shard_len * sizeof(KeyType),
                     shard_len * sizeof(GradType));
    } else {
      create_storage(gpu_num, i, shard_len * sizeof(KeyType),
                     shard_len * grad_value_size);
    }
  }

  if (!multi_mf_dim_) {
    walk_to_dest(gpu_num, total_gpu, h_left, h_right, d_shard_keys_ptr,
               d_shard_grads_ptr);
  } else {
    walk_to_dest(gpu_num, total_gpu, h_left, h_right, d_shard_keys_ptr,
               reinterpret_cast<char*>(d_shard_grads_ptr), grad_value_size);
  }

  std::vector<platform::Timer> time_lines;
  time_lines.resize(total_gpu);

  for (int i = 0; i < total_gpu; ++i) {
    time_lines[i].Start();
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    auto& node = path_[gpu_num][i].nodes_.back();
    cudaStreamSynchronize(node.in_stream);

    platform::CUDADeviceGuard guard(resource_->dev_id(i));
    if (!multi_mf_dim_) {
      tables_[i]->rwlock_->WRLock();
      tables_[i]->update(reinterpret_cast<KeyType*>(node.key_storage),
                         reinterpret_cast<GradType*>(node.val_storage),
                         h_right[i] - h_left[i] + 1, sgd,
                         resource_->remote_stream(i, gpu_num));
    } else {
      /*
      char* test_grad_values =
          (char*)malloc(grad_value_size * (h_right[i] - h_left[i] + 1));
      char* test_grad_keys =
          (char*)malloc(sizeof(KeyType) * (h_right[i] - h_left[i] + 1));
      cudaMemcpy(test_grad_values, node.val_storage,
                grad_value_size * (h_right[i] - h_left[i] + 1), cudaMemcpyDeviceToHost);
      cudaMemcpy(test_grad_keys, node.key_storage,
                sizeof(KeyType) * (h_right[i] - h_left[i] + 1), cudaMemcpyDeviceToHost);
      for (int i = 0; i < 10; i++) {
        FeaturePushValue* cur =
            (FeaturePushValue*)(test_grad_values + i * grad_value_size);
        VLOG(0) << "yxfpush:: i: " << i << " key: " << ((uint64_t*)test_grad_keys)[i] << " cur->slot: " << cur->slot
                << " show: " << cur->show << " mf_g: " << cur->mf_g;
      }
      */
      ptr_tables_[i]->rwlock_->WRLock();
      ptr_tables_[i]->update(reinterpret_cast<KeyType*>(node.key_storage),
                             node.val_storage, h_right[i] - h_left[i] + 1, sgd,
                             resource_->remote_stream(i, gpu_num));
    }
  }
  for (int i = 0; i < total_gpu; ++i) {
    cudaStreamSynchronize(resource_->remote_stream(i, gpu_num));
    if (h_left[i] != -1) {
      if (!multi_mf_dim_) {
        tables_[i]->rwlock_->UNLock();
      } else {
        ptr_tables_[i]->rwlock_->UNLock();
      }
      time_lines[i].Pause();
      mg_time_3[i] += time_lines[i].ElapsedSec();
    }
  }
  for (int i = 0; i < total_gpu; ++i) {
    destroy_storage(gpu_num, i);
  }
}

template <typename KeyType, typename ValType, typename GradType>
template <typename Sgd>
void HeterComm<KeyType, ValType, GradType>::update_one_table(
    int gpu_num, KeyType* d_keys, GradType* d_grads, size_t len,
    Sgd& sgd) {  // NOLINT
  if (len == 0) {
    return;
  }

  int dev_id = resource_->dev_id(gpu_num);
  platform::CUDADeviceGuard guard(dev_id);
  tables_[gpu_num]->rwlock_->WRLock();
  tables_[gpu_num]->update(d_keys, d_grads, len, sgd,
                           resource_->remote_stream(gpu_num, gpu_num));
  tables_[gpu_num]->rwlock_->UNLock();
  cudaStreamSynchronize(resource_->remote_stream(gpu_num, gpu_num));
}

template <typename KeyType, typename ValType, typename GradType>
template <typename Sgd>
void HeterComm<KeyType, ValType, GradType>::push_sparse_multi_node(
    int gpu_num, KeyType* d_keys, GradType* d_grads, size_t len,
    Sgd& sgd) {  // NOLINT
  if (len == 0) {
    return;
  }

  int uniq_len = len;
  merge_grad(gpu_num, d_keys, d_grads, len, uniq_len);

  uniq_len = gather_one_node_grad(gpu_num, d_keys, d_grads, uniq_len);

  uniq_len = gather_multi_node_grad(gpu_num, storage_[gpu_num].local_keys,
                                    storage_[gpu_num].local_grads, uniq_len);

  update_one_table(gpu_num, storage_[gpu_num].local_keys,
                   storage_[gpu_num].local_grads, uniq_len, sgd);
}

template <typename KeyType, typename ValType, typename GradType>
int HeterComm<KeyType, ValType, GradType>::gather_one_node_grad(
    int gpu_num, KeyType* d_keys, GradType* d_grads, int len) {
  int total_gpu = resource_->total_gpu();
  int dev_id = resource_->dev_id(gpu_num);
  auto& storage = storage_[gpu_num];
  platform::CUDAPlace place = platform::CUDAPlace(dev_id);
  platform::CUDADeviceGuard guard(dev_id);
  auto stream = resource_->local_stream(gpu_num, 0);
  int max_size = 0;

  ncclComm_t nccl_inner_comm = nccl_inner_comms_[gpu_num];
  // alloc for size
  int h_node_len[total_gpu];  // NOLINT
  auto d_node_len_mem = memory::Alloc(place, total_gpu * sizeof(int));
  int* d_node_len = reinterpret_cast<int*>(d_node_len_mem->ptr());
  h_node_len[gpu_num] = len;

  cudaMemcpy(d_node_len + gpu_num, h_node_len + gpu_num, sizeof(int),
             cudaMemcpyHostToDevice);

  // allgather grad len
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupStart());
  PADDLE_ENFORCE_GPU_SUCCESS(
      platform::dynload::ncclAllGather((const void*)(d_node_len + gpu_num),
                                       (void*)d_node_len, 1, ncclInt,  // NOLINT
                                       nccl_inner_comm, stream));
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupEnd());
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
  cudaMemcpy(h_node_len, d_node_len, sizeof(int) * total_gpu,
             cudaMemcpyDeviceToHost);

  for (int i = 0; i < total_gpu; ++i) {
    if (h_node_len[i] > max_size) {
      max_size = h_node_len[i];
    }
  }
  storage.alloc(max_size * total_gpu);

  // allgather keys and grads
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupStart());
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllGather(
      d_keys, storage.all_keys, max_size, ncclUint64, nccl_inner_comm, stream));

  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllGather(
      d_grads, storage.all_grads, max_size * sizeof(GradType), ncclUint8,
      nccl_inner_comm, stream));
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupEnd());
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));

  int h_left[total_gpu];   // NOLINT
  int h_right[total_gpu];  // NOLINT
  auto d_left = memory::Alloc(place, total_gpu * sizeof(int));
  auto d_right = memory::Alloc(place, total_gpu * sizeof(int));
  int* d_left_ptr = reinterpret_cast<int*>(d_left->ptr());
  int* d_right_ptr = reinterpret_cast<int*>(d_right->ptr());

  int merge_num = 0;
  for (int i = 0; i < total_gpu; ++i) {
    int index = i * max_size;
    auto d_idx = memory::Alloc(place, h_node_len[i] * sizeof(int));
    int* d_idx_ptr = reinterpret_cast<int*>(d_idx->ptr());

    cudaMemset(d_left_ptr, -1, total_gpu * sizeof(int));
    cudaMemset(d_right_ptr, -1, total_gpu * sizeof(int));

    split_input_to_shard(storage.all_keys + index, d_idx_ptr, h_node_len[i],
                         d_left_ptr, d_right_ptr, gpu_num);
    cudaMemcpy(h_left, d_left_ptr, total_gpu * sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_right, d_right_ptr, total_gpu * sizeof(int),
               cudaMemcpyDeviceToHost);

    int grid_size = (h_node_len[i] - 1) / block_size_ + 1;
    fill_shard_grads<<<grid_size, block_size_, 0, stream>>>(
        storage.local_keys + merge_num, storage.all_keys + index,
        storage.local_grads + merge_num, storage.all_grads + index,
        d_idx_ptr + h_left[gpu_num], h_right[gpu_num] - h_left[gpu_num] + 1);
    merge_num = merge_num + h_right[gpu_num] - h_left[gpu_num] + 1;
  }

  int ret = merge_num;
  merge_grad(gpu_num, storage.local_keys, storage.local_grads, merge_num, ret);
  return ret;
}

template <typename KeyType, typename ValType, typename GradType>
int HeterComm<KeyType, ValType, GradType>::gather_multi_node_grad(
    int gpu_num, KeyType* d_keys, GradType* d_grads, int len) {
  int dev_id = resource_->dev_id(gpu_num);
  auto& storage = storage_[gpu_num];
  platform::CUDAPlace place = platform::CUDAPlace(dev_id);
  platform::CUDADeviceGuard guard(dev_id);
  auto stream = resource_->local_stream(gpu_num, 0);
  int max_size = 0;
  ncclComm_t nccl_inter_comm = nccl_inter_comms_[gpu_num];
  // alloc for size
  int h_node_len[node_size_];  // NOLINT
  auto d_node_len_mem = memory::Alloc(place, node_size_ * sizeof(int));
  int* d_node_len = reinterpret_cast<int*>(d_node_len_mem->ptr());
  h_node_len[0] = len;

  cudaMemcpy(d_node_len, h_node_len, sizeof(int), cudaMemcpyHostToDevice);

  // allgather grad len
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupStart());
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllGather(
      d_node_len, d_node_len, 1, ncclInt, nccl_inter_comm, stream));
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupEnd());
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
  cudaMemcpy(h_node_len, d_node_len, sizeof(int) * node_size_,
             cudaMemcpyDeviceToHost);

  for (int i = 0; i < node_size_; ++i) {
    if (h_node_len[i] > max_size) {
      max_size = h_node_len[i];
    }
  }
  storage.alloc(max_size * node_size_);

  // allgather keys and grads
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupStart());
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllGather(
      d_keys, storage.all_keys, max_size, ncclUint64, nccl_inter_comm, stream));

  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllGather(
      d_grads, storage.all_grads, max_size * sizeof(GradType), ncclUint8,
      nccl_inter_comm, stream));
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupEnd());
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));

  int merge_num = 0;
  for (int i = 0; i < node_size_; ++i) {
    int index = i * max_size;
    cudaMemcpyAsync(storage.local_keys + merge_num, storage.all_keys + index,
                    h_node_len[i], cudaMemcpyDefault, stream);
    cudaMemcpyAsync(storage.local_grads + merge_num, storage.all_grads + index,
                    h_node_len[i], cudaMemcpyDefault, stream);
    merge_num += h_node_len[i];
  }

  int ret = merge_num;
  merge_grad(gpu_num, storage.local_keys, storage.local_grads, merge_num, ret);
  return ret;
}

template <typename KeyType, typename ValType, typename GradType>
void HeterComm<KeyType, ValType, GradType>::end_pass() {
  int total_gpu = resource_->total_gpu();
  std::vector<std::thread> threads;

  auto dump_to_cpu_func = [this](int index) {
    auto stream = resource_->local_stream(index, 0);
    int dev_id = resource_->dev_id(index);
    platform::CUDADeviceGuard guard(dev_id);
    tables_[index]->dump_to_cpu(dev_id, stream);
  };

  if (!multi_mf_dim_) {
    for (int i = 0; i < total_gpu; ++i) {
      threads.push_back(std::thread(dump_to_cpu_func, i));
    }
    for (auto& t : threads) {
      t.join();
    }
  }
}

// template <typename KeyType, typename ValType, typename GradType>
// void HeterComm<KeyType, ValType, GradType>::dump_to_cpu(int index) {
//  auto stream = resource_->local_stream(index, 0);
//  int dev_id = resource_->dev_id(index);
//  platform::CUDADeviceGuard guard(dev_id);
//  tables_[index]->dump_to_cpu(dev_id, stream);
//}

}  // end namespace framework
}  // end namespace paddle
#endif

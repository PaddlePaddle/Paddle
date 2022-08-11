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
#include "paddle/fluid/framework/fleet/heter_ps/heter_comm_kernel.h"
#include "paddle/fluid/platform/device_context.h"
#ifdef PADDLE_WITH_XPU_KP
#include "paddle/fluid/platform/device/xpu/xpu_info.h"
#if defined(PADDLE_WITH_XPU_BKCL)
#include "paddle/fluid/platform/device/xpu/bkcl_helper.h"
#include "paddle/fluid/platform/collective_helper.h"
#endif
#endif

namespace paddle {
namespace framework {

template <typename KeyType, typename ValType, typename GradType>
HeterComm<KeyType, ValType, GradType>::HeterComm(
    size_t capacity, std::shared_ptr<HeterPsResource> resource) {
  resource_ = resource;
  storage_.resize(resource_->total_device());
  for (int i = 0; i < resource_->total_device(); ++i) {
#if defined(PADDLE_WITH_CUDA)
    platform::CUDADeviceGuard guard(resource_->dev_id(i));
    allocators_.push_back(std::make_shared<cub::CachingDeviceAllocator>(
        8, 1, (unsigned int)-1, (size_t)-1, false, false));  // NOLINT
#endif
#if defined(PADDLE_WITH_XPU_KP)
    int dev_id = resource_->dev_id(i);
    AnyDeviceGuard guard(dev_id);
    DevPlace place = DevPlace(dev_id);
    auto table = new Table(capacity / load_factor_, place);
#else
    auto table = new Table(capacity / load_factor_);
#endif
#if defined(PADDLE_WITH_XPU_KP)
    int dev_idx = get_index_by_devid(dev_id);
    table->set_xpu_id(dev_id);
    table->set_xpu_idx(dev_idx);
    table->set_xpu_num(resource_->total_device());
    VLOG(3) << "init heter xpu table(id|idx|dev_num):"
            << dev_id << "|" << dev_idx << "|" << resource_->total_device();
#endif
    tables_.push_back(table);
    if (multi_node_) {
      storage_[i].init(feanum_, resource_->dev_id(i));
    }
  }
  heter_comm_kernel_ = std::make_unique<HeterCommKernel>(block_size_);
  init_path();

#if defined(PADDLE_WITH_XPU_KP)
  cache_mgr_ = std::make_shared<CacheManager>(resource_);
#endif
}

template <typename KeyType, typename ValType, typename GradType>
void HeterComm<KeyType, ValType, GradType>::init_path() {
  int total_device = resource_->total_device();
  path_.resize(total_device);
  if (!topo_aware_) {
    VLOG(3) << "init path without topo aware";
    for (int i = 0; i < total_device; ++i) {
      path_[i].resize(total_device);
      for (int j = 0; j < total_device; ++j) {
        auto& nodes = path_[i][j].nodes_;
        nodes.resize(1);
        nodes[0].in_stream = resource_->comm_stream(i, j);
        nodes[0].out_stream = resource_->comm_stream(i, j);
        nodes[0].key_storage = NULL;
        nodes[0].val_storage = NULL;
        nodes[0].sync = 0;
        nodes[0].dev_num = j;
      }
    }
  } else {
    VLOG(3) << "init path with topo aware";
    for (int i = 0; i < total_device; ++i) {
      path_[i].resize(total_device);
      for (int j = 0; j < total_device; ++j) {
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
          node.dev_num = transfer_id;
        }
        nodes.push_back(Node());
        Node& node = nodes.back();
        node.in_stream = resource_->comm_stream(i, transfer_id);
        node.out_stream = resource_->comm_stream(transfer_id, i);
        node.key_storage = NULL;
        node.val_storage = NULL;
        node.sync = 0;
        node.dev_num = j;
      }
    }
  }
}

template <typename KeyType, typename ValType, typename GradType>
template <typename DstPlace, typename SrcPlace, typename StreamType>
void HeterComm<KeyType, ValType, GradType>::memory_copy(
    DstPlace dst_place, void* dst, SrcPlace src_place, const void* src,
    size_t count, StreamType stream) {
#if defined(PADDLE_WITH_CUDA)
  cudaMemcpyAsync(dst, src, count, cudaMemcpyDefault, stream);
  if (stream == 0) {
    cudaStreamSynchronize(0);
  }
#elif defined(PADDLE_WITH_XPU_KP)
  memory::Copy(dst_place, dst, src_place, src, count);
#endif
}

template <typename KeyType, typename ValType, typename GradType>
void HeterComm<KeyType, ValType, GradType>::create_storage(int start_index,
                                                           int end_index,
                                                           int keylen,
                                                           int vallen) {
#if defined(PADDLE_WITH_CUDA)
  auto& allocator = allocators_[start_index];
  auto& nodes = path_[start_index][end_index].nodes_;
  for (size_t i = 0; i < nodes.size(); ++i) {
    platform::CUDADeviceGuard guard(resource_->dev_id(nodes[i].dev_num));
    allocator->DeviceAllocate(
        resource_->dev_id(nodes[i].dev_num),
        (void**)&(nodes[i].key_storage),  // NOLINT
        keylen, resource_->remote_stream(nodes[i].dev_num, start_index));
    allocator->DeviceAllocate(
        resource_->dev_id(nodes[i].dev_num),
        (void**)&(nodes[i].val_storage),  // NOLINT
        vallen, resource_->remote_stream(nodes[i].dev_num, start_index));
    nodes[i].key_bytes_len = keylen;
    nodes[i].val_bytes_len = vallen;
  }
#elif defined(PADDLE_WITH_XPU_KP)
  auto& nodes = path_[start_index][end_index].nodes_;
  for (size_t i = 0; i < nodes.size(); ++i) {
    platform::XPUDeviceGuard guard(resource_->dev_id(nodes[i].dev_num));
    auto place = DevPlace(resource_->dev_id(nodes[i].dev_num));
    auto node_keys_mem = memory::Alloc(place, keylen);
    nodes[i].key_storage = reinterpret_cast<char*>(node_keys_mem->ptr());
    auto node_vals_mem = memory::Alloc(place, vallen);
    nodes[i].val_storage = reinterpret_cast<char*>(node_vals_mem->ptr());
    nodes[i].key_bytes_len = keylen;
    nodes[i].val_bytes_len = vallen;
  }
#endif
}

template <typename KeyType, typename ValType, typename GradType>
void HeterComm<KeyType, ValType, GradType>::destroy_storage(int start_index,
                                                            int end_index) {
#if defined(PADDLE_WITH_CUDA)
  auto& allocator = allocators_[start_index];
  auto& nodes = path_[start_index][end_index].nodes_;
  for (size_t i = 0; i < nodes.size(); ++i) {
    platform::CUDADeviceGuard guard(resource_->dev_id(nodes[i].dev_num));

    allocator->DeviceFree(resource_->dev_id(nodes[i].dev_num),
                          nodes[i].key_storage);
    allocator->DeviceFree(resource_->dev_id(nodes[i].dev_num),
                          nodes[i].val_storage);
  }
#endif
}

template <typename KeyType, typename ValType, typename GradType>
void HeterComm<KeyType, ValType, GradType>::walk_to_dest(int start_index,
                                                         int num, int* h_left,
                                                         int* h_right,
                                                         KeyType* src_key,
                                                         GradType* src_val) {
  int need_copy_val = 0;
  if (src_val) {
    need_copy_val = 1;
  }
  std::queue<CopyTask> que;
  for (int i = 0; i < num; i++) {
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    // int size = path_[start_index][i].nodes_.size();
    auto& node = path_[start_index][i].nodes_[0];

    CopyTask t(&path_[start_index][i], 0);
    que.push(t);
    auto src_dev_id = resource_->dev_id(start_index);
    auto dst_dev_id = resource_->dev_id(i);
    auto src_place = DevPlace(src_dev_id);
    auto dst_place = DevPlace(dst_dev_id);

    memory_copy(dst_place, node.key_storage, src_place,
                reinterpret_cast<char*>(src_key + h_left[i]),
                node.key_bytes_len, node.in_stream);
    // #if defined(PADDLE_WITH_CUDA)  // adapt for gpu-graph
    //     cudaMemsetAsync(node.val_storage, -1, node.val_bytes_len,
    //     node.in_stream);
    // #endif

    if (need_copy_val) {
      memory_copy(dst_place, node.val_storage, src_place,
                  reinterpret_cast<char*>(src_val + h_left[i]),
                  node.val_bytes_len, node.in_stream);
    }
  }
  while (!que.empty()) {
    CopyTask& cur_task = que.front();
    que.pop();
    if (cur_task.path->nodes_[cur_task.step].sync) {
      sync_stream(cur_task.path->nodes_[cur_task.step].in_stream);
    }
    if (static_cast<size_t>(cur_task.step) !=
        cur_task.path->nodes_.size() - 1) {
      int cur_step = cur_task.step;
      CopyTask c(cur_task.path, cur_step + 1);
      que.push(c);

      auto src_dev_id =
          resource_->dev_id(cur_task.path->nodes_[cur_step].dev_num);
      auto dst_dev_id =
          resource_->dev_id(cur_task.path->nodes_[cur_step + 1].dev_num);
      auto src_place = DevPlace(src_dev_id);
      auto dst_place = DevPlace(dst_dev_id);

      memory_copy(dst_place, cur_task.path->nodes_[cur_step + 1].key_storage,
                  src_place, cur_task.path->nodes_[cur_step].key_storage,
                  cur_task.path->nodes_[cur_step + 1].key_bytes_len,
                  cur_task.path->nodes_[cur_step + 1].in_stream);
      if (need_copy_val) {
        memory_copy(dst_place, cur_task.path->nodes_[cur_step + 1].val_storage,
                    src_place, cur_task.path->nodes_[cur_step].val_storage,
                    cur_task.path->nodes_[cur_step + 1].val_bytes_len,
                    cur_task.path->nodes_[cur_step + 1].in_stream);
      }
    }
  }
}

template <typename KeyType, typename ValType, typename GradType>
void HeterComm<KeyType, ValType, GradType>::walk_to_src(int start_index,
                                                        int num, int* h_left,
                                                        int* h_right,
                                                        ValType* src_val) {
  std::queue<CopyTask> que;

  for (int i = 0; i < num; i++) {
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    int cur_step = path_[start_index][i].nodes_.size() - 1;
    auto& node = path_[start_index][i].nodes_[cur_step];

    auto src_dev_id = resource_->dev_id(i);
    auto src_place = DevPlace(src_dev_id);

    if (cur_step == 0) {
      auto dst_dev_id = resource_->dev_id(start_index);
      auto dst_place = DevPlace(dst_dev_id);
      memory_copy(dst_place, reinterpret_cast<char*>(src_val + h_left[i]),
                  src_place, node.val_storage, node.val_bytes_len,
                  node.out_stream);
    } else {
      CopyTask t(&path_[start_index][i], cur_step - 1);
      que.push(t);

      auto dst_dev_id =
          resource_->dev_id(path_[start_index][i].nodes_[cur_step - 1].dev_num);
      auto dst_place = DevPlace(dst_dev_id);

      memory_copy(dst_place,
                  path_[start_index][i].nodes_[cur_step - 1].val_storage,
                  src_place, node.val_storage,
                  path_[start_index][i].nodes_[cur_step - 1].val_bytes_len,
                  path_[start_index][i].nodes_[cur_step - 1].out_stream);
    }
  }

  while (!que.empty()) {
    CopyTask& cur_task = que.front();
    que.pop();
    int cur_step = cur_task.step;
    if (cur_task.path->nodes_[cur_step].sync) {
      sync_stream(cur_task.path->nodes_[cur_step].out_stream);
    }

    auto src_dev_id =
        resource_->dev_id(cur_task.path->nodes_[cur_step].dev_num);
    auto src_place = DevPlace(src_dev_id);

    if (cur_step > 0) {
      CopyTask c(cur_task.path, cur_step - 1);
      que.push(c);

      auto dst_dev_id =
          resource_->dev_id(cur_task.path->nodes_[cur_step - 1].dev_num);
      auto dst_place = DevPlace(dst_dev_id);

      memory_copy(dst_place, cur_task.path->nodes_[cur_step - 1].val_storage,
                  src_place, cur_task.path->nodes_[cur_step].val_storage,
                  cur_task.path->nodes_[cur_step - 1].val_bytes_len,
                  cur_task.path->nodes_[cur_step - 1].out_stream);

    } else if (cur_step == 0) {
      int end_index = cur_task.path->nodes_.back().dev_num;

      auto dst_dev_id = resource_->dev_id(end_index);
      auto dst_place = DevPlace(dst_dev_id);

      memory_copy(dst_place,
                  reinterpret_cast<char*>(src_val + h_left[end_index]),
                  src_place, cur_task.path->nodes_[cur_step].val_storage,
                  cur_task.path->nodes_[cur_step].val_bytes_len,
                  cur_task.path->nodes_[cur_step].out_stream);
    }
  }
}

template <typename KeyType, typename ValType, typename GradType>
HeterComm<KeyType, ValType, GradType>::~HeterComm() {
  for (auto& table : tables_) {
    delete table;
    table = nullptr;
  }
}

template <typename KeyType, typename ValType, typename GradType>
void HeterComm<KeyType, ValType, GradType>::show_one_table(int num) {
  tables_[num]->show();
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
void HeterComm<KeyType, ValType, GradType>::set_sparse_sgd(
    const OptimizerConfig& optimizer_config) {
  for (int i = 0; i < resource_->total_device(); ++i) {
    AnyDeviceGuard guard(resource_->dev_id(i));
    tables_[i]->set_sparse_sgd(optimizer_config);
  }
}

template <typename KeyType, typename ValType, typename GradType>
void HeterComm<KeyType, ValType, GradType>::set_embedx_sgd(
    const OptimizerConfig& optimizer_config) {
  for (int i = 0; i < resource_->total_device(); ++i) {
    AnyDeviceGuard guard(resource_->dev_id(i));
    tables_[i]->set_embedx_sgd(optimizer_config);
  }
}

template <typename KeyType, typename ValType, typename GradType>
void HeterComm<KeyType, ValType, GradType>::build_ps(
    int dev_num, KeyType* h_keys, ValType* h_vals, size_t len,
    size_t chunk_size, int stream_num) {
  if (len <= 0) {
    return;
  }
  int dev_id = resource_->dev_id(dev_num);

  std::vector<memory::allocation::AllocationPtr> d_key_bufs;
  std::vector<memory::allocation::AllocationPtr> d_val_bufs;

  DevPlace place = DevPlace(dev_id);
  AnyDeviceGuard guard(dev_id);
  ppStream streams[stream_num];  // NOLINT
  for (int i = 0; i < stream_num; ++i) {
    create_stream(&(streams[i]));
    auto d_k_buf = memory::Alloc(place, chunk_size * sizeof(KeyType));
    auto d_v_buf = memory::Alloc(place, chunk_size * sizeof(ValType));
    d_key_bufs.push_back(std::move(d_k_buf));
    d_val_bufs.push_back(std::move(d_v_buf));
  }

  int cur_len = 0;
  int cur_stream = 0;

  while (static_cast<size_t>(cur_len) < len) {
    cur_stream = cur_stream % stream_num;
    auto cur_use_stream = streams[cur_stream];
#if defined(PADDLE_WITH_XPU_KP)
    cur_use_stream = 0;
#endif

    int tmp_len = cur_len + chunk_size > len ? len - cur_len : chunk_size;

    auto dst_place = place;
    auto src_place = platform::CPUPlace();

    memory_copy(
        dst_place, reinterpret_cast<char*>(d_key_bufs[cur_stream]->ptr()),
        src_place, h_keys + cur_len, sizeof(KeyType) * tmp_len, cur_use_stream);
    memory_copy(
        dst_place, reinterpret_cast<char*>(d_val_bufs[cur_stream]->ptr()),
        src_place, h_vals + cur_len, sizeof(ValType) * tmp_len, cur_use_stream);

  #if defined(PADDLE_WITH_CUDA)
    tables_[dev_num]->insert(
        reinterpret_cast<KeyType*>(d_key_bufs[cur_stream]->ptr()),
        reinterpret_cast<ValType*>(d_val_bufs[cur_stream]->ptr()), tmp_len,
        cur_use_stream);
  #elif defined(PADDLE_WITH_XPU_KP)
    tables_[dev_num]->insert(
        place,
        reinterpret_cast<KeyType*>(d_key_bufs[cur_stream]->ptr()),
        reinterpret_cast<ValType*>(d_val_bufs[cur_stream]->ptr()), tmp_len,
        cur_use_stream);
  #endif
    cur_stream += 1;
    cur_len += tmp_len;
  }
  for (int i = 0; i < stream_num; ++i) {
    sync_stream(streams[i]);
    destroy_stream(streams[i]);
  }
}

template <typename KeyType, typename ValType, typename GradType>
void HeterComm<KeyType, ValType, GradType>::merge_grad(
    int dev_num, KeyType* d_keys, GradType* d_grads, size_t len,
    int& uniq_len) {  // NOLINT

  int dev_id = resource_->dev_id(dev_num);
  DevPlace place = DevPlace(dev_id);
  AnyDeviceGuard guard(dev_id);
  auto stream = resource_->local_stream(dev_num, 0);

  size_t temp_storage_bytes;

  auto d_merge_keys = memory::Alloc(place, len * sizeof(KeyType));
  KeyType* d_merge_keys_ptr = reinterpret_cast<KeyType*>(d_merge_keys->ptr());
  auto d_merge_grads = memory::Alloc(place, len * sizeof(GradType));
  GradType* d_merge_grads_ptr =
      reinterpret_cast<GradType*>(d_merge_grads->ptr());

#if defined(PADDLE_WITH_CUDA)
  heter_comm_kernel_->sort_pairs(NULL, temp_storage_bytes, d_keys,
                                 d_merge_keys_ptr, d_grads, d_merge_grads_ptr,
                                 len, 0, 8 * sizeof(KeyType), stream, false);
#elif defined(PADDLE_WITH_XPU_KP)
  heter_comm_kernel_->sort_pairs(place, NULL, temp_storage_bytes, d_keys,
                                 d_merge_keys_ptr, d_grads, d_merge_grads_ptr,
                                 len, 0, 8 * sizeof(KeyType), stream, false);
#endif

  auto d_temp_storage = memory::Alloc(place, temp_storage_bytes);

#if defined(PADDLE_WITH_CUDA)
  heter_comm_kernel_->sort_pairs(
      d_temp_storage->ptr(), temp_storage_bytes, d_keys, d_merge_keys_ptr,
      d_grads, d_merge_grads_ptr, len, 0, 8 * sizeof(KeyType), stream, false);
#elif defined(PADDLE_WITH_XPU_KP)
  heter_comm_kernel_->sort_pairs(
      place,
      d_temp_storage->ptr(), temp_storage_bytes, d_keys, d_merge_keys_ptr,
      d_grads, d_merge_grads_ptr, len, 0, 8 * sizeof(KeyType), stream, false);
#endif

  temp_storage_bytes = 0;

  auto d_num_runs_out_mem = memory::Alloc(place, sizeof(int));
  int* d_num_runs_out = reinterpret_cast<int*>(d_num_runs_out_mem->ptr());

#if defined(PADDLE_WITH_CUDA)
  heter_comm_kernel_->reduce_by_key(NULL, temp_storage_bytes, d_merge_keys_ptr,
                                    d_keys, d_merge_grads_ptr, d_grads,
                                    d_num_runs_out, len, stream, false);
#elif defined(PADDLE_WITH_XPU_KP)
  heter_comm_kernel_->reduce_by_key(place, NULL, temp_storage_bytes,
                                    d_merge_keys_ptr, d_keys, d_merge_grads_ptr,
                                    d_grads, d_num_runs_out, len, stream,
                                    false);
#endif

  if (d_temp_storage->size() < temp_storage_bytes) {
    d_temp_storage = NULL;
    d_temp_storage = memory::Alloc(place, temp_storage_bytes);
  }

#if defined(PADDLE_WITH_CUDA)
  heter_comm_kernel_->reduce_by_key(
      d_temp_storage->ptr(), temp_storage_bytes, d_merge_keys_ptr, d_keys,
      d_merge_grads_ptr, d_grads, d_num_runs_out, len, stream, false);
#elif defined(PADDLE_WITH_XPU_KP)
  heter_comm_kernel_->reduce_by_key(
      place, d_temp_storage->ptr(), temp_storage_bytes, d_merge_keys_ptr,
      d_keys, d_merge_grads_ptr, d_grads, d_num_runs_out, len, stream, false);
#endif

  auto dst_place = platform::CPUPlace();
  auto src_place = place;
  memory_copy(dst_place, &uniq_len, src_place, d_num_runs_out, sizeof(int),
              stream);

  sync_stream(stream);
}

template <typename KeyType, typename ValType, typename GradType>
void HeterComm<KeyType, ValType, GradType>::split_input_to_shard(
    KeyType* d_keys, int* d_idx_ptr, size_t len, int* left, int* right,
    int dev_num) {
  int total_device = resource_->total_device();
  int dev_id = resource_->dev_id(dev_num);
  DevPlace place = DevPlace(dev_id);
  AnyDeviceGuard guard(dev_id);
  auto stream = resource_->local_stream(dev_num, 0);

  auto d_idx_tmp = memory::Alloc(place, len * sizeof(int));
  int* d_idx_tmp_ptr = reinterpret_cast<int*>(d_idx_tmp->ptr());

  auto d_shard_index = memory::Alloc(place, len * sizeof(int));
  int* d_shard_index_ptr = reinterpret_cast<int*>(d_shard_index->ptr());

  auto d_shard_index_tmp = memory::Alloc(place, len * sizeof(int));
  int* d_shard_index_tmp_ptr = reinterpret_cast<int*>(d_shard_index_tmp->ptr());

  // int grid_size = (len - 1) / block_size_ + 1;

  heter_comm_kernel_->fill_idx(d_idx_tmp_ptr, len, stream);
  heter_comm_kernel_->calc_shard_index(d_keys, len, d_shard_index_tmp_ptr,
                                       total_device, stream);

  size_t temp_storage_bytes;
  const int num_bits = 1 + log2i(total_device);

#if defined(PADDLE_WITH_CUDA)
  heter_comm_kernel_->sort_pairs(
      NULL, temp_storage_bytes, d_shard_index_tmp_ptr, d_shard_index_ptr,
      d_idx_tmp_ptr, d_idx_ptr, len, 0, num_bits, stream);
#elif defined(PADDLE_WITH_XPU_KP)
  heter_comm_kernel_->sort_pairs(
      place,
      NULL, temp_storage_bytes, d_shard_index_tmp_ptr, d_shard_index_ptr,
      d_idx_tmp_ptr, d_idx_ptr, len, 0, num_bits, stream);
#endif

  auto d_temp_storage = memory::Alloc(place, temp_storage_bytes);

#if defined(PADDLE_WITH_CUDA)
  heter_comm_kernel_->sort_pairs(
      d_temp_storage->ptr(), temp_storage_bytes, d_shard_index_tmp_ptr,
      d_shard_index_ptr, d_idx_tmp_ptr, d_idx_ptr, len, 0, num_bits, stream);
#elif defined(PADDLE_WITH_XPU_KP)
  heter_comm_kernel_->sort_pairs(
      place,
      d_temp_storage->ptr(), temp_storage_bytes, d_shard_index_tmp_ptr,
      d_shard_index_ptr, d_idx_tmp_ptr, d_idx_ptr, len, 0, num_bits, stream);
#endif

  heter_comm_kernel_->calc_shard_offset(d_shard_index_ptr, left, right, len,
                                        total_device, stream);
  sync_stream(stream);
}

#if defined(PADDLE_WITH_XPU_KP)
static void reset_xpu_memory(DevPlace & place, void* in_ptr, int len, int8_t value, const XPUStream & stream) {
  int8_t * in_int8_ptr = reinterpret_cast<int8_t*>(in_ptr);
  auto dev_ctx = platform::DeviceContextPool::Instance().Get(place);
  auto xpu_context = static_cast<platform::XPUDeviceContext*>(dev_ctx)->x_context();
  xpu_context->xpu_stream = stream;
  int r = xpu::constant<int8_t>(xpu_context, in_int8_ptr, len, value);
  PADDLE_ENFORCE_EQ(r, XPU_SUCCESS,
                    platform::errors::External(
                        "reset_xpu_memory: XPU constant kernel return wrong value[%d %s]", r,
                        XPUAPIErrorMsg[r]));
}

template<class T>
static std::shared_ptr<std::vector<T>> copy_to_cpu(int dev_id, T * data, int data_len) {
  std::shared_ptr<std::vector<T>> buffer = std::make_shared<std::vector<T>>();
  buffer->resize(data_len);
  DevPlace place = DevPlace(dev_id);
  AnyDeviceGuard guard(dev_id);
  auto cpu_place = platform::CPUPlace();
  memory::Copy(cpu_place,
                &((*buffer)[0]),
                place,
                data,
                data_len * sizeof(T));
  xpu_wait(0);
  return buffer;
}
#endif

template <typename KeyType, typename ValType, typename GradType>
void HeterComm<KeyType, ValType, GradType>::pull_sparse(int num,
                                                        KeyType* d_keys,
                                                        ValType* d_vals,
                                                        size_t len) {
  if (len == 0) {
    return;
  }

  //platform::Timer timeline;
  //std::stringstream time_ss;
  //time_ss << "dev:" << num << ",key_len:" << len;
  //double total_time = 0.0;
  //timeline.Start();

  int dev_id = resource_->dev_id(num);

  DevPlace place = DevPlace(dev_id);
  AnyDeviceGuard guard(dev_id);
  auto stream = resource_->local_stream(num, 0);
  // int grid_size = (len - 1) / block_size_ + 1;

#if defined(PADDLE_WITH_XPU_CACHE_BFID)

  typedef int BfidType;
  typedef uint32_t FidType;

  //timeline.Pause();
  //total_time += timeline.ElapsedSec();
  //time_ss << ",init:" << timeline.ElapsedSec();

  // get fidseq
  //timeline.Start();
  FidType* d_fidseq_bucket_ptr = nullptr;
  int fidseq_bucket_len = 0;
  cache_mgr_->get_device_fidseq_bucket(dev_id, &d_fidseq_bucket_ptr, &fidseq_bucket_len);
  int bucket_mean_len = cache_mgr_->get_device_bucket_mean_len();
  //timeline.Pause();
  //total_time += timeline.ElapsedSec();
  //time_ss << ",get_fidseq:" << timeline.ElapsedSec();

  //timeline.Start();
  auto d_fidseq_bucket_vals = memory::Alloc(place, bucket_mean_len * sizeof(ValType));
  ValType* d_fidseq_bucket_vals_ptr = reinterpret_cast<ValType*>(d_fidseq_bucket_vals->ptr());
  reset_xpu_memory(place, d_fidseq_bucket_vals_ptr, bucket_mean_len * sizeof(ValType), 0, stream);
  //timeline.Pause();
  //total_time += timeline.ElapsedSec();
  //time_ss << ",reset_xpu_memory:" << timeline.ElapsedSec();

  // local search
  //timeline.Start();
  tables_[num]->get(place, reinterpret_cast<KeyType*>(d_fidseq_bucket_ptr),
                      reinterpret_cast<ValType*>(d_fidseq_bucket_vals_ptr),
                                                         fidseq_bucket_len,
                                                         stream);
  //timeline.Pause();
  //total_time += timeline.ElapsedSec();
  //time_ss << ",search_fid_in_table:" << timeline.ElapsedSec();

  // allreduce
  //timeline.Start();
  FidType* d_all_fidseq_bucket_ptr = nullptr;
  int all_fidseq_bucket_len = 0;
  cache_mgr_->get_device_all_fidseq_bucket(dev_id, &d_all_fidseq_bucket_ptr, &all_fidseq_bucket_len);
  //timeline.Pause();
  //total_time += timeline.ElapsedSec();
  //time_ss << ",get_all_fidseq_bucket:" << timeline.ElapsedSec();

  //timeline.Start();
  auto d_all_fidseq_bucket_vals = memory::Alloc(place, all_fidseq_bucket_len * sizeof(ValType));
  ValType* d_all_fidseq_bucket_vals_ptr = reinterpret_cast<ValType*>(d_all_fidseq_bucket_vals->ptr());
  reset_xpu_memory(place, d_all_fidseq_bucket_vals_ptr, all_fidseq_bucket_len * sizeof(ValType), 0, stream);
  //timeline.Pause();
  //total_time += timeline.ElapsedSec();
  //time_ss << ",reset_xpu_memory:" << timeline.ElapsedSec();

  if (resource_->total_device() > 1) {
    //timeline.Start();
    //sync_stream(stream);
    int bucket_mean_len = cache_mgr_->get_device_bucket_mean_len();
    auto comm = platform::BKCLCommContext::Instance().Get(0, place);
    //timeline.Pause();
    //total_time += timeline.ElapsedSec();
    //time_ss << ",wait-before-allgather:" << timeline.ElapsedSec();

    //timeline.Start();
    bkcl_all_gather(comm->comm(), d_fidseq_bucket_vals_ptr, bucket_mean_len * sizeof(ValType) / sizeof(float), d_all_fidseq_bucket_vals_ptr, BKCL_FLOAT, stream);
    //timeline.Pause();
    //total_time += timeline.ElapsedSec();
    //time_ss << ",allgather:" << timeline.ElapsedSec();
    VLOG(3) << "heter comm inl pull sparse all reduce finish";
  } else {
    VLOG(3) << "heter comm inl pull unnecessary all reduce";
    d_all_fidseq_bucket_vals_ptr = d_fidseq_bucket_vals_ptr;
  }

  // fill to d_val
  //timeline.Start();
  BfidType* d_bfids_ptr = nullptr;
  int bfid_len = 0;
  cache_mgr_->get_bfidseq(dev_id, &d_bfids_ptr, &bfid_len);
  PADDLE_ENFORCE_EQ(bfid_len, len);
  //timeline.Pause();
  //total_time += timeline.ElapsedSec();
  //time_ss << ",get_bfidseq:" << timeline.ElapsedSec();

  //timeline.Start();
  heter_comm_kernel_->fill_dvals_with_bfid(d_all_fidseq_bucket_vals_ptr, d_vals, d_bfids_ptr, len, stream);
  sync_stream(stream);
  //timeline.Pause();
  //total_time += timeline.ElapsedSec();
  //time_ss << ",fill_dvals_with_bfid:" << timeline.ElapsedSec();

  cache_mgr_->prepare_merge_grad(dev_id);

  //VLOG(0) << "pull_sparse time cost:" << total_time
  //       << " sec, detail:" << time_ss.str();
#else
  int total_device = resource_->total_device();
  int h_left[total_device];   // NOLINT
  int h_right[total_device];  // NOLINT

  auto d_left = memory::Alloc(place, total_device * sizeof(int));
  auto d_right = memory::Alloc(place, total_device * sizeof(int));
  int* d_left_ptr = reinterpret_cast<int*>(d_left->ptr());
  int* d_right_ptr = reinterpret_cast<int*>(d_right->ptr());

#if defined(PADDLE_WITH_CUDA)
  cudaMemsetAsync(d_left_ptr, -1, total_device * sizeof(int), stream);
  cudaMemsetAsync(d_right_ptr, -1, total_device * sizeof(int), stream);

#elif defined(PADDLE_WITH_XPU_KP)
  // get XPUDeviceContext according to xpu place
  auto dev_ctx = platform::DeviceContextPool::Instance().Get(place);
  auto xpu_context =
          static_cast<platform::XPUDeviceContext*>(dev_ctx)->x_context();
  xpu_context->xpu_stream = stream;

  int r = xpu::constant<int>(xpu_context, d_left_ptr, total_device, -1);
  PADDLE_ENFORCE_EQ(r, XPU_SUCCESS,
                    platform::errors::External(
                        "XPU constant kernel return wrong value[%d %s]", r,
                        XPUAPIErrorMsg[r]));
  int r2 = xpu::constant<int>(xpu_context, d_right_ptr, total_device, -1);
  PADDLE_ENFORCE_EQ(r2, XPU_SUCCESS,
                    platform::errors::External(
                        "XPU constant kernel return wrong value[%d %s]", r2,
                        XPUAPIErrorMsg[r2]));
#endif

  auto d_idx = memory::Alloc(place, len * sizeof(int));
  int* d_idx_ptr = reinterpret_cast<int*>(d_idx->ptr());

  auto d_shard_keys = memory::Alloc(place, len * sizeof(KeyType));
  KeyType* d_shard_keys_ptr = reinterpret_cast<KeyType*>(d_shard_keys->ptr());
  auto d_shard_vals = memory::Alloc(place, len * sizeof(ValType));
  ValType* d_shard_vals_ptr = reinterpret_cast<ValType*>(d_shard_vals->ptr());

  split_input_to_shard(d_keys, d_idx_ptr, len, d_left_ptr, d_right_ptr, num);

  heter_comm_kernel_->fill_shard_key(d_shard_keys_ptr, d_keys, d_idx_ptr, len,
                                     stream);

  sync_stream(stream);

  auto dst_place = platform::CPUPlace();
  auto src_place = place;

  memory_copy(dst_place, h_left, src_place, d_left_ptr,
              total_device * sizeof(int), stream);
  memory_copy(dst_place, h_right, src_place, d_right_ptr,
              total_device * sizeof(int), stream);

  for (int i = 0; i < total_device; ++i) {
    int shard_len = h_right[i] - h_left[i] + 1;
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    create_storage(num, i, shard_len * sizeof(KeyType),
                   shard_len * sizeof(ValType));
  }

  walk_to_dest(num, total_device, h_left, h_right, d_shard_keys_ptr, NULL);

  for (int i = 0; i < total_device; ++i) {
    if (h_left[i] == -1) {
      continue;
    }
    auto& node = path_[num][i].nodes_.back();
    sync_stream(node.in_stream);

    AnyDeviceGuard guard(resource_->dev_id(i));

    tables_[i]->rwlock_->RDLock();
#if defined(PADDLE_WITH_CUDA)
    tables_[i]->get(reinterpret_cast<KeyType*>(node.key_storage),
                    reinterpret_cast<ValType*>(node.val_storage),
                    h_right[i] - h_left[i] + 1,
                    resource_->remote_stream(i, num));
#elif defined(PADDLE_WITH_XPU_KP)
    tables_[i]->get(place, reinterpret_cast<KeyType*>(node.key_storage),
                    reinterpret_cast<ValType*>(node.val_storage),
                    h_right[i] - h_left[i] + 1,
                    resource_->remote_stream(i, num));
#endif
  }

  for (int i = 0; i < total_device; ++i) {
    sync_stream(resource_->remote_stream(i, num));
    if (h_left[i] == -1) {
      continue;
    }
    tables_[i]->rwlock_->UNLock();
  }

  walk_to_src(num, total_device, h_left, h_right, d_shard_vals_ptr);

  for (int i = 0; i < total_device; ++i) {
    auto& node = path_[num][i].nodes_.front();
    sync_stream(node.out_stream);
  }

  heter_comm_kernel_->fill_dvals(d_shard_vals_ptr, d_vals, d_idx_ptr, len,
                                 stream);

  sync_stream(stream);

  for (int i = 0; i < total_device; ++i) {
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    destroy_storage(num, i);
  }
#endif
}

#if defined(PADDLE_WITH_CUDA)
template <typename KeyType, typename ValType, typename GradType>
template <typename Sgd>
void HeterComm<KeyType, ValType, GradType>::push_sparse(int dev_num,
                                                        KeyType* d_keys,
                                                        GradType* d_grads,
                                                        size_t len,
                                                        Sgd& sgd) {  // NOLINT
  if (len == 0) {
    return;
  }

  int total_device = resource_->total_device();
  int dev_id = resource_->dev_id(dev_num);

  DevPlace place = DevPlace(dev_id);
  AnyDeviceGuard guard(dev_id);
  auto stream = resource_->local_stream(dev_num, 0);

  int h_left[total_device];   // NOLINT
  int h_right[total_device];  // NOLINT

  auto d_left = memory::Alloc(place, total_device * sizeof(int));
  auto d_right = memory::Alloc(place, total_device * sizeof(int));
  int* d_left_ptr = reinterpret_cast<int*>(d_left->ptr());
  int* d_right_ptr = reinterpret_cast<int*>(d_right->ptr());

  cudaMemsetAsync(d_left_ptr, -1, total_device * sizeof(int), stream);
  cudaMemsetAsync(d_right_ptr, -1, total_device * sizeof(int), stream);

  auto d_idx = memory::Alloc(place, len * sizeof(int));
  int* d_idx_ptr = reinterpret_cast<int*>(d_idx->ptr());

  auto d_shard_keys = memory::Alloc(place, len * sizeof(KeyType));
  KeyType* d_shard_keys_ptr = reinterpret_cast<KeyType*>(d_shard_keys->ptr());
  auto d_shard_grads = memory::Alloc(place, len * sizeof(GradType));
  GradType* d_shard_grads_ptr =
      reinterpret_cast<GradType*>(d_shard_grads->ptr());

  int uniq_len = len;
  merge_grad(dev_num, d_keys, d_grads, len, uniq_len);

  // int grid_size = (uniq_len - 1) / block_size_ + 1;

  split_input_to_shard(d_keys, d_idx_ptr, uniq_len, d_left_ptr, d_right_ptr,
                       dev_num);

  heter_comm_kernel_->fill_shard_grads(d_shard_keys_ptr, d_keys,
                                       d_shard_grads_ptr, d_grads, d_idx_ptr,
                                       uniq_len, stream);

  sync_stream(stream);

  auto dst_place = platform::CPUPlace();
  auto src_place = place;
  memory_copy(dst_place, h_left, src_place, d_left_ptr,
              total_device * sizeof(int), stream);
  memory_copy(dst_place, h_right, src_place, d_right_ptr,
              total_device * sizeof(int), stream);

  for (int i = 0; i < total_device; ++i) {
    int shard_len = h_right[i] - h_left[i] + 1;
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    create_storage(dev_num, i, shard_len * sizeof(KeyType),
                   shard_len * sizeof(GradType));
  }

  walk_to_dest(dev_num, total_device, h_left, h_right, d_shard_keys_ptr,
               d_shard_grads_ptr);

  for (int i = 0; i < total_device; ++i) {
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    auto& node = path_[dev_num][i].nodes_.back();
    sync_stream(node.in_stream);

    AnyDeviceGuard guard(resource_->dev_id(i));
    tables_[i]->rwlock_->WRLock();
    tables_[i]->update(reinterpret_cast<KeyType*>(node.key_storage),
                       reinterpret_cast<GradType*>(node.val_storage),
                       h_right[i] - h_left[i] + 1, sgd,
                       resource_->remote_stream(i, dev_num));
  }

  for (int i = 0; i < total_device; ++i) {
    sync_stream(resource_->remote_stream(i, dev_num));
    if (h_left[i] != -1) {
      tables_[i]->rwlock_->UNLock();
    }
  }

  for (int i = 0; i < total_device; ++i) {
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    destroy_storage(dev_num, i);
  }
}

#elif defined(PADDLE_WITH_XPU_KP)

template <typename KeyType, typename ValType, typename GradType>
void HeterComm<KeyType, ValType, GradType>::push_sparse(int dev_num,
                                                        KeyType* d_keys,
                                                        GradType* d_grads,
                                                        size_t len) {
  if (len == 0) {
    return;
  }

  int dev_id = resource_->dev_id(dev_num);

  DevPlace place = DevPlace(dev_id);
  AnyDeviceGuard guard(dev_id);
  auto stream = resource_->local_stream(dev_num, 0);

  //platform::Timer timeline;
  //std::stringstream time_ss;
  //time_ss << "dev:" << dev_num << ",key_len:" << len;
  //double total_time = 0.0;
  //timeline.Start();

#if defined(PADDLE_WITH_XPU_CACHE_BFID)

  typedef int BfidType;
  typedef uint32_t FidType;

  //timeline.Pause();
  //total_time += timeline.ElapsedSec();
  //time_ss << ",init:" << timeline.ElapsedSec();

  // merge grad
  //timeline.Start();
  FidType* d_all_fidseq_bucket_ptr = nullptr;
  int all_fidseq_bucket_len = 0;
  cache_mgr_->get_device_all_fidseq_bucket(dev_id, &d_all_fidseq_bucket_ptr, &all_fidseq_bucket_len);

  auto d_all_fidseq_bucket_grads = memory::Alloc(place, all_fidseq_bucket_len * sizeof(GradType));
  GradType* d_all_fidseq_bucket_grads_ptr = reinterpret_cast<GradType*>(d_all_fidseq_bucket_grads->ptr());
  reset_xpu_memory(place, d_all_fidseq_bucket_grads_ptr, all_fidseq_bucket_len * sizeof(GradType), 0, stream);

  BfidType* d_bfids_ptr = nullptr;
  int bfid_len = 0;
  cache_mgr_->get_bfidseq(dev_id, &d_bfids_ptr, &bfid_len);
  PADDLE_ENFORCE_EQ(bfid_len, len);

  //timeline.Start();
  // for new merge-grad impl
  int * fidseq_grad_idxs = nullptr;
  int fidseq_grad_idx_len = 0;
  int * fidseq_lods = nullptr;
  int fidseq_lod_len = 0;
  cache_mgr_->get_merge_grad_params(
      dev_id, &fidseq_grad_idxs, &fidseq_grad_idx_len, &fidseq_lods, &fidseq_lod_len);
  PADDLE_ENFORCE_EQ(fidseq_grad_idx_len, len);
  PADDLE_ENFORCE_EQ(fidseq_lod_len, all_fidseq_bucket_len + 1);
  // for new merge-grad impl end
  //timeline.Pause();
  //total_time += timeline.ElapsedSec();
  //time_ss << ",get_merge_grad_params:" << timeline.ElapsedSec();

  heter_comm_kernel_->merge_grad(d_bfids_ptr, d_grads, len, d_all_fidseq_bucket_grads_ptr, stream);
  //timeline.Pause();
  //total_time += timeline.ElapsedSec();
  //time_ss << ",merge_grad:" << timeline.ElapsedSec();

  // allreduce
  //timeline.Start();
  auto d_all_grads = memory::Alloc(place, all_fidseq_bucket_len * sizeof(GradType));
  GradType* d_all_grads_ptr = reinterpret_cast<GradType*>(d_all_grads->ptr());
  reset_xpu_memory(place, d_all_grads_ptr, all_fidseq_bucket_len * sizeof(GradType), 0, stream);
  //timeline.Pause();
  //total_time += timeline.ElapsedSec();
  //time_ss << ",prepare_allreduce:" << timeline.ElapsedSec();

  //timeline.Start();
  if (resource_->total_device() > 1) {
    auto comm = platform::BKCLCommContext::Instance().Get(0, place);
    heter_comm_kernel_->convert_feature_push_value_as_float(d_all_fidseq_bucket_grads_ptr, all_fidseq_bucket_len, true, stream);
    bkcl_all_reduce(comm->comm(), d_all_fidseq_bucket_grads_ptr, d_all_grads_ptr,
        all_fidseq_bucket_len * sizeof(GradType) / sizeof(float),
        BKCL_FLOAT, BKCL_ADD, stream);
    heter_comm_kernel_->convert_feature_push_value_as_float(d_all_grads_ptr, all_fidseq_bucket_len, false, stream);
    VLOG(3) << "heter comm inl push sparse all reduce finish";
  } else {
    VLOG(3) << "heter comm inl push sparse unnecessary all reduce";
    d_all_grads_ptr = d_all_fidseq_bucket_grads_ptr;
  }
  //timeline.Pause();
  //total_time += timeline.ElapsedSec();
  //time_ss << ",allreduce:" << timeline.ElapsedSec();
  //VLOG(0) << "push-allreduce dev:" << dev_num << ", size:" << all_fidseq_bucket_len << "*" << sizeof(GradType);

  // update
  //timeline.Start();
  int fidseq_len = 0;
  FidType* d_fidseq_ptr = nullptr;
  cache_mgr_->get_device_all_fidseq(dev_id, &d_fidseq_ptr, &fidseq_len);
  cache_mgr_->compress_bucket<GradType>(dev_id, d_all_grads_ptr, all_fidseq_bucket_len, stream);
  cache_mgr_->compress_bucket<uint32_t>(dev_id, d_all_fidseq_bucket_ptr, all_fidseq_bucket_len, stream);

  tables_[dev_num]->update(place, d_all_fidseq_bucket_ptr, d_all_grads_ptr, fidseq_len, stream);
  VLOG(3) << "heter comm inl push sparse update finish";
  sync_stream(stream);
  //timeline.Pause();
  //total_time += timeline.ElapsedSec();
  //time_ss << ",update:" << timeline.ElapsedSec();

  //VLOG(0) << "push_sparse time cost:" << total_time
  //        << " sec, detail:" << time_ss.str();

#else
  int total_device = resource_->total_device();

  int h_left[total_device];   // NOLINT
  int h_right[total_device];  // NOLINT

  auto d_left = memory::Alloc(place, total_device * sizeof(int));
  auto d_right = memory::Alloc(place, total_device * sizeof(int));
  int* d_left_ptr = reinterpret_cast<int*>(d_left->ptr());
  int* d_right_ptr = reinterpret_cast<int*>(d_right->ptr());

  // get XPUDeviceContext according to xpu place
  auto dev_ctx = platform::DeviceContextPool::Instance().Get(place);
  auto xpu_context =
          static_cast<platform::XPUDeviceContext*>(dev_ctx)->x_context();

  int r = xpu::constant<int>(xpu_context, d_left_ptr, total_device, -1);
  PADDLE_ENFORCE_EQ(r, XPU_SUCCESS,
                    platform::errors::External(
                        "XPU constant kernel return wrong value[%d %s]", r,
                        XPUAPIErrorMsg[r]));
  int r2 = xpu::constant<int>(xpu_context, d_right_ptr, total_device, -1);
  PADDLE_ENFORCE_EQ(r2, XPU_SUCCESS,
                    platform::errors::External(
                        "XPU constant kernel return wrong value[%d %s]", r2,
                        XPUAPIErrorMsg[r2]));

  auto d_idx = memory::Alloc(place, len * sizeof(int));
  int* d_idx_ptr = reinterpret_cast<int*>(d_idx->ptr());

  auto d_shard_keys = memory::Alloc(place, len * sizeof(KeyType));
  KeyType* d_shard_keys_ptr = reinterpret_cast<KeyType*>(d_shard_keys->ptr());
  auto d_shard_grads = memory::Alloc(place, len * sizeof(GradType));
  GradType* d_shard_grads_ptr =
      reinterpret_cast<GradType*>(d_shard_grads->ptr());

  int uniq_len = len;
  merge_grad(dev_num, d_keys, d_grads, len, uniq_len);

  // int grid_size = (uniq_len - 1) / block_size_ + 1;

  split_input_to_shard(d_keys, d_idx_ptr, uniq_len, d_left_ptr, d_right_ptr,
                       dev_num);

  heter_comm_kernel_->fill_shard_grads(d_shard_keys_ptr, d_keys,
                                       d_shard_grads_ptr, d_grads, d_idx_ptr,
                                       (long long)uniq_len, stream);

  sync_stream(stream);

  auto dst_place = platform::CPUPlace();
  auto src_place = place;
  memory_copy(dst_place, h_left, src_place, d_left_ptr,
              total_device * sizeof(int), stream);
  memory_copy(dst_place, h_right, src_place, d_right_ptr,
              total_device * sizeof(int), stream);

  for (int i = 0; i < total_device; ++i) {
    int shard_len = h_right[i] - h_left[i] + 1;
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    create_storage(dev_num, i, shard_len * sizeof(KeyType),
                   shard_len * sizeof(GradType));
  }

  walk_to_dest(dev_num, total_device, h_left, h_right, d_shard_keys_ptr,
               d_shard_grads_ptr);

  for (int i = 0; i < total_device; ++i) {
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    auto& node = path_[dev_num][i].nodes_.back();
    sync_stream(node.in_stream);

    AnyDeviceGuard guard(resource_->dev_id(i));
    tables_[i]->rwlock_->WRLock();
  #if defined(PADDLE_WITH_CUDA)
    tables_[i]->update(reinterpret_cast<KeyType*>(node.key_storage),
                       reinterpret_cast<GradType*>(node.val_storage),
                       h_right[i] - h_left[i] + 1,
                       resource_->remote_stream(i, dev_num));
  #elif defined(PADDLE_WITH_XPU_KP)
    tables_[i]->update(place, reinterpret_cast<KeyType*>(node.key_storage),
                    reinterpret_cast<GradType*>(node.val_storage),
                    h_right[i] - h_left[i] + 1,
                    resource_->remote_stream(i, dev_num));
  #endif
  }

  for (int i = 0; i < total_device; ++i) {
    sync_stream(resource_->remote_stream(i, dev_num));
    if (h_left[i] != -1) {
      tables_[i]->rwlock_->UNLock();
    }
  }

  for (int i = 0; i < total_device; ++i) {
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    destroy_storage(dev_num, i);
  }
#endif
}

#endif

#if defined(PADDLE_WITH_CUDA)
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
  int total_gpu = resource_->total_device();
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
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllGather(
      (const void*)(d_node_len + gpu_num), (void*)d_node_len, 1,  // NOLINT
      ncclInt,                                                    // NOLINT
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

    // int grid_size = (h_node_len[i] - 1) / block_size_ + 1;
    heter_comm_kernel_->fill_shard_grads(
        storage.local_keys + merge_num, storage.all_keys + index,
        storage.local_grads + merge_num, storage.all_grads + index,
        d_idx_ptr + h_left[gpu_num], h_right[gpu_num] - h_left[gpu_num] + 1,
        stream);
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
#endif

template <typename KeyType, typename ValType, typename GradType>
void HeterComm<KeyType, ValType, GradType>::end_pass() {
  int total_device = resource_->total_device();
  std::vector<std::thread> threads;

  auto dump_to_cpu_func = [this](int index) {
    auto stream = resource_->local_stream(index, 0);
    int dev_id = resource_->dev_id(index);
    AnyDeviceGuard guard(dev_id);
    tables_[index]->dump_to_cpu(dev_id, stream);
  };

  for (int i = 0; i < total_device; ++i) {
    threads.push_back(std::thread(dump_to_cpu_func, i));
  }
  for (auto& t : threads) {
    t.join();
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

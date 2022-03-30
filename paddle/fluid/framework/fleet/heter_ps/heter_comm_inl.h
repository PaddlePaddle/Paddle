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
#ifdef PADDLE_WITH_XPU_KP
#include "paddle/fluid/platform/device/xpu/xpu_info.h"
#endif

namespace paddle {
namespace framework {

template <typename KeyType, typename ValType, typename GradType>
HeterComm<KeyType, ValType, GradType>::HeterComm(
    size_t capacity, std::shared_ptr<HeterPsResource> resource) {
  resource_ = resource;
  storage_.resize(resource_->total_gpu());
  for (int i = 0; i < resource_->total_gpu(); ++i) {
#ifdef PADDLE_WITH_CUDA
    platform::CUDADeviceGuard guard(resource_->dev_id(i));
    allocators_.push_back(std::make_shared<cub::CachingDeviceAllocator>(
        8, 1, (unsigned int)-1, (size_t)-1, false, false));  // NOLINT
#endif
#ifdef PADDLE_WITH_XPU_KP
    platform::XPUDeviceGuard guard(resource_->dev_id(i));
    // allocators_.push_back();
#endif
    // table interface not change
    // auto table = new Table(capacity / load_factor_);
    // tables_.push_back(table);
    if (multi_node_) {
      storage_[i].init(feanum_, resource_->dev_id(i));
    }
  }
  init_path();
}

template <typename KeyType, typename ValType, typename GradType>
void HeterComm<KeyType, ValType, GradType>::init_path() {
  int total_device = resource_->total_device();
  path_.resize(total_device);
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

#ifdef PADDLE_WITH_CUDA
template <typename KeyType, typename ValType, typename GradType>
template <typename DstPlace, typename SrcPlace, typename StreamType>
void HeterComm<KeyType, ValType, GradType>::memory_copy(DstPlace dst_place, void* dst, SrcPlace src_place, const void* src, size_t count, StreamType stream = 0) {
  // stream=0 means launch copy operation in default stream
  if (paddle::platform::is_xpu_place(dst_place)) {
    memory::Copy(dst_place, dst, src_place, src,
               count);
  } else if (paddle::platform::is_gpu_place(dst_place)) {
    cudaMemcpyAsync(dst,
                    src,
                    count, cudaMemcpyDefault, stream);
    if (stream == 0) {
      cudaStreamSynchronize(0);
    }
  } else if (paddle::platform::is_cpu_place(dst_place)) { // ds_place == cpu place
    if (paddle::platform::is_xpu_place(src_place)) {
      memory::Copy(dst_place, dst, src_place, src, count);
    } else if (paddle::platform::is_gpu_place(src_place)) {
      cudaMemcpyAsync(dst,
                      src,
                      count, cudaMemcpyDefault, stream);
      if (stream == 0) {
        cudaStreamSynchronize(0);
      }
    }
  }
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
#elif defined(PADDLE_WITH_XPU_KP)
  auto& nodes = path_[start_index][end_index].nodes_;
  for (size_t i = 0; i < nodes.size(); ++i) {
    platform::XPUDeviceGuard guard(resource_->dev_id(nodes[i].gpu_num));
    auto node_keys_mem = memory::Alloc(place_, keylen);
    nodes[i].key_storage = reinterpret_cast<char*>(node_keys_mem->ptr());
    auto node_vals_mem = memory::Alloc(place_, vallen);
    nodes[i].val_storage = reinterpret_cast<char*>(node_vals_mem->ptr());
    nodes[i].key_bytes_len = keylen;
    nodes[i].val_bytes_len = vallen;
  }
#endif
}
#endif

#ifdef PADDLE_WITH_CUDA
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
#endif

#ifdef PADDLE_WITH_CUDA
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
  
    auto src_dev_id = resource_->dev_id(start_index);
    auto dst_dev_id = resource_->dev_id(i);
    auto src_place = DevPlace(src_dev_id);
    auto dst_place = DevPlace(dst_dev_id);
  
    memory_copy(dst_place, node.key_storage, src_place, reinterpret_cast<char*>(src_key + h_left[i]), node.key_bytes_len, node.in_stream);
    if (need_copy_val) {
      memory_copy(dst_place, node.key_storage, src_place, reinterpret_cast<char*>(src_key + h_left[i]), node.key_bytes_len, node.in_stream);
    }

  }
  while (!que.empty()) {
    CopyTask& cur_task = que.front();
    que.pop();
    if (cur_task.path->nodes_[cur_task.step].sync) {
      sync_stream(cur_task.path->nodes_[cur_task.step].in_stream);

    }
    if (cur_task.step != cur_task.path->nodes_.size() - 1) {
      int cur_step = cur_task.step;
      CopyTask c(cur_task.path, cur_step + 1);
      que.push(c);

      auto src_dev_id = resource_->dev_id(cur_task.path->nodes_[cur_step].gpu_num);
      auto dst_dev_id = resource_->dev_id(cur_task.path->nodes_[cur_step + 1].gpu_num);
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
                    cur_task.path->nodes_[cur_step + 1].in_stream)
      }
    }
  }
}
#endif

#ifdef PADDLE_WITH_CUDA
template <typename KeyType, typename ValType, typename GradType>
void HeterComm<KeyType, ValType, GradType>::walk_to_src(
    int start_index, int gpu_num, int* h_left, int* h_right, ValType* src_val) {
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
                src_place, node.val_storage,  node.val_bytes_len, node.out_stream);
    else {
      CopyTask t(&path_[start_index][i], cur_step - 1);
      que.push(t);

      auto dst_dev_id = resource_->dev_id(path_[start_index][i].nodes_[cur_step - 1].gpu_num);
      auto dst_place = DevPlace(dst_dev_id);

      memory_copy(dst_place, path_[start_index][i].nodes_[cur_step - 1].val_storage,
                  src_place, node.val_storage, path_[start_index][i].nodes_[cur_step - 1].val_bytes_len,
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
     
    auto src_dev_id = resource_->dev_id(cur_task.path->nodes_[cur_step].gpu_num);
    auto src_place = DevPlace(src_dev_id);

    if (cur_step > 0) {

      CopyTask c(cur_task.path, cur_step - 1);
      que.push(c);
     
      auto dst_dev_id = resource_->dev_id(cur_task.path->nodes_[cur_step - 1].gpu_num);
      auto dst_place = DevPlace(dst_dev_id);
     
      memory_copy(dst_place, cur_task.path->nodes_[cur_step - 1].val_storage,
                  src_place, cur_task.path->nodes_[cur_step].val_storage,
                  cur_task.path->nodes_[cur_step - 1].val_bytes_len,
                  cur_task.path->nodes_[cur_step - 1].out_stream) 

    } else if (cur_step == 0) {

      int end_index = cur_task.path->nodes_.back().gpu_num;

      auto dst_dev_id = resource_->dev_id(end_index);
      auto dst_place = DevPlace(dst_dev_id);

      memory_copy(dst_place, reinterpret_cast<char*>(src_val + h_left[end_index]),
                  src_place, cur_task.path->nodes_[cur_step].val_storage,
                  cur_task.path->nodes_[cur_step].val_bytes_len,
                  cur_task.path->nodes_[cur_step].out_stream) 
    }
  }

}
#endif

template <typename KeyType, typename ValType, typename GradType>
HeterComm<KeyType, ValType, GradType>::~HeterComm() {
  // for (auto& table : tables_) {
  //   delete table;
  //   table = nullptr;
  // }
}

template <typename KeyType, typename ValType, typename GradType>
void HeterComm<KeyType, ValType, GradType>::show_one_table(int gpu_num) {
  // tables_[gpu_num]->show();
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
#ifdef PADDLE_WITH_CUDA
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
#endif
#ifdef PADDLE_WITH_XPU_KP
  platform::XPUPlace place = platform::XPUPlace(dev_id);
  platform::XPUDeviceGuard guard(dev_id);
  KeyType* d_k_buf = nullptr;
  ValType* d_v_buf = nullptr;
  PADDLE_ENFORCE_EQ(xpu_malloc(reinterpret_cast<void**>(&d_k_buf),
                               chunk_size * sizeof(KeyType)),
                    XPU_SUCCESS, platform::errors::ResourceExhausted(
                                     "XPU has no enough memory"));
  PADDLE_ENFORCE_EQ(xpu_malloc(reinterpret_cast<void**>(&d_v_buf),
                               chunk_size * sizeof(ValType)),
                    XPU_SUCCESS, platform::errors::ResourceExhausted(
                                     "XPU has no enough memory"));

  int cur_len = 0;

  while (cur_len < len) {
    int tmp_len = cur_len + chunk_size > len ? len - cur_len : chunk_size;
    PADDLE_ENFORCE_XPU_SUCCESS(xpu_memcpy(d_k_buf, h_keys + cur_len,
                                          sizeof(KeyType) * tmp_len,
                                          XPU_HOST_TO_DEVICE));
    PADDLE_ENFORCE_XPU_SUCCESS(xpu_memcpy(d_v_buf, h_vals + cur_len,
                                          sizeof(ValType) * tmp_len,
                                          XPU_HOST_TO_DEVICE));
    // TODO(zhiping): xpu kv insert
    // tables_[num]->insert(
    //     reinterpret_cast<KeyType*>(d_k_buf),
    //     reinterpret_cast<ValType*>(d_v_buf)), tmp_len,
    //     NULL);
    cur_len += tmp_len;
  }

#endif
}

template <typename KeyType, typename ValType, typename GradType>
void HeterComm<KeyType, ValType, GradType>::merge_grad(
    int gpu_num, KeyType* d_keys, GradType* d_grads, size_t len,
    int& uniq_len) {  // NOLINT
  int dev_id = resource_->dev_id(gpu_num);
#ifdef PADDLE_WITH_CUDA
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
#endif
#ifdef PADDLE_WITH_XPU_KP
  platform::XPUPlace place = platform::XPUPlace(dev_id);
  platform::XPUDeviceGuard guard(dev_id);
  auto stream = resource_->local_stream(gpu_num, 0);

  KeyType* d_merge_keys_ptr = nullptr;
  xpu_malloc(reinterpret_cast<void**>(&d_merge_keys_ptr),
             len * sizeof(KeyType));

  GradType* d_merge_grads_ptr = nullptr;
  xpu_malloc(reinterpret_cast<void**>(&d_merge_grads_ptr),
             len * sizeof(GradType));

  // TODO(zhipeng): xpu::SortPairs 接口待实现
  // size_t temp_storage_bytes;
  // PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceRadixSort::SortPairs(
  //     NULL, temp_storage_bytes, d_keys, d_merge_keys_ptr, d_grads,
  //     d_merge_grads_ptr, len, 0, 8 * sizeof(KeyType), stream, false));

  // void* d_buff = NULL;
  // auto d_temp_storage = memory::Alloc(place, temp_storage_bytes);

  // PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceRadixSort::SortPairs(
  //     d_temp_storage->ptr(), temp_storage_bytes, d_keys, d_merge_keys_ptr,
  //     d_grads, d_merge_grads_ptr, len, 0, 8 * sizeof(KeyType), stream,
  //     false));
  // temp_storage_bytes = 0;

  int* d_num_runs_out = nullptr;
  xpu_malloc(reinterpret_cast<void**>(&d_num_runs_out), sizeof(int));

// TODO(zhipeng): xpu::ReduceByKey 接口待实现
// PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceReduce::ReduceByKey(
//     NULL, temp_storage_bytes, d_merge_keys_ptr, d_keys, d_merge_grads_ptr,
//     d_grads, d_num_runs_out, merger_, len, stream, false));

// if (d_temp_storage->size() < temp_storage_bytes) {
//   d_temp_storage = NULL;
//   d_temp_storage = memory::Alloc(place, temp_storage_bytes);
// }

// PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceReduce::ReduceByKey(
//     d_temp_storage->ptr(), temp_storage_bytes, d_merge_keys_ptr, d_keys,
//     d_merge_grads_ptr, d_grads, d_num_runs_out, merger_, len, stream,
//     false));

// cudaMemcpyAsync(&uniq_len, d_num_runs_out, sizeof(int),
//                 cudaMemcpyDeviceToHost, stream);

// PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
#endif
}

template <typename KeyType, typename ValType, typename GradType>
void HeterComm<KeyType, ValType, GradType>::split_input_to_shard(
    KeyType* d_keys, int* d_idx_ptr, size_t len, int* left, int* right,
    int gpu_num) {
  int total_gpu = resource_->total_gpu();
  int dev_id = resource_->dev_id(gpu_num);
#ifdef PADDLE_WITH_CUDA
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

  fill_idx(d_idx_tmp_ptr, len, stream);
  calc_shard_index(d_keys, len, d_shard_index_tmp_ptr, total_gpu, stream);

  size_t temp_storage_bytes;
  const int num_bits = 1 + log2i(total_gpu);
  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceRadixSort::SortPairs(
      NULL, temp_storage_bytes, d_shard_index_tmp_ptr, d_shard_index_ptr,
      d_idx_tmp_ptr, d_idx_ptr, len, 0, num_bits, stream));

  auto d_temp_storage = memory::Alloc(place, temp_storage_bytes);
  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceRadixSort::SortPairs(
      d_temp_storage->ptr(), temp_storage_bytes, d_shard_index_tmp_ptr,
      d_shard_index_ptr, d_idx_tmp_ptr, d_idx_ptr, len, 0, num_bits, stream));
  calc_shard_offset(d_shard_index_ptr, left, right, len, total_gpu, stream);
  cudaStreamSynchronize(stream);
}
#elif defined(PADDLE_WITH_XPU_KP)

template <typename KeyType, typename ValType, typename GradType>
void HeterComm<KeyType, ValType, GradType>::split_input_to_shard(
    KeyType* d_keys, int* d_idx_ptr, size_t len, int* left, int* right,
    int xpu_num) {

  int total_xpu = resource_->total_device();
  int dev_id = resource_->dev_id(xpu_num);

  platform::XPUPlace place = platform::XPUPlace(dev_id);
  platform::XPUDeviceGuard guard(dev_id);
  auto stream = resource_->local_stream(gpu_num, 0);

  int* d_idx_tmp_ptr = nullptr;
  xpu_malloc(reinterpret_cast<void**>(&d_idx_tmp_ptr), len * sizeof(int));

  int* d_shard_index_ptr = nullptr;
  xpu_malloc(reinterpret_cast<void**>(&d_shard_index_ptr), len * sizeof(int));

  int* d_shard_index_tmp_ptr = nullptr;
  xpu_malloc(reinterpret_cast<void**>(&d_shard_index_tmp_ptr),
             len * sizeof(int));


  heter_comm_kernel_->fill_idx(d_idx_tmp_ptr, len, stream);
  heter_comm_kernel_->calc_shard_index(d_keys, len, d_shard_index_tmp_ptr, total_xpu, stream);

  size_t temp_storage_bytes;
  const int num_bits = 1 + log2i(total_xpu);

  // TODO(zhipeng): xpu::SortPairs 接口待实现
  // size_t temp_storage_bytes;
  // const int num_bits = 1 + log2i(total_gpu);
  // PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceRadixSort::SortPairs(
  //     NULL, temp_storage_bytes, d_shard_index_tmp_ptr, d_shard_index_ptr,
  //     d_idx_tmp_ptr, d_idx_ptr, len, 0, num_bits, stream));


  auto d_temp_storage = memory::Alloc(place, temp_storage_bytes);
  PADDLE_ENFORCE_XPU_SUCCESS(SortPairs(
      d_temp_storage->ptr(), temp_storage_bytes, d_shard_index_tmp_ptr,
      d_shard_index_ptr, d_idx_tmp_ptr, d_idx_ptr, len, 0, num_bits, stream));
  heter_comm_kernel_->calc_shard_offset(d_shard_index_ptr, left, right, len, total_xpu, stream);
  xpu_wait(stream);
#endif
}

template <typename KeyType, typename ValType, typename GradType>
void HeterComm<KeyType, ValType, GradType>::pull_sparse(int num,
                                                        KeyType* d_keys,
                                                        ValType* d_vals,
                                                        size_t len) {
  if (len == 0) {
    return;
  }


  int total_gpu = resource_->total_device();
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


  auto d_idx = memory::Alloc(place, len * sizeof(int));
  int* d_idx_ptr = reinterpret_cast<int*>(d_idx->ptr());

  auto d_shard_keys = memory::Alloc(place, len * sizeof(KeyType));
  KeyType* d_shard_keys_ptr = reinterpret_cast<KeyType*>(d_shard_keys->ptr());
  auto d_shard_vals = memory::Alloc(place, len * sizeof(ValType));
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
                   shard_len * sizeof(ValType));
  }

  walk_to_dest(num, total_gpu, h_left, h_right, d_shard_keys_ptr, NULL);

  for (int i = 0; i < total_gpu; ++i) {
    if (h_left[i] == -1) {
      continue;
    }
    auto& node = path_[num][i].nodes_.back();
    cudaStreamSynchronize(node.in_stream);
    platform::CUDADeviceGuard guard(resource_->dev_id(i));
    tables_[i]->rwlock_->RDLock();
    tables_[i]->get(reinterpret_cast<KeyType*>(node.key_storage),
                    reinterpret_cast<ValType*>(node.val_storage),
                    h_right[i] - h_left[i] + 1,
                    resource_->remote_stream(i, num));
  }
  for (int i = 0; i < total_gpu; ++i) {
    cudaStreamSynchronize(resource_->remote_stream(i, num));
    if (h_left[i] == -1) {
      continue;
    }
    tables_[i]->rwlock_->UNLock();
  }

  walk_to_src(num, total_gpu, h_left, h_right, d_shard_vals_ptr);

  for (int i = 0; i < total_gpu; ++i) {
    auto& node = path_[num][i].nodes_.front();
    cudaStreamSynchronize(node.out_stream);
  }

  fill_dvals(d_shard_vals_ptr, d_vals, d_idx_ptr, len, stream);
  cudaStreamSynchronize(stream);
  for (int i = 0; i < total_gpu; ++i) {
    destroy_storage(num, i);
  }

}
#elif defined(PADDLE_WITH_XPU_KP)

template <typename KeyType, typename ValType, typename GradType>
void HeterComm<KeyType, ValType, GradType>::pull_sparse(int num,
                                                        KeyType* d_keys,
                                                        ValType* d_vals,
                                                        size_t len) {
  if (len == 0) {
    return;
  }

  int total_xpu = resource_->total_device();
  int dev_id = resource_->dev_id(num);
  platform::XPUPlace place = platform::XPUPlace(dev_id);
  platform::XPUDeviceGuard guard(dev_id);
  auto stream = resource_->local_stream(num, 0);

  // int grid_size = (len - 1) / block_size_ + 1;


  int h_left[total_xpu];   // NOLINT
  int h_right[total_xpu];  // NOLINT

  auto d_left = memory::Alloc(place, total_xpu * sizeof(int));
  auto d_right = memory::Alloc(place, total_xpu * sizeof(int));
  int* d_left_ptr = reinterpret_cast<int*>(d_left->ptr());
  int* d_right_ptr = reinterpret_cast<int*>(d_right->ptr());

  // get XPUDeviceContext according to xpu place
  paddle::platform::XPUDeviceContext xpu_dev_ctx(place);
  auto xpu_context = xpu_dev_ctx.x_context();

  int r = xpu::constant<int>(xpu_context, d_left_ptr, total_xpu, -1);
  PADDLE_ENFORCE_EQ(r, XPU_SUCCESS,
                    platform::errors::External(
                        "XPU constant kernel return wrong value[%d %s]", r,
                        XPUAPIErrorMsg[r]));
  int r2 = xpu::constant<int>(xpu_context, d_right_ptr, total_xpu, -1);
  PADDLE_ENFORCE_EQ(r2, XPU_SUCCESS,
                    platform::errors::External(
                        "XPU constant kernel return wrong value[%d %s]", r2,
                        XPUAPIErrorMsg[r2]));

  T* d_idx = nullptr;
  xpu_malloc(reinterpret_cast<*void**>(&d_idx), len * sizeof(int));
  int* d_idx_ptr = reinterpret_cast<int*>(d_idx);

  int* d_shard_keys_ptr = nullptr;
  int* d_shard_vals_ptr = nullptr;
  xpu_malloc(reinterpret_cast<*void**>(&d_shard_keys_ptr),
                len * sizeof(KeyType));
  xpu_malloc(reinterpret_cast<*void**>(&d_shard_vals_ptr),
             len * sizeof(ValType));

  split_input_to_shard(d_keys, d_idx_ptr, len, d_left_ptr, d_right_ptr, num);

  heter_comm_kernel_->fill_shard_key(d_shard_keys_ptr, d_keys, d_idx_ptr, len);

  auto dst_place = platform::CPUPlace();
  auto src_place = place;
  memory::Copy(dst_place, h_left, src_place, d_left_ptr, total_xpu * sizeof(int));
  memory::Copy(dst_place, h_right, src_place, d_right_ptr, total_xpu * sizeof(int));

  for (int i = 0; i < total_xpu; ++i) {
    int shard_len = h_right[i] - h_left[i] + 1;
    if (shard_len == 0) {
      continue;
    }
    // create_storage(num, i, shard_len * sizeof(KeyType),
    //                shard_len * sizeof(ValType));
  }

  walk_to_dest(num, total_xpu, h_left, h_right, d_shard_keys_ptr, NULL);

  for (int i = 0; i < total_xpu; ++i) {
    if (h_left[i] == -1) {
      continue;
    }
    auto& node = path_[num][i].nodes_.back();
    platform::XPUDeviceGuard guard(resource_->dev_id(i));
    // TODO(zhipeng): xpu kv table
    // tables_[i]->rwlock_->RDLock();
    // tables_[i]->get(reinterpret_cast<KeyType*>(node.key_storage),
    //                 reinterpret_cast<ValType*>(node.val_storage),
    //                 h_right[i] - h_left[i] + 1,
    //                 resource_->remote_stream(i, num));
  }

  for (int i = 0; i < total_xpu; ++i) {
    xpu_wait(resource_->remote_stream(i, num)); 
    if (h_left[i] == -1) {
      continue;
    }
    // TODO(zhipeng): xpu kv table
    // tables_[i]->rwlock_->UNLock();
  }


  walk_to_src(num, total_xpu, h_left, h_right, d_shard_vals_ptr);

  fill_dvals<<<2, 64, stream>>>(d_shard_vals_ptr, d_vals, d_idx_ptr, len);
  xpu_wait(stream);

  for (int i = 0; i < total_xpu; ++i) {
    destroy_storage(num, i);
  }

}

// TODO(zhangfan): free node.key_storage & node.val_storage
// for (int i = 0; i < total_gpu; ++i) {
//   // destroy_storage(num, i);
// }
#endif
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

#ifdef PADDLE_WITH_CUDA
  int total_gpu = resource_->total_gpu();
  int dev_id = resource_->dev_id(gpu_num);
  platform::CUDAPlace place = platform::CUDAPlace(dev_id);
  platform::CUDADeviceGuard guard(dev_id);
  auto stream = resource_->local_stream(gpu_num, 0);

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
  auto d_shard_grads = memory::Alloc(place, len * sizeof(GradType));
  GradType* d_shard_grads_ptr =
      reinterpret_cast<GradType*>(d_shard_grads->ptr());

  int uniq_len = len;
  merge_grad(gpu_num, d_keys, d_grads, len, uniq_len);

  int grid_size = (uniq_len - 1) / block_size_ + 1;

  split_input_to_shard(d_keys, d_idx_ptr, uniq_len, d_left_ptr, d_right_ptr,
                       gpu_num);

  fill_shard_grads<<<grid_size, block_size_, 0, stream>>>(
      d_shard_keys_ptr, d_keys, d_shard_grads_ptr, d_grads, d_idx_ptr,
      uniq_len);

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
    create_storage(gpu_num, i, shard_len * sizeof(KeyType),
                   shard_len * sizeof(GradType));
  }

  walk_to_dest(gpu_num, total_gpu, h_left, h_right, d_shard_keys_ptr,
               d_shard_grads_ptr);

  for (int i = 0; i < total_gpu; ++i) {
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    auto& node = path_[gpu_num][i].nodes_.back();
    cudaStreamSynchronize(node.in_stream);

    platform::CUDADeviceGuard guard(resource_->dev_id(i));
    tables_[i]->rwlock_->WRLock();
    tables_[i]->update(reinterpret_cast<KeyType*>(node.key_storage),
                       reinterpret_cast<GradType*>(node.val_storage),
                       h_right[i] - h_left[i] + 1, sgd,
                       resource_->remote_stream(i, gpu_num));
  }
  for (int i = 0; i < total_gpu; ++i) {
    cudaStreamSynchronize(resource_->remote_stream(i, gpu_num));
    if (h_left[i] != -1) {
      tables_[i]->rwlock_->UNLock();
    }
  }
  for (int i = 0; i < total_gpu; ++i) {
    destroy_storage(gpu_num, i);
  }
#endif
#ifdef PADDLE_WITH_XPU_KP
  int total_gpu = resource_->total_gpu();
  int dev_id = resource_->dev_id(gpu_num);
  platform::XPUPlace place = platform::XPUPlace(dev_id);
  platform::XPUDeviceGuard guard(dev_id);
  auto stream = resource_->local_stream(gpu_num, 0);

  int h_left[total_gpu];   // NOLINT
  int h_right[total_gpu];  // NOLINT

  T* d_left = nullptr;
  xpu_malloc(reinterpret_cast<*void**>(&d_left), total_gpu * sizeof(int));
  int* d_left_ptr = reinterpret_cast<int*>(d_left);

  T* d_right = nullptr;
  xpu_malloc(reinterpret_cast<*void**>(&d_right), total_gpu * sizeof(int));
  int* d_right_ptr = reinterpret_cast<int*>(d_right);

  // cudaMemsetAsync(d_left_ptr, -1, total_gpu * sizeof(int), stream);
  // cudaMemsetAsync(d_right_ptr, -1, total_gpu * sizeof(int), stream);

  T* d_idx = nullptr;
  xpu_malloc(reinterpret_cast<*void**>(&d_idx), len * sizeof(int));
  int* d_idx_ptr = reinterpret_cast<int*>(d_idx);

  T* d_shard_keys = nullptr;
  xpu_malloc(reinterpret_cast<*void**>(&d_shard_keys),
                len * sizeof(KeyType));
  KeyType* d_shard_keys_ptr = reinterpret_cast<KeyType*>(d_shard_keys);

  T* d_shard_grads = nullptr;
  xpu_malloc(reinterpret_cast<*void**>(&d_shard_grads),
                len * sizeof(GradType));
  GradType* d_shard_grads_ptr = reinterpret_cast<GradType*>(d_shard_grads);

  int uniq_len = len;
  // TODO(zhiping): SortPairs和ReduceByKey接口待实现
  // merge_grad(gpu_num, d_keys, d_grads, len, uniq_len);

  // int grid_size = (uniq_len - 1) / block_size_ + 1;

  split_input_to_shard(d_keys, d_idx_ptr, uniq_len, d_left_ptr, d_right_ptr,
                       gpu_num);

  fill_shard_grads<<<3, 64, stream>>>(d_shard_keys_ptr, d_keys,
                                      d_shard_grads_ptr, d_grads, d_idx_ptr,
                                      uniq_len);

  xpu_wait(stream);

  auto dst_place = platform::CPUPlace();
  auto src_place = place;
  memory::Copy(dst_place, h_left, src_place, d_left_ptr, total_xpu * sizeof(int));
  memory::Copy(dst_place, h_right, src_place, d_right_ptr, total_xpu * sizeof(int));

  for (int i = 0; i < total_xpu; ++i) {
    int shard_len = h_right[i] - h_left[i] + 1;
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    create_storage(xpu_num, i, shard_len * sizeof(KeyType),
                   shard_len * sizeof(GradType));
  }

  walk_to_dest(xpu_num, total_xpu, h_left, h_right, d_shard_keys_ptr,
               d_shard_grads_ptr);

// TODO(zhiping): XPU kv table
// for (int i = 0; i < total_gpu; ++i) {
//   if (h_left[i] == -1 || h_right[i] == -1) {
//     continue;
//   }
//   auto& node = path_[gpu_num][i].nodes_.back();
//   xpu_wait(node.in_stream);

//   platform::XPUDeviceGuard guard(resource_->dev_id(i));
//   tables_[i]->rwlock_->WRLock();
//   tables_[i]->update(reinterpret_cast<KeyType*>(node.key_storage),
//                      reinterpret_cast<GradType*>(node.val_storage),
//                      h_right[i] - h_left[i] + 1, sgd,
//                      resource_->remote_stream(i, gpu_num));
// }
// for (int i = 0; i < total_gpu; ++i) {
//   xpu_wait(resource_->remote_stream(i, gpu_num));
//   if (h_left[i] != -1) {
//     tables_[i]->rwlock_->UNLock();
//   }
// }

// TODO(zhangfan): free node.key_storage & node.val_storage
// for (int i = 0; i < total_gpu; ++i) {
//   destroy_storage(gpu_num, i);
// }
#endif
}

#ifdef PADDLE_WITH_CUDA
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
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllGather(
      (const reinterpret_cast<void*>)(d_node_len + gpu_num),
      (reinterpret_cast<void*>)d_node_len, 1,
      ncclInt,  // NOLINT
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
#endif

template <typename KeyType, typename ValType, typename GradType>
void HeterComm<KeyType, ValType, GradType>::end_pass() {
  int total_gpu = resource_->total_gpu();
  std::vector<std::thread> threads;

#ifdef PADDLE_WITH_CUDA
  auto dump_to_cpu_func = [this](int index) {
    auto stream = resource_->local_stream(index, 0);
    int dev_id = resource_->dev_id(index);
    platform::CUDADeviceGuard guard(dev_id);
    tables_[index]->dump_to_cpu(dev_id, stream);
  };
#endif
#ifdef PADDLE_WITH_XPU_KP
  auto dump_to_cpu_func = [this](int index) {
    auto stream = resource_->local_stream(index, 0);
    int dev_id = resource_->dev_id(index);
    platform::XPUDeviceGuard guard(dev_id);
    // TODO(张帆): xpu kv table
    // tables_[index]->dump_to_cpu(dev_id, stream);
  };
#endif

  for (int i = 0; i < total_gpu; ++i) {
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

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

#include "paddle/fluid/framework/fleet/heter_ps/feature_value.h"
#include "paddle/fluid/framework/fleet/heter_ps/gpu_graph_utils.h"
#include "paddle/fluid/framework/fleet/heter_ps/heter_comm_kernel.h"
#include "paddle/fluid/platform/device_context.h"
#ifdef PADDLE_WITH_XPU_KP
#include "paddle/fluid/platform/device/xpu/xpu_info.h"
#endif

DECLARE_double(gpugraph_hbm_table_load_factor);
DECLARE_bool(gpugraph_enable_gpu_direct_access);
DECLARE_bool(gpugraph_enable_segment_merge_grads);
DECLARE_uint64(gpugraph_merge_grads_segment_size);
DECLARE_int32(gpugraph_dedup_pull_push_mode);

namespace paddle {
namespace framework {

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
HeterComm<KeyType, ValType, GradType, GPUAccessor>::HeterComm(
    size_t capacity, std::shared_ptr<HeterPsResource> resource) {
  VLOG(1) << "Construct new HeterComm";
  resource_ = resource;
  storage_.resize(resource_->total_device());
  multi_mf_dim_ = resource->multi_mf();
  load_factor_ = FLAGS_gpugraph_hbm_table_load_factor;
  VLOG(0) << "load_factor = " << load_factor_;
  for (int i = 0; i < resource_->total_device(); ++i) {
#if defined(PADDLE_WITH_CUDA)
    platform::CUDADeviceGuard guard(resource_->dev_id(i));
    allocators_.push_back(std::make_shared<cub::CachingDeviceAllocator>(
        8, 1, (unsigned int)-1, (size_t)-1, false, false));  // NOLINT
#endif
    if (!multi_mf_dim_) {
      auto table = new Table(capacity / load_factor_);
      tables_.push_back(table);
    } else {
      max_mf_dim_ = resource_->max_mf_dim();
      auto accessor_wrapper_ptr =
          GlobalAccessorFactory::GetInstance().GetAccessorWrapper();
      size_t val_type_size =
          accessor_wrapper_ptr->GetFeatureValueSize(max_mf_dim_);
      size_t grad_type_size =
          accessor_wrapper_ptr->GetPushValueSize(max_mf_dim_);
      size_t pull_type_size =
          accessor_wrapper_ptr->GetPullValueSize(max_mf_dim_);

      VLOG(0) << " HeterComm init, max feature_value_size:" << val_type_size
              << ", feature_value_push_size:" << grad_type_size
              << ", feature_pull_type_size:" << pull_type_size;
      auto ptr_table = new PtrTable(capacity / load_factor_);
      ptr_table->set_feature_value_size(pull_type_size, grad_type_size);
      ptr_tables_.push_back(ptr_table);
    }
    if (multi_node_) {
      storage_[i].init(feanum_, resource_->dev_id(i));
    }
  }
  heter_comm_kernel_ = std::make_unique<HeterCommKernel>(block_size_);
  init_path();
}

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
HeterComm<KeyType, ValType, GradType, GPUAccessor>::HeterComm(
    size_t capacity,
    std::shared_ptr<HeterPsResource> resource,
    GPUAccessor& gpu_accessor) {
  VLOG(1) << "Construct new HeterComm";
  resource_ = resource;
  storage_.resize(resource_->total_device());
  multi_mf_dim_ = resource->multi_mf();
  gpu_accessor_ = gpu_accessor;
  load_factor_ = FLAGS_gpugraph_hbm_table_load_factor;
  VLOG(0) << "load_factor = " << load_factor_;
  for (int i = 0; i < resource_->total_device(); ++i) {
#if defined(PADDLE_WITH_CUDA)
    platform::CUDADeviceGuard guard(resource_->dev_id(i));
    allocators_.push_back(std::make_shared<cub::CachingDeviceAllocator>(
        8, 1, (unsigned int)-1, (size_t)-1, false, false));  // NOLINT
#endif
    if (!multi_mf_dim_) {
      auto table = new Table(capacity / load_factor_);
      tables_.push_back(table);
    } else {
      max_mf_dim_ = resource_->max_mf_dim();
      auto accessor_wrapper_ptr =
          GlobalAccessorFactory::GetInstance().GetAccessorWrapper();
      size_t val_type_size =
          accessor_wrapper_ptr->GetFeatureValueSize(max_mf_dim_);
      size_t grad_type_size =
          accessor_wrapper_ptr->GetPushValueSize(max_mf_dim_);
      size_t pull_type_size =
          accessor_wrapper_ptr->GetPullValueSize(max_mf_dim_);

      VLOG(0) << " HeterComm init, max feature_value_size:" << val_type_size
              << ", feature_value_push_size:" << grad_type_size
              << ", feature_pull_type_size:" << pull_type_size;
      auto ptr_table = new PtrTable(capacity / load_factor_);
      ptr_table->set_feature_value_size(pull_type_size, grad_type_size);
      ptr_tables_.push_back(ptr_table);
    }
    if (multi_node_) {
      storage_[i].init(feanum_, resource_->dev_id(i));
    }
  }
  heter_comm_kernel_ = std::make_unique<HeterCommKernel>(block_size_);
  init_path();
}

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::init_path() {
  int total_device = resource_->total_device();
  path_.resize(total_device);
  if (!topo_aware_) {
    VLOG(0) << "init path without topo aware";
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
    VLOG(0) << "init path with topo aware";
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

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
template <typename DstPlace, typename SrcPlace, typename StreamType>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::memory_copy(
    DstPlace dst_place,
    void* dst,
    SrcPlace src_place,
    const void* src,
    size_t count,
    StreamType stream) {
#if defined(PADDLE_WITH_CUDA)
  CUDA_CHECK(cudaMemcpyAsync(dst, src, count, cudaMemcpyDefault, stream));
  if (stream == 0) {
    CUDA_CHECK(cudaStreamSynchronize(0));
  }
#elif defined(PADDLE_WITH_XPU_KP)
  memory::Copy(dst_place, dst, src_place, src, count);
#endif
}

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::create_storage(
    int start_index, int end_index, size_t keylen, size_t vallen) {
#if defined(PADDLE_WITH_CUDA)
  auto& allocator = allocators_[start_index];
  auto& nodes = path_[start_index][end_index].nodes_;
  for (size_t i = 0; i < nodes.size(); ++i) {
    platform::CUDADeviceGuard guard(resource_->dev_id(nodes[i].dev_num));
    PADDLE_ENFORCE_GPU_SUCCESS(allocator->DeviceAllocate(
        resource_->dev_id(nodes[i].dev_num),
        (void**)&(nodes[i].key_storage),  // NOLINT
        keylen,
        resource_->remote_stream(nodes[i].dev_num, start_index)));
    PADDLE_ENFORCE_GPU_SUCCESS(allocator->DeviceAllocate(
        resource_->dev_id(nodes[i].dev_num),
        (void**)&(nodes[i].val_storage),  // NOLINT
        vallen,
        resource_->remote_stream(nodes[i].dev_num, start_index)));
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

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::destroy_storage(
    int start_index, int end_index) {
#if defined(PADDLE_WITH_CUDA)
  auto& allocator = allocators_[start_index];
  auto& nodes = path_[start_index][end_index].nodes_;
  for (size_t i = 0; i < nodes.size(); ++i) {
    platform::CUDADeviceGuard guard(resource_->dev_id(nodes[i].dev_num));

    PADDLE_ENFORCE_GPU_SUCCESS(allocator->DeviceFree(
        resource_->dev_id(nodes[i].dev_num), nodes[i].key_storage));
    PADDLE_ENFORCE_GPU_SUCCESS(allocator->DeviceFree(
        resource_->dev_id(nodes[i].dev_num), nodes[i].val_storage));
  }
#endif
}

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::walk_to_dest(
    int start_index,
    int num,
    int* h_left,
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

    memory_copy(dst_place,
                node.key_storage,
                src_place,
                reinterpret_cast<char*>(src_key + h_left[i]),
                node.key_bytes_len,
                node.in_stream);
    // #if defined(PADDLE_WITH_CUDA)  // adapt for gpu-graph
    //     cudaMemsetAsync(node.val_storage, -1, node.val_bytes_len,
    //     node.in_stream);
    // #endif

    if (need_copy_val) {
      memory_copy(dst_place,
                  node.val_storage,
                  src_place,
                  reinterpret_cast<char*>(src_val + h_left[i]),
                  node.val_bytes_len,
                  node.in_stream);
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

      memory_copy(dst_place,
                  cur_task.path->nodes_[cur_step + 1].key_storage,
                  src_place,
                  cur_task.path->nodes_[cur_step].key_storage,
                  cur_task.path->nodes_[cur_step + 1].key_bytes_len,
                  cur_task.path->nodes_[cur_step + 1].in_stream);
      if (need_copy_val) {
        memory_copy(dst_place,
                    cur_task.path->nodes_[cur_step + 1].val_storage,
                    src_place,
                    cur_task.path->nodes_[cur_step].val_storage,
                    cur_task.path->nodes_[cur_step + 1].val_bytes_len,
                    cur_task.path->nodes_[cur_step + 1].in_stream);
      }
    }
  }
}

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::walk_to_dest(
    int start_index,
    int gpu_num,
    int* h_left,
    int* h_right,
    KeyType* src_key,
    char* src_val,
    size_t val_size) {
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
    CUDA_CHECK(cudaMemcpyAsync(node.key_storage,
                               reinterpret_cast<char*>(src_key + h_left[i]),
                               node.key_bytes_len,
                               cudaMemcpyDefault,
                               node.in_stream));
    if (need_copy_val) {
      CUDA_CHECK(
          cudaMemcpyAsync(node.val_storage,
                          src_val + uint64_t(h_left[i]) * uint64_t(val_size),
                          node.val_bytes_len,
                          cudaMemcpyDefault,
                          node.in_stream));
    }
  }
  while (!que.empty()) {
    CopyTask& cur_task = que.front();
    que.pop();
    if (cur_task.path->nodes_[cur_task.step].sync) {
      CUDA_CHECK(cudaStreamSynchronize(
          cur_task.path->nodes_[cur_task.step].in_stream));
    }
    if (cur_task.step != cur_task.path->nodes_.size() - 1) {
      int cur_step = cur_task.step;
      CopyTask c(cur_task.path, cur_step + 1);
      que.push(c);
      CUDA_CHECK(
          cudaMemcpyAsync(cur_task.path->nodes_[cur_step + 1].key_storage,
                          cur_task.path->nodes_[cur_step].key_storage,
                          cur_task.path->nodes_[cur_step + 1].key_bytes_len,
                          cudaMemcpyDefault,
                          cur_task.path->nodes_[cur_step + 1].in_stream));
      if (need_copy_val) {
        CUDA_CHECK(
            cudaMemcpyAsync(cur_task.path->nodes_[cur_step + 1].val_storage,
                            cur_task.path->nodes_[cur_step].val_storage,
                            cur_task.path->nodes_[cur_step + 1].val_bytes_len,
                            cudaMemcpyDefault,
                            cur_task.path->nodes_[cur_step + 1].in_stream));
      }
    }
  }
}

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::walk_to_src(
    int start_index,
    int gpu_num,
    int* h_left,
    int* h_right,
    char* src_val,
    size_t val_size) {
  std::queue<CopyTask> que;
  for (int i = 0; i < gpu_num; i++) {
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    int cur_step = path_[start_index][i].nodes_.size() - 1;
    auto& node = path_[start_index][i].nodes_[cur_step];
    if (cur_step == 0) {
      CUDA_CHECK(cudaMemcpyAsync(src_val + uint64_t(h_left[i]) * val_size,
                                 node.val_storage,
                                 node.val_bytes_len,
                                 cudaMemcpyDefault,
                                 node.out_stream));
    } else {
      CopyTask t(&path_[start_index][i], cur_step - 1);
      que.push(t);
      CUDA_CHECK(cudaMemcpyAsync(
          path_[start_index][i].nodes_[cur_step - 1].val_storage,
          node.val_storage,
          path_[start_index][i].nodes_[cur_step - 1].val_bytes_len,
          cudaMemcpyDefault,
          path_[start_index][i].nodes_[cur_step - 1].out_stream));
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
      CUDA_CHECK(
          cudaMemcpyAsync(cur_task.path->nodes_[cur_step - 1].val_storage,
                          cur_task.path->nodes_[cur_step].val_storage,
                          cur_task.path->nodes_[cur_step - 1].val_bytes_len,
                          cudaMemcpyDefault,
                          cur_task.path->nodes_[cur_step - 1].out_stream));
    } else if (cur_step == 0) {
      int end_index = cur_task.path->nodes_.back().dev_num;
      CUDA_CHECK(
          cudaMemcpyAsync(src_val + uint64_t(h_left[end_index]) * val_size,
                          cur_task.path->nodes_[cur_step].val_storage,
                          cur_task.path->nodes_[cur_step].val_bytes_len,
                          cudaMemcpyDefault,
                          cur_task.path->nodes_[cur_step].out_stream));
    }
  }
}

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
HeterComm<KeyType, ValType, GradType, GPUAccessor>::~HeterComm() {
  if (!multi_mf_dim_) {
    for (auto& table : tables_) {
      delete table;
      table = nullptr;
    }
  } else {
    for (auto& table : ptr_tables_) {
      delete table;
      table = nullptr;
    }
    for (auto& table : tables_) {
      delete table;
      table = nullptr;
    }
  }
}

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::show_one_table(
    int gpu_num) {
  if (!multi_mf_dim_) {
    tables_[gpu_num]->show();
  }
}

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::
    show_table_collisions() {
  size_t idx = 0;
  for (auto& table : tables_) {
    if (table != nullptr) {
      table->show_collision(idx++);
    }
  }
  idx = 0;
  for (auto& table : ptr_tables_) {
    if (table != nullptr) {
      table->show_collision(idx++);
    }
  }
}

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
int HeterComm<KeyType, ValType, GradType, GPUAccessor>::log2i(int x) {
  unsigned res = 0;
  while (x >>= 1) {
    ++res;
  }
  return res;
}

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
int HeterComm<KeyType, ValType, GradType, GPUAccessor>::get_index_by_devid(
    int devid) {
  return resource_->get_index_by_devid(devid);
}

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::set_sparse_sgd(
    const OptimizerConfig& optimizer_config) {
  for (int i = 0; i < resource_->total_device(); ++i) {
    AnyDeviceGuard guard(resource_->dev_id(i));
    if (!multi_mf_dim_) {
      tables_[i]->set_sparse_sgd(optimizer_config);
    } else {
      ptr_tables_[i]->set_sparse_sgd(optimizer_config);
    }
  }
}

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::set_embedx_sgd(
    const OptimizerConfig& optimizer_config) {
  for (int i = 0; i < resource_->total_device(); ++i) {
    AnyDeviceGuard guard(resource_->dev_id(i));
    if (!multi_mf_dim_) {
      tables_[i]->set_embedx_sgd(optimizer_config);
    } else {
      ptr_tables_[i]->set_embedx_sgd(optimizer_config);
    }
  }
}

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::build_ps(
    int dev_num,
    KeyType* h_keys,
    ValType* h_vals,
    size_t len,
    size_t chunk_size,
    int stream_num,
    int offset) {
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

    memory_copy(dst_place,
                reinterpret_cast<char*>(d_key_bufs[cur_stream]->ptr()),
                src_place,
                h_keys + cur_len,
                sizeof(KeyType) * tmp_len,
                cur_use_stream);
    memory_copy(dst_place,
                reinterpret_cast<char*>(d_val_bufs[cur_stream]->ptr()),
                src_place,
                h_vals + cur_len,
                sizeof(ValType) * tmp_len,
                cur_use_stream);
    if (offset == -1) offset = dev_num;
    tables_[offset]->insert(
        reinterpret_cast<KeyType*>(d_key_bufs[cur_stream]->ptr()),
        reinterpret_cast<ValType*>(d_val_bufs[cur_stream]->ptr()),
        (size_t)tmp_len,
        cur_use_stream);

    cur_stream += 1;
    cur_len += tmp_len;
  }
  for (int i = 0; i < stream_num; ++i) {
    sync_stream(streams[i]);
    destroy_stream(streams[i]);
  }
}

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::build_ps(
    int num,
    KeyType* h_keys,
    char* pool,
    size_t len,
    size_t feature_value_size,
    size_t chunk_size,
    int stream_num) {
  if (len <= 0) {
    return;
  }
  int dev_id = resource_->dev_id(num);

  DevPlace place = DevPlace(dev_id);
  AnyDeviceGuard guard(dev_id);

  // use hbm pool
  std::vector<memory::allocation::AllocationPtr> d_key_bufs;

  ppStream streams[stream_num];  // NOLINT
  for (int i = 0; i < stream_num; ++i) {
    create_stream(&(streams[i]));
    auto d_k_buf = memory::Alloc(place, chunk_size * sizeof(KeyType));
    d_key_bufs.push_back(std::move(d_k_buf));
  }

  int cur_len = 0;
  int cur_stream = 0;

  while (cur_len < len) {
    cur_stream = cur_stream % stream_num;
    auto cur_use_stream = streams[cur_stream];
#if defined(PADDLE_WITH_XPU_KP)
    cur_use_stream = 0;
#endif
    int tmp_len = cur_len + chunk_size > len ? len - cur_len : chunk_size;

    auto dst_place = place;
    auto src_place = platform::CPUPlace();

    memory_copy(dst_place,
                reinterpret_cast<char*>(d_key_bufs[cur_stream]->ptr()),
                src_place,
                h_keys + cur_len,
                sizeof(KeyType) * tmp_len,
                cur_use_stream);
    ptr_tables_[num]->insert(
        reinterpret_cast<KeyType*>(d_key_bufs[cur_stream]->ptr()),
        tmp_len,
        pool,
        feature_value_size,
        cur_len,
        cur_use_stream);
    cur_stream += 1;
    cur_len += tmp_len;
  }
  for (int i = 0; i < stream_num; ++i) {
    sync_stream(streams[i]);
    destroy_stream(streams[i]);
  }
}

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::merge_grad(
    int dev_num,
    KeyType* d_keys,
    GradType* d_grads,
    size_t len,
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
  heter_comm_kernel_->sort_pairs(NULL,
                                 temp_storage_bytes,
                                 d_keys,
                                 d_merge_keys_ptr,
                                 d_grads,
                                 d_merge_grads_ptr,
                                 len,
                                 0,
                                 8 * sizeof(KeyType),
                                 stream,
                                 false);
  auto d_temp_storage = memory::Alloc(place, temp_storage_bytes);
  heter_comm_kernel_->sort_pairs(d_temp_storage->ptr(),
                                 temp_storage_bytes,
                                 d_keys,
                                 d_merge_keys_ptr,
                                 d_grads,
                                 d_merge_grads_ptr,
                                 len,
                                 0,
                                 8 * sizeof(KeyType),
                                 stream,
                                 false);
  temp_storage_bytes = 0;
  auto d_num_runs_out_mem = memory::Alloc(place, sizeof(int));
  int* d_num_runs_out = reinterpret_cast<int*>(d_num_runs_out_mem->ptr());
  heter_comm_kernel_->reduce_by_key(NULL,
                                    temp_storage_bytes,
                                    d_merge_keys_ptr,
                                    d_keys,
                                    d_merge_grads_ptr,
                                    d_grads,
                                    d_num_runs_out,
                                    len,
                                    stream,
                                    false);
  if (d_temp_storage->size() < temp_storage_bytes) {
    d_temp_storage = NULL;
    d_temp_storage = memory::Alloc(place, temp_storage_bytes);
  }
  heter_comm_kernel_->reduce_by_key(d_temp_storage->ptr(),
                                    temp_storage_bytes,
                                    d_merge_keys_ptr,
                                    d_keys,
                                    d_merge_grads_ptr,
                                    d_grads,
                                    d_num_runs_out,
                                    len,
                                    stream,
                                    false);
  auto dst_place = platform::CPUPlace();
  auto src_place = place;
  memory_copy(
      dst_place, &uniq_len, src_place, d_num_runs_out, sizeof(int), stream);
  sync_stream(stream);
}

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::dynamic_merge_grad(
    int gpu_num,
    KeyType* d_keys,
    float* d_grads,
    size_t len,
    int& uniq_len,
    size_t& segment_len,
    bool enable_segment_merge_grad) {
  int dev_id = resource_->dev_id(gpu_num);
  platform::CUDAPlace place = platform::CUDAPlace(dev_id);
  platform::CUDADeviceGuard guard(dev_id);
  auto stream = resource_->local_stream(gpu_num, 0);

  size_t temp_storage_bytes;
  size_t grad_dim = max_mf_dim_;
  auto accessor_wrapper_ptr =
      GlobalAccessorFactory::GetInstance().GetAccessorWrapper();
  size_t grad_value_size = accessor_wrapper_ptr->GetPushValueSize(max_mf_dim_);

  auto d_merge_keys = memory::Alloc(place, len * sizeof(KeyType));
  KeyType* d_merge_keys_ptr = reinterpret_cast<KeyType*>(d_merge_keys->ptr());
  auto d_fea_num_info = memory::Alloc(place, sizeof(uint32_t) * (len * 3 + 1));
  uint32_t* d_fea_num_info_ptr =
      reinterpret_cast<uint32_t*>(d_fea_num_info->ptr());
  uint32_t* d_index = (uint32_t*)&d_fea_num_info_ptr[len];
  uint32_t* d_idx = (uint32_t*)&d_index[len];
  int* d_merged_size = (int*)&d_idx[len];
  heter_comm_kernel_->fill_idx(d_idx, len, stream);

  PADDLE_ENFORCE_GPU_SUCCESS(
      cub::DeviceRadixSort::SortPairs(NULL,
                                      temp_storage_bytes,
                                      d_keys,
                                      d_merge_keys_ptr,
                                      d_idx,
                                      d_index,
                                      len,
                                      0,
                                      8 * sizeof(KeyType),
                                      stream));
  auto d_temp_storage = memory::Alloc(place, temp_storage_bytes);
  PADDLE_ENFORCE_GPU_SUCCESS(
      cub::DeviceRadixSort::SortPairs(d_temp_storage->ptr(),
                                      temp_storage_bytes,
                                      d_keys,
                                      d_merge_keys_ptr,
                                      d_idx,
                                      d_index,
                                      len,
                                      0,
                                      8 * sizeof(KeyType),
                                      stream));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));

  temp_storage_bytes = 0;
  PADDLE_ENFORCE_GPU_SUCCESS(
      cub::DeviceRunLengthEncode::Encode(NULL,
                                         temp_storage_bytes,
                                         d_merge_keys_ptr,
                                         d_keys,
                                         d_fea_num_info_ptr,
                                         d_merged_size,
                                         len,
                                         stream));
  if (d_temp_storage->size() < temp_storage_bytes) {
    d_temp_storage = NULL;
    d_temp_storage = memory::Alloc(place, temp_storage_bytes);
  }
  PADDLE_ENFORCE_GPU_SUCCESS(
      cub::DeviceRunLengthEncode::Encode(d_temp_storage->ptr(),
                                         temp_storage_bytes,
                                         d_merge_keys_ptr,
                                         d_keys,
                                         d_fea_num_info_ptr,
                                         d_merged_size,
                                         len,
                                         stream));

  cudaMemcpyAsync((void*)&uniq_len,
                  d_merged_size,
                  sizeof(int),
                  cudaMemcpyDeviceToHost,
                  stream);
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));

  assert(d_merged_size > 0);
  uint32_t* d_offset = (uint32_t*)&d_index[len];
  temp_storage_bytes = 0;
  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceScan::ExclusiveSum(NULL,
                                                           temp_storage_bytes,
                                                           d_fea_num_info_ptr,
                                                           d_offset,
                                                           uniq_len,
                                                           stream));
  if (d_temp_storage->size() < temp_storage_bytes) {
    d_temp_storage = NULL;
    d_temp_storage = memory::Alloc(place, temp_storage_bytes);
  }
  PADDLE_ENFORCE_GPU_SUCCESS(
      cub::DeviceScan::ExclusiveSum(d_temp_storage->ptr(),
                                    temp_storage_bytes,
                                    d_fea_num_info_ptr,
                                    d_offset,
                                    uniq_len,
                                    stream));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));

  if (enable_segment_merge_grad) {
    segment_merge_grad(gpu_num,
                       d_merge_keys_ptr,
                       d_grads,
                       d_index,
                       len,
                       d_fea_num_info_ptr,
                       uniq_len,
                       segment_len);
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(d_keys,
                                               d_merge_keys_ptr,
                                               sizeof(KeyType) * segment_len,
                                               cudaMemcpyDeviceToDevice,
                                               stream));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
  } else {
    auto d_merge_grads = memory::Alloc(place, len * grad_value_size);
    float* d_merge_grads_ptr = reinterpret_cast<float*>(d_merge_grads->ptr());

    heter_comm_kernel_->merge_gradient(d_keys,
                                       d_offset,
                                       d_fea_num_info_ptr,
                                       d_index,
                                       (char*)d_grads,
                                       (char*)d_merge_grads_ptr,
                                       uniq_len,
                                       grad_dim,
                                       grad_value_size,
                                       merger_,
                                       stream,
                                       gpu_accessor_);
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(d_grads,
                                               d_merge_grads_ptr,
                                               grad_value_size * uniq_len,
                                               cudaMemcpyDeviceToDevice,
                                               stream));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
  }
}

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::segment_merge_grad(
    int gpu_num,  // the device number
    KeyType*
        d_keys,  // the sorted keys list, which will be modified after merged
    float* d_grads,  // the raw grads list, which will be modified after merged
    const uint32_t*
        d_index,  // the storage position of d_keys, its length is len.
    size_t len,   // the number of raw input keys
    const uint32_t*
        d_fea_num_info,      // prefix sum array, its length is uniq_len+1
    size_t uniq_len,         // the number of unique keys
    size_t& segments_num) {  // the number of segment merged keys

  int dev_id = resource_->dev_id(gpu_num);
  platform::CUDAPlace place = platform::CUDAPlace(dev_id);
  platform::CUDADeviceGuard guard(dev_id);
  auto stream = resource_->local_stream(gpu_num, 0);

  auto grad_dim = max_mf_dim_;
  auto accessor_wrapper_ptr =
      GlobalAccessorFactory::GetInstance().GetAccessorWrapper();
  size_t grad_value_size = accessor_wrapper_ptr->GetPushValueSize(max_mf_dim_);

  auto d_buffer1 = memory::Alloc(place, sizeof(uint32_t) * len);
  auto d_segments = reinterpret_cast<uint32_t*>(d_buffer1->ptr());
  auto d_buffer2 = memory::Alloc(place, sizeof(uint32_t) * len);
  auto d_segments_offset = reinterpret_cast<uint32_t*>(d_buffer2->ptr());
  auto d_buffer3 = memory::Alloc(place, sizeof(uint32_t) * len);
  auto d_segments_fea_num_info = reinterpret_cast<uint32_t*>(d_buffer3->ptr());
  auto d_buffer4 = memory::Alloc(place, sizeof(uint32_t) * len);
  auto d_segments_fea_num_offset =
      reinterpret_cast<uint32_t*>(d_buffer4->ptr());
  auto d_buffer5 = memory::Alloc(place, sizeof(uint32_t));
  auto d_segments_num = reinterpret_cast<uint32_t*>(d_buffer5->ptr());
  CUDA_CHECK(cudaMemsetAsync(d_segments_num, 0, sizeof(uint32_t), stream));

  uint32_t segment_size = FLAGS_gpugraph_merge_grads_segment_size;
  heter_comm_kernel_->split_segments(d_fea_num_info,
                                     uniq_len,
                                     d_segments,
                                     d_segments_num,
                                     segment_size,
                                     stream);
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));

  size_t temp_storage_bytes = 0;
  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceReduce::Sum(
      NULL, temp_storage_bytes, d_segments, d_segments_num, uniq_len, stream));
  auto d_temp_storage = memory::Alloc(place, temp_storage_bytes);
  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceReduce::Sum(d_temp_storage->ptr(),
                                                    temp_storage_bytes,
                                                    d_segments,
                                                    d_segments_num,
                                                    uniq_len,
                                                    stream));
  CUDA_CHECK(cudaMemcpyAsync(&segments_num,
                             d_segments_num,
                             sizeof(uint32_t),
                             cudaMemcpyDeviceToHost,
                             stream));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));

  temp_storage_bytes = 0;
  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceScan::ExclusiveSum(NULL,
                                                           temp_storage_bytes,
                                                           d_segments,
                                                           d_segments_offset,
                                                           uniq_len,
                                                           stream));
  if (d_temp_storage->size() < temp_storage_bytes) {
    d_temp_storage = NULL;
    d_temp_storage = memory::Alloc(place, temp_storage_bytes);
  }
  PADDLE_ENFORCE_GPU_SUCCESS(
      cub::DeviceScan::ExclusiveSum(d_temp_storage->ptr(),
                                    temp_storage_bytes,
                                    d_segments,
                                    d_segments_offset,
                                    uniq_len,
                                    stream));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));

  heter_comm_kernel_->expand_segments(d_fea_num_info,
                                      d_segments_offset,
                                      uniq_len,
                                      d_segments_fea_num_info,
                                      segment_size,
                                      stream);
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));

  PADDLE_ENFORCE_GPU_SUCCESS(
      cub::DeviceScan::ExclusiveSum(NULL,
                                    temp_storage_bytes,
                                    d_segments_fea_num_info,
                                    d_segments_fea_num_offset,
                                    segments_num,
                                    stream));
  if (d_temp_storage->size() < temp_storage_bytes) {
    d_temp_storage = NULL;
    d_temp_storage = memory::Alloc(place, temp_storage_bytes);
  }
  PADDLE_ENFORCE_GPU_SUCCESS(
      cub::DeviceScan::ExclusiveSum(d_temp_storage->ptr(),
                                    temp_storage_bytes,
                                    d_segments_fea_num_info,
                                    d_segments_fea_num_offset,
                                    segments_num,
                                    stream));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));

  auto d_segments_keys = memory::Alloc(place, sizeof(KeyType) * segments_num);
  auto d_segments_keys_ptr = reinterpret_cast<KeyType*>(d_segments_keys->ptr());
  heter_comm_kernel_->shrink_keys(d_keys,
                                  d_segments_fea_num_offset,
                                  d_segments_keys_ptr,
                                  segments_num,
                                  stream);
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));

  auto d_segment_grads = memory::Alloc(place, segments_num * grad_value_size);
  auto d_segment_grads_ptr = reinterpret_cast<float*>(d_segment_grads->ptr());
  heter_comm_kernel_->merge_gradient(d_segments_keys_ptr,
                                     d_segments_fea_num_offset,
                                     d_segments_fea_num_info,
                                     d_index,
                                     (char*)d_grads,
                                     (char*)d_segment_grads_ptr,
                                     segments_num,
                                     grad_dim,
                                     grad_value_size,
                                     merger_,
                                     stream,
                                     gpu_accessor_);
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));

  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(d_keys,
                                             d_segments_keys_ptr,
                                             sizeof(KeyType) * segments_num,
                                             cudaMemcpyDeviceToDevice,
                                             stream));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(d_grads,
                                             d_segment_grads_ptr,
                                             grad_value_size * segments_num,
                                             cudaMemcpyDeviceToDevice,
                                             stream));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
}

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::split_input_to_shard(
    KeyType* d_keys,
    int* d_idx_ptr,
    size_t len,
    int* left,
    int* right,
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

  heter_comm_kernel_->fill_idx(d_idx_tmp_ptr, len, stream);
  heter_comm_kernel_->calc_shard_index(
      d_keys, len, d_shard_index_tmp_ptr, total_device, stream);

  size_t temp_storage_bytes;
  const int num_bits = 1 + log2i(total_device);
  heter_comm_kernel_->sort_pairs(NULL,
                                 temp_storage_bytes,
                                 d_shard_index_tmp_ptr,
                                 d_shard_index_ptr,
                                 d_idx_tmp_ptr,
                                 d_idx_ptr,
                                 len,
                                 0,
                                 num_bits,
                                 stream);

  auto d_temp_storage = memory::Alloc(place, temp_storage_bytes);
  heter_comm_kernel_->sort_pairs(d_temp_storage->ptr(),
                                 temp_storage_bytes,
                                 d_shard_index_tmp_ptr,
                                 d_shard_index_ptr,
                                 d_idx_tmp_ptr,
                                 d_idx_ptr,
                                 len,
                                 0,
                                 num_bits,
                                 stream);

  heter_comm_kernel_->calc_shard_offset(
      d_shard_index_ptr, left, right, len, total_device, stream);
  sync_stream(stream);
}

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::merge_keys(
    int gpu_num,
    const KeyType* d_keys,
    size_t len,               // input
    KeyType* d_sorted_keys,   // output
    KeyType* d_merged_keys,   // output
    uint32_t* d_restore_idx,  // output
    size_t& uniq_len) {       // output
  int dev_id = resource_->dev_id(gpu_num);
  platform::CUDAPlace place = platform::CUDAPlace(dev_id);
  platform::CUDADeviceGuard guard(dev_id);
  auto stream = resource_->local_stream(gpu_num, 0);

  size_t grad_dim = max_mf_dim_;
  auto accessor_wrapper_ptr =
      GlobalAccessorFactory::GetInstance().GetAccessorWrapper();
  size_t grad_value_size = accessor_wrapper_ptr->GetPushValueSize(max_mf_dim_);

  auto d_fea_num_info = memory::Alloc(place, sizeof(uint32_t) * (len * 4 + 1));
  uint32_t* d_fea_num_info_ptr =
      reinterpret_cast<uint32_t*>(d_fea_num_info->ptr());
  uint32_t* d_idx = (uint32_t*)&d_fea_num_info_ptr[len];
  uint32_t* d_index = (uint32_t*)&d_idx[len];
  uint32_t* d_offset = (uint32_t*)&d_index[len];
  uint32_t* d_merged_size = (uint32_t*)&d_offset[len];
  heter_comm_kernel_->fill_idx(d_idx, len, stream);

  size_t temp_storage_bytes;
  PADDLE_ENFORCE_GPU_SUCCESS(
      cub::DeviceRadixSort::SortPairs(NULL,
                                      temp_storage_bytes,
                                      d_keys,
                                      d_sorted_keys,
                                      d_idx,
                                      d_index,
                                      len,
                                      0,
                                      8 * sizeof(KeyType),
                                      stream));
  auto d_temp_storage = memory::Alloc(place, temp_storage_bytes);
  PADDLE_ENFORCE_GPU_SUCCESS(
      cub::DeviceRadixSort::SortPairs(d_temp_storage->ptr(),
                                      temp_storage_bytes,
                                      d_keys,
                                      d_sorted_keys,
                                      d_idx,
                                      d_index,
                                      len,
                                      0,
                                      8 * sizeof(KeyType),
                                      stream));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));

  temp_storage_bytes = 0;
  PADDLE_ENFORCE_GPU_SUCCESS(
      cub::DeviceRunLengthEncode::Encode(NULL,
                                         temp_storage_bytes,
                                         d_sorted_keys,
                                         d_merged_keys,
                                         d_fea_num_info_ptr,
                                         d_merged_size,
                                         len,
                                         stream));
  if (d_temp_storage->size() < temp_storage_bytes) {
    d_temp_storage = NULL;
    d_temp_storage = memory::Alloc(place, temp_storage_bytes);
  }
  PADDLE_ENFORCE_GPU_SUCCESS(
      cub::DeviceRunLengthEncode::Encode(d_temp_storage->ptr(),
                                         temp_storage_bytes,
                                         d_sorted_keys,
                                         d_merged_keys,
                                         d_fea_num_info_ptr,
                                         d_merged_size,
                                         len,
                                         stream));
  cudaMemcpyAsync((void*)&uniq_len,
                  d_merged_size,
                  sizeof(int),
                  cudaMemcpyDeviceToHost,
                  stream);
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));

  temp_storage_bytes = 0;
  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceScan::ExclusiveSum(NULL,
                                                           temp_storage_bytes,
                                                           d_fea_num_info_ptr,
                                                           d_offset,
                                                           uniq_len,
                                                           stream));
  if (d_temp_storage->size() < temp_storage_bytes) {
    d_temp_storage = NULL;
    d_temp_storage = memory::Alloc(place, temp_storage_bytes);
  }
  PADDLE_ENFORCE_GPU_SUCCESS(
      cub::DeviceScan::ExclusiveSum(d_temp_storage->ptr(),
                                    temp_storage_bytes,
                                    d_fea_num_info_ptr,
                                    d_offset,
                                    uniq_len,
                                    stream));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));

  heter_comm_kernel_->fill_restore_idx(true,
                                       len,
                                       uniq_len,
                                       d_merged_keys,
                                       d_index,
                                       d_offset,
                                       d_fea_num_info_ptr,
                                       d_restore_idx,
                                       stream);
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
}

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::pull_merge_sparse(
    int num, KeyType* d_keys, float* d_vals, size_t len) {
  int total_device = resource_->total_device();
  int dev_id = resource_->dev_id(num);
  DevPlace place = DevPlace(dev_id);
  AnyDeviceGuard guard(dev_id);
  auto stream = resource_->local_stream(num, 0);

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
  paddle::platform::XPUDeviceContext xpu_dev_ctx(place);
  auto xpu_context = xpu_dev_ctx.x_context();

  int r = xpu::constant<int>(xpu_context, d_left_ptr, total_device, -1);
  PADDLE_ENFORCE_EQ(r,
                    XPU_SUCCESS,
                    platform::errors::External(
                        "XPU constant kernel return wrong value[%d %s]",
                        r,
                        XPUAPIErrorMsg[r]));
  int r2 = xpu::constant<int>(xpu_context, d_right_ptr, total_device, -1);
  PADDLE_ENFORCE_EQ(r2,
                    XPU_SUCCESS,
                    platform::errors::External(
                        "XPU constant kernel return wrong value[%d %s]",
                        r2,
                        XPUAPIErrorMsg[r2]));
#endif

  auto accessor_wrapper_ptr =
      GlobalAccessorFactory::GetInstance().GetAccessorWrapper();
  size_t val_type_size = accessor_wrapper_ptr->GetPullValueSize(max_mf_dim_);
  VLOG(3) << "pull_sparse len:" << len << "  val_type_size: " << val_type_size;
  auto d_sorted_keys = memory::Alloc(place, len * sizeof(KeyType));
  auto d_sorted_keys_ptr = reinterpret_cast<KeyType*>(d_sorted_keys->ptr());
  auto d_merged_keys = memory::Alloc(place, len * sizeof(KeyType));
  auto d_merged_keys_ptr = reinterpret_cast<KeyType*>(d_merged_keys->ptr());
  auto d_restore_idx = memory::Alloc(place, len * sizeof(uint32_t));
  auto d_restore_idx_ptr = reinterpret_cast<uint32_t*>(d_restore_idx->ptr());
  auto d_shard_keys = memory::Alloc(place, len * sizeof(KeyType));
  auto d_shard_keys_ptr = reinterpret_cast<KeyType*>(d_shard_keys->ptr());
  auto d_shard_vals = memory::Alloc(place, len * val_type_size);
  auto d_shard_vals_ptr = reinterpret_cast<float*>(d_shard_vals->ptr());

  size_t uniq_len = 0;
  merge_keys(num,
             d_keys,
             len,
             d_sorted_keys_ptr,
             d_merged_keys_ptr,
             d_restore_idx_ptr,
             uniq_len);
  sync_stream(stream);

  auto d_idx = memory::Alloc(place, uniq_len * sizeof(int));
  auto d_idx_ptr = reinterpret_cast<int*>(d_idx->ptr());
  split_input_to_shard(
      d_merged_keys_ptr, d_idx_ptr, uniq_len, d_left_ptr, d_right_ptr, num);
  heter_comm_kernel_->fill_shard_key(
      d_shard_keys_ptr, d_merged_keys_ptr, d_idx_ptr, uniq_len, stream);
  sync_stream(stream);

  auto dst_place = platform::CPUPlace();
  auto src_place = place;

  memory_copy(dst_place,
              h_left,
              src_place,
              d_left_ptr,
              total_device * sizeof(int),
              stream);
  memory_copy(dst_place,
              h_right,
              src_place,
              d_right_ptr,
              total_device * sizeof(int),
              stream);

  if (!FLAGS_gpugraph_enable_gpu_direct_access) {
    for (int i = 0; i < total_device; ++i) {
      int shard_len = h_right[i] - h_left[i] + 1;
      if (h_left[i] == -1 || h_right[i] == -1) {
        continue;
      }
      create_storage(
          num, i, shard_len * sizeof(KeyType), shard_len * val_type_size);
    }
    walk_to_dest(num, total_device, h_left, h_right, d_shard_keys_ptr, NULL);
  }

  for (int i = 0; i < total_device; ++i) {
    if (h_left[i] == -1) {
      continue;
    }
    auto& node = path_[num][i].nodes_.back();
    if (!FLAGS_gpugraph_enable_gpu_direct_access) {
      sync_stream(node.in_stream);
    }
    AnyDeviceGuard guard(resource_->dev_id(i));
    ptr_tables_[i]->rwlock_->RDLock();
    if (!FLAGS_gpugraph_enable_gpu_direct_access) {
      ptr_tables_[i]->get(reinterpret_cast<KeyType*>(node.key_storage),
                          node.val_storage,
                          h_right[i] - h_left[i] + 1,
                          resource_->remote_stream(i, num),
                          gpu_accessor_);
    } else {
      ptr_tables_[i]->get(
          d_shard_keys_ptr + h_left[i],
          reinterpret_cast<char*>(d_shard_vals_ptr) + h_left[i] * val_type_size,
          h_right[i] - h_left[i] + 1,
          resource_->remote_stream(i, num),
          gpu_accessor_);
    }
  }

  for (int i = 0; i < total_device; ++i) {
    sync_stream(resource_->remote_stream(i, num));
    if (h_left[i] == -1) {
      continue;
    }
    ptr_tables_[i]->rwlock_->UNLock();
  }

  if (!FLAGS_gpugraph_enable_gpu_direct_access) {
    walk_to_src(num,
                total_device,
                h_left,
                h_right,
                reinterpret_cast<char*>(d_shard_vals_ptr),
                val_type_size);
    for (int i = 0; i < total_device; ++i) {
      auto& node = path_[num][i].nodes_.front();
      sync_stream(node.out_stream);
    }
  }

  auto d_merged_vals = memory::Alloc(place, uniq_len * val_type_size);
  auto d_merged_vals_ptr = reinterpret_cast<float*>(d_merged_vals->ptr());
  heter_comm_kernel_->dy_mf_fill_dvals(d_shard_vals_ptr,
                                       d_merged_vals_ptr,
                                       d_idx_ptr,
                                       uniq_len,
                                       val_type_size,
                                       stream);
  sync_stream(stream);

  heter_comm_kernel_->unpack_merged_vals(len,
                                         d_keys,
                                         d_merged_vals_ptr,
                                         d_restore_idx_ptr,
                                         d_vals,
                                         val_type_size,
                                         stream);
  sync_stream(stream);

  if (!FLAGS_gpugraph_enable_gpu_direct_access) {
    for (int i = 0; i < total_device; ++i) {
      if (h_left[i] == -1 || h_right[i] == -1) {
        continue;
      }
      destroy_storage(num, i);
    }
  }
}
template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::pull_normal_sparse(
    int num, KeyType* d_keys, float* d_vals, size_t len) {
  int total_device = resource_->total_device();
  int dev_id = resource_->dev_id(num);
  DevPlace place = DevPlace(dev_id);
  AnyDeviceGuard guard(dev_id);
  auto stream = resource_->local_stream(num, 0);

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
  paddle::platform::XPUDeviceContext xpu_dev_ctx(place);
  auto xpu_context = xpu_dev_ctx.x_context();

  int r = xpu::constant<int>(xpu_context, d_left_ptr, total_device, -1);
  PADDLE_ENFORCE_EQ(r,
                    XPU_SUCCESS,
                    platform::errors::External(
                        "XPU constant kernel return wrong value[%d %s]",
                        r,
                        XPUAPIErrorMsg[r]));
  int r2 = xpu::constant<int>(xpu_context, d_right_ptr, total_device, -1);
  PADDLE_ENFORCE_EQ(r2,
                    XPU_SUCCESS,
                    platform::errors::External(
                        "XPU constant kernel return wrong value[%d %s]",
                        r2,
                        XPUAPIErrorMsg[r2]));
#endif

  auto d_idx = memory::Alloc(place, len * sizeof(int));
  int* d_idx_ptr = reinterpret_cast<int*>(d_idx->ptr());

  auto accessor_wrapper_ptr =
      GlobalAccessorFactory::GetInstance().GetAccessorWrapper();
  size_t val_type_size = accessor_wrapper_ptr->GetPullValueSize(max_mf_dim_);
  VLOG(3) << "pull_sparse len:" << len << "  val_type_size: " << val_type_size;
  auto d_shard_keys = memory::Alloc(place, len * sizeof(KeyType));
  KeyType* d_shard_keys_ptr = reinterpret_cast<KeyType*>(d_shard_keys->ptr());
  auto d_shard_vals = memory::Alloc(place, len * val_type_size);
  float* d_shard_vals_ptr = reinterpret_cast<float*>(d_shard_vals->ptr());

  split_input_to_shard(d_keys, d_idx_ptr, len, d_left_ptr, d_right_ptr, num);

  heter_comm_kernel_->fill_shard_key(
      d_shard_keys_ptr, d_keys, d_idx_ptr, len, stream);

  sync_stream(stream);

  auto dst_place = platform::CPUPlace();
  auto src_place = place;

  memory_copy(dst_place,
              h_left,
              src_place,
              d_left_ptr,
              total_device * sizeof(int),
              stream);
  memory_copy(dst_place,
              h_right,
              src_place,
              d_right_ptr,
              total_device * sizeof(int),
              stream);

  if (!FLAGS_gpugraph_enable_gpu_direct_access) {
    for (int i = 0; i < total_device; ++i) {
      int shard_len = h_right[i] - h_left[i] + 1;
      if (h_left[i] == -1 || h_right[i] == -1) {
        continue;
      }
      create_storage(
          num, i, shard_len * sizeof(KeyType), shard_len * val_type_size);
    }
    walk_to_dest(num, total_device, h_left, h_right, d_shard_keys_ptr, NULL);
  }
  for (int i = 0; i < total_device; ++i) {
    if (h_left[i] == -1) {
      continue;
    }
    auto& node = path_[num][i].nodes_.back();
    if (!FLAGS_gpugraph_enable_gpu_direct_access) {
      sync_stream(node.in_stream);
    }
    AnyDeviceGuard guard(resource_->dev_id(i));
    ptr_tables_[i]->rwlock_->RDLock();
    if (!FLAGS_gpugraph_enable_gpu_direct_access) {
      ptr_tables_[i]->get(reinterpret_cast<KeyType*>(node.key_storage),
                          node.val_storage,
                          h_right[i] - h_left[i] + 1,
                          resource_->remote_stream(i, num),
                          gpu_accessor_);
    } else {
      ptr_tables_[i]->get(
          d_shard_keys_ptr + h_left[i],
          reinterpret_cast<char*>(d_shard_vals_ptr) + h_left[i] * val_type_size,
          h_right[i] - h_left[i] + 1,
          resource_->remote_stream(i, num),
          gpu_accessor_);
    }
  }

  for (int i = 0; i < total_device; ++i) {
    sync_stream(resource_->remote_stream(i, num));
    if (h_left[i] == -1) {
      continue;
    }
    ptr_tables_[i]->rwlock_->UNLock();
  }
  if (!FLAGS_gpugraph_enable_gpu_direct_access) {
    walk_to_src(num,
                total_device,
                h_left,
                h_right,
                reinterpret_cast<char*>(d_shard_vals_ptr),
                val_type_size);
    for (int i = 0; i < total_device; ++i) {
      auto& node = path_[num][i].nodes_.front();
      sync_stream(node.out_stream);
    }
  }
  heter_comm_kernel_->dy_mf_fill_dvals(
      d_shard_vals_ptr, d_vals, d_idx_ptr, len, val_type_size, stream);

  sync_stream(stream);

  if (!FLAGS_gpugraph_enable_gpu_direct_access) {
    for (int i = 0; i < total_device; ++i) {
      if (h_left[i] == -1 || h_right[i] == -1) {
        continue;
      }
      destroy_storage(num, i);
    }
  }
}

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::pull_sparse(
    int num, KeyType* d_keys, float* d_vals, size_t len) {
  if (len == 0) {
    return;
  }
  if (!FLAGS_gpugraph_dedup_pull_push_mode) {
    pull_merge_sparse(num, d_keys, d_vals, len);
  } else {
    pull_normal_sparse(num, d_keys, d_vals, len);
  }
}

#if defined(PADDLE_WITH_CUDA)
template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
template <typename Sgd>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::push_sparse(
    int dev_num,
    KeyType* d_keys,
    float* d_grads,
    size_t len,
    Sgd& sgd) {  // NOLINT
  if (len == 0) {
    return;
  }

  int total_device = resource_->total_device();
  int dev_id = resource_->dev_id(dev_num);

  auto accessor_wrapper_ptr =
      GlobalAccessorFactory::GetInstance().GetAccessorWrapper();
  size_t grad_value_size = accessor_wrapper_ptr->GetPushValueSize(max_mf_dim_);
  DevPlace place = DevPlace(dev_id);
  AnyDeviceGuard guard(dev_id);
  auto stream = resource_->local_stream(dev_num, 0);

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
  paddle::platform::XPUDeviceContext xpu_dev_ctx(place);
  auto xpu_context = xpu_dev_ctx.x_context();

  int r = xpu::constant<int>(xpu_context, d_left_ptr, total_device, -1);
  PADDLE_ENFORCE_EQ(r,
                    XPU_SUCCESS,
                    platform::errors::External(
                        "XPU constant kernel return wrong value[%d %s]",
                        r,
                        XPUAPIErrorMsg[r]));
  int r2 = xpu::constant<int>(xpu_context, d_right_ptr, total_device, -1);
  PADDLE_ENFORCE_EQ(r2,
                    XPU_SUCCESS,
                    platform::errors::External(
                        "XPU constant kernel return wrong value[%d %s]",
                        r2,
                        XPUAPIErrorMsg[r2]));
#endif

  auto d_idx = memory::Alloc(place, len * sizeof(int));
  int* d_idx_ptr = reinterpret_cast<int*>(d_idx->ptr());

  auto d_shard_keys = memory::Alloc(place, len * sizeof(KeyType));
  KeyType* d_shard_keys_ptr = reinterpret_cast<KeyType*>(d_shard_keys->ptr());

  float* d_shard_grads_ptr;
  auto d_shard_grads = memory::Alloc(place, len * grad_value_size);
  d_shard_grads_ptr = reinterpret_cast<float*>(d_shard_grads->ptr());

  int uniq_len = len;
  if (!FLAGS_gpugraph_dedup_pull_push_mode) {
    size_t segment_len = 0;
    if (FLAGS_gpugraph_enable_segment_merge_grads) {
      // do two gradient merge
      // 1st. do segmented gradient merge
      // 2nd. do global gradient merge
      dynamic_merge_grad(
          dev_num, d_keys, d_grads, len, uniq_len, segment_len, true);
      len = segment_len;
      uniq_len = 0;
      segment_len = 0;
      dynamic_merge_grad(
          dev_num, d_keys, d_grads, len, uniq_len, segment_len, false);
    } else {
      // Perform gradient merge only once
      dynamic_merge_grad(
          dev_num, d_keys, d_grads, len, uniq_len, segment_len, false);
    }
  }

  split_input_to_shard(
      d_keys, d_idx_ptr, uniq_len, d_left_ptr, d_right_ptr, dev_num);

  heter_comm_kernel_->dy_mf_fill_shard_grads(d_shard_keys_ptr,
                                             d_keys,
                                             d_shard_grads_ptr,
                                             d_grads,
                                             d_idx_ptr,
                                             uniq_len,
                                             grad_value_size,
                                             stream,
                                             gpu_accessor_);

  sync_stream(stream);

  auto dst_place = platform::CPUPlace();
  auto src_place = place;
  memory_copy(dst_place,
              h_left,
              src_place,
              d_left_ptr,
              total_device * sizeof(int),
              stream);
  memory_copy(dst_place,
              h_right,
              src_place,
              d_right_ptr,
              total_device * sizeof(int),
              stream);

  if (!FLAGS_gpugraph_enable_gpu_direct_access) {
    for (int i = 0; i < total_device; ++i) {
      int shard_len = h_right[i] - h_left[i] + 1;
      if (h_left[i] == -1 || h_right[i] == -1) {
        continue;
      }
      create_storage(
          dev_num, i, shard_len * sizeof(KeyType), shard_len * grad_value_size);
    }

    walk_to_dest(dev_num,
                 total_device,
                 h_left,
                 h_right,
                 d_shard_keys_ptr,
                 reinterpret_cast<char*>(d_shard_grads_ptr),
                 grad_value_size);
  }

  for (int i = 0; i < total_device; ++i) {
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    auto& node = path_[dev_num][i].nodes_.back();
    if (!FLAGS_gpugraph_enable_gpu_direct_access) {
      sync_stream(node.in_stream);
    }

    AnyDeviceGuard guard(resource_->dev_id(i));
    ptr_tables_[i]->rwlock_->WRLock();
    if (!FLAGS_gpugraph_enable_gpu_direct_access) {
      ptr_tables_[i]->update(reinterpret_cast<KeyType*>(node.key_storage),
                             node.val_storage,
                             h_right[i] - h_left[i] + 1,
                             sgd,
                             resource_->remote_stream(i, dev_num));
    } else {
      ptr_tables_[i]->update(d_shard_keys_ptr + h_left[i],
                             reinterpret_cast<char*>(d_shard_grads_ptr) +
                                 grad_value_size * h_left[i],
                             h_right[i] - h_left[i] + 1,
                             sgd,
                             resource_->remote_stream(i, dev_num));
    }
  }

  for (int i = 0; i < total_device; ++i) {
    sync_stream(resource_->remote_stream(i, dev_num));
    if (h_left[i] != -1) {
      if (!multi_mf_dim_) {
        tables_[i]->rwlock_->UNLock();
      } else {
        ptr_tables_[i]->rwlock_->UNLock();
      }
    }
  }

  if (!FLAGS_gpugraph_enable_gpu_direct_access) {
    for (int i = 0; i < total_device; ++i) {
      if (h_left[i] == -1 || h_right[i] == -1) {
        continue;
      }
      destroy_storage(dev_num, i);
    }
  }
}

#elif defined(PADDLE_WITH_XPU_KP)
template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::push_sparse(
    int dev_num, KeyType* d_keys, GradType* d_grads, size_t len) {
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

#if defined(PADDLE_WITH_CUDA)
  cudaMemsetAsync(d_left_ptr, -1, total_device * sizeof(int), stream);
  cudaMemsetAsync(d_right_ptr, -1, total_device * sizeof(int), stream);

#elif defined(PADDLE_WITH_XPU_KP)
  // get XPUDeviceContext according to xpu place
  paddle::platform::XPUDeviceContext xpu_dev_ctx(place);
  auto xpu_context = xpu_dev_ctx.x_context();

  int r = xpu::constant<int>(xpu_context, d_left_ptr, total_device, -1);
  PADDLE_ENFORCE_EQ(r,
                    XPU_SUCCESS,
                    platform::errors::External(
                        "XPU constant kernel return wrong value[%d %s]",
                        r,
                        XPUAPIErrorMsg[r]));
  int r2 = xpu::constant<int>(xpu_context, d_right_ptr, total_device, -1);
  PADDLE_ENFORCE_EQ(r2,
                    XPU_SUCCESS,
                    platform::errors::External(
                        "XPU constant kernel return wrong value[%d %s]",
                        r2,
                        XPUAPIErrorMsg[r2]));
#endif

  auto d_idx = memory::Alloc(place, len * sizeof(int));
  int* d_idx_ptr = reinterpret_cast<int*>(d_idx->ptr());

  auto d_shard_keys = memory::Alloc(place, len * sizeof(KeyType));
  KeyType* d_shard_keys_ptr = reinterpret_cast<KeyType*>(d_shard_keys->ptr());
  auto d_shard_grads = memory::Alloc(place, len * sizeof(GradType));
  GradType* d_shard_grads_ptr =
      reinterpret_cast<GradType*>(d_shard_grads->ptr());

  int uniq_len = len;
  merge_grad(dev_num, d_keys, d_grads, len, uniq_len);

  split_input_to_shard(
      d_keys, d_idx_ptr, uniq_len, d_left_ptr, d_right_ptr, dev_num);

  heter_comm_kernel_->fill_shard_grads(d_shard_keys_ptr,
                                       d_keys,
                                       d_shard_grads_ptr,
                                       d_grads,
                                       d_idx_ptr,
                                       (long long)uniq_len,
                                       stream);

  sync_stream(stream);

  auto dst_place = platform::CPUPlace();
  auto src_place = place;
  memory_copy(dst_place,
              h_left,
              src_place,
              d_left_ptr,
              total_device * sizeof(int),
              stream);
  memory_copy(dst_place,
              h_right,
              src_place,
              d_right_ptr,
              total_device * sizeof(int),
              stream);

  for (int i = 0; i < total_device; ++i) {
    int shard_len = h_right[i] - h_left[i] + 1;
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    create_storage(
        dev_num, i, shard_len * sizeof(KeyType), shard_len * sizeof(GradType));
  }

  walk_to_dest(dev_num,
               total_device,
               h_left,
               h_right,
               d_shard_keys_ptr,
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
                       h_right[i] - h_left[i] + 1,
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

#endif

#if defined(PADDLE_WITH_CUDA)
template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
template <typename Sgd>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::update_one_table(
    int gpu_num,
    KeyType* d_keys,
    GradType* d_grads,
    size_t len,
    Sgd& sgd) {  // NOLINT
  if (len == 0) {
    return;
  }

  int dev_id = resource_->dev_id(gpu_num);
  platform::CUDADeviceGuard guard(dev_id);
  tables_[gpu_num]->rwlock_->WRLock();
  tables_[gpu_num]->update(
      d_keys, d_grads, len, sgd, resource_->remote_stream(gpu_num, gpu_num));
  tables_[gpu_num]->rwlock_->UNLock();
  cudaStreamSynchronize(resource_->remote_stream(gpu_num, gpu_num));
}

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
template <typename Sgd>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::push_sparse_multi_node(
    int gpu_num,
    KeyType* d_keys,
    GradType* d_grads,
    size_t len,
    Sgd& sgd) {  // NOLINT
  if (len == 0) {
    return;
  }

  int uniq_len = len;
  merge_grad(gpu_num, d_keys, d_grads, len, uniq_len);

  uniq_len = gather_one_node_grad(gpu_num, d_keys, d_grads, uniq_len);

  uniq_len = gather_multi_node_grad(gpu_num,
                                    storage_[gpu_num].local_keys,
                                    storage_[gpu_num].local_grads,
                                    uniq_len);

  update_one_table(gpu_num,
                   storage_[gpu_num].local_keys,
                   storage_[gpu_num].local_grads,
                   uniq_len,
                   sgd);
}

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
int HeterComm<KeyType, ValType, GradType, GPUAccessor>::gather_one_node_grad(
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

  cudaMemcpy(d_node_len + gpu_num,
             h_node_len + gpu_num,
             sizeof(int),
             cudaMemcpyHostToDevice);

  // allgather grad len
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupStart());
  PADDLE_ENFORCE_GPU_SUCCESS(
      platform::dynload::ncclAllGather((const void*)(d_node_len + gpu_num),
                                       (void*)d_node_len,
                                       1,        // NOLINT
                                       ncclInt,  // NOLINT
                                       nccl_inner_comm,
                                       stream));
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupEnd());
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
  cudaMemcpy(
      h_node_len, d_node_len, sizeof(int) * total_gpu, cudaMemcpyDeviceToHost);

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

  PADDLE_ENFORCE_GPU_SUCCESS(
      platform::dynload::ncclAllGather(d_grads,
                                       storage.all_grads,
                                       max_size * sizeof(GradType),
                                       ncclUint8,
                                       nccl_inner_comm,
                                       stream));
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

    split_input_to_shard(storage.all_keys + index,
                         d_idx_ptr,
                         h_node_len[i],
                         d_left_ptr,
                         d_right_ptr,
                         gpu_num);
    cudaMemcpy(
        h_left, d_left_ptr, total_gpu * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(
        h_right, d_right_ptr, total_gpu * sizeof(int), cudaMemcpyDeviceToHost);

    heter_comm_kernel_->fill_shard_grads(storage.local_keys + merge_num,
                                         storage.all_keys + index,
                                         storage.local_grads + merge_num,
                                         storage.all_grads + index,
                                         d_idx_ptr + h_left[gpu_num],
                                         h_right[gpu_num] - h_left[gpu_num] + 1,
                                         stream);
    merge_num = merge_num + h_right[gpu_num] - h_left[gpu_num] + 1;
  }

  int ret = merge_num;
  merge_grad(gpu_num, storage.local_keys, storage.local_grads, merge_num, ret);
  return ret;
}

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
int HeterComm<KeyType, ValType, GradType, GPUAccessor>::gather_multi_node_grad(
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
  cudaMemcpy(
      h_node_len, d_node_len, sizeof(int) * node_size_, cudaMemcpyDeviceToHost);

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

  PADDLE_ENFORCE_GPU_SUCCESS(
      platform::dynload::ncclAllGather(d_grads,
                                       storage.all_grads,
                                       max_size * sizeof(GradType),
                                       ncclUint8,
                                       nccl_inter_comm,
                                       stream));
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupEnd());
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));

  int merge_num = 0;
  for (int i = 0; i < node_size_; ++i) {
    int index = i * max_size;
    cudaMemcpyAsync(storage.local_keys + merge_num,
                    storage.all_keys + index,
                    h_node_len[i],
                    cudaMemcpyDefault,
                    stream);
    cudaMemcpyAsync(storage.local_grads + merge_num,
                    storage.all_grads + index,
                    h_node_len[i],
                    cudaMemcpyDefault,
                    stream);
    merge_num += h_node_len[i];
  }

  int ret = merge_num;
  merge_grad(gpu_num, storage.local_keys, storage.local_grads, merge_num, ret);
  return ret;
}
#endif

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::end_pass() {
  int total_device = resource_->total_device();
  std::vector<std::thread> threads;

  auto dump_to_cpu_func = [this](int index) {
    auto stream = resource_->local_stream(index, 0);
    int dev_id = resource_->dev_id(index);
    AnyDeviceGuard guard(dev_id);
    tables_[index]->dump_to_cpu(dev_id, stream);
  };

  if (!multi_mf_dim_) {
    for (int i = 0; i < total_device; ++i) {
      threads.push_back(std::thread(dump_to_cpu_func, i));
    }
    for (auto& t : threads) {
      t.join();
    }
  }
}

#if defined(PADDLE_WITH_CUDA)
template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
int HeterComm<KeyType, ValType, GradType, GPUAccessor>::dedup_keys_and_fillidx(
    const int gpu_id,
    const int total_fea_num,
    const KeyType* d_keys,   // input
    KeyType* d_merged_keys,  // output
    KeyType* d_sorted_keys,
    uint32_t* d_restore_idx,
    uint32_t* d_sorted_idx,
    uint32_t* d_offset,
    uint32_t* d_merged_cnts,
    bool filter_zero) {
  int dev_id = resource_->dev_id(gpu_id);
  platform::CUDAPlace place = platform::CUDAPlace(dev_id);
  platform::CUDADeviceGuard guard(dev_id);
  auto stream = resource_->local_stream(gpu_id, 0);

  assert(total_fea_num > 0);
  int merged_size = 0;
  size_t byte_size = sizeof(uint32_t) * (total_fea_num + 1);

  auto d_index_ptr = memory::Alloc(place, byte_size);
  uint32_t* d_index_in = reinterpret_cast<uint32_t*>(d_index_ptr->ptr());
  int* d_merged_size = reinterpret_cast<int*>(&d_index_in[total_fea_num]);

  heter_comm_kernel_->fill_idx(d_index_in, total_fea_num, stream);

  void* d_buf = NULL;
  size_t temp_storage_bytes = 0;
  PADDLE_ENFORCE_GPU_SUCCESS(
      cub::DeviceRadixSort::SortPairs(NULL,
                                      temp_storage_bytes,
                                      d_keys,
                                      d_sorted_keys,
                                      d_index_in,
                                      d_sorted_idx,
                                      total_fea_num,
                                      0,
                                      8 * sizeof(KeyType),
                                      stream,
                                      false));
  auto d_cache_ptr = memory::Alloc(place, temp_storage_bytes);
  d_buf = reinterpret_cast<int*>(d_cache_ptr->ptr());
  PADDLE_ENFORCE_GPU_SUCCESS(
      cub::DeviceRadixSort::SortPairs(d_buf,
                                      temp_storage_bytes,
                                      d_keys,
                                      d_sorted_keys,
                                      d_index_in,
                                      d_sorted_idx,
                                      total_fea_num,
                                      0,
                                      8 * sizeof(KeyType),
                                      stream,
                                      false));

  PADDLE_ENFORCE_GPU_SUCCESS(
      cub::DeviceRunLengthEncode::Encode(NULL,
                                         temp_storage_bytes,
                                         d_sorted_keys,
                                         d_merged_keys,
                                         d_merged_cnts,
                                         d_merged_size,
                                         total_fea_num,
                                         stream));
  if (d_cache_ptr->size() < temp_storage_bytes) {
    d_cache_ptr = NULL;
    d_cache_ptr = memory::Alloc(place, temp_storage_bytes);
  }
  d_buf = reinterpret_cast<int*>(d_cache_ptr->ptr());
  PADDLE_ENFORCE_GPU_SUCCESS(
      cub::DeviceRunLengthEncode::Encode(d_buf,
                                         temp_storage_bytes,
                                         d_sorted_keys,
                                         d_merged_keys,
                                         d_merged_cnts,
                                         d_merged_size,
                                         total_fea_num,
                                         stream));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync((void*)&merged_size,
                                             (void*)d_merged_size,
                                             sizeof(int),
                                             cudaMemcpyDeviceToHost,
                                             stream));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));

  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceScan::ExclusiveSum(
      NULL, temp_storage_bytes, d_merged_cnts, d_offset, merged_size, stream));
  if (d_cache_ptr->size() < temp_storage_bytes) {
    d_cache_ptr = NULL;
    d_cache_ptr = memory::Alloc(place, temp_storage_bytes);
  }
  d_buf = reinterpret_cast<int*>(d_cache_ptr->ptr());
  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceScan::ExclusiveSum(
      d_buf, temp_storage_bytes, d_merged_cnts, d_offset, merged_size, stream));

  if (filter_zero) {
    cudaMemsetAsync(d_restore_idx, 0, total_fea_num * sizeof(uint32_t), stream);
  }
  // fill restore idx [1,3,5,2,4,6] = [1,2,1,3,2,1]
  heter_comm_kernel_->fill_restore_idx(filter_zero,
                                       total_fea_num,
                                       merged_size,
                                       d_merged_keys,
                                       d_sorted_idx,
                                       d_offset,
                                       d_merged_cnts,
                                       d_restore_idx,
                                       stream);

  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));

  return merged_size;
}
#endif
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

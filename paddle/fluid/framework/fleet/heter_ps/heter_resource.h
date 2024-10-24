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
#include <cstddef>
#include <map>
#include <memory>
#include <vector>

#ifdef PADDLE_WITH_CUDA
#include "paddle/phi/core/platform/cuda_device_guard.h"
#endif

#ifdef PADDLE_WITH_XPU_KP
#include <xpu/runtime.h>  // NOLINT

#include "paddle/phi/core/platform/device/xpu/xpu_info.h"
#endif

#include "paddle/fluid/platform/enforce.h"

#ifdef PADDLE_WITH_HETERPS

namespace paddle {
namespace framework {

#if defined(PADDLE_WITH_CUDA)
using ppStream = cudaStream_t;

#elif defined(PADDLE_WITH_XPU_KP)
using ppStream = XPUStream;
#endif

#if defined(PADDLE_WITH_CUDA)
class GPUResource {
 public:
  GPUResource(std::vector<int>& device_id, int index);  // NOLINT
  virtual ~GPUResource();
  GPUResource(const GPUResource&) = delete;
  GPUResource& operator=(const GPUResource&) = delete;

  int dev_id() const { return dev_id_; }
  int index() const { return index_; }
  gpuStream_t local_stream(int num) { return local_streams_[num]; }
  gpuStream_t remote_stream(int num) { return remote_streams_[num]; }
  gpuStream_t comm_stream(int num) { return comm_streams_[num]; }

  int dev_id_;
  int index_;
  std::vector<int> dev_ids_;
  std::vector<gpuStream_t> remote_streams_;
  std::vector<gpuStream_t> local_streams_;
  std::vector<gpuStream_t> comm_streams_;
};

#elif defined(PADDLE_WITH_XPU_KP)
class XPUResource {
 public:
  XPUResource(std::vector<int>& device_id, int index);  // NOLINT
  virtual ~XPUResource();
  XPUResource(const XPUResource&) = delete;
  XPUResource& operator=(const XPUResource&) = delete;

  int dev_id() const { return dev_id_; }
  int index() const { return index_; }
  XPUStream local_stream(int num) { return local_streams_[num]; }
  XPUStream remote_stream(int num) { return remote_streams_[num]; }
  XPUStream comm_stream(int num) { return comm_streams_[num]; }

  int dev_id_;
  int index_;
  std::vector<int> dev_ids_;
  std::vector<XPUStream> remote_streams_;
  std::vector<XPUStream> local_streams_;
  std::vector<XPUStream> comm_streams_;
};
#endif

#if defined(PADDLE_WITH_CUDA)
using DevResource = GPUResource;
using DevPlace = phi::GPUPlace;
using AnyDeviceGuard = platform::CUDADeviceGuard;
#elif defined(PADDLE_WITH_XPU_KP)
using DevResource = XPUResource;
using DevPlace = phi::XPUPlace;
using AnyDeviceGuard = phi::backends::xpu::XPUDeviceGuard;
#endif

#if defined(PADDLE_WITH_CUDA)
class GpuRDMAChecker {
 public:
  static GpuRDMAChecker* get(int device_num);

 public:
  explicit GpuRDMAChecker(int device_num);
  // rdma
  bool need_rdma_trans(void);
  bool is_device_support_rdma(int devid);
  // device num
  int device_num(void) { return device_num_; }
  // topo_aware
  bool topo_aware(void) { return topo_aware_; }

 private:
  bool check_device_status(const int& device_count,
                           std::vector<int>* gpu_status);

 private:
  int device_num_ = 0;
  bool topo_aware_ = false;
  // rdma
  bool rdma_trans_ = false;
  std::vector<int> rdma_status_;
};
#endif

template <typename KeyType, typename ValType>
class HashTable;

class HeterPsResource {
 public:
  explicit HeterPsResource(const std::vector<int>& dev_ids);
  HeterPsResource(const HeterPsResource&) = delete;
  HeterPsResource& operator=(const HeterPsResource&) = delete;
  virtual ~HeterPsResource() {}
  void enable_p2p();
  int total_device();
  int get_index_by_devid(int devid);
  int dev_id(int num);
  void set_multi_mf(int multi_mf_dim, int max_mf_dim);
  std::shared_ptr<HashTable<uint64_t, uint32_t>> keys2rank(int gpu_id) {
    return keys2rank_vec_[gpu_id];
  }
  void set_keys2rank(int gpu_id,
                     std::shared_ptr<HashTable<uint64_t, uint32_t>> keys2rank) {
    keys2rank_vec_[gpu_id] = keys2rank;
  }
  int multi_mf() { return multi_mf_dim_; }
  int max_mf_dim() { return max_mf_dim_; }

  ppStream local_stream(int dev_num, int stream_num);
  ppStream remote_stream(int dev_num, int stream_num);
  ppStream comm_stream(int dev_num, int stream_num);
  // node
  bool multi_node(void) { return multi_node_; }
  void set_multi_node(bool multi_node) { multi_node_ = multi_node; }

  std::vector<std::shared_ptr<DevResource>> resources_;
  std::vector<int> dev_ids_;
  std::map<int, int> devid_2_index_;
  int multi_mf_dim_{0};
  int max_mf_dim_{0};

  // multi node
  bool multi_node_ = false;
  std::vector<std::shared_ptr<HashTable<uint64_t, uint32_t>>> keys2rank_vec_;
};

}  // namespace framework
}  // namespace paddle
#endif

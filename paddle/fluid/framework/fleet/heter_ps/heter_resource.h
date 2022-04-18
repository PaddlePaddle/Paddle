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
#include "paddle/fluid/platform/cuda_device_guard.h"
#endif

#ifdef PADDLE_WITH_XPU_KP
#include <xpu/runtime.h>  // NOLINT
#include "paddle/fluid/platform/device/xpu/xpu_info.h"
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
using DevPlace = platform::CUDAPlace;
using AnyDeviceGuard = platform::CUDADeviceGuard;
#elif defined(PADDLE_WITH_XPU_KP)
using DevResource = XPUResource;
using DevPlace = platform::XPUPlace;
using AnyDeviceGuard = platform::XPUDeviceGuard;
#endif

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

  ppStream local_stream(int dev_num, int stream_num);
  ppStream remote_stream(int dev_num, int stream_num);
  ppStream comm_stream(int dev_num, int stream_num);

  std::vector<std::shared_ptr<DevResource>> resources_;
  std::vector<int> dev_ids_;
  std::map<int, int> devid_2_index_;
  int multi_mf_dim_{0};
  int max_mf_dim_{0};
};

}  // end namespace framework
}  // end namespace paddle
#endif

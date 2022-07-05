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

#ifdef PADDLE_WITH_HETERPS
#include "paddle/fluid/framework/fleet/heter_ps/heter_resource.h"

#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/cuda_device_guard.h"
#endif

#ifdef PADDLE_WITH_XPU_KP
#include "paddle/fluid/platform/device/xpu/enforce_xpu.h"
#include "paddle/fluid/platform/device/xpu/xpu_info.h"
#endif

namespace paddle {
namespace framework {

#if defined(PADDLE_WITH_CUDA)
GPUResource::GPUResource(std::vector<int>& dev_ids, int index) {
  index_ = index;
  dev_ids_ = dev_ids;
  dev_id_ = dev_ids_[index];

  platform::CUDADeviceGuard guard(dev_id_);
  local_streams_.resize(dev_ids_.size());
  comm_streams_.resize(dev_ids_.size());
  remote_streams_.resize(dev_ids_.size());

  for (size_t i = 0; i < dev_ids_.size(); ++i) {
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaStreamCreateWithFlags(&local_streams_[i], cudaStreamNonBlocking));
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaStreamCreateWithFlags(&comm_streams_[i], cudaStreamNonBlocking));
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaStreamCreateWithFlags(&remote_streams_[i], cudaStreamNonBlocking));
  }
}

GPUResource::~GPUResource() {
  platform::CUDADeviceGuard guard(dev_id_);
  for (size_t i = 0; i < local_streams_.size(); ++i) {
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamDestroy(local_streams_[i]));
  }
  for (size_t i = 0; i < comm_streams_.size(); ++i) {
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamDestroy(comm_streams_[i]));
  }
  for (size_t i = 0; i < remote_streams_.size(); ++i) {
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamDestroy(remote_streams_[i]));
  }
}

#elif defined(PADDLE_WITH_XPU_KP)
XPUResource::XPUResource(std::vector<int>& dev_ids, int index) {
  index_ = index;
  dev_ids_ = dev_ids;
  dev_id_ = dev_ids_[index];

  platform::XPUDeviceGuard guard(dev_id_);
  local_streams_.resize(dev_ids_.size());

  comm_streams_.resize(dev_ids_.size(), NULL);
  remote_streams_.resize(dev_ids_.size());

  for (size_t i = 0; i < dev_ids_.size(); ++i) {
    PADDLE_ENFORCE_XPU_SUCCESS(xpu_stream_create(&local_streams_[i]));
    // PADDLE_ENFORCE_XPU_SUCCESS(xpu_stream_create(&comm_streams_[i]));
    PADDLE_ENFORCE_XPU_SUCCESS(xpu_stream_create(&remote_streams_[i]));
  }
}

XPUResource::~XPUResource() {
  platform::XPUDeviceGuard guard(dev_id_);
  for (size_t i = 0; i < local_streams_.size(); ++i) {
    PADDLE_ENFORCE_XPU_SUCCESS(xpu_stream_destroy(local_streams_[i]));
  }

  // for (size_t i = 0; i < comm_streams_.size(); ++i) {
  //  PADDLE_ENFORCE_XPU_SUCCESS(xpu_stream_destroy(comm_streams_[i]));
  // }
  for (size_t i = 0; i < remote_streams_.size(); ++i) {
    PADDLE_ENFORCE_XPU_SUCCESS(xpu_stream_destroy(remote_streams_[i]));
  }
}

#endif

void HeterPsResource::enable_p2p() {
#if defined(PADDLE_WITH_CUDA)
  for (size_t i = 0; i < dev_ids_.size(); ++i) {
    platform::CUDADeviceGuard guard(dev_ids_[i]);
    for (size_t j = 0; j < dev_ids_.size(); ++j) {
      if (i != j) {
        int p2p_flag;
        PADDLE_ENFORCE_GPU_SUCCESS(
            cudaDeviceCanAccessPeer(&p2p_flag, dev_ids_[i], dev_ids_[j]));
        if (p2p_flag == 1) {
          cudaError_t ret = cudaDeviceEnablePeerAccess(dev_ids_[j], 0);
          if (ret != cudaSuccess && ret != cudaErrorPeerAccessAlreadyEnabled) {
            VLOG(0) << " Cuda error(" << ret << "), " << cudaGetErrorString(ret)
                    << ".";
          } else {
            cudaGetLastError();
          }
        }
      }
    }
  }
#endif
}

HeterPsResource::HeterPsResource(const std::vector<int>& dev_ids) {
  dev_ids_ = dev_ids;
  for (size_t i = 0; i < dev_ids_.size(); ++i) {
    std::shared_ptr<DevResource> resource =
        std::make_shared<DevResource>(dev_ids_, i);
    resources_.push_back(resource);
    devid_2_index_[dev_ids_[i]] = i;
  }
}

ppStream HeterPsResource::comm_stream(int dev_num, int stream_num) {
  return resources_[dev_num]->comm_stream(stream_num);
}
ppStream HeterPsResource::local_stream(int dev_num, int stream_num) {
  return resources_[dev_num]->local_stream(stream_num);
}

ppStream HeterPsResource::remote_stream(int dev_num, int stream_num) {
  return resources_[dev_num]->remote_stream(stream_num);
}

int HeterPsResource::dev_id(int num) { return dev_ids_[num]; }

int HeterPsResource::get_index_by_devid(int devid) {
  return devid_2_index_[devid];
}

int HeterPsResource::total_device() { return dev_ids_.size(); }

void HeterPsResource::set_multi_mf(int multi_mf_dim, int max_mf_dim) {
  multi_mf_dim_ = multi_mf_dim;
  max_mf_dim_ = max_mf_dim;
  VLOG(3) << "heter resource set mf dim: " << multi_mf_dim_
          << " max_mf_dim_: " << max_mf_dim_;
}

}  // end namespace framework
}  // end namespace paddle
#endif

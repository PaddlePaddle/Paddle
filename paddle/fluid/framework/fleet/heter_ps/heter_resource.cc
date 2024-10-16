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
#include "paddle/phi/core/platform/cuda_device_guard.h"
#endif

#ifdef PADDLE_WITH_XPU_KP
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/platform/device/xpu/xpu_info.h"
#endif
#include "paddle/common/flags.h"
#include "paddle/utils/string/string_helper.h"

COMMON_DECLARE_bool(enable_auto_detect_gpu_topo);
COMMON_DECLARE_bool(enable_auto_rdma_trans);

namespace paddle::framework {

#if defined(PADDLE_WITH_CUDA)
GPUResource::GPUResource(std::vector<int> &dev_ids, int index) {
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
    PADDLE_WARN_GPU_SUCCESS(cudaStreamDestroy(local_streams_[i]));
  }
  for (size_t i = 0; i < comm_streams_.size(); ++i) {
    PADDLE_WARN_GPU_SUCCESS(cudaStreamDestroy(comm_streams_[i]));
  }
  for (size_t i = 0; i < remote_streams_.size(); ++i) {
    PADDLE_WARN_GPU_SUCCESS(cudaStreamDestroy(remote_streams_[i]));
  }
}

#elif defined(PADDLE_WITH_XPU_KP)
XPUResource::XPUResource(std::vector<int> &dev_ids, int index) {
  index_ = index;
  dev_ids_ = dev_ids;
  dev_id_ = dev_ids_[index];

  phi::backends::xpu::XPUDeviceGuard guard(dev_id_);
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
  phi::backends::xpu::XPUDeviceGuard guard(dev_id_);
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
static std::string execute_cmd_result(const std::string &cmd) {
  FILE *fp = popen(cmd.c_str(), "r");
  if (fp == NULL) {
    fprintf(stderr, "cmd %s open failed\n", cmd.c_str());
    return "";
  }

  std::string out;
  size_t ret = 0;
  char szline[1024] = {0};
  while ((ret = fread(szline, sizeof(char), sizeof(szline), fp)) > 0) {
    out.append(szline, ret);
  }
  pclose(fp);
  fprintf(stderr, "cmd: %s, ret:\n%s\n", cmd.c_str(), out.c_str());
  return paddle::string::trim_spaces(out);
}
#if defined(PADDLE_WITH_CUDA)
static std::shared_ptr<GpuRDMAChecker> g_checker = nullptr;
GpuRDMAChecker *GpuRDMAChecker::get(int device_num) {
  if (g_checker == nullptr) {
    g_checker = std::make_shared<GpuRDMAChecker>(device_num);
  }
  // check gpu num
  PADDLE_ENFORCE_EQ(
      device_num,
      g_checker->device_num(),
      common::errors::InvalidArgument(
          "Invalid number of device. Should be %d. But received %d.",
          device_num,
          g_checker->device_num()));
  return g_checker.get();
}
GpuRDMAChecker::GpuRDMAChecker(int device_num) {
  device_num_ = device_num;
  rdma_trans_ = check_device_status(device_num, &rdma_status_);
}
bool GpuRDMAChecker::need_rdma_trans(void) {
  return (FLAGS_enable_auto_rdma_trans && rdma_trans_);
}
bool GpuRDMAChecker::is_device_support_rdma(int devid) {
  if (rdma_status_.empty()) {
    return true;
  }
  return rdma_status_[devid];
}
bool GpuRDMAChecker::check_device_status(const int &device_count,
                                         std::vector<int> *gpu_status) {
  // not need auto detect gpu topo aware
  if (!FLAGS_enable_auto_detect_gpu_topo) {
    return false;
  }
  // a100
  std::string str =
      execute_cmd_result("source ~/.bashrc && nvidia-smi topo -m");
  if (str.empty()) {  // a100 auto gpu card rdma status
    return false;
  }
  // mlx5_0  PXB PXB SYS SYS SYS SYS SYS SYS  X  SYS SYS
  // mlx5_2  SYS SYS PXB PXB SYS SYS SYS SYS SYS NODE     X
  std::vector<std::string> lines = paddle::string::split_string(str, "\n");
  if (lines.empty()) {
    fprintf(stdout, "%s\n", str.c_str());
    return false;
  }
  std::vector<std::string> gpu_mlxs;
  gpu_status->resize(device_count, 0);
  gpu_mlxs.resize(device_count);
  for (auto line : lines) {
    std::vector<std::string> tags = paddle::string::split_string(line);
    if (tags.size() < static_cast<size_t>(device_count + 1)) {
      continue;
    }
    std::string &card_name = tags[0];
    if (strncmp(card_name.c_str(), "GPU0", 4) == 0) {
      // check topo_aware
      topo_aware_ = false;
      for (int j = 1; j < device_count; ++j) {
        std::string &tag = tags[j + 1];
        if (strncmp(tag.c_str(), "NV", 2) == 0) {
          continue;
        }
        topo_aware_ = true;
      }
      continue;
    }
    if ((strncmp(card_name.c_str(), "mlx5", 4) != 0) &&
        (strncmp(card_name.c_str(), "NIC", 3) != 0)) {
      continue;
    }
    for (int j = 0; j < device_count; ++j) {
      std::string &tag = tags[j + 1];
      if (strcmp(tag.c_str(), "PXB") != 0 && strcmp(tag.c_str(), "PIX") != 0) {
        continue;
      }
      (*gpu_status)[j] = 1;
      if (!gpu_mlxs[j].empty()) {
        gpu_mlxs[j].append(",");
      }
      gpu_mlxs[j].append(card_name);
    }
  }
  int not_trans_cnt = 0;
  int need_trans_cnt = 0;
  // check all rdma
  for (int j = 0; j < device_count; ++j) {
    if ((*gpu_status)[j] > 0) {
      fprintf(
          stdout, "GPU%d: rdma check ok, used %s\n", j, gpu_mlxs[j].c_str());
      continue;
    }
    int trans_id = (j + device_count / 2) % device_count;
    if ((*gpu_status)[trans_id] > 0) {
      fprintf(
          stdout, "GPU%d: rdma check pcie, used trans id %d\n", j, trans_id);
      ++need_trans_cnt;
    } else {
      ++not_trans_cnt;
    }
  }
  // need trans device all connect to other device
  return (need_trans_cnt > 0 && not_trans_cnt == 0);
}
#endif

HeterPsResource::HeterPsResource(const std::vector<int> &dev_ids) {
  dev_ids_ = dev_ids;
  for (size_t i = 0; i < dev_ids_.size(); ++i) {
    std::shared_ptr<DevResource> resource =
        std::make_shared<DevResource>(dev_ids_, i);
    resources_.push_back(resource);
    devid_2_index_[dev_ids_[i]] = i;
  }
  keys2rank_vec_.resize(dev_ids.size());
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

}  // namespace paddle::framework
#endif

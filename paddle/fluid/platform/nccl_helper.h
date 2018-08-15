//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <stdio.h>
#include <string>
#include <thread>  // NOLINT
#include <typeindex>
#include <vector>
#include "paddle/fluid/platform/dynload/nccl.h"
#include "paddle/fluid/platform/enforce.h"

#define NCCL_ID_VARNAME "NCCLID"

namespace paddle {
namespace platform {

inline ncclDataType_t ToNCCLDataType(std::type_index type) {
  if (type == typeid(float)) {  // NOLINT
    return ncclFloat;
  } else if (type == typeid(double)) {  // NOLINT
    return ncclDouble;
  } else if (type == typeid(int)) {  // NOLINT
    return ncclInt;
  } else if (type == typeid(int64_t)) {  // NOLINT
    return ncclInt64;
  } else {
    PADDLE_THROW("Not supported");
  }
}

// NOTE(minqiyang): according to the ncclGroupEnd documentations:
// https://docs.nvidia.com/deeplearning/sdk/nccl-api/ncclapidoc.html,
// ncclGroupEnd will wait for all communicators to be initialized, which will
// cause blocking problem when a runtime_error was thrown, so try only guard
// NCCL actions when use it.
class NCCLGroupGuard {
 public:
  static std::mutex &NCCLMutex() {
    static std::mutex mtx;
    return mtx;
  }

  inline NCCLGroupGuard() {
    NCCLMutex().lock();
    PADDLE_ENFORCE(dynload::ncclGroupStart());
  }

  inline ~NCCLGroupGuard() {
    CHECK_EQ(dynload::ncclGroupEnd(), ncclSuccess);
    NCCLMutex().unlock();
  }
};

struct NCCLContext {
  std::unique_ptr<CUDADeviceContext> ctx_;
  ncclComm_t comm_;

  explicit NCCLContext(int dev_id)
      : ctx_(new CUDADeviceContext(CUDAPlace(dev_id))), comm_{nullptr} {}

  cudaStream_t stream() const { return ctx_->stream(); }

  int device_id() const {
    return boost::get<platform::CUDAPlace>(ctx_->GetPlace()).device;
  }
};

struct NCCLContextMap {
  std::unordered_map<int, NCCLContext> contexts_;
  std::vector<int> order_;

  explicit NCCLContextMap(const std::vector<platform::Place> &places,
                          ncclUniqueId *nccl_id = nullptr,
                          size_t num_trainers = 1, size_t trainer_id = 0) {
    PADDLE_ENFORCE(!places.empty());
    order_.reserve(places.size());
    for (auto &p : places) {
      int dev_id = boost::get<CUDAPlace>(p).device;
      order_.emplace_back(dev_id);
      contexts_.emplace(dev_id, NCCLContext(dev_id));
    }
    PADDLE_ENFORCE_EQ(
        order_.size(), contexts_.size(),
        "NCCL Context Map does not support contain two or more same device");

    if (places.size() <= 1) {
      return;
    }
    std::unique_ptr<ncclComm_t[]> comms(new ncclComm_t[order_.size()]);
    // if pass nccl_id here, can assume we are doing multi node training
    if (nccl_id == nullptr) {
      std::lock_guard<std::mutex> guard(NCCLGroupGuard::NCCLMutex());
      PADDLE_ENFORCE(platform::dynload::ncclCommInitAll(
          comms.get(), static_cast<int>(order_.size()), order_.data()));
    } else {
      PADDLE_ENFORCE_GT(num_trainers, 1);
      // TODO(wuyi): need to ensure each node have same number of GPUs
      {
        int nranks = num_trainers * order_.size();
        NCCLGroupGuard gurad;
        for (auto &gpu_id : order_) {
          int rank = trainer_id * order_.size() + gpu_id;
          VLOG(3) << "init nccl rank: " << rank << " nranks: " << nranks;
          PADDLE_ENFORCE(cudaSetDevice(gpu_id));
          PADDLE_ENFORCE(platform::dynload::ncclCommInitRank(
              comms.get() + gpu_id, nranks, *nccl_id, rank));
        }
      }
    }
    int i = 0;
    for (auto &dev_id : order_) {
      contexts_.at(dev_id).comm_ = comms[i++];
    }
  }

  NCCLContextMap(const NCCLContextMap &other) = delete;
  NCCLContextMap &operator=(const NCCLContextMap &other) = delete;

  CUDADeviceContext *DevCtx(int dev_id) const { return at(dev_id).ctx_.get(); }

  CUDADeviceContext *DevCtx(platform::Place p) const {
    return DevCtx(boost::get<CUDAPlace>(p).device);
  }

  const NCCLContext &at(platform::Place p) const {
    return this->at(boost::get<CUDAPlace>(p).device);
  }

  const NCCLContext &at(int dev_id) const { return contexts_.at(dev_id); }

  void WaitAll() {
    for (auto &p : contexts_) {
      p.second.ctx_->Wait();
    }
  }
};

}  // namespace platform
}  // namespace paddle

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

#ifndef _WIN32
#pragma once

#include <stdio.h>
#include <string>
#include <thread>  // NOLINT
#include <typeindex>
#include <vector>
#include "paddle/fluid/platform/device_context.h"
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
  std::unique_ptr<platform::CUDADeviceContext> ctx_;
  ncclComm_t comm_;
  int rank_;

  explicit NCCLContext(int dev_id)
      : ctx_(new platform::CUDADeviceContext(CUDAPlace(dev_id))),
        comm_{nullptr} {}

  cudaStream_t stream() const { return ctx_->stream(); }

  int device_id() const {
    return boost::get<platform::CUDAPlace>(ctx_->GetPlace()).device;
  }
};

class NCCLContextMap {
 public:
  std::unordered_map<int, NCCLContext> contexts_;
  std::vector<int> order_;
  int nranks_;

  static NCCLContextMap *Init(const std::vector<platform::Place> &places,
                              ncclUniqueId *nccl_id = nullptr,
                              size_t num_trainers = 1, size_t trainer_id = 0) {
    VLOG(10) << "init NCCLContextMap instance...";
    if (ctx_map_ptr_ == nullptr) {
      VLOG(10) << "first time initialize.";
      ctx_map_ptr_ =
          new NCCLContextMap(places, nccl_id, num_trainers, trainer_id);
    }
    return ctx_map_ptr_;
  }

  static NCCLContextMap &Instance() {
    if (ctx_map_ptr_ == nullptr)
      PADDLE_THROW("NCCLContextMap singleton should be initalized first.");
    return *ctx_map_ptr_;
  }

  explicit NCCLContextMap(const std::vector<platform::Place> &places,
                          ncclUniqueId *nccl_id = nullptr,
                          size_t num_trainers = 1, size_t trainer_id = 0);

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

  size_t size() const { return contexts_.size(); }

  void WaitAll() {
    for (auto &p : contexts_) {
      p.second.ctx_->Wait();
    }
  }

 private:
  static NCCLContextMap *ctx_map_ptr_;
};

}  // namespace platform
}  // namespace paddle
#endif

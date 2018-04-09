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

#include <thread>
#include <typeindex>
#include "paddle/fluid/platform/dynload/rccl.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace platform {

inline rcclDataType_t ToNCCLDataType(std::type_index type) {
  if (type == typeid(float)) {  // NOLINT
    return rcclFloat;
  } else if (type == typeid(double)) {  // NOLINT
    return rcclDouble;
  } else if (type == typeid(int)) {  // NOLINT
    return rcclInt;
  } else {
    PADDLE_THROW("Not supported");
  }
}

class NCCLGroupGuard {
 public:
  inline NCCLGroupGuard() {
    mutex().lock();
    //PADDLE_ENFORCE(dynload::rcclGroupStart());
  }

  inline ~NCCLGroupGuard() {
    //PADDLE_ENFORCE(dynload::rcclGroupEnd());
    mutex().unlock();
  }

 private:
  static std::mutex &mutex() {
    static std::mutex mtx;
    return mtx;
  }
};

struct NCCLContext {
  std::unique_ptr<CUDADeviceContext> ctx_;
  rcclComm_t comm_;

  explicit NCCLContext(int dev_id)
      : ctx_(new CUDADeviceContext(CUDAPlace(dev_id))) {}

  hipStream_t stream() const { return ctx_->stream(); }

  int device_id() const {
    return boost::get<platform::CUDAPlace>(ctx_->GetPlace()).device;
  }

  static void InitNCCLContext(std::unordered_map<int, NCCLContext> &contexts,
                              const std::vector<platform::Place> &places) {
    std::vector<rcclComm_t> comms;
    std::vector<int> devs;
    comms.resize(contexts.size());
    devs.reserve(contexts.size());

    for (auto &p : places) {
      devs.push_back(boost::get<platform::CUDAPlace>(p).device);
    }

    PADDLE_ENFORCE(platform::dynload::rcclCommInitAll(
        &comms[0], static_cast<int>(contexts.size()), &devs[0]));

    int i = 0;
    for (auto &dev_id : devs) {
      contexts.at(dev_id).comm_ = comms[i++];
    }
  }
};

struct NCCLContextMap {
  std::unordered_map<int, NCCLContext> contexts_;
  std::vector<int> order_;

  NCCLContextMap(const std::vector<platform::Place> &places) {
    order_.reserve(places.size());
    for (auto &p : places) {
      int dev_id = boost::get<CUDAPlace>(p).device;
      order_.emplace_back(dev_id);
      contexts_.emplace(dev_id, NCCLContext(dev_id));
    }
    PADDLE_ENFORCE_EQ(
        order_.size(), contexts_.size(),
        "RCCL Context Map does not support contain two or more same device");

    std::vector<rcclComm_t> comms;
    comms.resize(order_.size());

    PADDLE_ENFORCE(platform::dynload::rcclCommInitAll(
        &comms[0], static_cast<int>(order_.size()), &order_[0]));

    int i = 0;
    for (auto &dev_id : order_) {
      contexts_.at(dev_id).comm_ = comms[i++];
    }
  }

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

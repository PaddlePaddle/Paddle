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
#include <memory>
#include <string>
#include <thread>  // NOLINT
#include <typeindex>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/platform/dynload/nccl.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/float16.h"

#define NCCL_ID_VARNAME "NCCLID"

namespace paddle {
namespace platform {

inline ncclDataType_t ToNCCLDataType(framework::proto::VarType::Type type) {
  if (type == framework::proto::VarType::FP32) {
    return ncclFloat;
  } else if (type == framework::proto::VarType::FP64) {
    return ncclDouble;
  } else if (type == framework::proto::VarType::INT32) {
    return ncclInt;
  } else if (type == framework::proto::VarType::INT64) {
    return ncclInt64;
  } else if (type == framework::proto::VarType::FP16) {
    return ncclFloat16;
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
  static std::mutex& NCCLMutex() {
    static std::mutex mtx;
    return mtx;
  }

  inline NCCLGroupGuard() {
    NCCLMutex().lock();
    PADDLE_ENFORCE(dynload::ncclGroupStart());
  }

  inline ~NCCLGroupGuard() {
    PADDLE_ENFORCE(dynload::ncclGroupEnd());
    NCCLMutex().unlock();
  }
};

struct NCCLContext {
  std::unique_ptr<CUDADeviceContext> ctx_;
  ncclComm_t comm_;

  explicit NCCLContext(int dev_id)
      : ctx_(new CUDADeviceContext(CUDAPlace(dev_id))), comm_{nullptr} {}

  cudaStream_t stream() const { return ctx_->stream(); }
  ncclComm_t comm() const { return comm_; }

  int device_id() const {
    return boost::get<platform::CUDAPlace>(ctx_->GetPlace()).device;
  }
};

struct NCCLContextMap {
  std::unordered_map<int, NCCLContext> contexts_;
  std::vector<int> order_;

  explicit NCCLContextMap(const std::vector<platform::Place>& places,
                          ncclUniqueId* nccl_id = nullptr,
                          size_t num_trainers = 1, size_t trainer_id = 0) {
    PADDLE_ENFORCE(!places.empty());
    order_.reserve(places.size());
    for (auto& p : places) {
      int dev_id = boost::get<CUDAPlace>(p).device;
      order_.emplace_back(dev_id);
      contexts_.emplace(dev_id, NCCLContext(dev_id));
    }
    PADDLE_ENFORCE_EQ(
        order_.size(), contexts_.size(),
        "NCCL Context Map does not support contain two or more same device");

    std::unique_ptr<ncclComm_t[]> comms(new ncclComm_t[order_.size()]);
    // if num_trainers == 1, should create a new nccl id for local comms.
    if (num_trainers == 1 && nccl_id == nullptr) {
      std::lock_guard<std::mutex> guard(NCCLGroupGuard::NCCLMutex());
      PADDLE_ENFORCE(platform::dynload::ncclCommInitAll(
          comms.get(), static_cast<int>(order_.size()), order_.data()));
    } else {
      PADDLE_ENFORCE_NOT_NULL(nccl_id);
      {
        int nranks = num_trainers * order_.size();
        NCCLGroupGuard gurad;
        for (size_t i = 0; i < order_.size(); ++i) {
          int gpu_id = order_[i];
          int rank;
          if (order_.size() > 1) {
            rank = trainer_id * order_.size() + i;
          } else {
            rank = trainer_id;
          }
          VLOG(3) << "init nccl rank: " << rank << " nranks: " << nranks
                  << " gpu id: " << gpu_id;
          PADDLE_ENFORCE(cudaSetDevice(gpu_id));
          PADDLE_ENFORCE(platform::dynload::ncclCommInitRank(
              comms.get() + i, nranks, *nccl_id, rank));
        }
      }
    }
    int i = 0;
    for (auto& dev_id : order_) {
      contexts_.at(dev_id).comm_ = comms[i++];
    }
  }

  NCCLContextMap(const NCCLContextMap& other) = delete;
  NCCLContextMap& operator=(const NCCLContextMap& other) = delete;

  CUDADeviceContext* DevCtx(int dev_id) const { return at(dev_id).ctx_.get(); }

  CUDADeviceContext* DevCtx(platform::Place p) const {
    return DevCtx(boost::get<CUDAPlace>(p).device);
  }

  const NCCLContext& at(platform::Place p) const {
    return this->at(boost::get<CUDAPlace>(p).device);
  }

  const NCCLContext& at(int dev_id) const { return contexts_.at(dev_id); }

  void WaitAll() {
    for (auto& p : contexts_) {
      p.second.ctx_->Wait();
    }
  }
};

// a NCCL communication group contains NCCL communicators associated to
// a global ncclUniqueId. The instances are created and reversed
// in the NCCLContextPool singleton, which assigns unique group ids for them
class NCCLCommGroup {
 public:
  NCCLCommGroup(ncclUniqueId* nccl_id, int nranks, int group_id) {
    PADDLE_ENFORCE_NOT_NULL(nccl_id);
    PADDLE_ENFORCE_GT(nranks, 1);
    nccl_id_ = nccl_id;
    nranks_ = nranks;
    group_id_ = group_id;
  }
  ~NCCLCommGroup() {}

  // bind the global rank id to the device id, or rather the local rank
  void InitRank(int dev_id, int rank_id) {
    std::lock_guard<std::mutex> lg(mutex_);

    PADDLE_ENFORCE(dev_id >= 0 && rank_id >= 0,
                   "invalid arguments: dev_id=%d, rank_id=%d", dev_id, rank_id);
    PADDLE_ENFORCE(ctx_map_.count(dev_id) == 0,
                   "device %d is used in the communication group %d", dev_id,
                   group_id_);

    std::shared_ptr<NCCLContext> nccl_ctx(new NCCLContext(dev_id));

    PADDLE_ENFORCE(cudaSetDevice(dev_id));
    PADDLE_ENFORCE(platform::dynload::ncclCommInitRank(
        &(nccl_ctx->comm_), nranks_, *nccl_id_, rank_id));

    ctx_map_.emplace(dev_id, nccl_ctx);
  }

  void InitRank(Place place, int rank_id) {
    InitRank(boost::get<CUDAPlace>(place).device, rank_id);
  }

  int group_id() const { return group_id_; }

  CUDADeviceContext* DevCtx(int dev_id) const { return at(dev_id).ctx_.get(); }

  CUDADeviceContext* DevCtx(platform::Place p) const {
    return DevCtx(boost::get<CUDAPlace>(p).device);
  }

  const NCCLContext& at(int dev_id) const { return *ctx_map_.at(dev_id); }

  const NCCLContext& at(Place p) const {
    return at(boost::get<CUDAPlace>(p).device);
  }

 private:
  int group_id_;

  ncclUniqueId* nccl_id_;

  int nranks_;

  std::mutex mutex_;

  // Mapping dev_id to NCCLContext
  std::unordered_map<int, std::shared_ptr<NCCLContext>> ctx_map_;

  NCCLCommGroup(const NCCLCommGroup& other) = delete;
  NCCLCommGroup& operator=(const NCCLCommGroup& other) = delete;
};

// a singleton NCCL context pool reserves NCCLContext instances
// which are grouped by communication group
class NCCLContextPool {
 public:
  static NCCLContextPool& Instance() {
    static NCCLContextPool pool;
    return pool;
  }

  // create a communication group with a ncclUniqueId and the
  // world rank number. An integer group id, starting at 0
  // and increasing by 1, is generated for retrieval.
  int CreateCommGroup(ncclUniqueId* nccl_id, int nranks) {
    PADDLE_ENFORCE_NOT_NULL(nccl_id);
    PADDLE_ENFORCE(nranks > 1, "NCCL ranks must be greater than 1");

    groups_.emplace_back(
        new NCCLCommGroup(nccl_id, nranks, static_cast<int>(groups_.size())));

    return static_cast<int>(groups_.size() - 1);
  }

  // retrieve a communication group by the group id
  NCCLCommGroup* Group(int gid) const {
    PADDLE_ENFORCE_LT(gid, static_cast<int>(groups_.size()));
    return groups_[gid].get();
  }

  // NOTE: short cut for group 0
  void InitRank(int dev_id, int rank_id) const {
    Group(0)->InitRank(dev_id, rank_id);
  }

  // NOTE: short cut for group 0
  void InitRank(Place place, int rank_id) const {
    Group(0)->InitRank(place, rank_id);
  }

  // NOTE: short cut for group 0
  CUDADeviceContext* DevCtx(int dev_id) const {
    return Group(0)->DevCtx(dev_id);
  }

  // NOTE: short cut for group 0
  CUDADeviceContext* DevCtx(platform::Place p) const {
    return Group(0)->DevCtx(p);
  }

  // NOTE: short cut for group 0
  const NCCLContext& at(int dev_id) const { return Group(0)->at(dev_id); }

  // NOTE: short cut for group 0
  const NCCLContext& at(Place p) const { return Group(0)->at(p); }

 private:
  std::vector<std::shared_ptr<NCCLCommGroup>> groups_;

  NCCLContextPool() {}
  NCCLContextPool(const NCCLContextPool& other) = delete;
  NCCLContextPool& operator=(const NCCLContextPool& other) = delete;
};

}  // namespace platform
}  // namespace paddle

#endif

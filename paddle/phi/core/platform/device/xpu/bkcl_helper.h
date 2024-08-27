// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#if defined(PADDLE_WITH_XPU_BKCL)
#pragma once

#include <stdio.h>

#include <memory>
#include <string>
#include <thread>  // NOLINT
#include <typeindex>
#include <unordered_map>
#include <vector>

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/platform/collective_helper.h"
#include "paddle/phi/core/platform/device_context.h"
#include "xpu/bkcl.h"
#include "xpu/runtime.h"

#define BKCL_ID_VARNAME "BKCLID"

namespace paddle {
namespace platform {

inline int GetBKCLRankID(BKCLContext_t comm) {
  return reinterpret_cast<int *>(comm)[0];
}

inline int GetBKCLDevID(BKCLContext_t comm) {
  return reinterpret_cast<int *>(comm)[1];
}

inline int GetBKCLNRanks(BKCLContext_t comm) {
  return reinterpret_cast<int *>(comm)[2];
}

class BKCLGroupGuard {
 public:
  static std::mutex &BKCLMutex() {
    static std::mutex mtx;
    return mtx;
  }

  inline BKCLGroupGuard() {
    BKCLMutex().lock();
    PADDLE_ENFORCE_XPU_SUCCESS(bkcl_group_start());
  }

  inline ~BKCLGroupGuard() PADDLE_MAY_THROW {
    PADDLE_ENFORCE_XPU_SUCCESS(bkcl_group_end());
    BKCLMutex().unlock();
  }
};

struct BKCLContext {
  std::unique_ptr<phi::XPUContext> ctx_;
  BKCLContext_t comm_;

  explicit BKCLContext(int dev_id)
      : ctx_(new phi::XPUContext(phi::XPUPlace(dev_id))), comm_{nullptr} {}

  XPUStream stream() const { return ctx_->stream(); }
  BKCLContext_t comm() const { return comm_; }

  int device_id() const { return ctx_->GetPlace().device; }
};

struct InitBKCLPara {
  BKCLUniqueId *bkcl_id;
  int rank;
  int nranks;
  int dev_id;
  BKCLContext_t *ctx;
};

static void *init_bkcl_context_func(void *args) {
  struct InitBKCLPara *para = (struct InitBKCLPara *)args;
  platform::SetXPUDeviceId(para->dev_id);
  PADDLE_ENFORCE_XPU_SUCCESS(
      bkcl_init_rank(para->ctx, para->rank, para->nranks, para->bkcl_id));
  return nullptr;
}

struct BKCLContextMap {
  std::unordered_map<int, BKCLContext> contexts_;
  std::vector<int> order_;
  std::vector<phi::Place> places_;
  size_t num_trainers_;
  size_t trainer_id_;
  BKCLUniqueId *bkcl_id_;

  explicit BKCLContextMap(const std::vector<phi::Place> &places,
                          BKCLUniqueId *bkcl_id = nullptr,
                          size_t num_trainers = 1,
                          size_t trainer_id = 0) {
    places_ = places;
    bkcl_id_ = bkcl_id;
    num_trainers_ = num_trainers;
    trainer_id_ = trainer_id;
  }

  // Synchronization is required and can only be initialized with
  // multithreading.
  int init() {
    PADDLE_ENFORCE_EQ(
        !places_.empty(),
        true,
        common::errors::InvalidArgument("The BKCL place should not be empty."));
    order_.reserve(places_.size());
    for (auto &p : places_) {
      int dev_id = p.device;
      order_.emplace_back(dev_id);
      contexts_.emplace(dev_id, BKCLContext(dev_id));
    }
    PADDLE_ENFORCE_EQ(
        order_.size(),
        contexts_.size(),
        common::errors::Unavailable("BKCL Context Map does not support "
                                    "contain two or more same device"));

    std::unique_ptr<BKCLContext_t[]> comms(new BKCLContext_t[order_.size()]);
    std::unique_ptr<InitBKCLPara[]> paras(new InitBKCLPara[order_.size()]);
    std::unique_ptr<pthread_t[]> pids(new pthread_t[order_.size()]);
    BKCLResult_t ret;
    BKCLUniqueId id;
    // if num_trainers == 1, should create a new bkcl id for local comms.
    if (num_trainers_ == 1 && bkcl_id_ == nullptr) {
      ret = bkcl_get_unique_id(&id);
      PADDLE_ENFORCE_EQ(BKCL_SUCCESS,
                        ret,
                        common::errors::PreconditionNotMet(
                            "bkcl get unique id failed [%d]", ret));
      bkcl_id_ = &id;
    }
    PADDLE_ENFORCE_NOT_NULL(
        bkcl_id_,
        common::errors::InvalidArgument("The BKCL id should not be null."));
    {
      int nranks = num_trainers_ * order_.size();
      for (size_t i = 0; i < order_.size(); ++i) {
        int rank;
        if (order_.size() > 1) {
          rank = trainer_id_ * order_.size() + i;
        } else {
          rank = trainer_id_;
        }

        paras[i].rank = rank;
        paras[i].nranks = nranks;
        paras[i].dev_id = order_[i];
        paras[i].bkcl_id = bkcl_id_;
        paras[i].ctx = &comms[i];
        PADDLE_ENFORCE_EQ(pthread_create(&pids[i],
                                         nullptr,
                                         init_bkcl_context_func,
                                         reinterpret_cast<void *>(&paras[i])),
                          0,
                          common::errors::External("pthread_create failed"));
      }
      for (size_t i = 0; i < order_.size(); i++) {
        pthread_join(pids[i], nullptr);
      }
    }
    int i = 0;
    for (auto &dev_id : order_) {
      contexts_.at(dev_id).comm_ = comms[i++];
    }
    return 0;
  }

  BKCLContextMap(const BKCLContextMap &other) = delete;
  BKCLContextMap &operator=(const BKCLContextMap &other) = delete;

  phi::XPUContext *DevCtx(int dev_id) const { return at(dev_id).ctx_.get(); }

  phi::XPUContext *DevCtx(phi::Place p) const { return DevCtx(p.device); }

  const BKCLContext &at(phi::Place p) const { return this->at(p.device); }

  const BKCLContext &at(int dev_id) const { return contexts_.at(dev_id); }

  void WaitAll() {
    for (auto &p : contexts_) {
      p.second.ctx_->Wait();
    }
  }
};

inline std::string GetFlatBKCLVarName(size_t pos) {
  if (pos == 0) {
    return BKCL_ID_VARNAME;
  }
  return string::Sprintf("%s_%d", BKCL_ID_VARNAME, static_cast<int>(pos));
}

}  // namespace platform
}  // namespace paddle

#endif  // PADDLE_WITH_XPU_BKCL
#endif

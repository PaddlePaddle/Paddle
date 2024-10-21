//   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "glog/logging.h"

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/scope.h"
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

inline BKCLDataType ToBKCLDataType(framework::proto::VarType::Type type) {
  if (type == framework::proto::VarType::FP32) {
    return BKCL_FLOAT;
  } else if (type == framework::proto::VarType::INT64) {
    return BKCL_INT64;
  } else if (type == framework::proto::VarType::INT32) {
    return BKCL_INT32;
  } else if (type == framework::proto::VarType::FP64) {
    return BKCL_FLOAT64;
  } else if (type == framework::proto::VarType::FP16) {
    return BKCL_FLOAT16;
  } else if (type == framework::proto::VarType::BF16) {
    return BKCL_BFLOAT16;
  } else if (type == framework::proto::VarType::UINT8) {
    return BKCL_UINT8;
  } else if (type == framework::proto::VarType::BOOL) {
    return BKCL_UINT8;
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "BKCL currently only support FP32, INT64, INT32, FP64, FP16, BF16, "
        "UINT8 and BOOL, "
        "other data types are not supported."));
  }
}

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
        VLOG(1) << "init bkcl rank:" << rank << ", nranks:" << nranks
                << ", xpu_id:" << order_[i];
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

class BKCLCommunicator {
 public:
  BKCLCommunicator() {}
  virtual ~BKCLCommunicator() {}

  BKCLContextMap *DefaultFlatCtx() const {
    if (flat_ctxs_.size() == 0) {
      return nullptr;
    }

    return flat_ctxs_[0].get();
  }

  std::vector<std::unique_ptr<BKCLContextMap>> *GetFlatCtxs() {
    return &flat_ctxs_;
  }

  BKCLContextMap *GetFlatCtx(size_t run_order) const {
    return flat_ctxs_[run_order % flat_ctxs_.size()].get();
  }

  BKCLContextMap *GetRunEnvBKCLCtx(size_t run_order,
                                   bool use_hierarchical_allreduce) const {
    PADDLE_ENFORCE_EQ(use_hierarchical_allreduce,
                      false,
                      common::errors::Unimplemented(
                          "Hierarchical all reduce is not support for XPU"));
    return GetFlatCtx(run_order);
  }

  /*
   *It meets error when allreduce ophandle and sync_batch_norm_op use
   *bkcl_all_reduce
   *parallelly. So create a new bkcl comm for sync_batch_norm_op. And these
   *codes should be polished with a unified bkcl management.
   */
  BKCLContextMap *GetSyncBatchNormCtx(framework::Scope *scope,
                                      const std::vector<phi::Place> &places) {
    auto *bkcl_id_var = scope->FindVar(BKCL_ID_VARNAME);
    if (bkcl_id_var != nullptr) {
      return DefaultFlatCtx();
    }

    if (sync_batch_norm_ctx_.get() == nullptr) {
      sync_batch_norm_ctx_.reset(new BKCLContextMap(places));
      sync_batch_norm_ctx_->init();
    }
    return sync_batch_norm_ctx_.get();
  }

  void InitFlatCtxs(const std::vector<phi::Place> &places,
                    const std::vector<BKCLUniqueId *> &bkcl_ids,
                    size_t trainers_num,
                    size_t trainer_id) {
    if (bkcl_ids.size() == 0) {
      auto ptr = new platform::BKCLContextMap(places);
      ptr->init();
      VLOG(1) << "init local trainer";
      flat_ctxs_.emplace_back(ptr);
    } else {
      PADDLE_ENFORCE_EQ(bkcl_ids.size(),
                        1,
                        common::errors::Unimplemented(
                            "Multi-all-reduce-ring is not support for XPU"));
      for (size_t i = 0; i < bkcl_ids.size(); i++) {
        auto ptr = new platform::BKCLContextMap(
            places, bkcl_ids[i], trainers_num, trainer_id);
        ptr->init();
        VLOG(1) << "init trainer_id:" << trainer_id << ", comm no:" << i;
        flat_ctxs_.emplace_back(ptr);
      }
    }

    // as Executor have no way to use BKCLComm created by ParallelExecutor,
    // we assign all flatten contexts to BKCLCommContext to fix.
    int nranks = static_cast<int>(trainers_num * places.size());
    int nrings = static_cast<int>(flat_ctxs_.size());
    for (int ring_id = 0; ring_id < nrings; ++ring_id) {
      for (size_t p = 0; p < places.size(); ++p) {
        int rank = trainer_id * places.size() + p;
        int dev_id = places[p].device;
        platform::SetXPUDeviceId(dev_id);
        auto &ctx = flat_ctxs_[ring_id]->contexts_.at(dev_id);
        BKCLCommContext::Instance().AssignBKCLComm(
            ctx.comm_, nranks, rank, dev_id, ring_id);
      }
    }
  }

 protected:
  // Support multi bkcl comm on default bkcl ring while BKCLContextMap can't.
  std::vector<std::unique_ptr<BKCLContextMap>> flat_ctxs_;

  // just used for sync_batch_norm op.
  std::unique_ptr<BKCLContextMap> sync_batch_norm_ctx_;
};

}  // namespace platform
}  // namespace paddle

#endif  // PADDLE_WITH_XPU_BKCL
#endif

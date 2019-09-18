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
#include <map>
#include <memory>
#include <set>
#include <string>
#include <thread>  // NOLINT
#include <typeindex>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/scope.h"
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
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::ncclGroupStart());
  }

  inline ~NCCLGroupGuard() {
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::ncclGroupEnd());
    NCCLMutex().unlock();
  }
};

// The communication unit holding a connected ncclComm_t and an independent
// cudaStream_t. The lifetime of the ncclComm_t is maintained by this unit,
// so we should not destroy the ncclComm_t explicitly.
class NCCLContext {
 public:
  NCCLContext(int dev_id, ncclComm_t comm)
      : dev_ctx_(new CUDADeviceContext(CUDAPlace(dev_id))) {
    dev_ctx_->set_nccl_comm(comm);
  }
  virtual ~NCCLContext() {}

  cudaStream_t stream() const { return dev_ctx_->stream(); }

  ncclComm_t comm() const { return dev_ctx_->nccl_comm(); }

  int device_id() const {
    return boost::get<platform::CUDAPlace>(dev_ctx_->GetPlace()).device;
  }

  CUDADeviceContext* dev_ctx() const { return dev_ctx_.get(); }

 protected:
  // CUDADeviceContext is the real container of the ncclComm_t, and it destroys
  // ncclComm_t in its destructor
  // TODO(liuyi05): try to move the destruction of the ncclComm_t out and keep
  // it
  // in the same class with its construction
  std::unique_ptr<CUDADeviceContext> dev_ctx_;

  DISABLE_COPY_AND_ASSIGN(NCCLContext);
};

// A container preserving the mapping from device id to NCCLContext which has
// a corresponding level to "ring" in NCCL communication. Users retrieve the
// NCCLContext instance from it by the device id of a given CUDAPlace.
//
// The status of this container may be diverse in different training modes:
//
// 1. Multiprocess (with one thread each) mode:
//    The minimal training unit is a process in this mode, and each process,
//    taking up one device, has a rank id in a communication ring. The
//    NCCLContextMap holds the ring information and contains only one
//    NCCLContext.
//
// 2. One process with multithread mode:
//    In this mode, the minimal training unit is a thread, and all threads are
//    working in one process. Within a process, we can initialize all
//    NCCLContext instances once in a ring and put them all to a single
//    NCCLContextMap.
//
// 3. Multiprocess with multithread mode:
//    This mode is in the case of training with multimachine. The ring consists
//    of all threads across processes over machines. The NCCLContext is created
//    by each thread, taking up one device, and all NCCLContext instances in
//    one process preserved in a single NCCLContextMap.
//
// As one training procedure may use several rings to communicate, there could
// be several NCCLContextMaps.
//
// TODO(liuyi05): Do not expose this class. Do retrieve NCCLContext from
// a global singleton
class NCCLContextMap {
 public:
  explicit NCCLContextMap(const std::vector<platform::Place>& places,
                          ncclUniqueId* nccl_id = nullptr,
                          size_t num_trainers = 1, size_t trainer_id = 0);

  const std::map<int, std::unique_ptr<NCCLContext>>& contexts() const {
    return contexts_;
  }

  CUDADeviceContext* DevCtx(int dev_id) const { return at(dev_id)->dev_ctx(); }

  CUDADeviceContext* DevCtx(platform::Place p) const {
    return DevCtx(boost::get<CUDAPlace>(p).device);
  }

  NCCLContext* at(platform::Place p) const {
    return at(boost::get<CUDAPlace>(p).device);
  }

  NCCLContext* at(int dev_id) const { return contexts_.at(dev_id).get(); }

  size_t size() const { return contexts_.size(); }

  size_t count(int dev_id) const { return contexts_.count(dev_id); }

  NCCLContext* GetSingle() const {
    PADDLE_ENFORCE_EQ(contexts_.size(), 1);
    return contexts_.begin()->second.get();
  }

  void WaitAll() {
    for (auto& p : contexts_) {
      p.second->dev_ctx()->Wait();
    }
  }

 private:
  std::map<int, std::unique_ptr<NCCLContext>> contexts_;
  std::vector<int> order_;

  ncclUniqueId* nccl_id_;
  int ntrainers_;
  int trainer_id_;

  DISABLE_COPY_AND_ASSIGN(NCCLContextMap);
};

inline std::string GetFlatNCCLVarName(size_t pos) {
  if (pos == 0) {
    return NCCL_ID_VARNAME;
  }
  return string::Sprintf("%s_%d", NCCL_ID_VARNAME, static_cast<int>(pos));
}

inline std::string GetHierarchicalExterNCCLVarName(size_t pos) {
  return string::Sprintf("Hierarchical_exter_%s_%d", NCCL_ID_VARNAME,
                         static_cast<int>(pos));
}
inline std::string GetHierarchicalInterNCCLVarName(size_t pos) {
  return string::Sprintf("Hierarchical_inter_%s_%d", NCCL_ID_VARNAME,
                         static_cast<int>(pos));
}

// A singleton of NCCL communicator context. It creates NCCLContext in
// communication rings and maintains lifetime of all NCCL resources.
// Each ring has an unique ID, which could be used to retrieve
// the NCCLContextMap. So if both ring id and device id are given, the
// NCCLContext instance could be determined.
//
// For compatibility, we preserve the range [0, 1000) of ids for default
// rings and users could use ids beyond 1000.
//
// E.g. for a hierarchical communication case,
//
//    11 - 12   21 - 22
//     |    |    |    |
//    13 - 14 - 23 - 24
//          |    |
//    31 - 32 - 41 - 42
//     |    |    |    |
//    33 - 34   43 - 44
//
// we create (14,23,32,41) as the top ring, and (11,12,13,14), (21,22,23,24),
// (31,32,33,34), (41,42,43,44) as bottoms rings respectively.
//
// We could also use a single communication ring for the flatten case
class NCCLCommunicator {
 public:
  static NCCLCommunicator& Instance() {
    static NCCLCommunicator instance;
    return instance;
  }

  NCCLCommunicator() = default;
  ~NCCLCommunicator() = default;

  NCCLContextMap* GetContextMap(int ring_id) const {
    return ring2map_.at(ring_id).get();
  }

  NCCLContextMap* GetDefaultContextMap() const { return GetContextMap(0); }

  // done
  NCCLContextMap* DefaultFlatCtx() const {
    if (flat_ctxs_.size() == 0) {
      return nullptr;
    }

    return flat_ctxs_[0].get();
  }

  // check usage
  std::vector<std::unique_ptr<NCCLContextMap>>* GetFlatCtxs() {
    return &flat_ctxs_;
  }

  // check usage
  NCCLContextMap* GetFlatCtx(size_t run_order) const {
    return flat_ctxs_[run_order % flat_ctxs_.size()].get();
  }

  // check usage
  NCCLContextMap* GetRunEnvNCCLCtx(size_t run_order,
                                   bool use_hierarchical_allreduce) const {
    if (!use_hierarchical_allreduce) {
      return GetFlatCtx(run_order);
    }

    return GetHierarchicalInterCtx(run_order);
  }

  // TODO(liuyi05)
  /*
   *When nccl inits nccl comm using ncclCommInitAll, it meets error when
   *allreduce ophandle and sync_batch_norm_op use ncclallreduce parallelly. So
   *create a new nccl comm for sync_batch_norm_op. And these codes should be
   *polished with a unified nccl management.
  */
  NCCLContextMap* GetSyncBatchNormCtx(
      framework::Scope* scope, const std::vector<platform::Place>& places) {
    auto* nccl_id_var = scope->FindVar(NCCL_ID_VARNAME);
    if (nccl_id_var != nullptr) {
      return DefaultFlatCtx();
    }

    if (sync_batch_norm_ctx_.get() == nullptr) {
      sync_batch_norm_ctx_.reset(new NCCLContextMap(places));
    }
    return sync_batch_norm_ctx_.get();
  }

  // call InitNCCLContexts
  void InitFlatCtxs(const std::vector<platform::Place>& places,
                    const std::vector<ncclUniqueId*>& nccl_ids,
                    size_t trainers_num, size_t trainer_id) {
    if (nccl_ids.size() == 0) {
      auto ptr = new platform::NCCLContextMap(places);
      VLOG(1) << "init local trainer";
      flat_ctxs_.emplace_back(ptr);
      return;
    }

    for (size_t i = 0; i < nccl_ids.size(); i++) {
      auto ptr = new platform::NCCLContextMap(places, nccl_ids[i], trainers_num,
                                              trainer_id);
      VLOG(1) << "init trainer_id:" << trainer_id << ", comm no:" << i;
      flat_ctxs_.emplace_back(ptr);
    }
  }

  void InitNCCLContexts(const std::vector<Place>& places,
                        ncclUniqueId* nccl_id = nullptr, int ntrainers = 1,
                        int trainer_id = 0, int ring_id = 0);

  void InitAllNCCLContexts(const std::vector<Place>& places, int ring_id = 0) {
    InitNCCLContexts(places, nullptr, 1, 0, ring_id);
  }

  void InitNCCLContext(ncclUniqueId* nccl_id, int nranks, int rank, Place place,
                       int ring_id = 0) {
    InitNCCLContexts({place}, nccl_id, nranks, rank, ring_id);
  }

  // check necessary
  void InitHierarchicalCtxs(const std::vector<platform::Place>& places,
                            const std::vector<ncclUniqueId*>& inter_nccl_ids,
                            const std::vector<ncclUniqueId*>& exter_nccl_ids,
                            size_t trainers_num, size_t trainer_id,
                            size_t inter_trainers_num,
                            size_t exter_trainers_num) {
    PADDLE_ENFORCE_EQ(trainers_num, inter_trainers_num * exter_trainers_num,
                      "trainers_num:%llu != inter_trainers_num:%llu * "
                      "exter_trainers_num:%llu",
                      trainers_num, inter_trainers_num, exter_trainers_num);

    PADDLE_ENFORCE_GT(inter_trainers_num, 1, "inter_trainers_num:%llu must > 1",
                      inter_trainers_num);

    int inter_trainer_id = trainer_id % inter_trainers_num;
    for (size_t i = 0; i < inter_nccl_ids.size(); i++) {
      VLOG(1) << "init inter_trainer_id:" << inter_trainer_id
              << ", comm no:" << i;
      auto local = new NCCLContextMap(places, inter_nccl_ids[i],
                                      inter_trainers_num, inter_trainer_id);

      h_inter_ctxs_.emplace_back(local);
    }

    int exter_trainer_id = -1;
    if (trainer_id % inter_trainers_num == 0) {
      exter_trainer_id = trainer_id / inter_trainers_num;
    }

    if (exter_trainer_id >= 0) {
      for (size_t i = 0; i < exter_nccl_ids.size(); i++) {
        auto ex = new NCCLContextMap(places, exter_nccl_ids[i],
                                     exter_trainers_num, exter_trainer_id);
        VLOG(1) << "init exter_trainer_id:" << exter_trainer_id
                << ", comm no:" << i;
        h_exter_ctxs_.emplace_back(ex);
      }
    }
  }

  // check
  bool NeedExterAllReduce() const { return h_exter_ctxs_.size() > 0; }

  NCCLContextMap* GetHierarchicalInterCtx(size_t run_order) const {
    PADDLE_ENFORCE(h_inter_ctxs_.size() > 0,
                   "must init hierarchical ctxs first!");
    return h_inter_ctxs_[run_order % h_inter_ctxs_.size()].get();
  }

  NCCLContextMap* GetHierarchicalExterCtx(size_t run_order) const {
    PADDLE_ENFORCE(h_exter_ctxs_.size() > 0,
                   "must init hierarchical ctxs first!");
    return h_exter_ctxs_[run_order % h_exter_ctxs_.size()].get();
  }

  std::vector<std::unique_ptr<NCCLContextMap>>* GetHierarchicalInterCtxs() {
    return &h_inter_ctxs_;
  }

  std::vector<std::unique_ptr<NCCLContextMap>>* GetHierarchicalExterCtxs() {
    return &h_exter_ctxs_;
  }

  // retrieve a communicator by the ring id
  NCCLContext* Get(int ring_id) const {
    PADDLE_ENFORCE_GT(ring2map_.count(ring_id), 0,
                      "comunicator in ring id %d has not been initialized",
                      ring_id);
    return ring2map_.at(ring_id)->GetSingle();
  }

  // retrieve a communicator by the ring id and the device id
  NCCLContext* Get(int ring_id, int dev_id) const {
    PADDLE_ENFORCE_GT(ring2map_.count(ring_id), 0,
                      "comunicator of ring id %d has not been initialized",
                      ring_id);
    PADDLE_ENFORCE_GT(
        ring2map_.at(ring_id)->count(dev_id), 0,
        "comunicator at device id %d has not been initialized in ring %d",
        dev_id, ring_id);
    return ring2map_.at(ring_id)->at(dev_id);
  }

  // retrieve a communicator by the ring id and place
  NCCLContext* Get(int ring_id, Place place) const {
    return Get(ring_id, boost::get<CUDAPlace>(place).device);
  }

 protected:
  // we preserve the first 100 ring ids for system use
  enum SystemRingID { DEFAULT = 0, SYNC_BN = 1 };

  std::once_flag once_flag_;
  std::mutex ring2map_mutex_;
  // std::map<int, std::map<int, std::unique_ptr<NCCLContext>>> ring2map_;
  // ring id to NCCLContextMap
  std::map<int, std::unique_ptr<NCCLContextMap>> ring2map_;

  void ReleaseNCCLResource();

  // Support multi nccl comm on default nccl ring while NCCLContextMap can't.
  std::vector<std::unique_ptr<NCCLContextMap>> flat_ctxs_;

  // h_inter_ctxs_ and h_exter_ctxs_ are for 2d allreduce.
  // And h_exter_ctxs_ can support multi comm too.
  std::vector<std::unique_ptr<NCCLContextMap>> h_inter_ctxs_;
  std::vector<std::unique_ptr<NCCLContextMap>> h_exter_ctxs_;

  // just used for sync_batch_norm op.
  std::unique_ptr<NCCLContextMap> sync_batch_norm_ctx_;

  DISABLE_COPY_AND_ASSIGN(NCCLCommunicator);
};

}  // namespace platform
}  // namespace paddle
#endif

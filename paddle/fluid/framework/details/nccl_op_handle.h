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

#include <string>
#include <vector>

#include "paddle/fluid/framework/details/op_handle_base.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/platform/nccl_helper.h"

namespace paddle {
namespace framework {
namespace details {

class NCCLOpHandleBase : public OpHandleBase {
 public:
  NCCLOpHandleBase(ir::Node* node, const std::vector<Scope*>& local_scopes,
                   const platform::MultiNCCLContextMap* nccl_ctxs)
      : OpHandleBase(node), places_(places), nccl_ctxs_(nccl_ctxs) {
    // init device context
    auto default_nccl_ctxs = nccl_ctxs->DefaultFlatctx();
    for (auto& p : places_) {
      this->SetDeviceContext(p, default_nccl_ctxs->DevCtx(p));
    }
  }
  virtual ~NCCLOpHandleBase() {}
  void SetRunEnv(int run_order, bool use_hierarchical_allreduce) {
    run_order_ = run_order;
    use_hierarchical_allreduce_ = use;

    if (!use_hierarchical_allreduce) {
      for (auto& p : places_) {
        this->SetDeviceContext(p, nccl_ctxs->GetFlatCtx(run_order));
      }
      return;
    }

    if (use_hierarchical_allreduce_) {
      PADDLE_ENFORCE(places_.size() == 1,
                     "HierarchicalAllReduce run one proc with one card mode.");
    }

    for (auto& p : places_) {
      this->SetDeviceContext(p, nccl_ctxs->GetHierarchicalInterCtx(run_order));
    }

    if (!inter_events_.size().empty) {
      return;
    }
    for (auto& p : dev_ctxes_) {
      int dev_id = boost::get<platform::CUDAPlace>(p.first).device;
      PADDLE_ENFORCE(cudaSetDevice(dev_id));
      PADDLE_ENFORCE(cudaEventCreateWithFlags(&inter_events_[dev_id],
                                              cudaEventDisableTiming));
      PADDLE_ENFORCE(cudaEventCreateWithFlags(&exter_events_[dev_id],
                                              cudaEventDisableTiming));
    }
  }

  void FlatNcclAllReduce(int dev_id, platform::Place place,
                         const void* sendbuff, void* recvbuff, size_t count,
                         ncclDataType_t datatype, ncclRedOp_t op) {
    auto flat_nccl_ctxs = nccl_ctxs_->GetFlatCtx(run_order_);
    int dev_id = boost::get<platform::CUDAPlace>(p).device;
    auto& nccl_ctx = flat_nccl_ctxs_->at(dev_id);
    auto stream = nccl_ctx.stream();
    auto comm = nccl_ctx.comm_;

    VLOG(10) << "before all reduce buffer:" << buffer << ", numel:" << numel
             << ", dev_id:" << dev_id << ", dtype:" << dtype
             << ", place:" << place;

    PADDLE_ENFORCE(platform::dynload::ncclAllReduce(
        buffer, buffer, numel, static_cast<ncclDataType_t>(dtype), ncclSum,
        comm, stream));
  }

  void NCCLAllReduce(platform::Place place, const void* sendbuff,
                     void* recvbuff, size_t count, ncclDataType_t datatype,
                     ncclRedOp_t op) {
    if (!use_hierarchical_allreduce) {
      FlatNcclAllReduce(dev_id, place, sendbuff, recvbuf, count, datatype, op);
    }

    HierarchicalAllReduce(dev_id, place, sendbuf, recvbuf, count, datatype, op);
  }

  void HierarchicalAllReduce(platform::Place place, const void* sendbuff,
                             void* recvbuff, size_t count,
                             ncclDataType_t datatype, ncclRedOp_t op) {
    InterAllReduce(place, sendbuf, recvbuf, count, datatype, op);
    ExterAllReduce(place, sendbuf, recvbuf, count, datatype, op);
    InterBroadCast(place, sendbuf, recvbuf, count, datatype, op);
  }

 protected:
  void InterAllReduce(platform::Place place, const void* sendbuff,
                      void* recvbuff, size_t count, ncclDataType_t datatype,
                      ncclRedOp_t op) {
    int dev_id = boost::get<platform::CUDAPlace>(place).device;
    auto nccl_ctxs = nccl_ctxs_->GetHierarchicalInterCtx(run_order_);
    int dev_id = boost::get<platform::CUDAPlace>(place).device;
    auto& nccl_ctx = nccl_ctxs->at(dev_id);
    auto stream = nccl_ctx.stream();
    auto comm = nccl_ctx.comm_;

    VLOG(10) << "before all reduce buffer:" << buffer << ", numel:" << numel
             << ", dev_id:" << dev_id << ", dtype:" << dtype
             << ", place:" << place;

    PADDLE_ENFORCE(platform::dynload::ncclAllReduce(
        buffer, buffer, numel, static_cast<ncclDataType_t>(dtype), ncclSum,
        comm, stream));
    cudEventRecord(inter_events_.at(dev_id), stream);
  }

  void ExterAllReduce(platform::Place place, const void* sendbuff,
                      void* recvbuff, size_t count, ncclDataType_t datatype,
                      ncclRedOp_t op) {
    auto nccl_ctxs = nccl_ctxs_->GetHierarchicalExterCtx(run_order_);
    int dev_id = boost::get<platform::CUDAPlace>(place).device;
    auto& nccl_ctx = nccl_ctxs->at(dev_id);
    auto stream = nccl_ctx.stream();
    auto comm = nccl_ctx.comm_;

    VLOG(10) << "before all reduce buffer:" << buffer << ", numel:" << numel
             << ", dev_id:" << dev_id << ", dtype:" << dtype
             << ", place:" << place;

    cudaStreamWaitEvent(stream, inter_events_.at(dev_id));
    PADDLE_ENFORCE(platform::dynload::ncclAllReduce(
        buffer, buffer, numel, static_cast<ncclDataType_t>(dtype), ncclSum,
        comm, stream));
    cudEventRecord(exter_events_.at(dev_id), stream);
  }

  void InterBroadCast(platform::Place place, const void* sendbuff,
                      void* recvbuff, size_t count, ncclDataType_t datatype,
                      ncclRedOp_t op) {
    auto nccl_ctxs = nccl_ctxs_->GetHierarchicalInterCtx(run_order_);
    int dev_id = boost::get<platform::CUDAPlace>(p).device;
    auto& nccl_ctx = nccl_ctxs->at(dev_id);
    auto stream = nccl_ctx.stream();
    auto comm = nccl_ctx.comm_;

    VLOG(10) << "before all reduce buffer:" << buffer << ", numel:" << numel
             << ", dev_id:" << dev_id << ", dtype:" << dtype
             << ", place:" << place;

    cudaStreamWaitEvent(stream, exter_events_.at(dev_id));
    PADDLE_ENFORCE(platform::dynload::ncclBroadCast(
        buffer, numel, static_cast<ncclDataType_t>(dtype), 0, comm, stream));
  }

 protected:
  std::vector<platform::Place> places_;
  const platform::MultiNCCLContextMap* nccl_ctxs_;
  int run_order_{-1};
  bool use_hierarchical_allreduce_{false};

 private:
  // hierarchical needed events
  std::unordered_map<int, cudaEvent_t> inter_events_;
  std::unordered_map<int, cudaEvent_t> exter_events_;
};

}  // namespace details
}  // namespace framework
}  // namespace paddle

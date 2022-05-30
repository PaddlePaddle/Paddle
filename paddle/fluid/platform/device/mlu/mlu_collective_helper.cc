/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#if defined(PADDLE_WITH_CNCL)
#include <utility>
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/mlu/enforce.h"

namespace paddle {
namespace platform {

class CNCLCommImpl : public CNCLComm {
 public:
  void set_ring_id(int ring_id) { ring_id_ = ring_id; }
  int ring_id() const override { return ring_id_; }

  void set_nranks(int nranks) { nranks_ = nranks; }
  int nranks() const override { return nranks_; }

  void set_rank(int rank) { rank_ = rank; }
  int rank() const override { return rank_; }

  int device_id() const override { return dev_ctx_->GetPlace().device; }

  void set_comm(cnclComm_t comm) { comm_ = comm; }
  cnclComm_t comm() const override { return comm_; }

  mluStream stream() const override { return dev_ctx_->stream(); }

  void set_dev_ctx(std::unique_ptr<MLUDeviceContext>&& dev_ctx) {
    dev_ctx_ = std::move(dev_ctx);
  }
  MLUDeviceContext* dev_context() const override { return dev_ctx_.get(); }

  ~CNCLCommImpl() {
    if (comm_) {
      PADDLE_ENFORCE_MLU_SUCCESS(cnclFreeComm(comm_));
    }
  }

 private:
  int ring_id_;
  int nranks_;
  int rank_;
  cnclComm_t comm_;
  std::unique_ptr<MLUDeviceContext> dev_ctx_;
};

CNCLComm* CNCLCommContext::CreateComm(cnclCliqueId* cncl_id, int nranks,
                                      int rank, int dev_id, int ring_id) {
  PADDLE_ENFORCE_NOT_NULL(cncl_id,
                          platform::errors::InvalidArgument(
                              "The cncl unique id should not be null."));
  PADDLE_ENFORCE_GT(
      nranks, 1,
      platform::errors::InvalidArgument(
          "Expected nranks > 1. But received nranks is %d.", nranks));
  PADDLE_ENFORCE_GE(rank, 0,
                    platform::errors::InvalidArgument(
                        "Expected rank >= 0. But received rank is %d.", rank));
  PADDLE_ENFORCE_LT(
      rank, nranks,
      platform::errors::InvalidArgument(
          "Expected rank < nranks. But received rank is %d, nranks is %d.",
          rank, nranks));
  PADDLE_ENFORCE_GE(
      dev_id, 0,
      platform::errors::InvalidArgument(
          "Expected dev_id >= 0. But received dev_id is %d.", dev_id));

  cnclComm_t comm;
  int dev_list[] = {dev_id};
  int rank_list[] = {rank};
  SetMLUDeviceId(dev_id);
  PADDLE_ENFORCE_MLU_SUCCESS(
      cnclInitComms(&comm, 1, dev_list, rank_list, nranks, cncl_id));

  auto* comm_wrapper = AssignCNCLComm(comm, nranks, rank, dev_id, ring_id);

  VLOG(1) << "cncl communicator of rank " << rank << " in ring " << ring_id
          << " has been created on device " << dev_id;

  std::call_once(once_flag_, []() {
    std::atexit([]() { CNCLCommContext::Instance().ReleaseCNCLComms(); });
  });

  return comm_wrapper;
}

void CNCLCommContext::CreateAllCNCLComms(const std::vector<int>& dev_ids,
                                         int ring_id) {
  PADDLE_ENFORCE_GT(
      dev_ids.size(), 0,
      platform::errors::InvalidArgument("Expected the size of dev_ids > 0. But "
                                        "received the size of dev_ids is %d.",
                                        dev_ids.size()));

  const int kDevices = dev_ids.size();
  cnclComm_t comms[kDevices];
  int* rank_list = new int[kDevices];
  for (int i = 0; i < kDevices; i++) {
    rank_list[i] = i;
  }
  cnclCliqueId clique_id;
  PADDLE_ENFORCE_MLU_SUCCESS(cnclGetCliqueId(&clique_id));
  PADDLE_ENFORCE_MLU_SUCCESS(cnclInitComms(comms, dev_ids.size(),
                                           dev_ids.data(), rank_list,
                                           dev_ids.size(), &clique_id));

  PADDLE_ENFORCE_EQ(comm_map_.count(ring_id), 0,
                    platform::errors::InvalidArgument(
                        "Expected comm_map_.count(ring_id) = 0. But received "
                        "comm_map_.count(ring_id) is %d.",
                        comm_map_.count(ring_id)));
  for (size_t i = 0; i < dev_ids.size(); ++i) {
    AssignCNCLComm(comms[i], dev_ids.size(), i, dev_ids[i], ring_id);
    VLOG(1) << "cncl communicator of rank " << i << " in ring " << ring_id
            << " has been created on device " << dev_ids[i];
  }

  std::call_once(once_flag_, []() {
    std::atexit([]() { CNCLCommContext::Instance().ReleaseCNCLComms(); });
  });
  delete[] rank_list;
}

CNCLComm* CNCLCommContext::AssignCNCLComm(cnclComm_t comm, int nranks, int rank,
                                          int dev_id, int ring_id) {
  std::unique_ptr<MLUDeviceContext> dev_ctx(
      new MLUDeviceContext(MLUPlace(dev_id)));

  CNCLCommImpl* c = new CNCLCommImpl;
  c->set_ring_id(ring_id);
  c->set_nranks(nranks);
  c->set_rank(rank);
  c->set_comm(comm);
  c->set_dev_ctx(std::move(dev_ctx));

  comm_map_mutex_.lock();
  if (comm_map_.count(ring_id) == 0) {
    comm_map_.emplace(ring_id, std::map<int, std::unique_ptr<CNCLComm>>());
  }
  auto& dev2comm = comm_map_[ring_id];

  dev2comm.emplace(dev_id, std::unique_ptr<CNCLComm>(c));
  comm_map_mutex_.unlock();

  if (ring_id == 0) {
    auto* dev_ctx = static_cast<platform::MLUDeviceContext*>(
        platform::DeviceContextPool::Instance().Get(
            platform::MLUPlace(dev_id)));
    dev_ctx->set_cncl_comm(comm);
  }

  return comm_map_[ring_id][dev_id].get();
}

void CNCLCommContext::ReleaseCNCLComms() {
  for (auto& p : comm_map_) {
    for (auto& q : p.second) {
      q.second.reset();
    }
  }
}

}  // namespace platform
}  // namespace paddle
#endif

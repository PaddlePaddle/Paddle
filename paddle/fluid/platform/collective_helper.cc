//   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/platform/collective_helper.h"

#include <utility>

#include "paddle/fluid/memory/allocation/allocator_facade.h"
#include "paddle/fluid/platform/device/device_wrapper.h"
#include "paddle/fluid/platform/device/gpu/gpu_resource_pool.h"

namespace paddle {
namespace platform {
#if defined(PADDLE_WITH_XPU_BKCL)

class BKCLCommImpl : public BKCLComm {
 public:
  void set_ring_id(int ring_id) { ring_id_ = ring_id; }
  int ring_id() const override { return ring_id_; }

  void set_nranks(int nranks) { nranks_ = nranks; }
  int nranks() const override { return nranks_; }

  void set_rank(int rank) { rank_ = rank; }
  int rank() const override { return rank_; }

  int device_id() const override { return dev_ctx_->GetPlace().device; }

  void set_comm(BKCLContext_t comm) { comm_ = comm; }
  BKCLContext_t comm() const override { return comm_; }

  XPUStream stream() const override {
    return dev_ctx_->x_context()->xpu_stream;
  }

  void set_dev_ctx(std::unique_ptr<XPUDeviceContext>&& dev_ctx) {
    dev_ctx_ = std::move(dev_ctx);
  }
  XPUDeviceContext* dev_context() const override { return dev_ctx_.get(); }

 private:
  int ring_id_;
  int nranks_;
  int rank_;
  BKCLContext_t comm_;
  std::unique_ptr<XPUDeviceContext> dev_ctx_;
};

BKCLComm* BKCLCommContext::CreateComm(
    BKCLUniqueId* bkcl_id, int nranks, int rank, int dev_id, int ring_id) {
  PADDLE_ENFORCE_NOT_NULL(bkcl_id,
                          platform::errors::InvalidArgument(
                              "The bkcl unique id should not be null."));
  PADDLE_ENFORCE_GT(
      nranks,
      1,
      platform::errors::InvalidArgument(
          "Expected nranks > 1. But received nranks is %d.", nranks));
  PADDLE_ENFORCE_GE(rank,
                    0,
                    platform::errors::InvalidArgument(
                        "Expected rank >= 0. But received rank is %d.", rank));
  PADDLE_ENFORCE_LT(
      rank,
      nranks,
      platform::errors::InvalidArgument(
          "Expected rank < nranks. But received rank is %d, nranks is %d.",
          rank,
          nranks));
  PADDLE_ENFORCE_GE(
      dev_id,
      0,
      platform::errors::InvalidArgument(
          "Expected dev_id >= 0. But received dev_id is %d.", dev_id));

  BKCLContext_t comm = nullptr;
  platform::SetXPUDeviceId(dev_id);
  PADDLE_ENFORCE_XPU_SUCCESS(bkcl_init_rank(&comm, rank, nranks, bkcl_id));

  auto* comm_wrapper = AssignBKCLComm(comm, nranks, rank, dev_id, ring_id);

  VLOG(1) << "bkcl communicator of rank " << rank << " in ring " << ring_id
          << " has been created on device " << dev_id;

  std::call_once(once_flag_, []() {
    std::atexit([]() { BKCLCommContext::Instance().ReleaseBKCLComms(); });
  });

  return comm_wrapper;
}

BKCLComm* BKCLCommContext::AssignBKCLComm(
    BKCLContext_t comm, int nranks, int rank, int dev_id, int ring_id) {
  std::unique_ptr<XPUDeviceContext> dev_ctx(
      new XPUDeviceContext(XPUPlace(dev_id)));
  dev_ctx->SetAllocator(paddle::memory::allocation::AllocatorFacade::Instance()
                            .GetAllocator(XPUPlace(dev_id))
                            .get());
  dev_ctx->SetHostAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(paddle::platform::CPUPlace())
          .get());
  dev_ctx->SetZeroAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetZeroAllocator(XPUPlace(dev_id))
          .get());
  dev_ctx->SetHostZeroAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetZeroAllocator(paddle::platform::CPUPlace())
          .get());

  BKCLCommImpl* c = new BKCLCommImpl;
  c->set_ring_id(ring_id);
  c->set_nranks(nranks);
  c->set_rank(rank);
  c->set_comm(comm);
  c->set_dev_ctx(std::move(dev_ctx));

  comm_map_mutex_.lock();
  if (comm_map_.count(ring_id) == 0) {
    comm_map_.emplace(ring_id, std::map<int, std::unique_ptr<BKCLComm>>());
  }
  auto& dev2comm = comm_map_[ring_id];

  dev2comm.emplace(dev_id, std::unique_ptr<BKCLComm>(c));
  comm_map_mutex_.unlock();

  if (ring_id == 0) {
    auto* dev_ctx = static_cast<platform::XPUDeviceContext*>(
        platform::DeviceContextPool::Instance().Get(
            platform::XPUPlace(dev_id)));
    dev_ctx->SetBkclContext(comm);
  }

  return comm_map_[ring_id][dev_id].get();
}

void BKCLCommContext::ReleaseBKCLComms() {
  for (auto& p : comm_map_) {
    for (auto& q : p.second) {
      q.second.reset();
    }
  }
}

#endif

}  // namespace platform
}  // namespace paddle

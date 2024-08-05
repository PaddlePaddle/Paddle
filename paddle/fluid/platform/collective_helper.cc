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
#include <vector>

#include "paddle/fluid/platform/device/device_wrapper.h"
#include "paddle/fluid/platform/device/gpu/gpu_resource_pool.h"
#include "paddle/phi/core/memory/allocation/allocator_facade.h"

namespace paddle {
namespace platform {

#if defined(PADDLE_WITH_CUSTOM_DEVICE)
class XCCLCommImpl : public XCCLComm {
 public:
  void set_ring_id(int ring_id) { ring_id_ = ring_id; }
  int ring_id() const override { return ring_id_; }

  void set_nranks(int nranks) { nranks_ = nranks; }
  int nranks() const override { return nranks_; }

  void set_rank(int rank) { rank_ = rank; }
  int rank() const override { return rank_; }

  int device_id() const override { return dev_ctx_->GetPlace().device; }

  void set_comm(phi::ccl::CCLComm comm) { comm_ = comm; }
  phi::ccl::CCLComm comm() const override { return comm_; }

  std::shared_ptr<phi::stream::Stream> stream() const override {
    return dev_ctx_->GetStream();
  }

  void set_dev_ctx(std::unique_ptr<phi::CustomContext>&& dev_ctx) {
    dev_ctx_ = std::move(dev_ctx);
  }
  phi::CustomContext* dev_context() const override { return dev_ctx_.get(); }

  std::shared_ptr<phi::event::Event> compute_event() const override {
    return compute_event_;
  }

  std::shared_ptr<phi::event::Event> comm_event() const override {
    return comm_event_;
  }

  void set_compute_event(std::shared_ptr<phi::event::Event>&& compute_event) {
    compute_event_ = std::move(compute_event);
  }

  void set_comm_event(std::shared_ptr<phi::event::Event>&& comm_event) {
    comm_event_ = std::move(comm_event);
  }

 private:
  int ring_id_;
  int nranks_;
  int rank_;
  phi::ccl::CCLComm comm_;
  std::unique_ptr<phi::CustomContext> dev_ctx_;

  // used for comm wait compute, compute_stream-->event-->comm_stream
  std::shared_ptr<phi::event::Event> compute_event_;

  // used for compute wait comm, comm_stream-->event-->compute_stream
  std::shared_ptr<phi::event::Event> comm_event_;
};

static std::unordered_map<std::string, std::unique_ptr<XCCLCommContext>>
    g_xccl_comm_ctx_map;

void XCCLCommContext::Release() {
  for (auto& it : g_xccl_comm_ctx_map) {
    it.second->ReleaseXCCLComms();
  }
  g_xccl_comm_ctx_map.clear();
}

XCCLCommContext& XCCLCommContext::Instance(const std::string& device_type) {
  if (g_xccl_comm_ctx_map.find(device_type) == g_xccl_comm_ctx_map.end()) {
    g_xccl_comm_ctx_map.insert(
        {device_type,
         std::unique_ptr<XCCLCommContext>(new XCCLCommContext(device_type))});
  }
  return *g_xccl_comm_ctx_map[device_type];
}

XCCLComm* XCCLCommContext::CreateComm(phi::ccl::CCLRootId* xccl_id,
                                      int nranks,
                                      int rank,
                                      int dev_id,
                                      int ring_id) {
  PADDLE_ENFORCE_NOT_NULL(xccl_id,
                          common::errors::InvalidArgument(
                              "The xccl unique id should not be null."));
  PADDLE_ENFORCE_GT(
      nranks,
      1,
      common::errors::InvalidArgument(
          "Expected nranks > 1. But received nranks is %d.", nranks));
  PADDLE_ENFORCE_GE(rank,
                    0,
                    common::errors::InvalidArgument(
                        "Expected rank >= 0. But received rank is %d.", rank));
  PADDLE_ENFORCE_LT(
      rank,
      nranks,
      common::errors::InvalidArgument(
          "Expected rank < nranks. But received rank is %d, nranks is %d.",
          rank,
          nranks));
  PADDLE_ENFORCE_GE(
      dev_id,
      0,
      common::errors::InvalidArgument(
          "Expected dev_id >= 0. But received dev_id is %d.", dev_id));

  phi::ccl::CCLComm comm = nullptr;
  phi::DeviceManager::SetDevice(device_type_, dev_id);
  phi::DeviceManager::CCLCommInitRank(
      device_type_, nranks, xccl_id, rank, &comm);

  auto* comm_wrapper = AssignXCCLComm(comm, nranks, rank, dev_id, ring_id);

  VLOG(1) << "xccl communicator of rank " << rank << " in ring " << ring_id
          << " has been created on device " << dev_id;

  return comm_wrapper;
}

void XCCLCommContext::CreateXCCLCommMultiTrainer(
    const std::vector<int>& dev_ids,
    phi::ccl::CCLRootId* xccl_id,
    int ntrainers,
    int train_id,
    int ring_id) {
  PADDLE_ENFORCE_GT(
      dev_ids.size(),
      0,
      common::errors::InvalidArgument(
          "dev ids = [%d], it should greater than 0.", dev_ids.size()));
  const int kDevices = dev_ids.size();
  VLOG(1) << "Begin CreateXCCLCommMultiTrainer. device number: " << kDevices
          << ", ntrainers: " << ntrainers << ", train_id: " << train_id
          << ", rind_id: " << ring_id;
  std::vector<phi::ccl::CCLComm> comms;
  comms.resize(kDevices);
  {
    for (int i = 0; i < kDevices; i++) {
      phi::DeviceManager::SetDevice(device_type_, i);
      phi::DeviceManager::CCLCommInitRank(device_type_,
                                          kDevices * ntrainers,
                                          xccl_id,
                                          train_id * kDevices + i,
                                          comms.data() + i);
      VLOG(1) << "CCLCommInitRank: " << i;
    }
  }
  PADDLE_ENFORCE_EQ(comm_map_.count(ring_id),
                    0,
                    common::errors::InvalidArgument(
                        "comm_map_ of ring_id: %s should be 0. %s is provided",
                        ring_id,
                        comm_map_.count(ring_id)));
  for (int i = 0; i < kDevices; ++i) {
    AssignXCCLComm(comms[i],
                   kDevices * ntrainers,
                   train_id * kDevices + i,
                   dev_ids[i],
                   ring_id);
    VLOG(1) << "xccl communicator of train_id " << train_id * kDevices + i
            << " in ring " << ring_id << " has been created on device "
            << dev_ids[i];
  }
}

XCCLComm* XCCLCommContext::AssignXCCLComm(
    phi::ccl::CCLComm comm, int nranks, int rank, int dev_id, int ring_id) {
  auto place = phi::CustomPlace(device_type_, dev_id);
  std::unique_ptr<phi::CustomContext> dev_ctx(new phi::CustomContext(place));
  dev_ctx->SetAllocator(paddle::memory::allocation::AllocatorFacade::Instance()
                            .GetAllocator(place)
                            .get());
  dev_ctx->SetHostAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(phi::CPUPlace())
          .get());
  dev_ctx->SetZeroAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetZeroAllocator(place)
          .get());
  dev_ctx->SetHostZeroAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetZeroAllocator(phi::CPUPlace())
          .get());
  dev_ctx->SetPinnedAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(phi::CPUPlace())
          .get());
  // dev_ctx->PartialInitWithAllocator();

  auto compute_event = std::make_shared<phi::event::Event>();
  auto comm_event = std::make_shared<phi::event::Event>();
  compute_event->Init(place);
  comm_event->Init(place);

  auto* c = new XCCLCommImpl;
  c->set_ring_id(ring_id);
  c->set_nranks(nranks);
  c->set_rank(rank);
  c->set_comm(comm);
  c->set_dev_ctx(std::move(dev_ctx));
  c->set_compute_event(std::move(compute_event));
  c->set_comm_event(std::move(comm_event));

  comm_map_mutex_.lock();
  if (comm_map_.count(ring_id) == 0) {
    comm_map_.emplace(ring_id, std::map<int, std::unique_ptr<XCCLComm>>());
  }
  auto& dev2comm = comm_map_[ring_id];

  dev2comm.emplace(dev_id, std::unique_ptr<XCCLComm>(c));
  comm_map_mutex_.unlock();

  if (ring_id == 0) {
    auto* dev_ctx = static_cast<phi::CustomContext*>(
        phi::DeviceContextPool::Instance().Get(place));
    dev_ctx->set_xccl_comm(comm);
  }
  VLOG(4) << "add xccl comm: " << comm_map_[ring_id][dev_id].get()
          << ", ring_id:" << ring_id << ", dev_id:" << dev_id;
  return comm_map_[ring_id][dev_id].get();
}

void XCCLCommContext::ReleaseXCCLComms() {
  for (auto& p : comm_map_) {
    for (auto& q : p.second) {
      q.second.reset();
    }
  }
}

#endif
}  // namespace platform
}  // namespace paddle

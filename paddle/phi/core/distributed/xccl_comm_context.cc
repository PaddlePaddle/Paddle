// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/distributed/xccl_comm_context.h"

#include "glog/logging.h"

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/distributed/check/static_check.h"
#include "paddle/phi/core/distributed/utils.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/utils/data_type.h"

namespace phi {
namespace distributed {

XCCLCommContext::XCCLCommContext(const std::string& device_type,
                                 int rank,
                                 int size,
                                 const ccl::CCLRootId& xccl_id)
    : CommContext(rank, size) {
  device_type_ = device_type;
  phi::DeviceManager::CCLCommInitRank(device_type,
                                      size_,
                                      const_cast<ccl::CCLRootId*>(&xccl_id),
                                      rank,
                                      &xccl_comm_);
}

void XCCLCommContext::Broadcast(phi::DenseTensor* out_tensor,
                                const phi::DenseTensor& in_tensor,
                                int root,
                                const phi::stream::Stream& stream) const {
  CommStaticCheck::SameShape(*out_tensor,
                             in_tensor,
                             /*dst_rank*/ rank_,
                             /*cur_rank*/ rank_,
                             size_,
                             phi::AllocationType::CUSTOM);
  if (rank_ == root) {
    phi::DeviceManager::CCLBroadcast(device_type_,
                                     const_cast<void*>(in_tensor.data()),
                                     in_tensor.numel(),
                                     phi::ccl::ToCCLDataType(in_tensor.dtype()),
                                     root,
                                     xccl_comm_,
                                     stream);
  } else {
    phi::DeviceManager::CCLBroadcast(device_type_,
                                     out_tensor->data(),
                                     out_tensor->numel(),
                                     phi::ccl::ToCCLDataType(in_tensor.dtype()),
                                     root,
                                     xccl_comm_,
                                     stream);
  }
}

void XCCLCommContext::AllGather(phi::DenseTensor* out_tensor,
                                const phi::DenseTensor& in_tensor,
                                const phi::stream::Stream& stream) const {
  phi::distributed::CommStaticCheck::GatherLikeShape(
      *out_tensor,
      in_tensor,
      /*dst_rank*/ rank_,
      /*cur_rank*/ rank_,
      size_,
      phi::AllocationType::CUSTOM);
  phi::DeviceManager::CCLAllGather(device_type_,
                                   const_cast<void*>(in_tensor.data()),
                                   out_tensor->data(),
                                   in_tensor.numel(),
                                   phi::ccl::ToCCLDataType(in_tensor.dtype()),
                                   xccl_comm_,
                                   stream);
}
void XCCLCommContext::ReduceScatter(phi::DenseTensor* out_tensor,
                                    const phi::DenseTensor& in_tensor,
                                    phi::ccl::CCLReduceOp reduce_type,
                                    const phi::stream::Stream& stream) const {
  phi::distributed::CommStaticCheck::ScatterLikeShape(
      *out_tensor,
      in_tensor,
      /*dst_rank*/ rank_,
      /*cur_rank*/ rank_,
      size_,
      phi::AllocationType::CUSTOM);
  phi::DeviceManager::CCLReduceScatter(
      device_type_,
      const_cast<void*>(in_tensor.data()),
      out_tensor->data(),
      out_tensor->numel(),
      phi::ccl::ToCCLDataType(in_tensor.type()),
      reduce_type,
      xccl_comm_,
      stream);
}

void XCCLCommContext::Send(const phi::DenseTensor& in_tensor,
                           const int64_t& count,
                           const int& peer,
                           const phi::stream::Stream& stream) const {
  phi::distributed::CommStaticCheck::CheckShape(
      in_tensor, rank_, size_, phi::AllocationType::CUSTOM);
  phi::DeviceManager::CCLSend(device_type_,
                              const_cast<void*>(in_tensor.data()),
                              count,
                              phi::ccl::ToCCLDataType(in_tensor.type()),
                              peer,
                              xccl_comm_,
                              stream);
  VLOG(3) << "rank " << GetRank() << " send " << phi::product(in_tensor.dims())
          << " to " << peer;
}

void XCCLCommContext::Recv(phi::DenseTensor* out_tensor,
                           const int64_t& count,
                           const int& peer,
                           const phi::stream::Stream& stream) const {
  phi::distributed::CommStaticCheck::CheckShape(
      *out_tensor, rank_, size_, phi::AllocationType::CUSTOM);
  phi::DeviceManager::CCLRecv(device_type_,
                              out_tensor->data(),
                              count,
                              phi::ccl::ToCCLDataType(out_tensor->type()),
                              peer,
                              xccl_comm_,
                              stream);
  VLOG(3) << "rank " << GetRank() << " recv "
          << phi::product(out_tensor->dims()) << " from " << peer;
}

void XCCLCommContext::AllReduce(phi::DenseTensor* out_tensor,
                                const phi::DenseTensor& in_tensor,
                                phi::ccl::CCLReduceOp reduce_type,
                                const phi::stream::Stream& stream) const {
  phi::distributed::CommStaticCheck::SameShape(*out_tensor,
                                               in_tensor,
                                               /*dst_rank*/ rank_,
                                               /*cur_rank*/ rank_,
                                               size_,
                                               phi::AllocationType::CUSTOM);
  phi::DeviceManager::CCLAllReduce(device_type_,
                                   const_cast<void*>(in_tensor.data()),
                                   out_tensor->data(),
                                   in_tensor.numel(),
                                   phi::ccl::ToCCLDataType(in_tensor.type()),
                                   reduce_type,
                                   xccl_comm_,
                                   stream);
}

void XCCLCommContext::Reduce(phi::DenseTensor* out_tensor,
                             const phi::DenseTensor& in_tensor,
                             phi::ccl::CCLReduceOp reduce_type,
                             int root,
                             const phi::stream::Stream& stream) const {
  phi::distributed::CommStaticCheck::SameShape(*out_tensor,
                                               in_tensor,
                                               /*dst_rank*/ root,
                                               /*cur_rank*/ rank_,
                                               size_,
                                               phi::AllocationType::CUSTOM);
  phi::DeviceManager::CCLReduce(device_type_,
                                const_cast<void*>(in_tensor.data()),
                                out_tensor->data(),
                                in_tensor.numel(),
                                phi::ccl::ToCCLDataType(in_tensor.type()),
                                reduce_type,
                                root,
                                xccl_comm_,
                                stream);
}

void XCCLCommContext::GroupStart() const {
  phi::DeviceManager::CCLGroupStart(device_type_);
}
void XCCLCommContext::GroupEnd() const {
  phi::DeviceManager::CCLGroupEnd(device_type_);
}

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
  PADDLE_ENFORCE_NOT_NULL(
      bkcl_id,
      phi::errors::InvalidArgument("The bkcl unique id should not be null."));
  PADDLE_ENFORCE_GT(
      nranks,
      1,
      phi::errors::InvalidArgument(
          "Expected nranks > 1. But received nranks is %d.", nranks));
  PADDLE_ENFORCE_GE(rank,
                    0,
                    phi::errors::InvalidArgument(
                        "Expected rank >= 0. But received rank is %d.", rank));
  PADDLE_ENFORCE_LT(
      rank,
      nranks,
      phi::errors::InvalidArgument(
          "Expected rank < nranks. But received rank is %d, nranks is %d.",
          rank,
          nranks));
  PADDLE_ENFORCE_GE(
      dev_id,
      0,
      phi::errors::InvalidArgument(
          "Expected dev_id >= 0. But received dev_id is %d.", dev_id));

  BKCLContext_t comm = nullptr;
  phi::backends::xpu::SetXPUDeviceId(dev_id);
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
    auto* dev_ctx =
        static_cast<phi::XPUContext*>(phi::DeviceContextPool::Instance().Get(
            paddle::platform::XPUPlace(dev_id)));
    dev_ctx->SetBkclContext(comm);
  }
  VLOG(3) << "add bkcl comm: " << comm_map_[ring_id][dev_id].get()
          << ", ring_id:" << ring_id << ", dev_id:" << dev_id;
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

}  // namespace distributed
}  // namespace phi

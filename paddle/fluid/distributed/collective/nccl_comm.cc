// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/distributed/collective/nccl_comm.h"
#include "paddle/fluid/distributed/collective/nccl_tools.h"
#include "paddle/fluid/platform/device/gpu/gpu_resource_pool.h"

namespace paddle {
namespace distributed {

class NCCLCommImpl : public NCCLComm {
 public:
  void set_ring_id(int ring_id) { ring_id_ = ring_id; }
  int ring_id() const override { return ring_id_; }

  void set_nranks(int nranks) { nranks_ = nranks; }
  int nranks() const override { return nranks_; }

  void set_rank(int rank) { rank_ = rank; }
  int rank() const override { return rank_; }

  int device_id() const override { return dev_ctx_->GetPlace().device; }

  void set_comm(ncclComm_t comm) { comm_ = comm; }
  ncclComm_t comm() const override { return comm_; }

  gpuStream_t stream() const override { return dev_ctx_->stream(); }

  void set_dev_ctx(std::unique_ptr<phi::GPUContext>&& dev_ctx) {
    dev_ctx_ = std::move(dev_ctx);
  }
  phi::GPUContext* dev_context() const override { return dev_ctx_.get(); }

  gpuEvent_t compute_event() const override { return compute_event_.get(); }

  gpuEvent_t comm_event() const override { return comm_event_.get(); }

  void set_compute_event(
      std::shared_ptr<platform::CudaEventObject>&& compute_event) {
    compute_event_ = std::move(compute_event);
  }

  void set_comm_event(std::shared_ptr<platform::CudaEventObject>&& comm_event) {
    comm_event_ = std::move(comm_event);
  }

  void ReduceScatter(phi::DenseTensor* out_tensor,
                     const phi::DenseTensor& in_tensor,
                     ReduceOp reduce_type,
                     gpuStream_t stream) override;

 private:
  int ring_id_;
  int nranks_;
  int rank_;
  ncclComm_t comm_;
  std::unique_ptr<phi::GPUContext> dev_ctx_;

  // used for comm wait compute, compute_stream-->event-->comm_stream
  std::shared_ptr<platform::CudaEventObject> compute_event_;

  // used for compute wait comm, comm_stream-->event-->compute_stream
  std::shared_ptr<platform::CudaEventObject> comm_event_;
};

void NCCLCommImpl::ReduceScatter(phi::DenseTensor* out_tensor,
                                 const phi::DenseTensor& in_tensor,
                                 ReduceOp reduce_type,
                                 gpuStream_t stream) {
  NCCL_CHECK(
      platform::dynload::ncclReduceScatter(in_tensor.data(),
                                           out_tensor->data(),
                                           out_tensor->numel(),
                                           ToNCCLDataType(in_tensor.dtype()),
                                           ToNCCLRedType(reduce_type),
                                           comm_,
                                           stream));
}

NCCLCommContext& NCCLCommContext::Instance() {
  static NCCLCommContext comm_ctx;
  return comm_ctx;
}

NCCLComm* NCCLCommContext::CreateComm(
    ncclUniqueId* nccl_id, int nranks, int rank, int dev_id, int ring_id) {
  PADDLE_ENFORCE_NOT_NULL(nccl_id,
                          platform::errors::InvalidArgument(
                              "The nccl unique id should not be null."));
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

  ncclComm_t comm = nullptr;
  platform::SetDeviceId(dev_id);
  PADDLE_ENFORCE_GPU_SUCCESS(
      platform::dynload::ncclCommInitRank(&comm, nranks, *nccl_id, rank));

  auto* comm_wrapper = AssignNCCLComm(comm, nranks, rank, dev_id, ring_id);

  VLOG(1) << "nccl communicator of rank " << rank << " in ring " << ring_id
          << " has been created on device " << dev_id;

  std::call_once(once_flag_, []() {
    std::atexit([]() { NCCLCommContext::Instance().ReleaseNCCLComms(); });
  });

  return comm_wrapper;
}

void NCCLCommContext::CreateAllNCCLComms(const std::vector<int>& dev_ids,
                                         int ring_id) {
  PADDLE_ENFORCE_GT(
      dev_ids.size(),
      0,
      platform::errors::InvalidArgument("Expected the size of dev_ids > 0. But "
                                        "received the size of dev_ids is %d.",
                                        dev_ids.size()));

  const int kDevices = dev_ids.size();
  ncclComm_t comms[kDevices];
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclCommInitAll(
      comms, dev_ids.size(), dev_ids.data()));

  PADDLE_ENFORCE_EQ(comm_map_.count(ring_id),
                    0,
                    platform::errors::InvalidArgument(
                        "Expected comm_map_.count(ring_id) = 0. But received "
                        "comm_map_.count(ring_id) is %d.",
                        comm_map_.count(ring_id)));
  for (size_t i = 0; i < dev_ids.size(); ++i) {
    AssignNCCLComm(comms[i], dev_ids.size(), i, dev_ids[i], ring_id);
    VLOG(1) << "nccl communicator of rank " << i << " in ring " << ring_id
            << " has been created on device " << dev_ids[i];
  }

  std::call_once(once_flag_, []() {
    std::atexit([]() { NCCLCommContext::Instance().ReleaseNCCLComms(); });
  });
}

void NCCLCommContext::CreateNCCLCommMultiTrainer(
    const std::vector<int>& dev_ids,
    ncclUniqueId* nccl_id,
    int ntrainers,
    int train_id,
    int ring_id) {
  PADDLE_ENFORCE_GT(
      dev_ids.size(),
      0,
      paddle::platform::errors::InvalidArgument(
          "dev ids = [%d], it should greater than 0.", dev_ids.size()));
  const int kDevices = dev_ids.size();
  VLOG(1) << "Begin CreateNCCLCommMultiTrainer. device number: " << kDevices
          << ", ntrainers: " << ntrainers << ", train_id: " << train_id
          << ", rind_id: " << ring_id;
  ncclComm_t comms[kDevices];
  {
    NCCL_CHECK(platform::dynload::ncclGroupStart());
    for (int i = 0; i < kDevices; i++) {
#ifdef PADDLE_WITH_HIP
      PADDLE_ENFORCE_GPU_SUCCESS(hipSetDevice(i));
#else
      PADDLE_ENFORCE_GPU_SUCCESS(cudaSetDevice(i));
#endif
      platform::dynload::ncclCommInitRank(
          comms + i, kDevices * ntrainers, *nccl_id, train_id * kDevices + i);
      VLOG(1) << "ncclCommInitRank: " << i;
    }
    NCCL_CHECK(platform::dynload::ncclGroupEnd());
    VLOG(1) << "nccl group end seccessss";
  }
  PADDLE_ENFORCE_EQ(comm_map_.count(ring_id),
                    0,
                    platform::errors::InvalidArgument(
                        "comm_map_ of ring_id: %s should be 0. %s is provided",
                        ring_id,
                        comm_map_.count(ring_id)));
  for (int i = 0; i < kDevices; ++i) {
    AssignNCCLComm(comms[i],
                   kDevices * ntrainers,
                   train_id * kDevices + i,
                   dev_ids[i],
                   ring_id);
    VLOG(1) << "nccl communicator of train_id " << train_id * kDevices + i
            << " in ring " << ring_id << " has been created on device "
            << dev_ids[i];
  }

  std::call_once(once_flag_, []() {
    std::atexit([]() { NCCLCommContext::Instance().ReleaseNCCLComms(); });
  });
}

NCCLComm* NCCLCommContext::AssignNCCLComm(
    ncclComm_t comm, int nranks, int rank, int dev_id, int ring_id) {
  std::unique_ptr<phi::GPUContext> dev_ctx(
      new phi::GPUContext(platform::CUDAPlace(dev_id)));
  dev_ctx->SetAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(platform::CUDAPlace(dev_id), dev_ctx->stream())
          .get());
  dev_ctx->SetHostAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(paddle::platform::CPUPlace())
          .get());
  dev_ctx->SetZeroAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetZeroAllocator(platform::CUDAPlace(dev_id))
          .get());
  dev_ctx->SetHostZeroAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetZeroAllocator(paddle::platform::CPUPlace())
          .get());
  dev_ctx->SetPinnedAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(paddle::platform::CUDAPinnedPlace())
          .get());
  dev_ctx->PartialInitWithAllocator();

  std::shared_ptr<platform::CudaEventObject> compute_event(
      platform::CudaEventResourcePool::Instance().New(dev_id));
  std::shared_ptr<platform::CudaEventObject> comm_event(
      platform::CudaEventResourcePool::Instance().New(dev_id));

  NCCLCommImpl* c = new NCCLCommImpl;
  c->set_ring_id(ring_id);
  c->set_nranks(nranks);
  c->set_rank(rank);
  c->set_comm(comm);
  c->set_dev_ctx(std::move(dev_ctx));
  c->set_compute_event(std::move(compute_event));
  c->set_comm_event(std::move(comm_event));

  comm_map_mutex_.lock();
  if (comm_map_.count(ring_id) == 0) {
    comm_map_.emplace(ring_id, std::map<int, std::unique_ptr<NCCLComm>>());
  }
  auto& dev2comm = comm_map_[ring_id];

  dev2comm.emplace(dev_id, std::unique_ptr<NCCLComm>(c));
  comm_map_mutex_.unlock();

  if (ring_id == 0) {
    auto* dev_ctx = static_cast<phi::GPUContext*>(
        platform::DeviceContextPool::Instance().Get(
            platform::CUDAPlace(dev_id)));
    dev_ctx->set_nccl_comm(comm);
  }
  VLOG(4) << "add mccl comm: " << comm_map_[ring_id][dev_id].get()
          << ", ring_id:" << ring_id << ", dev_id:" << dev_id;
  return comm_map_[ring_id][dev_id].get();
}

void NCCLCommContext::ReleaseNCCLComms() {
  for (auto& p : comm_map_) {
    for (auto& q : p.second) {
      q.second.reset();
    }
  }
}

}  // namespace distributed
}  // namespace paddle

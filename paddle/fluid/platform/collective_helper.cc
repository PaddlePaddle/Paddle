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

#if defined(PADDLE_WITH_NCCL)
#include "paddle/fluid/platform/collective_helper.h"

#include <memory>
#include <utility>

#include "paddle/fluid/platform/dynload/nccl.h"

namespace paddle {
namespace platform {

class NCCLCommImpl : public NCCLComm {
 public:
  void set_ring_id(int ring_id) { ring_id_ = ring_id; }
  int ring_id() const override { return ring_id_; }

  void set_nranks(int nranks) { nranks_ = nranks; }
  int nranks() const override { return nranks_; }

  void set_rank(int rank) { rank_ = rank; }
  int rank() const override { return rank_; }

  int device_id() const override {
    return boost::get<CUDAPlace>(dev_ctx_->GetPlace()).device;
  }

  ncclComm_t comm() const override { return dev_ctx_->nccl_comm(); }

  cudaStream_t stream() const override { return dev_ctx_->stream(); }

  void set_dev_ctx(std::unique_ptr<CUDADeviceContext>&& dev_ctx) {
    dev_ctx_ = std::move(dev_ctx);
  }

 private:
  int ring_id_;
  int nranks_;
  int rank_;
  std::unique_ptr<CUDADeviceContext> dev_ctx_;
};

NCCLComm* NCCLCommContext::CreateNCCLComm(ncclUniqueId* nccl_id, int nranks,
                                          int rank, int dev_id, int ring_id) {
  PADDLE_ENFORCE_NOT_NULL(nccl_id);
  PADDLE_ENFORCE_GT(nranks, 1);
  PADDLE_ENFORCE_GE(rank, 0);
  PADDLE_ENFORCE_LT(rank, nranks);
  PADDLE_ENFORCE_GE(dev_id, 0);

  ncclComm_t comm = nullptr;
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaSetDevice(dev_id));
  PADDLE_ENFORCE_CUDA_SUCCESS(
      platform::dynload::ncclCommInitRank(&comm, nranks, *nccl_id, rank));

  std::unique_ptr<CUDADeviceContext> dev_ctx(
      new CUDADeviceContext(CUDAPlace(dev_id)));
  dev_ctx->set_nccl_comm(comm);

  NCCLCommImpl* c = new NCCLCommImpl;
  c->set_ring_id(ring_id);
  c->set_nranks(nranks);
  c->set_rank(rank);
  c->set_dev_ctx(std::move(dev_ctx));

  comm_map_mutex_.lock();
  if (comm_map_.count(ring_id) == 0) {
    comm_map_.emplace(ring_id, std::map<int, std::unique_ptr<NCCLComm>>());
  }
  auto& dev2comm = comm_map_[ring_id];

  dev2comm.emplace(dev_id, std::unique_ptr<NCCLComm>(c));
  comm_map_mutex_.unlock();

  VLOG(1) << "nccl communicator of rank " << rank << " in ring " << ring_id
          << " has been created";

  std::call_once(once_flag_, []() {
    std::atexit([]() { NCCLCommContext::Instance().ReleaseNCCLComms(); });
  });

  return comm_map_[ring_id][dev_id].get();
}

void NCCLCommContext::CreateAllNCCLComms(const std::vector<int>& dev_ids,
                                         int ring_id) {
  PADDLE_ENFORCE_GT(dev_ids.size(), 0);

  const int kDevices = dev_ids.size();
  ncclComm_t comms[kDevices];
  PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclCommInitAll(
      comms, dev_ids.size(), dev_ids.data()));

  PADDLE_ENFORCE_EQ(comm_map_.count(ring_id), 0);
  comm_map_.emplace(ring_id, std::map<int, std::unique_ptr<NCCLComm>>());

  auto& dev2comm = comm_map_[ring_id];
  for (size_t i = 0; i < dev_ids.size(); ++i) {
    std::unique_ptr<CUDADeviceContext> dev_ctx(
        new CUDADeviceContext(CUDAPlace(dev_ids[i])));
    dev_ctx->set_nccl_comm(comms[i]);

    NCCLCommImpl* c = new NCCLCommImpl;
    c->set_ring_id(ring_id);
    c->set_nranks(dev_ids.size());
    c->set_rank(i);
    c->set_dev_ctx(std::move(dev_ctx));

    dev2comm.emplace(dev_ids[i], std::unique_ptr<NCCLComm>(c));
  }

  std::call_once(once_flag_, []() {
    std::atexit([]() { NCCLCommContext::Instance().ReleaseNCCLComms(); });
  });
}

void NCCLCommContext::ReleaseNCCLComms() {
  // CUDADeviceContext maintain the lifetime of nccl_comm_t, so we should not
  // destroy nccl_comm_t explicitly. Please refer to
  // platform::CUDADeviceContext::~CUDADeviceContext()
  for (auto& p : comm_map_) {
    for (auto& q : p.second) {
      q.second.reset();
    }
  }
}

}  // namespace platform
}  // namespace paddle

#endif

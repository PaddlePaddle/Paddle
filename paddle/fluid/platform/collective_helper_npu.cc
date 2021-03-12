//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#if defined(PADDLE_WITH_ASCEND_CL)
#include "paddle/fluid/platform/collective_helper.h"
#include <utility>

namespace paddle {
namespace platform {

class HCCLCommImpl : public HCCLComm {
 public:
  void set_ring_id(int ring_id) { ring_id_ = ring_id; }
  int ring_id() const override { return ring_id_; }

  void set_nranks(int nranks) { nranks_ = nranks; }
  int nranks() const override { return nranks_; }

  void set_rank(int rank) { rank_ = rank; }
  int rank() const override { return rank_; }

  int device_id() const override {
    return BOOST_GET_CONST(NPUPlace, dev_ctx_->GetPlace()).device;
  }

  aclrtStream stream() const override { return dev_ctx_->stream(); }

  void set_dev_ctx(std::unique_ptr<NPUDeviceContext>&& dev_ctx) {
    dev_ctx_ = std::move(dev_ctx);
  }
  NPUDeviceContext* dev_context() const override { return dev_ctx_.get(); }

 private:
  int ring_id_;
  int nranks_;
  int rank_;
  std::unique_ptr<NPUDeviceContext> dev_ctx_;
};

HCCLComm* HCCLCommContext::CreateHCCLComm(const std::vector<int>& world_rank_ids, int rank, int dev_id, int ring_id) {
  PADDLE_ENFORCE_GT(
      world_rank_ids.size(), 1,
      platform::errors::InvalidArgument(
          "Expected world_rank_ids.size() > 1. But received size is %d.", world_rank_ids.size()));
  PADDLE_ENFORCE_GE(rank, 0,
                    platform::errors::InvalidArgument(
                        "Expected rank >= 0. But received rank is %d.", rank));
  PADDLE_ENFORCE_LT(
      rank, world_rank_ids.size(),
      platform::errors::InvalidArgument(
          "Expected rank < nranks. But received rank is %d, nranks is %d.",
          rank, world_rank_ids.size()));
  PADDLE_ENFORCE_GE(
      dev_id, 0,
      platform::errors::InvalidArgument(
          "Expected dev_id >= 0. But received dev_id is %d.", dev_id));
  PADDLE_ENFORCE_GE(
      ring_id, 0,
      platform::errors::InvalidArgument(
          "Expected ring_id >= 0. But received ring_id is %d.", ring_id));

  auto* comm_wrapper = AssignHCCLComm(world_rank_ids.size(), rank, dev_id, ring_id);

  // HACK(sunpeng17): hcom API requires bind stream to a model
  // but we don't need model in Paddle, so we feed stream pointer as model pointer
  PADDLE_ENFORCE_NPU_SUCCESS(
      platform::dynload::hcom_bind_model(comm_wrapper->stream(),
                                         comm_wrapper->stream()));

  // Get world_rank_ids registered in gen_nccl_id op
  std::string group_name = HCOM_GROUP_PREFIX + std::to_string(ring_id);
  PADDLE_ENFORCE_NPU_SUCCESS(platform::dynload::hcom_create_group(
      group_name.c_str(), world_rank_ids.size(), (unsigned int*)world_rank_ids.data()));

  VLOG(1) << "hccl communicator of rank " << rank << " in ring " << ring_id
          << " has been created on device " << dev_id << ", group name: " << group_name;

  std::call_once(once_flag_, []() {
    std::atexit([]() { HCCLCommContext::Instance().ReleaseHCCLComms(); });
  });

  return comm_wrapper;
}

HCCLComm* HCCLCommContext::AssignHCCLComm(int nranks, int rank, int dev_id, int ring_id) {
  std::unique_ptr<NPUDeviceContext> dev_ctx(
      new NPUDeviceContext(NPUPlace(dev_id)));

  HCCLCommImpl* c = new HCCLCommImpl;
  c->set_ring_id(ring_id);
  c->set_nranks(nranks);
  c->set_rank(rank);
  c->set_dev_ctx(std::move(dev_ctx));

  comm_map_mutex_.lock();
  if (comm_map_.count(ring_id) == 0) {
    comm_map_.emplace(ring_id, std::map<int, std::unique_ptr<HCCLComm>>());
  }
  auto& dev2comm = comm_map_[ring_id];

  dev2comm.emplace(dev_id, std::unique_ptr<HCCLComm>(c));
  comm_map_mutex_.unlock();

  return comm_map_[ring_id][dev_id].get();
}

void HCCLCommContext::InitHcomWorldGroup() {
  const char *rank_table_file = getenv(ENV_RANK_TABLE_FILE);
  PADDLE_ENFORCE_NOT_NULL(
      rank_table_file,
      platform::errors::InvalidArgument("The RANK_TABLE_FILE environment variable should not be null."));

  const char *rank_id = getenv(ENV_RANK_ID);
  PADDLE_ENFORCE_NOT_NULL(
      rank_id,
      platform::errors::InvalidArgument("The RANK_ID environment variable should not be null."));

  PADDLE_ENFORCE_NPU_SUCCESS(platform::dynload::hcom_init(rank_table_file, rank_id));
  VLOG(3) << "Successfully initialized hcom. rank_table_file: "
    << rank_table_file << ", rank_id " << rank_id;
}

void HCCLCommContext::ReleaseHCCLComms() {
  for (auto& p : comm_map_) {
    for (auto& q : p.second) {
      q.second.reset();
    }
  }
}

}  // namespace platform
}  // namespace paddle
#endif

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

#include "paddle/fluid/imperative/heter_ccl_context.h"

// NCCL first
#ifdef PADDLE_WITH_NCCL
#include "paddle/fluid/imperative/all_reduce.h"
#endif

#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/gen_comm_id_helper.h"
#include "paddle/fluid/framework/fleet/gloo_wrapper.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/string/split.h"
#include "paddle/fluid/string/string_helper.h"

namespace paddle {
namespace framework {
class Variable;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace imperative {

HeterParallelContext::HeterParallelContext(const ParallelStrategy &strategy,
                                           const int &device_id)
#ifdef PADDLE_WITH_NCCL
    : ParallelContext(strategy, platform::CUDAPlace(device_id)) {
#endif
#ifdef PADDLE_WITH_XPU_BKCL
    : ParallelContext(strategy, platform::XPUPlace(device_id)) {
#endif
#ifdef PADDLE_WITH_ASCEND_CL
    : ParallelContext(strategy, platform::NPUPlace(device_id)) {
#endif

  // construct node_strategy_ from global strategy by selecting the
  // endpoints with same ip address.

  std::vector<std::string> global_eps = strategy_.trainer_endpoints_;
  for (auto ep : global_eps) {
    std::string ip = ep.substr(0, ep.find(':'));
    // record ip of different nodes
    nodes_ips_.emplace(ip);
  }

  int local_nranks = strategy_.local_nranks_;
  PADDLE_ENFORCE_NE(local_nranks, 0,
                    platform::errors::InvalidArgument(
                        "The number of local nranks should not be zero."
                    ));
  // PADDLE_ENFORCE_GT(nodes_ips_.size(), 1,
  //                   platform::errors::InvalidArgument(
  //                       "The number of nodes should be greater than 1."
  //                   ));
  PADDLE_ENFORCE_EQ(nodes_ips_.size() * strategy_.local_nranks_,
                    strategy_.trainer_endpoints_.size(),
                    platform::errors::InvalidArgument(
                        "The number of nodes times the number of cards "
                        "in each node should equal to the number of global trainers."
                    ));

  node_strategy_.nranks_ = local_nranks;
  node_strategy_.local_rank_ = strategy_.local_rank_ % local_nranks;
  node_strategy_.current_endpoint_ = strategy_.current_endpoint_;

  if (nodes_ips_.size() == 1) {
    // NOTE(liubo48): this branch is only used for unittest with single node.
    node_strategy_.trainer_endpoints_ = strategy_.trainer_endpoints_;
  } else {
    // nodes with different ips
    std::string curr_ep = strategy_.current_endpoint_;
    std::string curr_ep_ip = paddle::string::Split(curr_ep, ':')[0];
    for (auto ep : global_eps) {
      std::string ip = paddle::string::Split(ep, ':')[0];
      if (ip == curr_ep_ip) {
        node_strategy_.trainer_endpoints_.push_back(ep);
      }
    }
  }

  // NOTE(liubo48): currently not support only one rank on each node.
  PADDLE_ENFORCE_GT(node_strategy_.trainer_endpoints_.size(), 1,
                    platform::errors::InvalidArgument(
                        "In heter mode, the number of ranks on each node "
                        "should be greater than 1."
                    ));

  // construct gloo_strategy_ from global strategy by selecting the
  // endpoints with local_rank which can be evenly divided by local_nranks_.
  if ((nodes_ips_.size() > 1) &&
      (strategy_.local_rank_ % local_nranks == 0)) {
    gloo_strategy_.nranks_ = strategy_.nranks_ / local_nranks;
    gloo_strategy_.local_rank_ = strategy_.local_rank_ / local_nranks;
    gloo_strategy_.current_endpoint_ = strategy_.current_endpoint_;
    for (size_t i = 0; i < strategy_.trainer_endpoints_.size(); i++) {
      if (i % local_nranks == 0) {
        gloo_strategy_.trainer_endpoints_.push_back(
            strategy_.trainer_endpoints_[i]);
      }
    }
    gloo_ctx_ = std::make_shared<GLOOParallelContext>(
        gloo_strategy_, platform::CPUPlace());
  }

#ifdef PADDLE_WITH_NCCL
  node_place_ = platform::CUDAPlace(device_id);
  heter_parallel_ctx_ =
      std::make_shared<NCCLParallelContext>(node_strategy_, node_place_);
#endif
#ifdef PADDLE_WITH_XPU_BKCL
  node_place_ = platform::XPUPlace(device_id);
  heter_parallel_ctx_ =
      std::make_shared<BKCLParallelContext>(node_strategy_, node_place_);
#endif
#ifdef PADDLE_WITH_ASCEND_CL
  node_place_ = platform::NPUPlace(device_id);
  heter_parallel_ctx_ =
      std::make_shared<HCCLParallelContext>(node_strategy_, node_place_);
#endif
}

void HeterParallelContext::Init() {
  PADDLE_ENFORCE_NE(heter_parallel_ctx_, nullptr,
                    platform::errors::Unavailable(
                        "The heter parallel context has not been initialized."
                    ));
  // NOTE(liubo48): call Init() to create ring_id(0) for parameters synchonization.
  heter_parallel_ctx_->Init();

  // create another ring for internal communication.
  heter_parallel_ctx_->InitWithRingID(1);

  if (gloo_ctx_ != nullptr) {
    gloo_ctx_->Init();
  }

  VLOG(3) << "/// DEBUG /// heter parallel env init done..." << std::endl;
}

void HeterParallelContext::InitWithRingID(int ring_id) {
  PADDLE_THROW(platform::errors::Unimplemented(
                   "Unimplemented InitWithRingID from heter ctx."));
}

void HeterParallelContext::AllReduceByStream(
    const framework::Variable &src, framework::Variable *dst, int ring_id,
    bool use_calc_stream) {
  // step 1: call reduce within node
  VLOG(3) << "/// DEBUG /// step 1: reduce within node... ";
  heter_parallel_ctx_->InterReduce(src, dst, 1);
  heter_parallel_ctx_->WaitComm(1);

  // step 2: call allreduce between nodes with gloo
  if (gloo_ctx_ != nullptr) {
    auto gloo_ptr = paddle::framework::GlooWrapper::GetInstance();
    PADDLE_ENFORCE_EQ(gloo_ptr->IsInitialized(), true,
                      paddle::platform::errors::Unavailable(
                          "Gloo context is not initialized."));

    auto src_dev_tensor = src.Get<framework::LoDTensor>();
    auto *dst_dev_tensor = dst->GetMutable<framework::LoDTensor>();
    dst_dev_tensor->Resize(src_dev_tensor.dims());

    // step 2.1: Dev Tensor to CPU Tensor
    VLOG(3) << "/// DEBUG /// step 2.1: Dev Tensor to CPU Tensor... ";
    framework::Tensor src_cpu_tensor;
    framework::Tensor dst_cpu_tensor;
    src_cpu_tensor.mutable_data<float>(src_dev_tensor.dims(),
                                       platform::CPUPlace());

    auto &place = src_dev_tensor.place();
    auto cpu_place = new platform::CPUPlace();
    framework::TensorCopySync(src_dev_tensor, *cpu_place, &src_cpu_tensor);

    // step 2.2: call gloo->AllReduce between cpus of nodes
    VLOG(3) << "/// DEBUG /// step 2.2: gloo allreduce between nodes... ";
    std::vector<float> send_vector;
    framework::TensorToVector<float>(src_cpu_tensor, &send_vector);
    auto recv_vector = gloo_ptr->AllReduce<float>(send_vector);
    framework::TensorFromVector<float>(recv_vector, &dst_cpu_tensor);

    // step 2.3: CPU Tensor to Dev tensor
    VLOG(3) << "/// DEBUG /// step 2.3: CPU Tensor to Dev tensor... ";
    framework::TensorCopySync(dst_cpu_tensor, place, dst_dev_tensor);

    gloo_ptr->Barrier();
  }

  // step 3: call broadcast within node
  VLOG(3) << "/// DEBUG /// step 3: broadcast within node... ";
  heter_parallel_ctx_->InterBroadCast(dst, 1);
  heter_parallel_ctx_->WaitComm(1);
}

void HeterParallelContext::InterReduce(
    const framework::Variable &src, framework::Variable *dst, int ring_id) {
  PADDLE_THROW(platform::errors::Unimplemented(
                   "Unimplemented InterReduce from heter ctx."));
}

void HeterParallelContext::InterBroadCast(framework::Variable * src,
                                          int ring_id) {
  PADDLE_THROW(platform::errors::Unimplemented(
                   "Unimplemented InterBroadCast from heter ctx."));
}

paddle::platform::DeviceContext *HeterParallelContext::GetDeviceContext(
    int ring_id) {
  // directly call the implementation of target parallel ctx.
  return heter_parallel_ctx_->GetDeviceContext(ring_id);
}

void HeterParallelContext::WaitCompute(int ring_id) {
  // directly call the implementation of target parallel ctx.
  heter_parallel_ctx_->WaitCompute(ring_id);
}

void HeterParallelContext::WaitComm(int ring_id) {
  // directly call the implementation of target parallel ctx.
  heter_parallel_ctx_->WaitComm(ring_id);
}

void HeterParallelContext::SynchronizeCompute() {
  // directly call the implementation of target parallel ctx.
  heter_parallel_ctx_->SynchronizeCompute();
}

}  //  namespace imperative
}  //  namespace paddle

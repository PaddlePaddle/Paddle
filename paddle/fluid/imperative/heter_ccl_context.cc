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

// TODO(liubo48)
#ifdef PADDLE_WITH_XPU_BKCL
#endif

// TODO(liubo48)
#ifdef PADDLE_WITH_ASCEND_CL
#endif

#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/gen_comm_id_helper.h"
#include "paddle/fluid/framework/fleet/gloo_wrapper.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/place.h"

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
  // endpoints
  // with same ip address.

  // NOTE(liubo48): hard code number of cards (2) for experiment only.
  node_strategy_.nranks_ = strategy_.nranks_ / 2;
  node_strategy_.local_rank_ = strategy_.local_rank_ % 2;
  node_strategy_.current_endpoint_ = strategy_.current_endpoint_;

  // for single node (same ip)
  if (strategy_.local_rank_ < 2) {
    node_strategy_.trainer_endpoints_.push_back(strategy_.trainer_endpoints_[0]);
    node_strategy_.trainer_endpoints_.push_back(strategy_.trainer_endpoints_[1]);
  } else {
    node_strategy_.trainer_endpoints_.push_back(strategy_.trainer_endpoints_[2]);
    node_strategy_.trainer_endpoints_.push_back(strategy_.trainer_endpoints_[3]);
  }

  // // for multi node with different ips
  // std::vector<std::string> global_eps = strategy_.trainer_endpoints_;
  // std::string curr_ep = strategy_.current_endpoint_;
  // std::string curr_ep_ip = curr_ep.substr(0, curr_ep.find(':'));
  // for (auto ep : global_eps) {
  //   std::string ip = ep.substr(0, ep.find(':'));
  //   if (ip == curr_ep_ip) {
  //     node_strategy_.trainer_endpoints_.push_back(ep);
  //   }
  // }

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

  // construct gloo_strategy_ from global strategy by selecting the
  // endpoints
  // with local_rank can be divided by 2

  // NOTE(liubo48): hard code number of cards (2) for experiment only.
  if (strategy_.local_rank_ % 2 == 0) {
    gloo_strategy_.nranks_ = strategy_.nranks_ / 2;
    gloo_strategy_.local_rank_ = strategy_.local_rank_ / 2;
    gloo_strategy_.current_endpoint_ = strategy_.current_endpoint_;
    for (size_t i = 0; i < strategy_.trainer_endpoints_.size(); i++) {
      if (i % 2 == 0) {
        gloo_strategy_.trainer_endpoints_.push_back(
            strategy_.trainer_endpoints_[i]);
      }
    }
    gloo_ctx_ = std::make_shared<GLOOParallelContext>(gloo_strategy_,
                                                      platform::CPUPlace());
  }
}

void HeterParallelContext::Init() {
  // check heter_parallel_ctx_
  // check gloo_ctx_

  // NOTE(liubo48): call Init() to create ring_id(0) for compatibility.
  heter_parallel_ctx_->Init();

  // NOTE(liubo48): hard code number of cards (2) for experiment only.
  if (strategy_.local_rank_ % 2 == 0) {
    gloo_ctx_->Init();
  }

  std::cout << "/// DEBUG /// heter parallel env init done..." << std::endl;
}

void HeterParallelContext::InitWithRingID(int ring_id) {
  // no need to implement.
  return;
}

void HeterParallelContext::AllReduceByStream(
    const framework::Variable &src, framework::Variable *dst, int ring_id,
    bool use_calc_stream) {
#ifdef PADDLE_WITH_NCCL
  // step 1: call reduce within node
  heter_parallel_ctx_->InterReduce(src, dst, 0);
  heter_parallel_ctx_->WaitComm(0);

  // step 2: call allreduce between nodes
  // NOTE(liubo48): hard code number of cards (2) for experiment only.
  if (strategy_.local_rank_ % 2 == 0) {
    auto gloo_ptr = paddle::framework::GlooWrapper::GetInstance();
    PADDLE_ENFORCE_EQ(gloo_ptr->IsInitialized(), true,
                      paddle::platform::errors::Unavailable(
                          "Gloo context is not initialized."));
    auto src_gpu_tensor = src.Get<framework::LoDTensor>();
    auto *dst_gpu_tensor = dst->GetMutable<framework::LoDTensor>();
    dst_gpu_tensor->Resize(src_gpu_tensor.dims());

    // step 2.1: GPU Tensor to CPU Tensor
    framework::Tensor src_cpu_tensor;
    framework::Tensor dst_cpu_tensor;
    src_cpu_tensor.mutable_data<float>(src_gpu_tensor.dims(),
                                        platform::CPUPlace());
    auto *dev_ctx = static_cast<platform::CUDADeviceContext *>(
        platform::DeviceContextPool::Instance().Get(
            src_gpu_tensor.place()));
    auto cpu_place = new platform::CPUPlace();
    framework::TensorCopy(src_gpu_tensor, *cpu_place, *dev_ctx,
                          &src_cpu_tensor);
    dev_ctx->Wait();

    // step 2.2: call gloo->AllReduce between cpus
    std::vector<float> send_vector;
    framework::TensorToVector<float>(src_cpu_tensor, &send_vector);
    auto recv_vector = gloo_ptr->AllReduce<float>(send_vector);
    framework::TensorFromVector<float>(recv_vector, &dst_cpu_tensor);

    // step 2.3: CPU Tensor to GPU tensor
    framework::TensorCopy(dst_cpu_tensor, dst_gpu_tensor->place(), *dev_ctx,
                          dst_gpu_tensor);
    dev_ctx->Wait();

    gloo_ptr->Barrier();
  }

  // step 3: call broadcast within node
  heter_parallel_ctx_->InterBroadCast(dst, 0);
  heter_parallel_ctx_->WaitComm(0);
#endif
}

void HeterParallelContext::InterReduce(
    const framework::Variable &src, framework::Variable *dst, int ring_id) {
  // no need to implement.
  return;
}

void HeterParallelContext::InterBroadCast(framework::Variable * src,
                                          int ring_id) {
  // no need to implement.
  return;
}

paddle::platform::DeviceContext *HeterParallelContext::GetDeviceContext(
    int ring_id) {
  return heter_parallel_ctx_->GetDeviceContext(ring_id);
}

void HeterParallelContext::WaitCompute(int ring_id) {
  // no need to implement.
  return;
}

void HeterParallelContext::WaitComm(int ring_id) {
  // no need to implement.
  return;
}

void HeterParallelContext::SynchronizeCompute() {
  // no need to implement.
  return;
}

}  //  namespace imperative
}  //  namespace paddle

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

#include "paddle/fluid/framework/fleet/gloo_wrapper.h"
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/gen_comm_id_helper.h"
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
    : ParallelContext(strategy, platform::CUDAPlace(device_id))
#elif PADDLE_WITH_XPU_BKCL
    : ParallelContext(strategy, platform::XPUPlace(device_id))
#elif PADDLE_WITH_ASCEND_CL
    : ParallelContext(strategy, platform::NPUPlace(device_id))
#else
    : ParallelContext(strategy, platform::CPUPlace())
#endif
{
  // construct node_strategy_ from global strategy by selecting the
  // endpoints with same ip address.
  std::string node_ip = strategy_.current_endpoint_.substr(
      0, strategy_.current_endpoint_.find(':'));
  int node_nranks = 0;
  int inter_rank = -1;

  std::vector<std::string> all_eps = strategy_.trainer_endpoints_;
  std::vector<std::string> inter_endpoints;
  std::set<std::string> nodes_ips;
  for (auto ep : all_eps) {
    std::string ip = ep.substr(0, ep.find(':'));
    // record ip of different nodes
    if (nodes_ips.find(ip) == nodes_ips.end()) {
      if (ep == strategy_.current_endpoint_) {
        inter_rank = nodes_ips.size();
      }
      inter_endpoints.push_back(ep);
      nodes_ips.emplace(ip);
    }

    if (ip == node_ip) {
      if (ep == strategy_.current_endpoint_) {
        node_strategy_.local_rank_ = node_nranks;
      }
      node_nranks++;
      node_strategy_.trainer_endpoints_.push_back(ep);
    }
  }

  VLOG(0) << "init node size " << node_nranks << " rank "
          << node_strategy_.local_rank_;

  PADDLE_ENFORCE_NE(node_nranks, 0,
                    platform::errors::InvalidArgument(
                        "The number of local nranks should not be zero."));
  node_strategy_.nranks_ = node_nranks;
  node_strategy_.current_endpoint_ = strategy_.current_endpoint_;

  if (inter_rank >= 0 && inter_endpoints.size() > 1) {
    inter_strategy_.nranks_ = inter_endpoints.size();
    inter_strategy_.local_rank_ = inter_rank;
    inter_strategy_.current_endpoint_ = strategy_.current_endpoint_;
    inter_strategy_.trainer_endpoints_ = inter_endpoints;
#ifdef PADDLE_WITH_GLOO
    inter_parallel_ctx_ = std::make_shared<GLOOParallelContext>(
        inter_strategy_, platform::CPUPlace());
#endif
  }

  VLOG(0) << "init inter size " << inter_endpoints.size() << " rank "
          << inter_rank;

#ifdef PADDLE_WITH_NCCL
  node_place_ = platform::CUDAPlace(device_id);
  node_parallel_ctx_ =
      std::make_shared<NCCLParallelContext>(node_strategy_, node_place_);
#endif
#ifdef PADDLE_WITH_XPU_BKCL
  node_place_ = platform::XPUPlace(device_id);
  node_parallel_ctx_ =
      std::make_shared<BKCLParallelContext>(node_strategy_, node_place_);
#endif
#ifdef PADDLE_WITH_ASCEND_CL
  node_place_ = platform::NPUPlace(device_id);
  node_parallel_ctx_ =
      std::make_shared<HCCLParallelContext>(node_strategy_, node_place_);
#endif
}

void HeterParallelContext::Init() {
  PADDLE_ENFORCE_NE(
      node_parallel_ctx_, nullptr,
      platform::errors::Unavailable(
          "The heter parallel context has not been initialized."));

  if (inter_parallel_ctx_ != nullptr) {
    inter_parallel_ctx_->Init();
  }

  node_parallel_ctx_->Init();

  VLOG(3) << "/// DEBUG /// heter parallel env init done..." << std::endl;
}

void HeterParallelContext::InitWithRingID(int ring_id) {
  PADDLE_THROW(platform::errors::Unimplemented(
      "Unimplemented InitWithRingID from heter ctx."));
}

void HeterParallelContext::AllReduceByStream(const framework::Variable &src,
                                             framework::Variable *dst,
                                             int ring_id,
                                             bool use_calc_stream) {
  // step 1: call reduce within node
  VLOG(3) << "/// DEBUG /// step 1: reduce in node... ";
  node_parallel_ctx_->AllReduceByStream(src, dst, ring_id, false);
  node_parallel_ctx_->WaitComm(ring_id);

  // step 2: call allreduce between nodes with gloo
  if (inter_parallel_ctx_ != nullptr) {
    // copy src to cpu
    // dst is now the src
    auto src_tensor = dst->Get<framework::LoDTensor>();
    framework::Variable src_cpu;
    auto src_cpu_tensor = src_cpu.GetMutable<framework::LoDTensor>();
    framework::TensorCopySync(src_tensor, platform::CPUPlace(), src_cpu_tensor);

    // allreduce src/cpu to dst/cpu
    framework::Variable dst_cpu;
    inter_parallel_ctx_->AllReduceByStream(src_cpu, &dst_cpu, ring_id, false);
    inter_parallel_ctx_->WaitComm(ring_id);

    // copy dst/cpu to dst
    auto dst_cpu_tensor = dst_cpu.Get<framework::LoDTensor>();
    auto dst_tensor = dst->GetMutable<framework::LoDTensor>();
    framework::TensorCopySync(dst_cpu_tensor, dst_tensor->place(), dst_tensor);

    inter_parallel_ctx_->WaitComm(ring_id);
  }

  // step 3: call broadcast within node
  VLOG(3) << "/// DEBUG /// step 3: broadcast within node... ";
  node_parallel_ctx_->WaitComm(ring_id);
  node_parallel_ctx_->Broadcast(dst, ring_id);
  node_parallel_ctx_->WaitComm(ring_id);
}

void HeterParallelContext::Broadcast(framework::Variable *src, int ring_id) {
  PADDLE_THROW(platform::errors::Unimplemented("Unimplemented function."));
}

paddle::platform::DeviceContext *HeterParallelContext::GetDeviceContext(
    int ring_id) {
  // directly call the implementation of target parallel ctx.
  return node_parallel_ctx_->GetDeviceContext(ring_id);
}

void HeterParallelContext::WaitCompute(int ring_id) {
  // directly call the implementation of target parallel ctx.
  node_parallel_ctx_->WaitCompute(ring_id);
}

void HeterParallelContext::WaitComm(int ring_id) {
  // directly call the implementation of target parallel ctx.
  node_parallel_ctx_->WaitComm(ring_id);
}

void HeterParallelContext::SynchronizeCompute() {
  // directly call the implementation of target parallel ctx.
  node_parallel_ctx_->SynchronizeCompute();
}

}  //  namespace imperative
}  //  namespace paddle

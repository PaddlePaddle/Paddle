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

#include "paddle/fluid/imperative/gloo_context.h"
#include "paddle/fluid/framework/fleet/gloo_wrapper.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/string/split.h"

namespace paddle {
namespace framework {
class Variable;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace imperative {

void GLOOParallelContext::Init() {
  // PADDLE_THROW(platform::errors::OutOfRange(
  //  "Still not implement Init"));
  VLOG(4) << "Start GLOOParallelContext initialization";
  auto gloo_wrapper = framework::GlooWrapper::GetInstance();
  gloo_wrapper->SetSize(strategy_.nranks_);
  gloo_wrapper->SetRank(strategy_.local_rank_);
  gloo_wrapper->SetPrefix("");
  gloo_wrapper->SetIface("lo");
  auto addr = paddle::string::Split(strategy_.trainer_endpoints_[0], ':');
  VLOG(4) << "Server is" << strategy_.trainer_endpoints_[0];
  std::string host = addr[0];
  int port = std::stoi(addr[1]);
  gloo_wrapper->SetHttpStore(host, port, "worker");
  gloo_wrapper->Init();
  device_ = std::unique_ptr<platform::CPUDeviceContext>(
      new platform::CPUDeviceContext(platform::CPUPlace()));
}

void GLOOParallelContext::InitWithRingID(int ring_id) {
  PADDLE_THROW(
      platform::errors::OutOfRange("Still not implement InitWithRingID"));
}

#define GLOO_CASE(type, T, gw)                                        \
  case type: {                                                        \
    VLOG(4) << "Use the gloo all reduce to sync. SRC:" << src_tensor; \
    std::vector<T> send_vector##T;                                    \
    framework::TensorToVector<T>(src_tensor, &send_vector##T);        \
    auto recv_vector##T = gw->AllReduce<T>(send_vector##T);           \
    framework::TensorFromVector<T>(recv_vector##T, dst_tensor);       \
    VLOG(4) << "DST:" << *dst_tensor;                                 \
    break;                                                            \
  }

void GLOOParallelContext::AllReduceByStream(const framework::Variable &src,
                                            framework::Variable *dst,
                                            int ring_id, bool use_calc_stream) {
  // AllReduce(src, dst, strategy_, ring_id, use_calc_stream);
  auto src_tensor = src.Get<framework::LoDTensor>();
  auto *dst_tensor = dst->GetMutable<framework::LoDTensor>();
  auto gloo_wrapper = framework::GlooWrapper::GetInstance();
  dst_tensor->Resize(src_tensor.dims());
  switch (src_tensor.type()) {
    GLOO_CASE(framework::proto::VarType::FP32, float, gloo_wrapper);
    GLOO_CASE(framework::proto::VarType::FP64, double, gloo_wrapper);
    GLOO_CASE(framework::proto::VarType::INT32, int, gloo_wrapper);
    GLOO_CASE(framework::proto::VarType::INT64, int64_t, gloo_wrapper);
    default: {
      PADDLE_THROW(
          platform::errors::InvalidArgument("Invalid datatype for allreduce"));
    }
  }
  gloo_wrapper->Barrier();
}

paddle::platform::DeviceContext *GLOOParallelContext::GetDeviceContext(
    int ring_id) {
  // return the CPUDeviceContext
  return device_.get();
}

void GLOOParallelContext::WaitCompute(int ring_id) {
  // do nothing because cpu don't need sync
  return;
}

void GLOOParallelContext::WaitComm(int ring_id) {
  // do nothing because cpu don't need sync
  return;
}

void GLOOParallelContext::SynchronizeCompute() {
  // do nothing because cpu don't need sync
  return;
}

}  //  namespace imperative
}  //  namespace paddle

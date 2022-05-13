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


#include "paddle/fluid/imperative/offload_scheduler.h"
#include "paddle/fluid/string/string_helper.h"

#include <iostream>

namespace paddle {
namespace imperative {

OffloadScheduler::OffloadScheduler(
  const std::vector<std::shared_ptr<imperative::VarBase>> &vars,
  const Place& src_place, const Place& dst_place, const int64_t num_streams)
    : vars_(vars),
      src_place_(src_place),
      dst_place_(dst_place),
      num_streams_(num_streams) {

  VLOG(3) << "Start construct the OffloadScheduler ...";

  contexts_.resize(num_streams);
  events_.resize(vars_.size());

  for (int64_t index = 0; index < num_streams; index++) {
    contexts_[index].reset(new CUDADeviceContext(dst_place));
  }

}

const std::shared_ptr<imperative::VarBase> OffloadScheduler::CopyVarToGPUPlace(
  const int64_t var_index, const int64_t device_id) {
  auto& var = vars_[var_index];
  Place var_place = var->Place();

  if (platform::is_gpu_place(var_place)) {
    return var;
  }

  size_t stream_index = var_index % num_streams_;

  PADDLE_ENFORCE_EQ(
    var->Var().IsType<framework::LoDTensor>(), true,
    platform::errors::PreconditionNotMet("Only support LoDTensor now."));
  
  PADDLE_ENFORCE_EQ(
    platform::is_cuda_pinned_place(var_place), true,
    platform::errors::PreconditionNotMet("Only support CUDAPinned->GPU now."));

  auto& src_tensor = var->Var().Get<framework::LoDTensor>();
  auto new_var = std::make_shared<imperative::VarBase>(
      /* need grad */true, var->Name() + "_in_gpu_place");

  auto* dst_tensor =
      new_var->MutableVar()->GetMutable<framework::LoDTensor>();
  dst_tensor->set_lod(src_tensor.lod());
  new_var->SetPersistable(var->Persistable());
  new_var->SetDataType(var->DataType());
  new_var->SetType(var->Type());
  new_var->SetOverridedStopGradient(var->OverridedStopGradient());

  auto gpu_place = platform::CUDAPlace(device_id);
  framework::TensorCopy(src_tensor, gpu_place, *contexts_[stream_index], dst_tensor);
  events_[var_index].Record(*contexts_[stream_index]);

  return new_var;

}

void OffloadScheduler::WaitStreamForCopyVar(
  const int64_t var_index, const int64_t device_id) {
  auto gpu_place = platform::CUDAPlace(device_id);
  auto* default_ctx = static_cast<CUDADeviceContext*>(
        platform::DeviceContextPool::Instance().Get(gpu_place));

  events_[var_index].Block(*default_ctx);

}

}  // namespace imperative
}  // namespace paddle

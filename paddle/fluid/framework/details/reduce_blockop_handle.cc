//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/details/reduce_blockop_handle.h"
#include "paddle/fluid/framework/details/container_cast.h"
#include "paddle/fluid/framework/details/reduce_and_gather.h"
#include "paddle/fluid/framework/details/variable_visitor.h"

namespace paddle {
namespace framework {
namespace details {

void ReduceBlockOpHandle::RunImpl() {
  if (places_.size() == 1) return;

  WaitInputVarGenerated();

  auto in_var_handles = DynamicCast<VarHandle>(inputs_);
  auto out_var_handles = DynamicCast<VarHandle>(outputs_);

  PADDLE_ENFORCE_EQ(in_var_handles.size(),
                    places_.size() * out_var_handles.size(),
                    "The number of input and output is not consistent.");

#ifdef PADDLE_WITH_CUDA
  std::vector<const LoDTensor *> lod_tensors = GetInputValues();
  auto type = lod_tensors[0]->type();
  auto numel = lod_tensors[0]->numel();
  std::vector<std::function<void()>> nccl_reduce_calls;
  auto root_dev_id =
      boost::get<platform::CUDAPlace>(places_[dst_scope_id_]).device;
  for (size_t i = 0; i < local_scopes_.size(); ++i) {
    int dev_id =
        boost::get<platform::CUDAPlace>(lod_tensors[i]->place()).device;
    auto &nccl_ctx = nccl_ctxs_->at(dev_id);

    void *buffer = const_cast<void *>(lod_tensors[i]->data<void>());
    void *recvbuffer = nullptr;
    if (i == dst_scope_id_) {
      auto reduce_var = local_scopes_[dst_scope_id_]->FindVar(var_name_);
      recvbuffer = reduce_var->GetMutable<framework::LoDTensor>()->mutable_data(
          lod_tensors[dst_scope_id_]->place());
    }

    int nccl_type = platform::ToNCCLDataType(type);
    nccl_reduce_calls.emplace_back(
        [buffer, recvbuffer, nccl_type, numel, root_dev_id, &nccl_ctx] {
          PADDLE_ENFORCE(platform::dynload::ncclReduce(
              buffer, recvbuffer, numel, static_cast<ncclDataType_t>(nccl_type),
              ncclSum, root_dev_id, nccl_ctx.comm_, nccl_ctx.stream()));
        });
  }
#endif

  std::vector<const Scope *> var_scopes;
  for (auto *s : local_scopes_) {
    auto var = s->FindVar(kLocalExecScopeName);
    PADDLE_ENFORCE_NOT_NULL(var);
    var_scopes.emplace_back(var->Get<Scope *>());
  }

  // Reduce the variable which is in CPU side.
  std::vector<std::vector<const LoDTensor *>> cpu_lod_tensors;
  std::vector<Variable *> cpu_out_vars;
  for (auto out_var_h : out_var_handles) {
    auto out_var = var_scopes[dst_scope_id_]->FindVar(out_var_h->name_);
    PADDLE_ENFORCE_NOT_NULL(out_var);

    if (platform::is_cpu_place(out_var->Get<LoDTensor>().place())) {
      std::vector<const LoDTensor *> lod_tensors;
      for (size_t i = 0; i < places_.size(); ++i) {
        lod_tensors.emplace_back(
            &(var_scopes[i]->FindVar(out_var_h->name_)->Get<LoDTensor>()));
        PADDLE_ENFORCE(platform::is_cpu_place(lod_tensors.back()->place()));
      }

      cpu_lod_tensors.push_back(lod_tensors);
      cpu_out_vars.push_back(out_var);
    }
  }
#ifdef PADDLE_WITH_CUDA
  this->RunAndRecordEvent([&] {
    {
      platform::NCCLGroupGuard guard;
      for (auto &call : nccl_reduce_calls) {
        call();
      }
    }
    for (size_t i = 0; i < cpu_lod_tensors.size(); ++i) {
      ReduceLoDTensor func(cpu_lod_tensors[i],
                           cpu_out_vars[i]->GetMutable<framework::LoDTensor>());
      VisitDataType(ToDataType(cpu_lod_tensors[i][0]->type()), func);
    }
  });
#endif
}

std::vector<const LoDTensor *> ReduceBlockOpHandle::GetInputValues() {
  std::vector<const LoDTensor *> lod_tensors;

  auto &value_0 = GetInputValue(0);
  lod_tensors.emplace_back(&value_0);

  std::unordered_set<int> dev_id_set;
  auto dev_id = boost::get<platform::CUDAPlace>(value_0.place()).device;
  dev_id_set.insert(dev_id);

  for (size_t i = 1; i < local_scopes_.size(); ++i) {
    auto &value = GetInputValue(i);
    PADDLE_ENFORCE_EQ(value_0.type(), value.type());
    PADDLE_ENFORCE_EQ(value_0.numel(), value.numel());
    lod_tensors.emplace_back(&value);

    auto dev_id = boost::get<platform::CUDAPlace>(value.place()).device;
    if (dev_id_set.count(dev_id)) {
      PADDLE_THROW("dev_%d has been in dev_id_set.", dev_id);
    }
    dev_id_set.insert(dev_id);
  }
  return lod_tensors;
}

const LoDTensor &ReduceBlockOpHandle::GetInputValue(size_t idx) {
  auto reduce_var = local_scopes_.at(idx)->FindVar(var_name_);
  PADDLE_ENFORCE_NOT_NULL(reduce_var, "%s is not found.", var_name_);
  auto &lod_tensor = reduce_var->Get<LoDTensor>();
  PADDLE_ENFORCE(platform::is_gpu_place(lod_tensor.place()));
  return lod_tensor;
}

std::string ReduceBlockOpHandle::Name() const { return "reduce_block"; }
}  // namespace details
}  // namespace framework
}  // namespace paddle

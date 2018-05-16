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

#include "paddle/fluid/framework/details/reduce_op_handle.h"
#include "paddle/fluid/framework/details/container_cast.h"
#include "paddle/fluid/framework/details/reduce_and_gather.h"
#include "paddle/fluid/framework/details/variable_visitor.h"

namespace paddle {
namespace framework {
namespace details {

void ReduceOpHandle::RunImpl() {
  if (places_.size() == 1) return;

  WaitInputVarGenerated();

  auto in_var_handles = DynamicCast<VarHandle>(inputs_);
  auto out_var_handles = DynamicCast<VarHandle>(outputs_);

  PADDLE_ENFORCE_EQ(in_var_handles.size(),
                    places_.size() * out_var_handles.size(),
                    "The number of input and output is not consistent.");

  std::vector<const Scope *> var_scopes;
  for (auto *s : local_scopes_) {
    auto var = s->FindVar(kLocalExecScopeName);
    PADDLE_ENFORCE_NOT_NULL(var);
    var_scopes.emplace_back(var->Get<Scope *>());
  }

  if (var_name_.size() != 0) {
    ReduceGroup(out_var_handles, var_scopes);
  } else {
    ReduceInput(in_var_handles, out_var_handles, var_scopes);
  }
}

void ReduceOpHandle::ReduceInput(
    const std::vector<VarHandle *> &in_var_handles,
    const std::vector<VarHandle *> &out_var_handles,
    const std::vector<const Scope *> &var_scopes) {
  auto get_var = [&](const VarHandle &var_handle) -> Variable * {
    auto var = var_scopes.at(var_handle.scope_idx_)->FindVar(var_handle.name_);
    PADDLE_ENFORCE_NOT_NULL(var);
    return var;
  };

  auto pre_in_var = get_var(*in_var_handles[0]);
  VarHandle *out_var_handle = out_var_handles.front();

  // NOTE: The Places of all input tensor must be all on CPU or all on GPU.
  std::vector<platform::Place> in_places;  // used to get dev_ctx
  for (auto in_var_h : in_var_handles) {
    in_places.emplace_back(in_var_h->place_);
    VariableVisitor::EnforceShapeAndDTypeEQ(*pre_in_var, *get_var(*in_var_h));
  }

  auto out_var = get_var(*out_var_handle);
  auto out_p = out_var_handle->place_;
  auto in_p = VariableVisitor::GetMutableTensor(pre_in_var).place();
  if (platform::is_gpu_place(in_p)) {
    PADDLE_ENFORCE(platform::is_gpu_place(out_p),
                   "Places of input and output must be all on GPU.");
  } else {
    out_p = platform::CPUPlace();
  }

  if (pre_in_var->IsType<framework::SelectedRows>()) {
    this->RunAndRecordEvent([&] {
      std::vector<const SelectedRows *> in_selected_rows =
          this->GetInputValues<SelectedRows>(in_var_handles, var_scopes);
      GatherSelectedRows(in_selected_rows, in_places, this->dev_ctxes_, out_p,
                         out_var->GetMutable<framework::SelectedRows>());
    });
  } else {
    std::vector<const LoDTensor *> lod_tensors =
        this->GetInputValues<LoDTensor>(in_var_handles, var_scopes);
    if (paddle::platform::is_cpu_place(lod_tensors[0]->place())) {
      this->RunAndRecordEvent([&] {
        ReduceLoDTensor func(lod_tensors,
                             out_var->GetMutable<framework::LoDTensor>());
        VisitDataType(ToDataType(lod_tensors[0]->type()), func);
      });
    } else if (paddle::platform::is_gpu_place(lod_tensors[0]->place())) {
#ifdef PADDLE_WITH_CUDA
      auto pre_in = pre_in_var->Get<framework::LoDTensor>();
      VariableVisitor::ShareDimsAndLoD(*pre_in_var, out_var);
      VariableVisitor::GetMutableTensor(out_var).mutable_data(
          out_var_handle->place_, pre_in.type());

      int dst_dev_id =
          boost::get<platform::CUDAPlace>(out_var_handle->place_).device;
      std::vector<std::function<void()>> all_reduce_calls;

      NCCLReduce(lod_tensors, dst_dev_id, out_var, &all_reduce_calls);

      this->RunAndRecordEvent([&] {
        platform::NCCLGroupGuard guard;
        for (auto &call : all_reduce_calls) {
          call();
        }
      });
#else
      PADDLE_THROW("CUDA is not enabled.");
#endif
    } else {
      PADDLE_THROW("Place should be CPUPlace or CUDAPlace.");
    }
  }
}

void ReduceOpHandle::ReduceGroup(
    const std::vector<VarHandle *> &out_var_handles,
    const std::vector<const Scope *> &var_scopes) {
#ifdef PADDLE_WITH_CUDA
  std::vector<const LoDTensor *> lod_tensors = GetGroupValues();
  std::vector<std::function<void()>> nccl_reduce_calls;
  auto dst_dev_id =
      boost::get<platform::CUDAPlace>(places_[dst_scope_id_]).device;
  auto reduce_var = local_scopes_[dst_scope_id_]->FindVar(var_name_);
  NCCLReduce(lod_tensors, dst_dev_id, reduce_var, &nccl_reduce_calls);

  this->RunAndRecordEvent([&] {
    platform::NCCLGroupGuard guard;
    for (auto &call : nccl_reduce_calls) {
      call();
    }
  });
#else
  PADDLE_THROW("CUDA is not enabled.");
#endif
  // Reduce the variable which is in CPU side.
  std::vector<std::vector<const LoDTensor *>> cpu_lod_tensors;
  std::vector<Variable *> cpu_out_vars;
  for (auto out_var_h : out_var_handles) {
    auto out_var = var_scopes[dst_scope_id_]->FindVar(out_var_h->name_);
    PADDLE_ENFORCE_NOT_NULL(out_var);

    if (is_cpu_place(out_var->Get<LoDTensor>().place())) {
      std::vector<const LoDTensor *> lod_tensors;
      for (size_t i = 0; i < places_.size(); ++i) {
        lod_tensors.emplace_back(
            &(var_scopes[i]->FindVar(out_var_h->name_)->Get<LoDTensor>()));
        PADDLE_ENFORCE(is_cpu_place(lod_tensors.back()->place()));
      }

      cpu_lod_tensors.push_back(lod_tensors);
      cpu_out_vars.push_back(out_var);
    }
  }
  RunAndRecordEvent([&] {
    for (size_t i = 0; i < cpu_lod_tensors.size(); ++i) {
      ReduceLoDTensor func(cpu_lod_tensors[i],
                           cpu_out_vars[i]->GetMutable<LoDTensor>());
      VisitDataType(ToDataType(cpu_lod_tensors[i][0]->type()), func);
    }
  });
}

#ifdef PADDLE_WITH_CUDA
void ReduceOpHandle::NCCLReduce(
    const std::vector<const LoDTensor *> &lod_tensors, const size_t dst_dev_id,
    Variable *out_var, std::vector<std::function<void()>> *nccl_reduce_calls) {
  int nccl_type = platform::ToNCCLDataType(lod_tensors[0]->type());
  size_t numel = static_cast<size_t>(lod_tensors[0]->numel());

  for (size_t i = 0; i < lod_tensors.size(); ++i) {
    int dev_id = get<platform::CUDAPlace>(lod_tensors[i]->place()).device;
    auto &nccl_ctx = nccl_ctxs_->at(dev_id);

    void *buffer = const_cast<void *>(lod_tensors[i]->data<void>());
    void *recvbuffer = nullptr;
    if (i == dst_dev_id) {
      recvbuffer = out_var->GetMutable<LoDTensor>()->mutable_data(
          lod_tensors[dst_dev_id]->place());
    }

    nccl_reduce_calls->emplace_back(
        [buffer, recvbuffer, nccl_type, numel, dst_dev_id, &nccl_ctx] {
          PADDLE_ENFORCE(platform::dynload::ncclReduce(
              buffer, recvbuffer, numel, static_cast<ncclDataType_t>(nccl_type),
              ncclSum, dst_dev_id, nccl_ctx.comm_, nccl_ctx.stream()));
        });
  }
}
#endif

std::vector<const LoDTensor *> ReduceOpHandle::GetGroupValues() {
  auto get_group_value = [&](int idx) -> const LoDTensor & {
    auto reduce_var = local_scopes_.at(idx)->FindVar(var_name_);
    PADDLE_ENFORCE_NOT_NULL(reduce_var, "%s is not found.", var_name_);
    auto &lod_tensor = reduce_var->Get<LoDTensor>();
    PADDLE_ENFORCE(platform::is_gpu_place(lod_tensor.place()));
    return lod_tensor;
  };

  std::vector<const LoDTensor *> lod_tensors;

  auto &value_0 = get_group_value(0);
  lod_tensors.emplace_back(&value_0);

  std::unordered_set<int> dev_id_set;
  auto dev_id = boost::get<platform::CUDAPlace>(value_0.place()).device;
  dev_id_set.insert(dev_id);

  for (size_t i = 1; i < local_scopes_.size(); ++i) {
    auto &value = get_group_value(i);
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

template <typename T>
std::vector<const T *> ReduceOpHandle::GetInputValues(
    const std::vector<VarHandle *> &in_var_handles,
    const std::vector<const Scope *> &var_scopes) {
  std::vector<const T *> in_selected_rows;
  for (auto *in_handle : in_var_handles) {
    auto &in_sr = var_scopes.at(in_handle->scope_idx_)
                      ->FindVar(in_handle->name_)
                      ->Get<T>();
    in_selected_rows.emplace_back(&in_sr);
  }
  return in_selected_rows;
}

std::string ReduceOpHandle::Name() const { return "reduce_block"; }
}  // namespace details
}  // namespace framework
}  // namespace paddle

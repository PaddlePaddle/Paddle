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
#include "paddle/fluid/operators/distributed/collective_client.h"
#include "paddle/fluid/operators/distributed/collective_server.h"
#include "paddle/fluid/operators/math/selected_rows_functor.h"
#include "paddle/fluid/platform/profiler.h"

DEFINE_bool(
    cpu_deterministic, false,
    "Whether to make the result of computation deterministic in CPU side.");

namespace paddle {
namespace framework {
namespace details {

void ReduceOpHandle::GatherRemoteSelectedRows(
    platform::DeviceContext *dev_ctx, Scope *scope, const std::string &var_name,
    std::vector<const SelectedRows *> *remote) {
  std::vector<std::string> eps = collective_context_.end_points_;
  eps.erase(eps.begin() + collective_context_.rank_id_);

  std::string ep_str;
  for (auto ep : eps) {
    ep_str += "," + ep;
  }
  VLOG(30) << "gather from eps:" << ep_str;

  operators::distributed::CollectiveClient *client =
      operators::distributed::CollectiveClient::GetInstance();
  PADDLE_ENFORCE(client->Gather(eps, *dev_ctx, *scope, var_name, remote));
}

void ReduceOpHandle::WaitLocalSelectedRows(
    const std::map<platform::Place, platform::DeviceContext *> &dev_ctxes) {
  // TODO(gongwb): use event wait?
  for (auto &dev_ctx : dev_ctxes) {
    dev_ctx.second->Wait();
  }
}

void ReduceOpHandle::GatherSelectedRows(
    const std::vector<const SelectedRows *> &src_selected_rows,
    const std::vector<platform::Place> &in_places,
    const std::map<platform::Place, platform::DeviceContext *> &dev_ctxes,
    VarHandle *out_var_handle, const platform::Place &out_place,
    SelectedRows *dst_selecte_rows) {
  if (collective_context_.end_points_.size() <= 1) {
    GatherLocalSelectedRows(src_selected_rows, in_places, dev_ctxes, out_place,
                            dst_selecte_rows);
    return;
  }

  // 1. gather local selected rows, merge them
  std::string gathered_var_name = out_var_handle->name_ + "_gathered_tmp";
  auto scope = local_scopes_.at(out_var_handle->scope_idx_);
  auto gathered_var_mid = scope->FindVar(gathered_var_name);
  auto gathered_select_rows =
      gathered_var_mid->GetMutable<framework::SelectedRows>();
  GatherLocalSelectedRows(src_selected_rows, in_places, dev_ctxes, out_place,
                          gathered_select_rows);

  VLOG(40) << "gathered_select_rows :" << gathered_select_rows->Info();

  // merge them
  auto merged_dev_ctx = dev_ctxes.at(out_place);
  std::string merged_var_name = out_var_handle->name_ + "_merged_tmp";
  auto merged_var_mid =
      local_scopes_.at(out_var_handle->scope_idx_)->FindVar(merged_var_name);
  // operators::math::scatter::MergeAdd<platform::DeviceContext,
  // gather_select_rows->value().type()> merge_func;
  // FIXME(gongwb):get type?
  auto merged_select_rows = merged_var_mid->GetMutable<SelectedRows>();
  operators::math::scatter::MergeAdd<platform::DeviceContext, float> merge_func;
  merge_func(*merged_dev_ctx, *gathered_select_rows, merged_select_rows);
  VLOG(40) << "merged_select_rows :" << merged_select_rows->Info();

  // 2. start collective server if it doesn't exist
  std::vector<const SelectedRows *> remote;
  operators::distributed::CollectiveServer *server =
      operators::distributed::CollectiveServer::GetInstance(
          collective_context_.end_points_[collective_context_.rank_id_],
          collective_context_.end_points_.size());
  server->WaitNotInService();
  server->ResetContext(scope, merged_dev_ctx);
  WaitLocalSelectedRows(dev_ctxes);
  server->SetInService();

  // 3. gather them from all nodes.
  std::vector<const SelectedRows *> all;
  GatherRemoteSelectedRows(merged_dev_ctx, scope, out_var_handle->name_, &all);
  all.insert(all.begin() + collective_context_.rank_id_, merged_select_rows);
  merge_func(*merged_dev_ctx, all, dst_selecte_rows);
  VLOG(40) << "all_select_rows :" << dst_selecte_rows->Info();

  server->WaitNotInService();
  std::vector<std::string> vars{gathered_var_name, merged_var_name};
  scope->EraseVars(vars);
}

void ReduceOpHandle::RunImpl() {
  platform::RecordEvent record_event(Name(), dev_ctxes_.cbegin()->second);

  if (places_.size() == 1) return;
  // the input and output may have dummy var.
  auto in_var_handles = DynamicCast<VarHandle>(inputs_);

  PADDLE_ENFORCE_EQ(
      in_var_handles.size(), places_.size(),
      "The number of output should equal to the number of places.");

  VarHandle *out_var_handle;
  {
    auto out_var_handles = DynamicCast<VarHandle>(outputs_);

    PADDLE_ENFORCE_EQ(out_var_handles.size(), 1,
                      "The number of output should be one.");
    out_var_handle = out_var_handles.front();
  }

  auto in_0_handle = in_var_handles[0];

  std::vector<const Scope *> var_scopes;
  for (auto *s : local_scopes_) {
    var_scopes.emplace_back(s->FindVar(kLocalExecScopeName)->Get<Scope *>());
  }

  auto pre_in_var =
      var_scopes.at(in_0_handle->scope_idx_)->FindVar(in_0_handle->name_);
  PADDLE_ENFORCE_NOT_NULL(pre_in_var);

  // Wait input done, this Wait is asynchronous operation
  WaitInputVarGenerated();

  // NOTE: The Places of all input tensor must be all on CPU or all on GPU.
  std::vector<platform::Place> in_places;  // used to get dev_ctx
  for (auto *in_handle : in_var_handles) {
    in_places.emplace_back(in_handle->place_);
    auto in_var =
        var_scopes.at(in_handle->scope_idx_)->FindVar(in_handle->name_);
    PADDLE_ENFORCE_NOT_NULL(in_var);
    VariableVisitor::EnforceShapeAndDTypeEQ(*pre_in_var, *in_var);
  }

  auto out_var =
      var_scopes.at(out_var_handle->scope_idx_)->FindVar(out_var_handle->name_);
  PADDLE_ENFORCE_NOT_NULL(out_var);

  // NOTE: The tensors' Place of input and output must be all on GPU or all on
  // CPU.
  auto in_p = VariableVisitor::GetMutableTensor(pre_in_var).place();
  platform::Place t_out_p;
  if (platform::is_gpu_place(in_p)) {
    PADDLE_ENFORCE(platform::is_gpu_place(out_var_handle->place_),
                   "Places of input and output must be all on GPU.");
    t_out_p = out_var_handle->place_;
  } else {
    t_out_p = platform::CPUPlace();
  }

  if (pre_in_var->IsType<framework::SelectedRows>()) {
    this->RunAndRecordEvent([&] {
      std::vector<const SelectedRows *> in_selected_rows =
          GetInputValues<SelectedRows>(in_var_handles, var_scopes);

      GatherSelectedRows(in_selected_rows, in_places, dev_ctxes_,
                         out_var_handle, t_out_p,
                         out_var->GetMutable<framework::SelectedRows>());
    });
  } else {
    std::vector<const LoDTensor *> lod_tensors =
        GetInputValues<LoDTensor>(in_var_handles, var_scopes);

    if (paddle::platform::is_cpu_place(lod_tensors[0]->place())) {
      this->RunAndRecordEvent([&] {
        // FIXME(zcd): The order of summing is important,
        // especially when the type of data is float or double.
        // For example, the result of `a+b+c+d` may be different
        // with the result of `c+a+b+d`, so the summing order should be fixed.
        if (!FLAGS_cpu_deterministic) {
          ReduceLoDTensor func(lod_tensors,
                               out_var->GetMutable<framework::LoDTensor>());
          VisitDataType(ToDataType(lod_tensors[0]->type()), func);
        } else {
          // We sum lod_tensors to reduce_sum_trg which is in local_scopes_0
          // here, but it doesn't mean reduce_sum_trg must be in local_scopes_0.
          auto &reduce_sum_trg = *this->local_scopes_[0]
                                      ->FindVar(kLocalExecScopeName)
                                      ->Get<Scope *>()
                                      ->FindVar(out_var_handle->name_)
                                      ->GetMutable<framework::LoDTensor>();
          ReduceLoDTensor func(lod_tensors, &reduce_sum_trg);
          VisitDataType(ToDataType(lod_tensors[0]->type()), func);

          auto trg = out_var->GetMutable<framework::LoDTensor>();
          if (reduce_sum_trg.data<void>() != trg->data<void>()) {
            TensorCopy(reduce_sum_trg, platform::CPUPlace(), trg);
          }
        }
      });
    } else if (paddle::platform::is_gpu_place(lod_tensors[0]->place())) {
#ifdef PADDLE_WITH_CUDA
      auto pre_in = pre_in_var->Get<framework::LoDTensor>();
      VariableVisitor::ShareDimsAndLoD(*pre_in_var, out_var);
      VariableVisitor::GetMutableTensor(out_var).mutable_data(
          out_var_handle->place_, pre_in.type());

      auto out_p = out_var_handle->place_;
      int root_id = boost::get<platform::CUDAPlace>(out_p).device;
      std::vector<std::function<void()>> all_reduce_calls;
      for (size_t i = 0; i < var_scopes.size(); ++i) {
        auto &p = in_places[i];
        auto &lod_tensor = *lod_tensors[i];

        int dev_id = boost::get<platform::CUDAPlace>(p).device;
        auto &nccl_ctx = nccl_ctxs_->at(dev_id);

        void *buffer = const_cast<void *>(lod_tensor.data<void>());
        void *recvbuffer = nullptr;
        if (root_id == dev_id) {
          recvbuffer =
              out_var->GetMutable<framework::LoDTensor>()->mutable_data(
                  out_var_handle->place_);
        }

        int type = platform::ToNCCLDataType(lod_tensor.type());
        size_t numel = static_cast<size_t>(lod_tensor.numel());
        all_reduce_calls.emplace_back(
            [buffer, recvbuffer, type, numel, root_id, &nccl_ctx] {
              PADDLE_ENFORCE(platform::dynload::ncclReduce(
                  buffer, recvbuffer, numel, static_cast<ncclDataType_t>(type),
                  ncclSum, root_id, nccl_ctx.comm_, nccl_ctx.stream()));
            });
      }

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

template <typename T>
std::vector<const T *> ReduceOpHandle::GetInputValues(
    const std::vector<VarHandle *> &in_var_handles,
    const std::vector<const Scope *> &var_scopes) const {
  std::vector<const T *> in_selected_rows;
  for (auto *in_handle : in_var_handles) {
    auto &in_sr = var_scopes.at(in_handle->scope_idx_)
                      ->FindVar(in_handle->name_)
                      ->Get<T>();
    in_selected_rows.emplace_back(&in_sr);
  }
  return in_selected_rows;
}

std::string ReduceOpHandle::Name() const { return "reduce"; }
}  // namespace details
}  // namespace framework
}  // namespace paddle

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
#include "paddle/fluid/framework/details/print_help.h"
#include "paddle/fluid/framework/details/reduce_and_gather.h"
#include "paddle/fluid/framework/details/variable_visitor.h"
#if defined PADDLE_WITH_CUDA && defined PADDLE_WITH_DISTRIBUTE
#include "paddle/fluid/operators/distributed/collective_client.h"
#include "paddle/fluid/operators/distributed/collective_server.h"
#include "paddle/fluid/operators/distributed/request_handler.h"
#endif
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/math/selected_rows_functor.h"
#include "paddle/fluid/platform/profiler.h"

DEFINE_bool(
    cpu_deterministic, false,
    "Whether to make the result of computation deterministic in CPU side.");

namespace paddle {
namespace framework {
namespace details {

std::once_flag CollectiveContext::init_flag_;
std::unique_ptr<CollectiveContext> CollectiveContext::context_;

static inline std::string GetRemoteVarName(const std::string &var_name,
                                           int trainer_id) {
  return string::Sprintf("%s_merged_tmp@trainer_%d", var_name, trainer_id);
}

void ReduceOpHandle::Wait(
    const std::map<platform::Place, platform::DeviceContext *> &dev_ctxes) {
  // TODO(gongwb): use event wait?
  for (auto &dev_ctx : dev_ctxes) {
    dev_ctx.second->Wait();
  }
}

inline void GatherLocalSelectedRows2(
    const std::vector<const SelectedRows *> &src_selecte_rows_,
    const std::vector<platform::Place> &in_places,
    const std::map<platform::Place, platform::DeviceContext *> &dev_ctxes,
    const platform::Place &out_place, SelectedRows *dst_selecte_rows) {
  PADDLE_ENFORCE(!src_selecte_rows_.empty());

  std::vector<Tensor> in_tensors;
  std::vector<int64_t> out_rows;

  for (auto in_sr_ptr : src_selecte_rows_) {
    auto &in_sr = *in_sr_ptr;
    in_tensors.emplace_back(in_sr.value());
    out_rows.insert(out_rows.end(), in_sr.rows().begin(), in_sr.rows().end());
  }

  auto &pre_in = src_selecte_rows_[0];

  auto &dst_tensor = *dst_selecte_rows;
  dst_tensor.set_height(pre_in->height());
  dst_tensor.set_rows(out_rows);
  size_t rows = out_rows.size();
  DDim out_dim = pre_in->GetCompleteDims();
  out_dim[0] = static_cast<int64_t>(rows);
  dst_tensor.mutable_value()->Resize(out_dim);
  dst_tensor.mutable_value()->mutable_data(out_place, pre_in->value().type());
  Tensor *out_tensor = dst_tensor.mutable_value();

  // copy
  /*
  int s = 0, e = 0;
  for (size_t j = 0; j < in_tensors.size(); ++j) {
    e += in_tensors[j].dims()[0];
    auto sub_out = out_tensor->Slice(s, e);
    paddle::framework::TensorCopy(in_tensors[j], out_place,
                                  *(dev_ctxes.at(in_places[j])), &sub_out);
    s = e;
  }
  */

  auto merged_dev_ctx2 =
      dynamic_cast<platform::CUDADeviceContext *>(dev_ctxes.at(out_place));
  operators::math::SetConstant<platform::CUDADeviceContext, float>
      constant_functor;
  constant_functor(*merged_dev_ctx2, out_tensor, static_cast<float>(0));
}

#if defined PADDLE_WITH_CUDA && defined PADDLE_WITH_DISTRIBUTE
template <typename DevCtx, typename DataType>
void ReduceOpHandle::GatherSelectedRows(
    const std::vector<const SelectedRows *> &src_selected_rows,
    const std::vector<platform::Place> &in_places,
    const std::map<platform::Place, platform::DeviceContext *> &dev_ctxes,
    VarHandle *out_var_handle, const platform::Place &out_place,
    SelectedRows *dst_selected_rows) {
  /*
  auto merged_dev_ctx2 = dynamic_cast<DevCtx *>(dev_ctxes.at(out_place));
  auto input_width = src_selected_rows[0]->value().dims()[1];
  auto input_height = src_selected_rows[0]->height();

  auto rows = src_selected_rows[0]->rows();
  dst_selected_rows->set_rows(rows);
  dst_selected_rows->set_height(input_height);
  dst_selected_rows->mutable_value()->mutable_data<float>(
      framework::make_ddim({static_cast<int64_t>(rows.size()), input_width}),
  merged_dev_ctx2->GetPlace());
  operators::math::SetConstant<platform::CUDADeviceContext, float>
  constant_functor;
  constant_functor(*merged_dev_ctx2, dst_selected_rows->mutable_value(),
  static_cast<float>(0));
  return;
  */

  const CollectiveContext &collective_context =
      *CollectiveContext::GetInstance();

  // 1. gather local selected rows, merge them
  std::string gathered_var_name = out_var_handle->name_ + "_gathered_tmp";
  auto scope = local_scopes_.at(out_var_handle->scope_idx_);
  auto gathered_var_mid = scope->Var(gathered_var_name);
  auto gathered_select_rows =
      gathered_var_mid->GetMutable<framework::SelectedRows>();
  GatherLocalSelectedRows(src_selected_rows, in_places, dev_ctxes, out_place,
                          gathered_select_rows);
  // GatherLocalSelectedRows2(src_selected_rows, in_places, dev_ctxes,
  // out_place,
  // dst_selected_rows);
  // return;

  // FIXME(gongwb): remove this Wait.
  Wait(dev_ctxes);

  VLOG(10) << "in reduce gathered_select_rows:"
           << GetVarInfo(scope, gathered_var_name);

  // merge them
  auto merged_dev_ctx = dynamic_cast<DevCtx *>(dev_ctxes.at(out_place));
  std::string merged_var_name =
      GetRemoteVarName(out_var_handle->name_, collective_context.trainer_id_);
  auto merged_select_rows =
      scope->Var(merged_var_name)->GetMutable<SelectedRows>();
  operators::math::scatter::MergeAdd<DevCtx, DataType> merge_func;
  merge_func(*merged_dev_ctx, *gathered_select_rows, merged_select_rows);
  merged_dev_ctx->Wait();

  VLOG(10) << "in reduce merged_select_rows:"
           << GetVarInfo(scope, merged_var_name);

  // 2. start collective server if it doesn't exist
  operators::distributed::CollectiveServer *server =
      operators::distributed::CollectiveServer::GetInstance(
          collective_context.endpoints_[collective_context.trainer_id_],
          collective_context.endpoints_.size() - 1);

  auto rpc_server = server->GetRPCServer();
  if (1) {
    rpc_server->RegisterVar(merged_var_name,
                            operators::distributed::kRequestGetMonomerVariable,
                            scope, merged_dev_ctx);
  }

  // 3. gather them from all remote nodes.
  std::vector<const SelectedRows *> remote;
  operators::distributed::CollectiveClient *client =
      operators::distributed::CollectiveClient::GetInstance();

  std::vector<operators::distributed::RemoteVar> vars;
  for (unsigned int i = 0; i < collective_context.endpoints_.size(); i++) {
    if (i == (unsigned)collective_context.trainer_id_) continue;

    operators::distributed::RemoteVar var;
    var.trainer_id_ = i;
    var.var_name_ = GetRemoteVarName(out_var_handle->name_, i);
    var.ep_ = collective_context.endpoints_[i];

    vars.push_back(var);
    VLOG(4) << "gather from:" << var.String();
  }

  // erase gathered vars
  // merged_dev_ctx->Wait();
  // scope->EraseVars(std::vector<std::string>{gathered_var_name});

  PADDLE_ENFORCE(client->Gather(vars, &remote, *merged_dev_ctx, scope));
  // merged_dev_ctx->Wait();
  if (1) {
    PADDLE_ENFORCE(remote.size() == vars.size());

    // 4. merged local selected rows.
    std::vector<const SelectedRows *> all;
    all.resize(collective_context.endpoints_.size());
    for (auto v : vars) {
      PADDLE_ENFORCE(v.trainer_id_ < collective_context.endpoints_.size() &&
                     v.trainer_id_ >= 0);
      all[v.trainer_id_] =
          scope->FindVar(v.var_name_)->GetMutable<SelectedRows>();
      PADDLE_ENFORCE(all[v.trainer_id_]);
    }
    PADDLE_ENFORCE(collective_context.trainer_id_ <
                       collective_context.endpoints_.size() &&
                   collective_context.trainer_id_ >= 0);
    all[collective_context.trainer_id_] = merged_select_rows;
    PADDLE_ENFORCE(all[collective_context.trainer_id_]);

    /*
    std::vector<const SelectedRows *> all2;
    all2.push_back(gathered_select_rows);
    merge_func(*merged_dev_ctx, gathered_select_rows, dst_selected_rows);
    */

    operators::math::scatter::MergeAdd<DevCtx, DataType> merge_func;
    merge_func(*merged_dev_ctx, all, dst_selected_rows);
    merged_dev_ctx->Wait();
    VLOG(10) << "in reduce dst_selecte_rows:"
             << GetVarInfo(scope, out_var_handle->name_);
    /*
    dst_selected_rows->mutable_value()->mutable_data<float>(
        framework::make_ddim({767,512}), merged_dev_ctx->GetPlace());
    operators::math::SetConstant<platform::CUDADeviceContext, float>
    constant_functor;
    constant_functor(*merged_dev_ctx, dst_selected_rows->mutable_value(),
    static_cast<float>(0));
    */
  }
  rpc_server->WaitVarBarrier(merged_var_name);
  rpc_server->ClearVar(merged_var_name);

  // 5. clear mid vars
  /*
  std::vector<std::string> tmp_vars{merged_var_name};
  for (auto r : vars) {
    tmp_vars.push_back(r.var_name_);
  }
  scope->EraseVars(tmp_vars);
  */

  for (auto input : src_selected_rows) {
    VLOG(10) << "in reduce: input place:" << input->value().place();
  }

  for (auto it : dev_ctxes) {
    VLOG(10) << "in reduce: ctx place:" << it.first
             << ", dev_ctx place:" << it.second->GetPlace();
  }

  VLOG(10) << "in reduce: out place:" << dst_selected_rows->value().place();
}
#endif

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

      const CollectiveContext &collective_context =
          *CollectiveContext::GetInstance();
      VLOG(10) << "GatherSelectedRows CollectiveContext:"
               << collective_context.String();

      // TODO(gongwb): add cpu support
      if (collective_context.endpoints_.size() <= 1 ||
          is_cpu_place(in_places[0]) || is_cpu_place(t_out_p)) {
        GatherLocalSelectedRows(in_selected_rows, in_places, dev_ctxes_,
                                t_out_p,
                                out_var->GetMutable<framework::SelectedRows>());
        return;
      }

#if defined PADDLE_WITH_CUDA && defined PADDLE_WITH_DISTRIBUTE
      if (in_selected_rows[0]->value().type() ==
          framework::proto::VarType::FP32) {
        VLOG(10) << "selected_rows tensor use float32";
        GatherSelectedRows<platform::CUDADeviceContext, float>(
            in_selected_rows, in_places, dev_ctxes_, out_var_handle, t_out_p,
            out_var->GetMutable<framework::SelectedRows>());
      } else if (in_selected_rows[0]->value().type() ==
                 framework::proto::VarType::FP64) {
        VLOG(10) << "selected_rows tensor use float64";
        GatherSelectedRows<platform::CUDADeviceContext, double>(
            in_selected_rows, in_places, dev_ctxes_, out_var_handle, t_out_p,
            out_var->GetMutable<framework::SelectedRows>());
      } else {
        PADDLE_THROW("only support double or float when gather SelectedRows");
      }
#endif
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
          VisitDataType(lod_tensors[0]->type(), func);
        } else {
          // We sum lod_tensors to reduce_sum_trg which is in local_scopes_0
          // here, but it doesn't mean reduce_sum_trg must be in local_scopes_0.
          auto &reduce_sum_trg = *this->local_scopes_[0]
                                      ->FindVar(kLocalExecScopeName)
                                      ->Get<Scope *>()
                                      ->FindVar(out_var_handle->name_)
                                      ->GetMutable<framework::LoDTensor>();
          ReduceLoDTensor func(lod_tensors, &reduce_sum_trg);
          VisitDataType(lod_tensors[0]->type(), func);

          auto trg = out_var->GetMutable<framework::LoDTensor>();
          if (reduce_sum_trg.data<void>() != trg->data<void>()) {
            TensorCopy(reduce_sum_trg, platform::CPUPlace(), trg);
          }
        }
      });
    } else if (paddle::platform::is_gpu_place(lod_tensors[0]->place())) {
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
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

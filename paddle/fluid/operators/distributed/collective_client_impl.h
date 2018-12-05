// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <condition_variable>  // NOLINT
#include <string>
#include <vector>
#include "gflags/gflags.h"

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/framework/selected_rows_util.h"
#include "paddle/fluid/operators/detail/macros.h"
#include "paddle/fluid/operators/math/selected_rows_functor.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace operators {
namespace distributed {

template <typename DataType>
void CollectiveClient::ReduceSelectedRows(
    const std::vector<std::string>& endpoints, const std::string& var_name,
    framework::Scope* local_scope, const std::string& dst_var_name,
    int64_t time_out) {
  distributed::RPCClient* client =
      distributed::RPCClient::GetInstance<RPCCLIENT_T>(0);

  auto cpu_place = platform::CPUPlace();
  // platform::CPUDeviceContext* cpu_ctx =
  // dynamic_cast<platform::CPUDeviceContext*>(dev_ctxes.at(cpu_place));
  platform::CPUDeviceContext* cpu_ctx =
      dynamic_cast<platform::CPUDeviceContext*>(
          platform::DeviceContextPool::Instance().Get(cpu_place));

  // 1. copy to cpu if on gpu
  auto local_slr =
      local_scope->FindVar(var_name)->GetMutable<framework::SelectedRows>();
  auto cpu_local_slr = local_slr;
  if (platform::is_gpu_place(local_slr->place())) {
    // copy to cpu
    cpu_local_slr = local_scope->Var(var_name + "_cpu_mirror_")
                        ->GetMutable<framework::SelectedRows>();
    framework::SelectedRowsCopy(*local_slr, cpu_place, cpu_local_slr);
  }

  // 2. get remote selectrows to cpu_ctx
  std::vector<framework::Scope*> scopes;
  for (auto ep : endpoints) {
    VLOG(4) << "begin gather from ep:" << ep;
    // auto* scope = &local_scope->NewScope();
    auto* scope = new framework::Scope();
    scope->Var(var_name)->GetMutable<framework::SelectedRows>();
    client->AsyncGetMonomerVariable(ep, *cpu_ctx, *scope, var_name, time_out);
    scopes.push_back(scope);
  }
  client->Wait();

  std::vector<const framework::SelectedRows*> slrs;
  for (unsigned int i = 0; i < endpoints.size(); i++) {
    auto select_rows =
        scopes[i]->FindVar(var_name)->GetMutable<framework::SelectedRows>();
    slrs.push_back(select_rows);

    VLOG(4) << "gather from ep:" << endpoints[i]
            << ", select_rows:" << GetSelectedRowsInfo(*select_rows);

    client->AsyncGetMonomerBarrier(endpoints[i], var_name);
  }
  client->Wait();

  // wait copy local from gpu to cpu.
  // auto local_dev_ctx = dev_ctxes.at(local_slr->place());
  auto local_dev_ctx =
      platform::DeviceContextPool::Instance().Get(local_slr->place());
  if (platform::is_gpu_place(local_slr->place())) {
    local_dev_ctx->Wait();
  }

  // 3. merge local_cpu and remotes on cpu for saving gpu memory.
  slrs.push_back(cpu_local_slr);
  auto cpu_name = var_name + "_reduced_cpu_";
  auto reduced_cpu_slr =
      local_scope->Var(cpu_name)->GetMutable<framework::SelectedRows>();
  operators::math::scatter::MergeAdd<platform::CPUDeviceContext, DataType>
      merge_func;
  merge_func(*cpu_ctx, slrs, reduced_cpu_slr);
  VLOG(9) << cpu_name << ":" << GetSelectedRowsInfo(*reduced_cpu_slr);

  if (platform::is_gpu_place(local_slr->place())) {
    VLOG(10) << "ReduceSelectedRows copy to gpu";
    // copy cpu to gpu.
    auto gpu_slr =
        local_scope->Var(dst_var_name)->GetMutable<framework::SelectedRows>();
    framework::SelectedRowsCopy(*reduced_cpu_slr, local_slr->place(), gpu_slr);
  } else {
    local_scope->EraseVars(std::vector<std::string>{dst_var_name});
    local_scope->Rename(cpu_name, dst_var_name);
  }

  for (unsigned int i = 0; i < endpoints.size(); i++) {
    // local_scope->DeleteScope(scopes[i]);
    delete scopes[i];
  }
}
}  // namespace distributed
}  // namespace operators
}  // namespace paddle

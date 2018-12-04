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

namespace paddle {
namespace operators {
namespace distributed {

template <typename DataType>
void CollectiveClient::ReduceSelectedRows(
    const std::vector<std::string>& endpoints, const std::string& var_name,
    framework::Scope* local_scope, int64_t time_out) {
  distributed::RPCClient* client =
      distributed::RPCClient::GetInstance<RPCCLIENT_T>(0);

  auto cpu_place = platform::CPUPlace();
  platform::CPUDeviceContext* cpu_ctx =
      dynamic_cast<platform::CPUDeviceContext*>(
          platform::DeviceContextPool::Instance().Get(cpu_place));

  auto local_slr =
      local_scope->FindVar(var_name)->GetMutable<framework::SelectedRows>();
  auto cpu_local_slr = local_slr;
  // copy if on gpu
  if (platform::is_gpu_place(local_slr->place())) {
    // copy to cpu
    cpu_local_slr = local_scope->Var(var_name + "_cpu_mirror_")
                        ->GetMutable<framework::SelectedRows>();
    framework::SelectedRowsCopy<DataType>(*local_slr, cpu_place, cpu_local_slr);
  }

  // get remote selectrows to cpu_ctx
  std::vector<framework::Scope*> scopes;
  for (auto ep : endpoints) {
    VLOG(4) << "begin gather from ep:" << ep;
    auto* scope = &local_scope->NewScope();
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

  auto local_dev_ctx =
      platform::DeviceContextPool::Instance().Get(local_slr->place());
  if (platform::is_gpu_place(local_slr->place())) {
    // wait copy local from gpu to cpu.
    local_dev_ctx->Wait();
  }

  // merge local_cpu and remote on cpu.
  slrs.push_back(cpu_local_slr);
  auto cpu_name = var_name + "_reduced_cpu_";
  auto mid_slr =
      local_scope->Var(cpu_name)->GetMutable<framework::SelectedRows>();
  operators::math::scatter::MergeAdd<platform::CPUDeviceContext, DataType>
      merge_func;
  merge_func(*cpu_ctx, slrs, mid_slr);
  VLOG(10) << cpu_name << ":" << GetSelectedRowsInfo(*mid_slr);

  if (platform::is_gpu_place(local_slr->place())) {
    // copy cpu to gpu.
    auto gpu_name = var_name + "_reduced_gpu_";
    auto gpu_slr =
        local_scope->Var(gpu_name)->GetMutable<framework::SelectedRows>();
    framework::SelectedRowsCopy<DataType>(*mid_slr, local_slr->place(),
                                          gpu_slr);

    // wait
    local_dev_ctx->Wait();
    VLOG(10) << gpu_name << ":" << GetSelectedRowsInfo(*gpu_slr);

    // rename
    std::cout << "rename" << std::endl;
    local_scope->EraseVars(std::vector<std::string>{var_name});
    local_scope->Rename(gpu_name, var_name);
    std::cout << "after rename" << std::endl;
  } else {
    local_scope->EraseVars(std::vector<std::string>{var_name});
    local_scope->Rename(cpu_name, var_name);
  }

  for (unsigned int i = 0; i < endpoints.size(); i++) {
    local_scope->DeleteScope(scopes[i]);
  }
}
}  // namespace distributed
}  // namespace operators
}  // namespace paddle

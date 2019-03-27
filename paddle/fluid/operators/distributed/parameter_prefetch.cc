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

#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/operators/distributed/parameter_prefetch.h"

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/framework/tensor.h"

#include "paddle/fluid/operators/distributed/distributed.h"
#include "paddle/fluid/operators/distributed/rpc_client.h"
#include "paddle/fluid/operators/distributed/variable_response.h"
#include "paddle/fluid/operators/distributed_ops/send_recv_util.h"

namespace paddle {
namespace operators {
namespace distributed {

using LoDTensor = framework::LoDTensor;
using LoDTensor = framework::LoDTensor;
using SelectedRows = framework::SelectedRows;
using DDim = framework::DDim;

static std::vector<std::vector<int64_t>> SplitIds(
    const std::vector<int64_t>& ids_vector,
    const std::vector<int64_t>& height_section) {
  std::set<int64_t> all_ids;
  for (auto id : ids_vector) {
    all_ids.insert(id);
  }

  auto abs_sections = ToAbsoluteSection(height_section);
  std::vector<std::vector<int64_t>> splited_ids;
  splited_ids.resize(height_section.size() + 1);
  for (auto& id : all_ids) {
    auto section_index = GetSectionIndex(id, abs_sections);
    splited_ids[section_index].push_back(id - abs_sections[section_index]);
  }
  return splited_ids;
}

static void SplitIdsIntoMultipleVarsBySection(
    const std::vector<std::string>& in_var_names,
    const std::vector<int64_t>& height_section,
    const std::vector<std::vector<int64_t>>& splited_ids,
    framework::Scope* scope) {
  PADDLE_ENFORCE_EQ(in_var_names.size(), height_section.size(), "");

  auto place = platform::CPUPlace();

  for (size_t i = 0; i < in_var_names.size(); ++i) {
    auto* id_tensor =
        scope->Var(in_var_names[i])->GetMutable<framework::LoDTensor>();
    auto& ids = splited_ids[i];
    if (!ids.empty()) {
      auto* id_tensor_data = id_tensor->mutable_data<int64_t>(
          framework::make_ddim({static_cast<int64_t>(ids.size()), 1}), place);
      memcpy(id_tensor_data, ids.data(), sizeof(int64_t) * ids.size());
    }
  }
}

static void MergeMultipleVarsIntoOneBySection(
    const std::string& id_name, const std::vector<int64_t>& ids_vector,
    const std::string& out_name, const std::vector<std::string>& out_var_names,
    const std::vector<int64_t>& height_section,
    const std::vector<std::vector<int64_t>>& splited_ids,
    const framework::ExecutionContext& context, framework::Scope* scope,
    platform::DeviceContext* actual_ctx) {
  PADDLE_ENFORCE_EQ(out_var_names.size(), height_section.size(), "");

  auto cpu_place = platform::CPUPlace();

  auto abs_sections = ToAbsoluteSection(height_section);
  std::unordered_map<int64_t, std::vector<size_t>> id_to_offset;
  for (size_t i = 0; i < ids_vector.size(); ++i) {
    id_to_offset[ids_vector[i]].push_back(i);
  }

  auto& id_tensor = scope->FindVar(id_name)->Get<framework::LoDTensor>();
  auto* out_tensor =
      scope->FindVar(out_name)->GetMutable<framework::LoDTensor>();

  PADDLE_ENFORCE_GT(
      out_tensor->numel(), 0,
      "When calling this method, the LoDTensor's numel must larger than zero. "
      "Please check LoDTensor::Resize has been called first.");

  auto* out_tensor_data = out_tensor->mutable_data<float>(id_tensor.place());

  bool is_on_cpu_place = true;
  if (!platform::is_cpu_place(id_tensor.place())) {
    is_on_cpu_place = false;
  }

  for (size_t section_idx = 0; section_idx < out_var_names.size();
       ++section_idx) {
    auto& ids_in_this_section = splited_ids[section_idx];
    if (!ids_in_this_section.empty()) {
      auto& prefetch_out_var =
          scope->Var(out_var_names[section_idx])->Get<framework::LoDTensor>();
      const auto* out_var_data = prefetch_out_var.data<float>();
      auto& dims = prefetch_out_var.dims();

      PADDLE_ENFORCE_EQ(dims.size(), 2, "");
      PADDLE_ENFORCE_EQ(ids_in_this_section.size(), dims[0]);

      auto row_numel = dims[1];

      for (int64_t i = 0; i < dims[0]; ++i) {
        auto id = ids_in_this_section[i];
        auto origin_id = id + abs_sections[section_idx];
        auto& offsets = id_to_offset[origin_id];
        for (auto& offset : offsets) {
          // should support GPU tensor
          if (is_on_cpu_place) {
            memory::Copy(cpu_place, out_tensor_data + offset * row_numel,
                         cpu_place, out_var_data + i * row_numel,
                         sizeof(float) * row_numel);
          } else {
#ifndef PADDLE_WITH_CUDA
            PADDLE_THROW("paddle is not compiled with CUDA!");
#else
            auto stream =
                static_cast<platform::CUDADeviceContext*>(actual_ctx)->stream();
            memory::Copy(boost::get<platform::CUDAPlace>(id_tensor.place()),
                         out_tensor_data + offset * row_numel, cpu_place,
                         out_var_data + i * row_numel,
                         sizeof(float) * row_numel, stream);
#endif
          }
        }
      }
    } else {
      VLOG(3) << "ids in this section is empty";
    }
  }
}

void prefetch(const std::string& id_name, const std::string& out_name,
              const std::vector<std::string>& table_names,
              const std::vector<std::string>& epmap,
              const std::vector<int64_t>& height_sections,
              const framework::ExecutionContext& context,
              const framework::Scope& scope) {
  framework::Scope* local_scope = scope.NewTmpScope();

  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  auto& cpu_ctx = *pool.Get(platform::CPUPlace());
  auto& actual_ctx = *pool.Get(context.GetPlace());

  distributed::RPCClient* rpc_client =
      distributed::RPCClient::GetInstance<RPCCLIENT_T>(
          context.Attr<int>("trainer_id"));

  std::vector<std::string> in_var_names;
  std::vector<std::string> out_var_names;
  for (size_t i = 0; i < epmap.size(); ++i) {
    in_var_names.push_back(id_name + "@" + epmap[i]);
    out_var_names.push_back(out_name + "@" + epmap[i]);
  }

  auto& id_tensor = scope.FindVar(id_name)->Get<framework::LoDTensor>();
  std::vector<int64_t> ids_vector;
  if (platform::is_cpu_place(id_tensor.place())) {
    auto* id_data = id_tensor.data<int64_t>();
    for (int64_t i = 0; i < id_tensor.numel(); ++i) {
      ids_vector.push_back(id_data[i]);
    }
  } else {
#ifndef PADDLE_WITH_CUDA
    PADDLE_THROW("paddle is not compiled with CUDA!");
#else
    auto cpu_place = platform::CPUPlace();
    framework::LoDTensor cpu_tensor;
    auto* cpu_tensor_data =
        cpu_tensor.mutable_data<int64_t>(id_tensor.dims(), cpu_place);
    auto stream =
        static_cast<platform::CUDADeviceContext*>(&actual_ctx)->stream();
    memory::Copy(cpu_place, cpu_tensor_data,
                 boost::get<platform::CUDAPlace>(id_tensor.place()),
                 id_tensor.data<int64_t>(), sizeof(int64_t) * id_tensor.numel(),
                 stream);
    for (size_t i = 0; i < cpu_tensor.numel(); ++i) {
      ids_vector.push_back(cpu_tensor_data[i]);
    }
#endif
  }

  auto splited_ids = SplitIds(ids_vector, height_sections);
  SplitIdsIntoMultipleVarsBySection(in_var_names, height_sections, splited_ids,
                                    local_scope);

  // create output var in local scope
  for (auto& name : out_var_names) {
    local_scope->Var(name)->GetMutable<framework::LoDTensor>();
  }

  std::vector<distributed::VarHandlePtr> rets;
  for (size_t i = 0; i < in_var_names.size(); i++) {
    if (NeedSend(*local_scope, in_var_names[i])) {
      VLOG(3) << "sending " << in_var_names[i] << " to " << epmap[i]
              << " to get " << out_var_names[i] << " back";
      rets.push_back(rpc_client->AsyncPrefetchVar(
          epmap[i], cpu_ctx, *local_scope, in_var_names[i], out_var_names[i],
          table_names[i]));
    } else {
      VLOG(3) << "don't send no-initialied variable: " << out_var_names[i];
    }
  }

  for (size_t i = 0; i < rets.size(); i++) {
    PADDLE_ENFORCE(rets[i]->Wait(), "internal error in RPCClient");
  }

  MergeMultipleVarsIntoOneBySection(id_name, ids_vector, out_name,
                                    out_var_names, height_sections, splited_ids,
                                    context, local_scope, &actual_ctx);
  delete local_scope;
}

};  // namespace distributed
};  // namespace operators
};  // namespace paddle

/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <future>  // NOLINT
#include <ostream>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/operators/detail/macros.h"
#include "paddle/fluid/operators/distributed_ops/send_recv_util.h"
#include "paddle/fluid/operators/math/blas.h"

namespace paddle {
namespace operators {

inline size_t GetSectionIndex(int64_t id,
                              const std::vector<int64_t>& abs_sections) {
  for (size_t i = 1; i < abs_sections.size(); ++i) {
    if (id < abs_sections[i]) {
      return i - 1;
    }
  }
  return abs_sections.size() - 1;
}

inline std::vector<int64_t> ToAbsoluteSection(
    const std::vector<int64_t>& height_sections) {
  std::vector<int64_t> abs_sections;
  abs_sections.resize(height_sections.size());
  abs_sections[0] = 0;
  for (size_t i = 1; i < height_sections.size(); ++i) {
    abs_sections[i] = height_sections[i - 1] + abs_sections[i - 1];
  }
  return abs_sections;
}

inline std::vector<std::vector<int64_t>> SplitIds(
    const std::string& id_name, const std::vector<int64_t>& height_section,
    framework::Scope* scope) {
  auto& id_tensor = scope->Var(id_name)->Get<framework::LoDTensor>();
  auto* id_data = id_tensor.data<int64_t>();
  std::set<int64_t> all_ids;
  for (size_t i = 0; i < id_tensor.numel(); ++i) {
    all_ids.insert(id_data[i]);
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

inline void SplitIdsIntoMultipleVarsBySection(
    const std::string& id_name, const std::vector<std::string>& in_var_names,
    const std::vector<int64_t>& height_section,
    const std::vector<std::vector<int64_t>>& splited_ids,
    framework::Scope* scope) {
  PADDLE_ENFORCE_EQ(in_var_names.size(), height_section.size() + 1, "");

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

inline void MergeMultipleVarsIntoOnBySection(
    const std::string& id_name, const std::string& out_name,
    const std::vector<std::string>& out_var_names,
    const std::vector<int64_t>& height_section,
    const std::vector<std::vector<int64_t>>& splited_ids,
    const framework::ExecutionContext& context, framework::Scope* scope) {
  PADDLE_ENFORCE_EQ(out_var_names.size(), height_section.size() + 1, "");

  auto cpu_place = platform::CPUPlace();

  auto abs_sections = ToAbsoluteSection(height_section);
  auto& id_tensor = scope->Var(id_name)->Get<framework::LoDTensor>();
  auto* id_data = id_tensor.data<int64_t>();
  std::unordered_map<int64_t, std::vector<size_t>> id_to_offset;
  for (size_t i = 0; i < id_tensor.numel(); ++i) {
    id_to_offset[id_data[i]].push_back(i);
  }

  auto* out_tensor = scope->Var(out_name)->GetMutable<framework::LoDTensor>();
  auto* out_tensor_data = out_tensor->mutable_data<float>(context.GetPlace());

  for (size_t section_idx = 0; section_idx < out_var_names.size();
       ++section_idx) {
    auto& ids_in_this_section = splited_ids[section_idx];
    auto& prefetch_out_var =
        scope->Var(out_var_names[section_idx])->Get<framework::LoDTensor>();
    const auto* out_var_data = prefetch_out_var.data<float>();
    auto& dims = prefetch_out_var.dims();

    PADDLE_ENFORCE_EQ(dims.size(), 2, "");
    PADDLE_ENFORCE_EQ(ids_in_this_section.size(), dims[0]);

    auto row_numel = dims[1];

    for (size_t i = 0; i < dims[0]; ++i) {
      auto id = ids_in_this_section[i];
      auto origin_id = id + abs_sections[section_idx];
      auto& offsets = id_to_offset[origin_id];
      for (auto& offset : offsets) {
        // should support GPU tensor
        memory::Copy(cpu_place, out_tensor_data + offset * row_numel, cpu_place,
                     out_var_data + i * row_numel, sizeof(float) * row_numel);
      }
    }
  }
}

// inline void prefetch(const std::string& table_name, const std::string&
// id_name,
//                     const std::string& out_name,
//                     const std::vector<std::string>& epmap,
//                     const std::vector<int64_t>& height_section,
//                     const framework::Scope& scope,
//                     const platform::Place& place) {
//  auto& local_scope = scope.NewScope();
//
//  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
//  auto& ctx = *pool.Get(place);
//
//  distributed::RPCClient* rpc_client =
//      distributed::RPCClient::GetInstance<RPCCLIENT_T>(Attr<int>("trainer_id"));
//
//  std::vector<std::string> in_var_names;
//  std::vector<std::string> out_var_names;
//  for (size_t i = 0; i < epmap.size(); ++i) {
//    in_var_names.push_back(id_name + "@" + epmap[i]);
//    out_var_names.push_back(out_name + "@" + epmap[i]);
//  }
//
//  auto splited_ids = SplitIds(id_name, height_section, &local_scope);
//  SplitIdsIntoMultipleVarsBySection(id_name, in_var_names, height_section,
//                                    splited_ids, &local_scope);
//
//  // create output var in local scope
//  for (auto& name : out_var_names) {
//    local_scope.Var(name)->GetMutable<framework::LoDTensor>();
//  }
//
//  std::vector<distributed::VarHandlePtr> rets;
//  for (size_t i = 0; i < in_var_names.size(); i++) {
//    if (NeedSend(local_scope, in_var_names[i])) {
//      VLOG(30) << "sending " << in_var_names[i] << " to " << epmap[i] << " to
//      get "
//               << out_var_names[i] << " back";
//      rets.push_back(rpc_client->AsyncPrefetchVar(
//          epmap[i], ctx, local_scope, in_var_names[i], out_var_names[i]));
//    } else {
//      VLOG(30) << "don't send no-initialied variable: " << out_var_names[i];
//    }
//  }
//  for (size_t i = 0; i < rets.size(); i++) {
//    PADDLE_ENFORCE(rets[i]->Wait(), "internal error in RPCClient");
//  }
//
//  MergeMultipleVarsIntoOnBySection(id_name, out_name, out_var_names,
//                                   height_section, splited_ids, &local_scope);
//
//  scope.DeleteScope(&local_scope);
//}

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using SelectedRows = framework::SelectedRows;
using DDim = framework::DDim;

constexpr int64_t kNoPadding = -1;

template <typename T>
class LookupRemoteTableKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    std::string id_name = context.Inputs("Ids").front();
    auto* ids_t = context.Input<LoDTensor>("Ids");  // int tensor

    std::string out_name = context.Outputs("Out").front();
    auto* output_t = context.Output<LoDTensor>("Out");  // float tensor

    std::string table_name = context.Inputs("W").front();
    auto* table_var = context.InputVar("W");

    int64_t padding_idx = context.Attr<int64_t>("padding_idx");
    int64_t* ids = const_cast<int64_t*>(ids_t->data<int64_t>());
    int64_t ids_numel = ids_t->numel();

    auto epmap = context.Attr<std::vector<std::string>>("epmap");
    auto height_sections =
        context.Attr<std::vector<int64_t>>("height_sections");

    auto& local_scope = context.scope().NewScope();

    platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
    auto& ctx = *pool.Get(context.GetPlace());

    distributed::RPCClient* rpc_client =
        distributed::RPCClient::GetInstance<RPCCLIENT_T>(
            context.Attr<int>("trainer_id"));

    std::vector<std::string> in_var_names;
    std::vector<std::string> out_var_names;
    for (size_t i = 0; i < epmap.size(); ++i) {
      in_var_names.push_back(id_name + "@" + epmap[i]);
      out_var_names.push_back(out_name + "@" + epmap[i]);
    }

    auto splited_ids = SplitIds(id_name, height_sections, &local_scope);
    SplitIdsIntoMultipleVarsBySection(id_name, in_var_names, height_sections,
                                      splited_ids, &local_scope);

    // create output var in local scope
    for (auto& name : out_var_names) {
      local_scope.Var(name)->GetMutable<framework::LoDTensor>();
    }

    std::vector<distributed::VarHandlePtr> rets;
    for (size_t i = 0; i < in_var_names.size(); i++) {
      if (NeedSend(local_scope, in_var_names[i])) {
        VLOG(30) << "sending " << in_var_names[i] << " to " << epmap[i]
                 << " to get " << out_var_names[i] << " back";
        rets.push_back(rpc_client->AsyncPrefetchVar(
            epmap[i], ctx, local_scope, in_var_names[i], out_var_names[i]));
      } else {
        VLOG(30) << "don't send no-initialied variable: " << out_var_names[i];
      }
    }
    for (size_t i = 0; i < rets.size(); i++) {
      PADDLE_ENFORCE(rets[i]->Wait(), "internal error in RPCClient");
    }

    MergeMultipleVarsIntoOnBySection(id_name, out_name, out_var_names,
                                     height_sections, splited_ids, context,
                                     &local_scope);

    context.scope().DeleteScope(&local_scope);
  }
};

}  // namespace operators
}  // namespace paddle

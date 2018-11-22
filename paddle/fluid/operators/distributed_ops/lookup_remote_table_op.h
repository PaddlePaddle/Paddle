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

#include <future>  // NOLINT
#include <ostream>
#include <set>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/operators/detail/macros.h"
#include "paddle/fluid/operators/distributed_ops/send_recv_util.h"

namespace paddle {
namespace operators {
namespace distributed {

inline size_t GetSectionIndex(int64_t id,
                              const std::vector<int64_t>& abs_sections) {
  for (size_t i = 1; i < abs_sections.size(); ++i) {
    if (row < abs_sections[i]) {
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
    auto section_index = GetSectionIndex(id);
    splited_ids[section_index].push_back(id - abs_sections[section_index]);
  }
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
          framework::make_ddim({ids.size(), 1}), place);
      memcpy(id_tensor_data, ids.data(), sizeof(int64_t) * ids.size());
    }
  }
}

inline void MergeMultipleVarsIntoOnBySection(
    const std::string& id_name, const std::string& out_name,
    const std::vector<std::string>& out_var_names,
    const std::vector<int64_t>& height_section,
    const std::vector<std::vector<int64_t>>& splited_ids,
    framework::Scope* scope) {
  PADDLE_ENFORCE_EQ(in_var_names.size(), height_section.size() + 1, "");

  auto cpu_place = platform::CPUPlace();

  auto abs_sections = ToAbsoluteSection(height_section);
  auto& id_tensor = scope->Var(id_name)->Get<framework::LoDTensor>();
  auto* id_data = id_tensor.data<int64_t>();
  std::unordered_map<int64_t, std::vector<size_t>> id_to_offset;
  for (size_t i = 0; i < id_tensor.numel(); ++i) {
    id_to_offset[id_data[i]].push_back(i);
  }

  auto& out_tensor = scope->Var(out_name)->Get<framework::LoDTensor>();
  auto* out_tensor_data = out_tensor.mutable_data<float>();

  for (size_t section_idx = 0; section_idx < out_var_names.size();
       ++section_idx) {
    auto& ids_in_this_section = splited_ids[section_idx];
    auto& prefetch_out_var =
        scope->Var(out_var_names[section_idx])->Get<framework::LoDTensor>();
    const auto* out_var_data = prefetch_out_var.mutable_data<float>();
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
                     out_var_data + i * grad_row_numel,
                     sizeof(T) * grad_row_numel);
      }
    }
  }
}

inline void prefetch(const std::string& table_name, const std::string& id_name,
                     const std::string& out_name,
                     const std::vector<std::string>& epmap,
                     const std::vector<int64_t>& height_section,
                     const framework::Scope& scope,
                     const platform::Place& place) const {
  auto local_scope = scope.NewScope();

  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  auto& ctx = *pool.Get(place);

  distributed::RPCClient* rpc_client =
      distributed::RPCClient::GetInstance<RPCCLIENT_T>(Attr<int>("trainer_id"));

  std::vector<std::string> in_var_names;
  std::vector<std::string> out_var_names;
  for (size_t i = 0; i < epmap.size(); ++i) {
    in_var_names.push_back(id_name + "@" + epmap[i]);
    out_var_names.push_back(out_name + "@" + epmap[i]);
  }

  auto splited_ids = SplitIds(id_name, height_section, local_scope);
  SplitIdsIntoMultipleVarsBySection(id_name, in_var_names, height_section,
                                    splited_ids, local_scope);

  // create output var in local scope
  for (auto& name : out_var_names) {
    local_scope.Var(name)->GetMutable<framework::LoDTensor>();
  }

  std::vector<distributed::VarHandlePtr> rets;
  for (size_t i = 0; i < ins.size(); i++) {
    if (NeedSend(local_scope, ins[i])) {
      VLOG(30) << "sending " << ins[i] << " to " << epmap[i] << " to get "
               << outs[i] << " back";
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
                                   height_section, plited_ids, scope)

      scope.DeleteScope(local_scope);
}

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using SelectedRows = framework::SelectedRows;
using DDim = framework::DDim;

constexpr int64_t kNoPadding = -1;

template <typename T>
class LookupRemoteTableKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* ids_t = context.Input<LoDTensor>("Ids");      // int tensor
    auto* output_t = context.Output<LoDTensor>("Out");  // float tensor
    auto* table_var = context.InputVar("W");

    int64_t padding_idx = context.Attr<int64_t>("padding_idx");
    int64_t* ids = const_cast<int64_t*>(ids_t->data<int64_t>());
    int64_t ids_numel = ids_t->numel();

    if (table_var->IsType<LoDTensor>()) {
      auto* table_t = context.Input<LoDTensor>("W");
      int64_t row_number = table_t->dims()[0];
      int64_t row_width = table_t->dims()[1];

      auto* table = table_t->data<T>();
      auto* output = output_t->mutable_data<T>(context.GetPlace());

      for (int64_t i = 0; i < ids_numel; ++i) {
        if (padding_idx != kNoPadding && ids[i] == padding_idx) {
          memset(output + i * row_width, 0, row_width * sizeof(T));
        } else {
          PADDLE_ENFORCE_LT(ids[i], row_number);
          PADDLE_ENFORCE_GE(ids[i], 0, "ids %d", i);
          memcpy(output + i * row_width, table + ids[i] * row_width,
                 row_width * sizeof(T));
        }
      }
    } else if (table_var->IsType<SelectedRows>()) {
      const auto& table_t = table_var->Get<SelectedRows>();
      int64_t row_width = table_t.value().dims()[1];
      const auto* table = table_t.value().data<T>();
      auto* output = output_t->mutable_data<T>(context.GetPlace());

      auto blas = math::GetBlas<platform::CPUDeviceContext, T>(context);
      for (int64_t i = 0; i < ids_numel; ++i) {
        if (padding_idx != kNoPadding && ids[i] == padding_idx) {
          memset(output + i * row_width, 0, row_width * sizeof(T));
        } else {
          PADDLE_ENFORCE_GE(ids[i], 0);
          auto id_index = table_t.Index(ids[i]);
          PADDLE_ENFORCE_GE(id_index, 0, "the input key should be exists.");
          blas.VCOPY(row_width, table + id_index * row_width,
                     output + i * row_width);
        }
      }
    }
  }
};

}  // namespace distributed
}  // namespace operators
}  // namespace paddle

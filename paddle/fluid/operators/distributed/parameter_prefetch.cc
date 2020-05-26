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

#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
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
    VLOG(1) << "Parameter Prefetch var: " << in_var_names[i]
            << "On CPU place: " << platform::is_cpu_place(id_tensor->place());
  }
}

typedef std::vector<std::pair<std::string, std::string>> TableAndEndpoints;

void prefetch_core(
    const std::vector<int64_t>& ids, const TableAndEndpoints& tables,
    const std::vector<int64_t>& height_sections,
    const framework::ExecutionContext& context, const framework::Scope& scope,
    std::unordered_map<int64_t, std::vector<float>>* recved_vec_map) {
  VLOG(3) << "Prefetch_core Begin";
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  auto& cpu_ctx = *pool.Get(platform::CPUPlace());

  std::unique_ptr<framework::Scope> local_scope = scope.NewTmpScope();

  std::vector<std::string> in_var_names;
  std::vector<std::string> out_var_names;
  for (size_t i = 0; i < tables.size(); ++i) {
    in_var_names.push_back("prefetch_send@" + tables[i].second);
    out_var_names.push_back("prefetch_recv@" + tables[i].second);
  }
  VLOG(3) << "Prefetch_core SplitIdsIntoMultipleVarsBySection Begin";
  auto splited_ids = SplitIds(ids, height_sections);
  SplitIdsIntoMultipleVarsBySection(in_var_names, height_sections, splited_ids,
                                    local_scope.get());

  VLOG(3) << "Prefetch_core local_scope->var GetMutable Begin";
  // create output var in local scope
  for (auto& name : out_var_names) {
    local_scope->Var(name)->GetMutable<framework::LoDTensor>();
  }

  distributed::RPCClient* rpc_client =
      distributed::RPCClient::GetInstance<RPCCLIENT_T>(
          context.Attr<int>("trainer_id"));

  VLOG(3) << "Prefetch_core NeedSend Begin";
  std::vector<distributed::VarHandlePtr> rets;
  for (size_t i = 0; i < in_var_names.size(); i++) {
    if (NeedSend(*local_scope.get(), in_var_names[i])) {
      VLOG(3) << "sending " << in_var_names[i] << " to " << tables[i].second
              << " to get " << out_var_names[i] << " back";
      rets.push_back(rpc_client->AsyncPrefetchVar(
          tables[i].second, cpu_ctx, *local_scope.get(), in_var_names[i],
          out_var_names[i], tables[i].first));
    } else {
      VLOG(3) << "don't send no-initialied variable: " << out_var_names[i];
    }
  }

  VLOG(3) << "Prefetch_core AsyncPrefetchVar Begin";
  for (size_t i = 0; i < rets.size(); i++) {
    PADDLE_ENFORCE(rets[i]->Wait(), "internal error in RPCClient");
  }

  PADDLE_ENFORCE_EQ(out_var_names.size(), height_sections.size(), "");

  VLOG(3) << "Prefetch_core OutVar Copy Begin";
  auto abs_sections = ToAbsoluteSection(height_sections);
  for (size_t section_idx = 0; section_idx < out_var_names.size();
       ++section_idx) {
    auto& ids_in_this_section = splited_ids[section_idx];
    if (!ids_in_this_section.empty()) {
      auto& prefetch_out_var = local_scope->Var(out_var_names[section_idx])
                                   ->Get<framework::LoDTensor>();
      const auto* out_var_data = prefetch_out_var.data<float>();
      auto& dims = prefetch_out_var.dims();

      PADDLE_ENFORCE_EQ(dims.size(), 2, "");
      PADDLE_ENFORCE_EQ(ids_in_this_section.size(), dims[0]);

      auto row_numel = dims[1];

      for (int64_t i = 0; i < dims[0]; ++i) {
        auto id = ids_in_this_section[i];
        auto origin_id = id + abs_sections[section_idx];
        std::vector<float> vecs(row_numel);
        std::copy_n(out_var_data + i * row_numel, row_numel, vecs.begin());
        (*recved_vec_map)[origin_id] = vecs;
      }
    } else {
      VLOG(3) << "ids in this section is empty";
    }
    // if (!ids_in_this_section.empty()) {
    //   auto& prefetch_out_var = local_scope->Var(out_var_names[section_idx])
    //                                ->Get<framework::LoDTensor>();

    //   auto& dims = prefetch_out_var->dims();
    //   PADDLE_ENFORCE_EQ(dims.size(), 2, "");
    //   PADDLE_ENFORCE_EQ(ids_in_this_section.size(), dims[0]);

    //   auto row_numel = dims[1];

    //   VLOG(3) << "Prefetch_core OutVar Copy";
    //   if (platform::is_cpu_place(prefetch_out_var.place())) {
    //     VLOG(3) << "Prefetch_core OutVar Copy CPU";
    //     const auto* out_var_data = prefetch_out_var.data<float>();
    //     for (int64_t i = 0; i < dims[0]; ++i) {
    //       auto id = ids_in_this_section[i];
    //       auto origin_id = id + abs_sections[section_idx];
    //       std::vector<float> vecs(row_numel);
    //       std::copy_n(out_var_data + i * row_numel, row_numel, vecs.begin());
    //       (*recved_vec_map)[origin_id] = vecs;
    //     }
    //   } else {
    //     VLOG(3) << "Prefetch_core OutVar Copy GPU";
    //     auto cpu_ctx = paddle::platform::CPUDeviceContext();
    //     auto* gpu_ctx = static_cast<platform::CUDADeviceContext*>(
    //         platform::DeviceContextPool::Instance().Get(
    //             prefetch_out_var->place()));
    //     auto& cpu_place =
    //         BOOST_GET_CONST(platform::CPUPlace, cpu_ctx.GetPlace());
    //     auto& gpu_place =
    //         BOOST_GET_CONST(platform::CUDAPlace, gpu_ctx->GetPlace());
    //     for (int64_t i = 0; i < dims[0]; ++i) {
    //       auto id = ids_in_this_section[i];
    //       auto origin_id = id + abs_sections[section_idx];
    //       std::vector<float> vecs(row_numel);
    //       paddle::memory::Copy(gpu_place,
    //                            prefetch_out_var->data<float>() + i *
    //                            row_numel,
    //                            cpu_place, &vecs[0], sizeof(float) *
    //                            row_numel,
    //                            gpu_ctx->stream());
    //       // cudaMemcpy(prefetch_out_var->data<float>() + i * row_numel,
    //       // &vecs[0],
    //       //            sizeof(float) * row_numel, cudaMemcpyHostToDevice);
    //       (*recved_vec_map)[origin_id] = vecs;
    //     }
    //   }
    // } else {
    //   VLOG(3) << "ids in this section is empty";
    // }
  }
}

void prefetch(const std::string& id_name, const std::string& out_name,
              const std::string& persistable_var_name, const bool backfill,
              const std::vector<std::string>& table_names,
              const std::vector<std::string>& endpoints,
              const std::vector<int64_t>& height_sections,
              const framework::ExecutionContext& context,
              const framework::Scope& scope) {
  prefetchs({id_name}, {out_name}, persistable_var_name, backfill, table_names,
            endpoints, height_sections, context, scope);
}

void prefetchs(const std::vector<std::string>& id_var_names,
               const std::vector<std::string>& out_var_names,
               const std::string& persistable_var_name, const bool backfill,
               const std::vector<std::string>& table_names,
               const std::vector<std::string>& endpoints,
               const std::vector<int64_t>& height_sections,
               const framework::ExecutionContext& context,
               const framework::Scope& scope) {
  PADDLE_ENFORCE_GT(id_var_names.size(), 0, "");
  PADDLE_ENFORCE_EQ(id_var_names.size(), out_var_names.size(), "");
  PADDLE_ENFORCE_EQ(table_names.size(), endpoints.size(), "");
  PADDLE_ENFORCE_EQ(table_names.size(), height_sections.size(), "");

  auto vec_dim_1 = 0;
  framework::Variable* var = scope.FindVar(persistable_var_name);

  PADDLE_ENFORCE_EQ(var->IsType<framework::LoDTensor>(), true,
                    platform::errors::InvalidArgument(
                        "prefetch can only support LodTensor only"));

  vec_dim_1 = var->Get<framework::LoDTensor>().dims()[1];

  PADDLE_ENFORCE_GT(vec_dim_1, 0,
                    platform::errors::InvalidArgument(
                        "lookup table var's dim must gather than 0"));

  const auto place =
      scope.FindVar(id_var_names[0])->Get<framework::LoDTensor>().place();

  // if (!platform::is_cpu_place(place)) {
  //   PADDLE_THROW("multi prefetch only support CPU currently");
  // }

  std::vector<std::vector<int64_t>> ids_group;
  std::vector<int64_t> ids_union;
  std::vector<framework::LoD> ids_lods;
  TableAndEndpoints tables;

  for (auto& id_name : id_var_names) {
    auto& id_tensor = scope.FindVar(id_name)->Get<framework::LoDTensor>();
    // auto* id_data = id_tensor.data<int64_t>();
    // for (int64_t i = 0; i < id_tensor.numel(); ++i) {
    //   ids.push_back(id_data[i]);
    //   ids_union.push_back(id_data[i]);
    // }
    std::vector<int64_t> ids;
    std::vector<int64_t> ids_union_part;
    TensorToVector(id_tensor, context.device_context(), &ids);
    VLOG(1) << "Parameter Prefetch: size(): " << ids.size() << " ids[0] "
            << ids[0];
    // TensorToVector(id_tensor, context.device_context(), &ids_union_part);
    ids_union.insert(ids_union.end(), ids.begin(), ids.end());
    ids_group.push_back(ids);
    ids_lods.push_back(id_tensor.lod());
  }

  std::unordered_set<int64_t> s(ids_union.begin(), ids_union.end());
  ids_union.assign(s.begin(), s.end());

  for (size_t i = 0; i < table_names.size(); i++) {
    tables.push_back(std::make_pair(table_names[i], endpoints[i]));
  }

  std::unordered_map<int64_t, std::vector<float>> recved_vec_map;
  prefetch_core(ids_union, tables, height_sections, context, scope,
                &recved_vec_map);

  auto padding_idx = distributed::kNoPadding;

  if (context.HasAttr("padding_idx")) {
    padding_idx = context.Attr<int64_t>("padding_idx");
  }

  // copy vectors to out vars
  for (size_t i = 0; i < out_var_names.size(); i++) {
    auto& ids = ids_group[i];
    auto* out_t =
        scope.FindVar(out_var_names[i])->GetMutable<framework::LoDTensor>();
    out_t->Resize(
        framework::make_ddim({static_cast<int64_t>(ids.size()), vec_dim_1}));
    out_t->set_lod(ids_lods[i]);

    if (platform::is_cpu_place(out_t->place())) {
      auto* out_d = out_t->mutable_data<float>(place);
      for (size_t idx = 0; idx < ids.size(); idx++) {
        const auto& id = ids[idx];
        if (padding_idx != distributed::kNoPadding && id == padding_idx) {
          memset(out_d + idx * vec_dim_1, 0, sizeof(float) * vec_dim_1);
        } else {
          std::copy_n(recved_vec_map[id].begin(), vec_dim_1,
                      out_d + idx * vec_dim_1);
        }
      }
    } else {
      auto cpu_ctx = paddle::platform::CPUDeviceContext();
      auto* gpu_ctx = reinterpret_cast<platform::CUDADeviceContext*>(
          platform::DeviceContextPool::Instance().Get(out_t->place()));
      auto& cpu_place = BOOST_GET_CONST(platform::CPUPlace, cpu_ctx.GetPlace());
      auto& gpu_place =
          BOOST_GET_CONST(platform::CUDAPlace, gpu_ctx->GetPlace());
      for (size_t idx = 0; idx < ids.size(); idx++) {
        const auto& id = ids[idx];
        if (padding_idx != distributed::kNoPadding && id == padding_idx) {
          paddle::platform::GpuMemsetAsync(
              out_t->data<float>() + idx * vec_dim_1, 0, vec_dim_1,
              gpu_ctx->stream());

        } else {
          paddle::memory::Copy(gpu_place,
                               out_t->data<float>() + idx * vec_dim_1,
                               cpu_place, &recved_vec_map[id][0],
                               sizeof(float) * vec_dim_1, gpu_ctx->stream());
        }
      }
    }
  }
}

};  // namespace distributed
};  // namespace operators
};  // namespace paddle

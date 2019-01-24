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
#include <vector>

#include "paddle/fluid/operators/distributed/parameter_send.h"

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

static size_t GetSectionIndex(int64_t id,
                              const std::vector<int64_t>& abs_sections) {
  for (size_t i = 1; i < abs_sections.size(); ++i) {
    if (id < abs_sections[i]) {
      return i - 1;
    }
  }
  return abs_sections.size() - 1;
}

static std::vector<int64_t> ToAbsoluteSection(
    const std::vector<int>& height_sections) {
  std::vector<int64_t> abs_sections;
  abs_sections.resize(height_sections.size());
  abs_sections[0] = 0;
  for (size_t i = 1; i < height_sections.size(); ++i) {
    abs_sections[i] = height_sections[i - 1] + abs_sections[i - 1];
  }
  return abs_sections;
}

static std::vector<std::vector<int64_t>> SplitIds(
    const std::vector<int64_t>& ids_vector,
    const std::vector<int>& height_section, framework::Scope* scope) {
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
    const std::vector<int>& height_section,
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

void send(const std::string& var_name,
          const std::vector<std::string>& send_varnames,
          const std::vector<std::string>& epmap,
          const std::vector<int>& height_sections,
          const framework::ExecutionContext& context,
          const framework::Scope& scope, bool sync) {
  framework::Scope* local_scope = scope.NewTmpScope();

  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  auto& cpu_ctx = *pool.Get(platform::CPUPlace());
  auto& actual_ctx = *pool.Get(context.GetPlace());

  distributed::RPCClient* rpc_client =
      distributed::RPCClient::GetInstance<RPCCLIENT_T>(
          context.Attr<int>("trainer_id"));

  auto* send_var = scope.FindVar(var_name);
  size_t out_num = send_varnames.size();
  if (send_var->IsType<framework::LoDTensor>()) {
    auto& send_tensor = send_var->Get<framework::LoDTensor>();
    auto& send_tensor_dims = send_tensor.dims();
    std::vector<framework::DDim> outs_dims;
    outs_dims.reserve(out_num);

    // infer output shape
    int num = context.Attr<int>("num");
    if (num > 0) {
      int64_t in_axis_dim = send_tensor_dims[0];
      PADDLE_ENFORCE_EQ(in_axis_dim % num, 0,
                        "tensor split does not result"
                        " in an equal division");
      size_t out_axis_dim = in_axis_dim / num;
      for (size_t i = 0; i < out_num; ++i) {
        auto dim = send_tensor_dims;
        dim[0] = out_axis_dim;
        outs_dims.push_back(dim);
      }
    } else if (height_sections.size() > 0) {
      PADDLE_ENFORCE_EQ(height_sections.size(), out_num,
                        "tensor split sections size"
                        "should be equal to output size.");
      for (size_t i = 0; i < out_num; ++i) {
        auto dim = send_tensor_dims;
        dim[0] = height_sections[i];
        outs_dims.push_back(dim);
      }
    }

    // create output var in local scope
    size_t row_offset = 0;
    for (auto i = 0; i < out_num; ++i) {
      auto* out =
          local_scope->Var(send_varnames[i])->GetMutable<framework::Tensor>();
      *out = send_tensor.Slice(row_offset, row_offset + outs_dims[i][0]);
      row_offset += outs_dims[i][0];
    }
  } else if (send_var->IsType<framework::LoDTensor>()) {
    // create output var in local scope
    for (auto& name : send_varnames) {
      local_scope->Var(name)->GetMutable<framework::SelectedRows>();
    }
  } else {
    PADDLE_THROW("unsupported var type");
  }

  std::vector<distributed::VarHandlePtr> rets;
  for (size_t i = 0; i < send_varnames.size(); i++) {
    auto& send_var_name = send_varnames[i];
    auto& endpoint = epmap[i];
    if (NeedSend(*local_scope, send_var_name)) {
      VLOG(3) << "sending " << send_var_name << " to " << endpoint;
      rets.push_back(rpc_client->AsyncSendVar(endpoint, cpu_ctx, *local_scope,
                                              send_var_name));
    } else {
      VLOG(3) << "don't send non-initialized variable: " << send_varnames[i];
    }
  }

  if (sync) {
    for (size_t i = 0; i < rets.size(); i++) {
      PADDLE_ENFORCE(rets[i]->Wait(), "internal error in RPCClient");
    }
  }

  delete local_scope;
}

};  // namespace distributed
};  // namespace operators
};  // namespace paddle

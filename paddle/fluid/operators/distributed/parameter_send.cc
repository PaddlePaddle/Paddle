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

template <typename T>
void ParameterSend<T>::operator()(const RpcContext &rpc_ctx,
                                  const framework::Scope &scope, bool sync) {
  framework::Scope *local_scope = scope.NewTmpScope();

  platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
  auto &cpu_ctx = *pool.Get(platform::CPUPlace());

  distributed::RPCClient *rpc_client =
      distributed::RPCClient::GetInstance<RPCCLIENT_T>(rpc_ctx.trainer_id);

  auto *send_var = scope.FindVar(rpc_ctx.var_name);
  size_t out_num = rpc_ctx.splited_var_names.size();
  if (send_var->IsType<framework::LoDTensor>()) {
    if (out_num > 1) {
      auto &send_tensor = send_var->Get<framework::LoDTensor>();
      auto &send_tensor_dims = send_tensor.dims();
      std::vector<framework::DDim> outs_dims;
      outs_dims.reserve(out_num);

      // infer output shape
      PADDLE_ENFORCE_EQ(rpc_ctx.height_sections.size(), out_num,
                        "tensor split sections size"
                        "should be equal to output size.");
      for (size_t i = 0; i < out_num; ++i) {
        auto dim = send_tensor_dims;
        dim[0] = rpc_ctx.height_sections[i];
        outs_dims.push_back(dim);
      }

      // create output var in local scope
      size_t row_offset = 0;
      for (auto i = 0; i < out_num; ++i) {
        framework::Tensor *out = local_scope->Var(rpc_ctx.splited_var_names[i])
                                     ->GetMutable<framework::LoDTensor>();
        *out = send_tensor.Slice(row_offset, row_offset + outs_dims[i][0]);
        row_offset += outs_dims[i][0];
      }
    }
  } else if (send_var->IsType<framework::SelectedRows>()) {
    auto &send_slr = send_var->Get<framework::SelectedRows>();
    auto abs_sections = ToAbsoluteSection(rpc_ctx.height_sections);

    auto &send_rows = send_slr.rows();
    std::vector<std::vector<int>> outs_rows_idx;
    std::vector<std::vector<int>> outs_dense_idx;

    outs_rows_idx.resize(out_num);
    outs_dense_idx.resize(out_num);

    auto row_numel = send_slr.value().numel() / send_slr.value().dims()[0];
    auto *src = send_slr.value().data<T>();

    // create output var in local scope
    std::vector<framework::SelectedRows *> outs;
    for (auto &name : rpc_ctx.splited_var_names) {
      auto *out = local_scope->Var(name)->GetMutable<framework::SelectedRows>();
      outs.push_back(out);
    }

    // split rows index into output sparse vars
    for (size_t i = 0; i < send_rows.size(); ++i) {
      int out_idx = GetSectionIndex(send_rows[i], abs_sections);
      outs_rows_idx[out_idx].push_back(send_rows[i]);
      outs_dense_idx[out_idx].push_back(i);
    }
    auto place = platform::CPUPlace();

    for (size_t i = 0; i < outs_rows_idx.size(); ++i) {
      auto rows_idx = outs_rows_idx[i];
      outs[i]->set_height(rpc_ctx.height_sections[i]);
      auto dims = send_slr.GetCompleteDims();
      dims[0] = rows_idx.size();
      outs[i]->mutable_rows()->clear();
      outs[i]->mutable_value()->mutable_data<T>(dims, send_slr.place());
      if (rows_idx.size() > 0) {
        for (auto idx : rows_idx) {
          outs[i]->mutable_rows()->push_back(idx - abs_sections[i]);
        }
        auto dst = outs[i]->mutable_value()->mutable_data<T>(place);
        for (size_t j = 0; j < rows_idx.size(); j++) {
          if (platform::is_cpu_place(place)) {
            memory::Copy(
                platform::CPUPlace(), dst + j * row_numel, platform::CPUPlace(),
                src + outs_dense_idx[i][j] * row_numel, sizeof(T) * row_numel);
          } else {
            PADDLE_THROW("do not support GPU now");
            /*
            #ifdef PADDLE_WITH_CUDA
                        auto stream = ctx.cuda_device_context().stream();
                        memory::Copy(platform::CUDAPlace(), dst + j * row_numel,
                                     platform::CUDAPlace(),
                                     src + outs_dense_idx[i][j] * row_numel,
                                     sizeof(T) * row_numel, stream);
            #else
                        PADDLE_THROW("Paddle is not compiled with GPU");
            #endif
            */
          }
        }
      }
      PADDLE_ENFORCE_EQ(rows_idx.size(), outs[i]->rows().size(),
                        "rows should has the same size with tensor dim 0");
    }

  } else {
    PADDLE_THROW("unsupported var type to send!");
  }

  std::vector<distributed::VarHandlePtr> rets;
  for (size_t i = 0; i < rpc_ctx.splited_var_names.size(); i++) {
    auto &send_var_name = rpc_ctx.splited_var_names[i];
    auto &endpoint = rpc_ctx.epmap[i];
    if (NeedSend(*local_scope, send_var_name)) {
      VLOG(3) << "sending " << send_var_name << " to " << endpoint;
      rets.push_back(rpc_client->AsyncSendVar(endpoint, cpu_ctx, *local_scope,
                                              send_var_name));
    } else {
      VLOG(3) << "don't send non-initialized variable: "
              << rpc_ctx.splited_var_names[i];
    }
  }

  // note!! only support sync send now
  if (true || sync) {
    for (size_t i = 0; i < rets.size(); i++) {
      PADDLE_ENFORCE(rets[i]->Wait(), "internal error in RPCClient");
    }
  }

  delete local_scope;
}

template struct ParameterSend<float>;

};  // namespace distributed
};  // namespace operators
};  // namespace paddle

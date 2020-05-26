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

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "paddle/fluid/operators/distributed/parameter_recv.h"

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/framework/tensor.h"

#include "paddle/fluid/operators/distributed/distributed.h"
#include "paddle/fluid/operators/distributed/rpc_client.h"
#include "paddle/fluid/operators/distributed/variable_response.h"
#include "paddle/fluid/operators/distributed_ops/send_recv_util.h"
#include "paddle/fluid/operators/strided_memcpy.h"

namespace paddle {
namespace operators {
namespace distributed {

using LoDTensor = framework::LoDTensor;
using LoDTensor = framework::LoDTensor;
using SelectedRows = framework::SelectedRows;
using DDim = framework::DDim;

template <typename T>
void ParameterRecv<T>::operator()(const RpcContext &rpc_ctx,
                                  const framework::Scope &scope) {
  VLOG(2) << "ParameterRecv in " << rpc_ctx.var_name;
  std::unique_ptr<framework::Scope> local_scope = scope.NewTmpScope();

  platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
  auto &cpu_ctx = *pool.Get(platform::CPUPlace());

  distributed::RPCClient *rpc_client =
      distributed::RPCClient::GetInstance<RPCCLIENT_T>(rpc_ctx.trainer_id);

  auto *recv_var = scope.FindVar(rpc_ctx.var_name);

  // recv all vars to local scope
  if (recv_var->IsType<framework::LoDTensor>() ||
      recv_var->IsType<framework::SelectedRows>()) {
    std::vector<distributed::VarHandlePtr> rets;
    for (size_t i = 0; i < rpc_ctx.splited_var_names.size(); i++) {
      auto &recv_var_name = rpc_ctx.splited_var_names[i];
      local_scope->Var(recv_var_name);
      VLOG(4) << "recv " << recv_var_name << " from " << rpc_ctx.epmap[i];
      if (recv_var->IsType<framework::LoDTensor>()) {
        // sparse param in recv_scope is LoDTensor
        rets.push_back(rpc_client->AsyncGetVar(rpc_ctx.epmap[i], cpu_ctx,
                                               *local_scope.get(),
                                               recv_var_name, recv_var_name));
      } else {
        // sparse param in pserver_scope is SelectedRows
        rets.push_back(rpc_client->AsyncGetVar(
            rpc_ctx.epmap[i], cpu_ctx, *local_scope.get(), recv_var_name,
            recv_var_name, recv_var_name));
      }
    }
    for (size_t i = 0; i < rets.size(); i++) {
      PADDLE_ENFORCE(rets[i]->Wait(), "internal error in RPCClient");
    }
  } else {
    PADDLE_THROW("unsupported var type to recv!");
  }

  // concat recved tensor into one var
  if (recv_var->IsType<framework::LoDTensor>()) {
    size_t output_offset = 0;
    size_t row_offset = 0;
    framework::Tensor *recv_tensor =
        recv_var->GetMutable<framework::LoDTensor>();

    int64_t recv_numel = 0;
    for (auto &recv_var_name : rpc_ctx.splited_var_names) {
      auto *recv_var = local_scope->FindVar(recv_var_name);
      if (recv_var->IsType<framework::LoDTensor>()) {
        auto &in = recv_var->Get<framework::LoDTensor>();
        recv_numel += in.numel();
        auto in_stride = framework::stride_numel(in.dims());
        auto out_stride = framework::stride_numel(recv_tensor->dims());
        if (platform::is_cpu_place(recv_tensor->place())) {
          VLOG(1) << "StridedNumelCopyWithAxis CPU Begin";
          auto cpu_ctx = paddle::platform::CPUDeviceContext();
          StridedNumelCopyWithAxis<T>(
              cpu_ctx, 0, recv_tensor->data<T>() + output_offset, out_stride,
              in.data<T>(), in_stride, in_stride[0]);
        } else {
          VLOG(1) << "StridedNumelCopyWithAxis CPU<->GPU Begin";
          auto cpu_ctx = paddle::platform::CPUDeviceContext();
          auto *gpu_ctx = reinterpret_cast<platform::CUDADeviceContext *>(
              platform::DeviceContextPool::Instance().Get(
                  recv_tensor->place()));

          StridedNumelCopyWithAxis<T>(
              gpu_ctx, cpu_ctx, 0, recv_tensor->data<T>() + output_offset,
              out_stride, in.data<T>(), in_stride, in_stride[0]);
        }

        output_offset += in_stride[0];
      } else if (recv_var->IsType<framework::SelectedRows>()) {
        auto &recv_slr = recv_var->Get<framework::SelectedRows>();
        auto &recv_dims = recv_tensor->dims();
        int64_t width = recv_dims[1];
        recv_numel += recv_slr.height() * width;
        PADDLE_ENFORCE_EQ(recv_slr.value().dims()[1], width);
        PADDLE_ENFORCE_EQ(recv_slr.value().dims()[0], recv_slr.rows().size());
        VLOG(3) << "recv slr " << recv_var_name << " dims "
                << recv_slr.value().dims();
        if (VLOG_IS_ON(3)) {
          std::ostringstream sstream;
          sstream << "[";
          for (auto &row_id : recv_slr.rows()) {
            sstream << row_id << ", ";
          }
          sstream << "]";
          VLOG(3) << "recv_slr size: " << recv_slr.rows().size() << " "
                  << sstream.str();
        }

        for (size_t i = 0; i < recv_slr.rows().size(); ++i) {
          auto row_id = recv_slr.rows()[i] + row_offset;
          PADDLE_ENFORCE_LT(row_id, recv_dims[0]);
          memcpy(recv_tensor->data<T>() + row_id * width,
                 recv_slr.value().data<T>() + i * width, sizeof(T) * width);
        }
        row_offset += recv_slr.height();
      } else {
        PADDLE_THROW("unsupported recieved var type");
      }
    }
    auto numel = recv_tensor->numel();
    if (recv_numel != numel) {
      LOG(FATAL) << "recv_numel: " << recv_numel << " acture numel: " << numel;
    }
    PADDLE_ENFORCE_EQ(recv_numel, numel);
  } else if (recv_var->IsType<framework::SelectedRows>()) {
    auto cpu_place = platform::CPUPlace();
    auto *slr = recv_var->GetMutable<framework::SelectedRows>();
    slr->mutable_rows()->clear();
    slr->mutable_value()->mutable_data<float>({{}}, cpu_place);
    int64_t width = 0;
    int64_t height = 0;
    std::vector<int64_t> new_rows{};

    // trans sparse ids from local to global
    std::vector<int64_t> abs_sections =
        ToAbsoluteSection(rpc_ctx.height_sections);

    for (size_t i = 0; i < rpc_ctx.splited_var_names.size(); i++) {
      auto &recv_var_name = rpc_ctx.splited_var_names[i];
      auto *var = local_scope->FindVar(recv_var_name);
      auto *var_slr = var->GetMutable<framework::SelectedRows>();
      auto *var_slr_row = var_slr->mutable_rows();
      width = var_slr->mutable_value()->dims()[1];
      height += var_slr->height();
      auto row_offset = abs_sections[i];
      VLOG(4) << "Recv split_var " << recv_var_name << " Row size "
              << var_slr_row->size();
      for (size_t j = 0; j < var_slr_row->size(); j++) {
        new_rows.push_back(row_offset + (*var_slr_row)[j]);
      }
    }
    slr->set_rows(new_rows);
    slr->set_height(height);
    slr->mutable_value()->mutable_data<float>(
        framework::make_ddim(
            {static_cast<int64_t>(slr->mutable_rows()->size()), width}),
        cpu_place);
    auto *slr_data = slr->mutable_value()->data<float>();

    size_t row_offset = 0;
    for (auto &recv_var_name : rpc_ctx.splited_var_names) {
      auto *var = local_scope->FindVar(recv_var_name);
      auto *var_slr = var->GetMutable<framework::SelectedRows>();
      auto *var_slr_row = var_slr->mutable_rows();
      auto var_slr_row_size = var_slr_row->size();
      auto *var_slr_data = var_slr->mutable_value()->data<float>();

      memcpy(slr_data + row_offset * width, var_slr_data,
             sizeof(float) * width * var_slr_row_size);
      row_offset += var_slr_row_size;
    }
  }

  VLOG(2) << "ParameterRecv out " << rpc_ctx.var_name;
}

template struct ParameterRecv<float>;

};  // namespace distributed
};  // namespace operators
};  // namespace paddle

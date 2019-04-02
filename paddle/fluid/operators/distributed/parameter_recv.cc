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
  VLOG(3) << "ParameterRecv in";
  std::unique_ptr<framework::Scope> local_scope = scope.NewTmpScope();

  platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
  auto &cpu_ctx = *pool.Get(platform::CPUPlace());

  distributed::RPCClient *rpc_client =
      distributed::RPCClient::GetInstance<RPCCLIENT_T>(rpc_ctx.trainer_id);

  auto *recv_var = scope.FindVar(rpc_ctx.var_name);

  // recv all vars to local scope
  if (recv_var->IsType<framework::LoDTensor>()) {
    std::vector<distributed::VarHandlePtr> rets;
    for (size_t i = 0; i < rpc_ctx.splited_var_names.size(); i++) {
      auto &recv_var_name = rpc_ctx.splited_var_names[i];
      local_scope->Var(recv_var_name);
      VLOG(3) << "recv " << recv_var_name << " from " << rpc_ctx.epmap[i];
      rets.push_back(rpc_client->AsyncGetVar(rpc_ctx.epmap[i], cpu_ctx,
                                             *local_scope.get(), recv_var_name,
                                             recv_var_name));
    }
    for (size_t i = 0; i < rets.size(); i++) {
      PADDLE_ENFORCE(rets[i]->Wait(), "internal error in RPCClient");
    }
  } else {
    PADDLE_THROW("unsupported var type to recv!");
  }

  // concat recved tensor into one var
  {
    size_t output_offset = 0;
    size_t row_offset = 0;
    framework::Tensor *recv_tensor =
        recv_var->GetMutable<framework::LoDTensor>();
    auto dev_ctx = paddle::platform::CPUDeviceContext();
    int64_t recv_numel = 0;
    for (auto &recv_var_name : rpc_ctx.splited_var_names) {
      auto *recv_var = local_scope->FindVar(recv_var_name);
      if (recv_var->IsType<framework::LoDTensor>()) {
        auto &in = recv_var->Get<framework::LoDTensor>();
        recv_numel += in.numel();
        auto in_stride = framework::stride_numel(in.dims());
        auto out_stride = framework::stride_numel(recv_tensor->dims());
        StridedNumelCopyWithAxis<T>(
            dev_ctx, 0, recv_tensor->data<T>() + output_offset, out_stride,
            in.data<T>(), in_stride, in_stride[0]);
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

        for (auto i = 0; i < recv_slr.rows().size(); ++i) {
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
  }

  VLOG(3) << "ParameterRecv out " << rpc_ctx.var_name;
}

template struct ParameterRecv<float>;

};  // namespace distributed
};  // namespace operators
};  // namespace paddle

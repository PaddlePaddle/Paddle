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
void ParameterRecv<T>::operator()(const std::string &var_name,
                                  const std::vector<std::string> &recv_varnames,
                                  const std::vector<std::string> &epmap,
                                  const framework::ExecutionContext &ctx,
                                  const framework::Scope &scope) {
  framework::Scope *local_scope = scope.NewTmpScope();

  platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
  auto &cpu_ctx = *pool.Get(platform::CPUPlace());

  distributed::RPCClient *rpc_client =
      distributed::RPCClient::GetInstance<RPCCLIENT_T>(
          ctx.Attr<int>("trainer_id"));

  auto *recv_var = scope.FindVar(var_name);

  std::vector<framework::Tensor *> recved_tensors;

  // recv all vars to local scope
  if (recv_var->IsType<framework::LoDTensor>()) {
    std::vector<distributed::VarHandlePtr> rets;
    for (size_t i = 0; i < recv_varnames.size(); i++) {
      auto &recv_var_name = recv_varnames[i];
      framework::Tensor *t =
          local_scope->Var(recv_var_name)->GetMutable<framework::LoDTensor>();
      recved_tensors.push_back(t);
      VLOG(3) << "recv " << recv_var_name << " from " << epmap[i];
      rets.push_back(rpc_client->AsyncGetVar(epmap[i], cpu_ctx, *local_scope,
                                             recv_var_name, recv_var_name));
    }
    for (size_t i = 0; i < rets.size(); i++) {
      PADDLE_ENFORCE(rets[i]->Wait(), "internal error in RPCClient");
    }
  } else {
    PADDLE_THROW("unsupported var type to send!");
  }

  // concat recved tensor into one var
  {
    size_t output_offset = 0;
    framework::Tensor *recv_tensor =
        recv_var->GetMutable<framework::LoDTensor>();
    for (auto *in : recved_tensors) {
      auto in_stride = framework::stride_numel(in->dims());
      auto out_stride = framework::stride_numel(recv_tensor->dims());
      StridedNumelCopyWithAxis<T>(
          ctx.device_context(), 0, recv_tensor->data<T>() + output_offset,
          out_stride, in->data<T>(), in_stride, in_stride[0]);
      output_offset += in_stride[0];
    }
  }

  delete local_scope;
}

template struct ParameterRecv<float>;

};  // namespace distributed
};  // namespace operators
};  // namespace paddle

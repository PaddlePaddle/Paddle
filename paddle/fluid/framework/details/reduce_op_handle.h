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

#pragma once

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/details/op_handle_base.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/platform/device_context.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/nccl_helper.h"
#endif

namespace paddle {
namespace framework {
namespace details {

struct ReduceOpHandle : public OpHandleBase {
  const std::vector<Scope *> &local_scopes_;
  const std::vector<platform::Place> &places_;
  const size_t dst_scope_id_;

#ifdef PADDLE_WITH_CUDA
  const platform::NCCLContextMap *nccl_ctxs_;
  ReduceOpHandle(const std::vector<Scope *> &local_scopes,
                 const std::vector<platform::Place> &places,
                 const platform::NCCLContextMap *nccl_ctxs,
                 const size_t dst_scope_id = -1,
                 const std::string var_name = "")
      : local_scopes_(local_scopes),
        places_(places),
        dst_scope_id_(dst_scope_id),
        nccl_ctxs_(nccl_ctxs),
        var_name_(var_name) {
    if (nccl_ctxs_) {
      for (auto &p_ctx : nccl_ctxs_->contexts_) {
        dev_ctxes_[platform::CUDAPlace(p_ctx.first)] = p_ctx.second.ctx_.get();
      }
    }
  }
#else
  ReduceOpHandle(const std::vector<Scope *> &local_scopes,
                 const std::vector<platform::Place> &places,
                 const size_t dst_scope_id = -1,
                 const std::string var_name = "")
      : local_scopes_(local_scopes),
        places_(places),
        dst_scope_id_(dst_scope_id),
        var_name_(var_name) {}
#endif

  const std::string var_name_;

  std::string Name() const override;

  bool IsMultiDeviceTransfer() override { return true; };

 protected:
  void RunImpl() override;

  std::vector<const LoDTensor *> GetGroupValues();

  template <typename T>
  std::vector<const T *> GetInputValues(
      const std::vector<VarHandle *> &in_var_handles,
      const std::vector<const Scope *> &var_scopes);

  void ReduceGroup(const std::vector<VarHandle *> &out_var_handles,
                   const std::vector<const Scope *> &var_scopes);

  void ReduceInput(const std::vector<VarHandle *> &in_var_handles,
                   const std::vector<VarHandle *> &out_var_handles,
                   const std::vector<const Scope *> &var_scopes);

#ifdef PADDLE_WITH_CUDA
  void NCCLReduce(const std::vector<const LoDTensor *> &lod_tensors,
                  const size_t dst_dev_id, Variable *out_var,
                  std::vector<std::function<void()>> *nccl_reduce_calls);
#endif
};

}  // namespace details
}  // namespace framework
}  // namespace paddle

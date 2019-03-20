// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>
#include <vector>
#include "paddle/fluid/framework/details/op_handle_base.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
#include "paddle/fluid/platform/nccl_helper.h"
#endif

namespace paddle {
namespace framework {
namespace details {

struct DataBalanceOpHandle : public OpHandleBase {
 public:
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
  DataBalanceOpHandle(ir::Node *node, const std::vector<Scope *> &local_scopes,
                      const std::vector<platform::Place> &places,
                      const platform::NCCLContextMap *ctxs);
#else
  DataBalanceOpHandle(ir::Node *node, const std::vector<Scope *> &local_scopes,
                      const std::vector<platform::Place> &places);
#endif

  std::string Name() const override;

  bool IsMultiDeviceTransfer() override { return false; };

 protected:
  void RunImpl() override;

 private:
  // std::vector<(src_dev_id, dst_dev_id, trans_size)>
  std::vector<std::array<int, 3>> GetBalancePlan(
      const std::vector<int> &batch_size_per_device);

  const std::vector<Scope *> local_scopes_;
  const std::vector<platform::Place> places_;
};

}  // namespace details
}  // namespace framework
}  // namespace paddle

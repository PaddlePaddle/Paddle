//   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/framework/feed_fetch_type.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/platform/device_context.h"

namespace pten {
class DenseTensor;
}  // namespace pten

namespace paddle {
namespace framework {

namespace ir {
class Node;
}  // namespace ir
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace details {

struct FetchAsyncOpHandle : public OpHandleBase {
 public:
  FetchAsyncOpHandle(ir::Node *node, FetchResultType *data, size_t offset,
                     std::vector<Scope *> *local_scopes,
                     std::vector<Scope *> *local_exec_scopes,
                     bool return_merged);

  ~FetchAsyncOpHandle();

  void RecordWaitEventOnCtx(platform::DeviceContext *waited_ctx) override;

  std::string Name() const override;

  bool IsMultiDeviceTransfer() override;

 protected:
  void RunImpl() override;

  std::vector<Scope *> GetLocalScopes() override { return *local_scopes_; }

  void FetchMergedLodTensor(
      const std::vector<const LoDTensor *> &src_lodtensors,
      LoDTensor *dst_lodtensor);

 private:
  FetchResultType *data_;
  size_t offset_;
  std::vector<Scope *> *local_scopes_;
  std::vector<Scope *> *local_exec_scopes_;
  bool return_merged_;
};

}  // namespace details
}  // namespace framework
}  // namespace paddle

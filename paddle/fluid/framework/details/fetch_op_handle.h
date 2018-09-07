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

#include <string>
#include <vector>

#include "paddle/fluid/framework/details/op_handle_base.h"
#include "paddle/fluid/framework/feed_fetch_type.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace framework {
namespace details {

struct FetchOpHandle : public OpHandleBase {
 public:
  FetchOpHandle(FeedFetchList *data, size_t offset,
                std::vector<Scope *> *local_scopes);

  ~FetchOpHandle();

  void RecordWaitEventOnCtx(platform::DeviceContext *waited_ctx) override;

  void WaitAndMergeCPUTensors() const;

  std::string Name() const override;

 protected:
  void RunImpl() override;

  void WaitInputVarGenerated(const platform::Place &place) override;

 private:
  FeedFetchList *data_;
  size_t offset_;
  std::vector<Scope *> *local_scopes_;
  std::vector<LoDTensor> tensors_;
};

}  // namespace details
}  // namespace framework
}  // namespace paddle

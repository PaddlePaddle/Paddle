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

#include <memory>
#include <string>

#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/operators/distributed/request.h"
#include "paddle/fluid/operators/distributed/request_handler.h"

namespace paddle {
namespace operators {
namespace distributed {

using Scope = paddle::framework::Scope;

class CheckpointHandler final : public RequestHandler {
 public:
  bool Handle(RPCRequest* request) override;
  framework::Variable* GetOrCreateRequestVar(const std::string& varname,
                                             RPCRequest* request) override;

  void SetId(int id) { checkpoint_block_id_ = id; }

  void SetCheckpointNotifyPreparedCtx(
      std::shared_ptr<framework::ExecutorPrepareContext> g) {
    checkpoint_prepared_ctx_ = g;
  }

 private:
  int checkpoint_block_id_ = -1;
  std::shared_ptr<framework::ExecutorPrepareContext> checkpoint_prepared_ctx_;
};

}  // namespace distributed
}  // namespace operators
}  // namespace paddle

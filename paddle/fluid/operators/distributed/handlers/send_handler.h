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
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/operators/distributed/request.h"
#include "paddle/fluid/operators/distributed/request_handler.h"

namespace paddle {
namespace operators {
namespace distributed {

class SendHandlerSync final : public RequestHandler {
 public:
  bool Handle(RPCRequest *request, Scope *scope) override;
};

class SendHandlerAsync final : public RequestHandler {
 public:
  bool Handle(RPCRequest *request, Scope *scope) override;
  void SetGradToPreparedCtx(
      std::unordered_map<
          std::string, std::shared_ptr<framework::ExecutorPrepareContext>> *g) {
    grad_to_prepared_ctx_ = g;
  }

 private:
  std::unordered_map<std::string,
                     std::shared_ptr<framework::ExecutorPrepareContext>>
      *grad_to_prepared_ctx_;
};

}  // namespace distributed
}  // namespace operators
}  // namespace paddle

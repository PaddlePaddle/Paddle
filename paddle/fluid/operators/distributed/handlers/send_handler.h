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

using Scope = paddle::framework::Scope;

class SendHandlerSync final : public RequestHandler {
 public:
  void Start(std::function<RPCRequest*(framework::Scope*)> start) override;
  bool Handle(RPCRequest* request) override;
};

class SendHandlerAsync final : public RequestHandler {
 public:
  void Start(std::function<RPCRequest*(framework::Scope*)> start) override;
  bool Handle(RPCRequest* request) override;

 private:
  framework::Scope* local_scope_;
};

}  // namespace distributed
}  // namespace operators
}  // namespace paddle

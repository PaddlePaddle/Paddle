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

#include "paddle/fluid/operators/distributed/handlers/get_no_barrier_handler.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/string/piece.h"

namespace paddle {
namespace operators {
namespace distributed {

framework::Variable* GetVarNoBarrierInternal(framework::Scope* scope,
                                             const std::string& varname) {
  // get var from pserver immediately without barriers
  string::Piece without_barrier_piece(WITHOUT_BARRIER_MESSAGE);
  string::Piece var_name_piece = string::Piece(varname);

  PADDLE_ENFORCE(string::Contains(var_name_piece, without_barrier_piece));
  var_name_piece = string::TrimSuffix(var_name_piece, without_barrier_piece);
  VLOG(4) << "Get var " << var_name_piece << " with "
          << WITHOUT_BARRIER_MESSAGE;
  return scope->FindVar(var_name_piece.ToString());
}

framework::Variable* GetNoBarrierHandler::GetOrCreateRequestVar(
    const std::string& varname, RPCRequest* request) {
  return GetVarNoBarrierInternal(scope_, varname);
}

bool GetNoBarrierHandler::Handle(RPCRequest* request) {
  VLOG(4) << "GetNoBarrierHandler " << request->varname_;
  request->out_var_ = GetVarNoBarrierInternal(scope_, request->varname_);
  return true;
}

}  // namespace distributed
}  // namespace operators
}  // namespace paddle

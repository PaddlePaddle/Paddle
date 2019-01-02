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

#include "paddle/fluid/operators/distributed/handlers/prefetch_handler.h"

namespace paddle {
namespace operators {
namespace distributed {

static inline void BuildVar(const std::string& param_name,
                            std::initializer_list<const char*> arguments,
                            paddle::framework::proto::OpDesc::Var* var) {
  var->set_parameter(param_name);
  for (auto& arg_name : arguments) {
    *var->mutable_arguments()->Add() = arg_name;
  }
}

bool Handle(RPCRequest* request, Scope* scope) {
  VLOG(4) << "RequestPrefetchHandler " << varname;

  if (request->table_name_.empty()) {
    auto var_desc = program_->Block(0).FindVar(request->out_var_name_);
    InitializeVariable(*(request->outvar_), var_desc->GetType());
    executor_->RunPreparedContext(
        (*prefetch_var_name_to_prepared_ctx_)[request->varname_].get(), scope);
  } else {
    (*(request->outvar_))->GetMutable<framework::LoDTensor>();
    auto lookup_table_op = BuildLookupTableOp(
        request->table_name_, request->varname_, request->out_var_name_);
    paddle::platform::CPUPlace cpu_place;
    lookup_table_op->Run(*scope, cpu_place);
  }
  return true;
}

}  // namespace distributed
}  // namespace operators
}  // namespace paddle

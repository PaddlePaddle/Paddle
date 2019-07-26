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
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/variable_helper.h"

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

framework::Variable* PrefetchHandler::GetOrCreateRequestVar(
    const std::string& varname, RPCRequest* request) {
  request->scope_ = &scope_->NewScope();
  return request->scope_->Var(varname);
}

bool PrefetchHandler::Handle(RPCRequest* request) {
  VLOG(4) << "RequestPrefetchHandler " << request->varname_
          << " outname: " << request->out_var_name_
          << " tablename: " << request->table_name_;
  request->out_var_ = request->scope_->Var(request->out_var_name_);

  if (request->table_name_.empty()) {
    auto var_desc = program_->Block(0).FindVar(request->out_var_name_);
    InitializeVariable(request->out_var_, var_desc->GetType());
    executor_->RunPreparedContext(
        (*prefetch_var_name_to_prepared_ctx_)[request->varname_].get(),
        request->scope_);
  } else {
    request->out_var_->GetMutable<framework::LoDTensor>();
    auto lookup_table_op = BuildLookupTableOp(
        request->table_name_, request->varname_, request->out_var_name_);
    paddle::platform::CPUPlace cpu_place;
    lookup_table_op->Run(*(request->scope_), cpu_place);
  }
  return true;
}

std::unique_ptr<paddle::framework::OperatorBase>
PrefetchHandler::BuildLookupTableOp(const std::string& table_name,
                                    const std::string& id_name,
                                    const std::string& out_name) {
  paddle::framework::proto::OpDesc op_desc;
  op_desc.set_type("lookup_table");
  BuildVar("W", {table_name.data()}, op_desc.add_inputs());
  BuildVar("Ids", {id_name.data()}, op_desc.add_inputs());
  BuildVar("Out", {out_name.data()}, op_desc.add_outputs());

  auto op = paddle::framework::OpRegistry::CreateOp(op_desc);
  return op;
}

}  // namespace distributed
}  // namespace operators
}  // namespace paddle

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
#include <unordered_map>

#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/platform/macros.h"

namespace paddle {
namespace operators {
namespace distributed {

enum RequestType {
  SEND = 0,
  RECV,
  PREFETCH,
  CHECKPOINT,
  RECV_NO_BARRIER,
  GET_MONOMER,
  GET_MONOMER_BARRIER,
};

class RPCRequest {
 public:
  RPCRequest() : out_var_(nullptr), scope_(nullptr) {}
  void Prepare(const std::string &varname, framework::Variable *invar,
               const std::string &out_var_name, const std::string &table_name,
               int trainer_id, RequestType req_type) {
    varname_ = varname;
    var_ = invar;
    out_var_name_ = out_var_name;
    table_name_ = table_name;
    trainer_id_ = trainer_id;
    type_ = req_type;
  }

 public:
  int trainer_id_;
  std::string varname_;
  framework::Variable *var_;
  std::string out_var_name_;
  framework::Variable *out_var_;
  std::string table_name_;
  RequestType type_;

  // scope for current request
  framework::Scope *scope_;

  DISABLE_COPY_AND_ASSIGN(RPCRequest);
};

struct EnumClassHash {
  template <typename T>
  std::size_t operator()(T t) const {
    return static_cast<std::size_t>(t);
  }
};

}  // namespace distributed
}  // namespace operators
}  // namespace paddle

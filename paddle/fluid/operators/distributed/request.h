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
  GET_MONOMER,
  GET_MONOMER_BARRIER,
};

class RPCRequest {
 public:
  RPCRequest() {}

 public:
  int trainer_id_;
  RequestType type_;
  std::string varname_;
  framework::Variable *var_;
  framework::Variable **out_var_;
  // prefetch table name and out_var_name.
  std::string out_var_name_;
  std::string table_name_;

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

// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/paddle2cinn/cinn_compiled_object.h"

#include <map>

#include "paddle/fluid/framework/feed_fetch_type.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"

namespace paddle {
namespace framework {
namespace paddle2cinn {

CinnCompiledObject::CinnCompiledObject() {
  // TODO(zhhsplendid): complete this function after CINN interface is ready
}
CinnCompiledObject::~CinnCompiledObject() {
  // TODO(zhhsplendid): complete this function after CINN interface is ready
}

void CinnCompiledObject::Compile(
    const ir::Graph& graph,
    std::map<std::string, const LoDTensor*>* feed_targets) {
  // TODO(zhhsplendid): complete this function after CINN interface is ready
}

std::map<std::string, FetchType*> CinnCompiledObject::Run(
    Scope* scope, std::map<std::string, const LoDTensor*>* feed_targets) {
  // TODO(zhhsplendid): complete this function after CINN interface is ready
  return std::map<std::string, FetchType*>();
}

}  // namespace paddle2cinn
}  // namespace framework
}  // namespace paddle

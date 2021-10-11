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

#include "paddle/fluid/framework/paddle2cinn/cinn_runner.h"

#include <memory>

#include "gtest/gtest.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"

namespace paddle {
namespace framework {
namespace paddle2cinn {

using ir::Graph;

TEST(CinnRunnerTest, TodoTest) {
  ProgramDesc empty_program;
  Graph empty_graph(empty_program);
  Scope empty_scope;
  std::map<std::string, const LoDTensor*> empty_feed;

  std::shared_ptr<CinnRunner> cinn_runner = CinnRunner::GetInstance();
  cinn_runner->ReplaceWithCinn(&empty_graph);
  cinn_runner->Run(empty_graph, &empty_scope, &empty_feed);
}

}  // namespace paddle2cinn
}  // namespace framework
}  // namespace paddle

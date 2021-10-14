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

#include <map>

#include "gtest/gtest.h"

#include "paddle/fluid/framework/feed_fetch_type.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/paddle2cinn/cinn_compiled_object.h"
#include "paddle/fluid/framework/program_desc.h"

namespace paddle {
namespace framework {
namespace paddle2cinn {

TEST(CinnCompiledObjecctTest, TodoTest) {
  ProgramDesc empty_program;
  ir::Graph empty_graph(empty_program);
  std::map<std::string, const LoDTensor*> empty_feed;
  Scope empty_scope;

  CinnCompiledObject compiled_obj;
  compiled_obj.Compile(empty_graph, &empty_feed);
  auto fetch = compiled_obj.Run(&empty_scope, &empty_feed);
}

}  // namespace paddle2cinn
}  // namespace framework
}  // namespace paddle

// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/lite/model_parser/model_parser.h"
#include <gtest/gtest.h>
#include "paddle/fluid/lite/core/scope.h"

namespace paddle {
namespace lite {

TEST(ModelParser, LoadProgram) {
  auto program = LoadProgram(
      "/home/chunwei/project2/models/fc/fluid_checkpoint/__model__");
}

TEST(ModelParser, LoadParam) {
  Scope scope;
  auto* v = scope.Var("xxx");
  LoadParam("/home/chunwei/project2/models/fc/fluid_checkpoint/b1", v);
  const auto& t = v->Get<TensorBase>();
  LOG(INFO) << "loaded\n";
  LOG(INFO) << t;
}

TEST(ModelParser, LoadModel) {
  Scope scope;
  framework::proto::ProgramDesc prog;
  LoadModel("/home/chunwei/project2/models/fc/fluid_checkpoint", &scope, &prog);
}

}  // namespace lite
}  // namespace paddle

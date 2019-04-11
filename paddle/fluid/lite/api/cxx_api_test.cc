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

#include "paddle/fluid/lite/api/cxx_api.h"
#include <gtest/gtest.h>
#include "paddle/fluid/lite/core/executor.h"
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {

TEST(CXXApi, test) {
  Scope scope;
  framework::proto::ProgramDesc prog;
  LoadModel("/home/chunwei/project2/models/model2", &scope, &prog);
  framework::ProgramDesc prog_desc(prog);

  lite::Executor executor(&scope,
                          {OpLite::Place{TARGET(kHost), PRECISION(kFloat)}});

  auto x = scope.Var("a")->GetMutable<Tensor>();
  x->Resize({100, 100});
  x->mutable_data<float>();

  executor.PrepareWorkspace(prog_desc, &scope);
  executor.Build(prog_desc);
  executor.Run();
}

}  // namespace lite
}  // namespace paddle

USE_LITE_OP(mul);
USE_LITE_OP(fc);
USE_LITE_OP(scale);
USE_LITE_KERNEL(fc, kHost, kFloat);
USE_LITE_KERNEL(mul, kHost, kFloat);
USE_LITE_KERNEL(scale, kHost, kFloat);

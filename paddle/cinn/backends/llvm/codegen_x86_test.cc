// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/backends/llvm/codegen_x86.h"

#include <gtest/gtest.h>

#include "paddle/cinn/backends/llvm/simple_jit.h"
#include "paddle/cinn/cinn.h"
#include "paddle/cinn/common/test_helper.h"
#include "paddle/cinn/runtime/cinn_runtime.h"

namespace cinn {
namespace backends {

TEST(Vectorize, basic) {
  Expr M(1024);
  Placeholder<float> A("A", {M});
  Placeholder<float> B("B", {M});

  auto C = Compute(
      {M}, [&](Expr i) { return A(i) + B(i); }, "C");
  auto stages = CreateStages({C});

  stages[C]->Vectorize(0, 8);

  auto fn = Lower("fn", stages, {A, B, C});

  LOG(INFO) << "fn: " << fn;

  Module::Builder builder("module", common::DefaultHostTarget());
  builder.AddFunction(fn);

  auto module = builder.Build();

  LOG(INFO) << "\n" << module->functions[0];

  auto jit = SimpleJIT::Create();
  jit->Link(builder.Build());

  auto fn_ = jit->Lookup("fn");

  auto* fn_ptr = reinterpret_cast<lower_func_ptr_t>(fn_);

  auto* A_buf = common::BufferBuilder(Float(32), {1024})
                    .set_random()
                    .set_align(64)
                    .Build();
  auto* B_buf = common::BufferBuilder(Float(32), {1024})
                    .set_random()
                    .set_align(64)
                    .Build();
  auto* C_buf =
      common::BufferBuilder(Float(32), {1024}).set_zero().set_align(64).Build();

  auto args = common::ArgsBuilder().Add(A_buf).Add(B_buf).Add(C_buf).Build();

  fn_ptr(reinterpret_cast<void**>(args.data()), args.size());

  auto* A_data = reinterpret_cast<float*>(A_buf->memory);
  auto* B_data = reinterpret_cast<float*>(B_buf->memory);
  auto* C_data = reinterpret_cast<float*>(C_buf->memory);
  for (int i = 0; i < C_buf->num_elements(); i++) {
    ASSERT_NEAR(A_data[i] + B_data[i], C_data[i], 1e-5);
  }
}

}  // namespace backends
}  // namespace cinn

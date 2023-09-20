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

#include <iostream>

#include "paddle/cinn/backends/compiler.h"
#include "paddle/cinn/cinn.h"
#include "paddle/cinn/common/test_helper.h"
#include "paddle/cinn/hlir/pe/elementwise.h"
#include "paddle/cinn/hlir/pe/nn.h"
#include "paddle/cinn/runtime/cpu/use_extern_funcs.h"
#include "paddle/cinn/utils/timer.h"

namespace cinn {
namespace backends {

// test x86 compiler
int run() {
  Expr M(4), N(4);

  auto create_module = [&]() {
    Placeholder<float> A("A", {M, N});
    Placeholder<float> B("B", {M, N});

    auto C = Compute(
        {M, N}, [=](Expr i, Expr j) { return A(i, j) + B(i, j); }, "C");
    return std::make_tuple(A, B, C);
  };

  // test x86
  auto module = create_module();  // NOLINT
  auto& A = std::get<0>(module);
  auto& B = std::get<1>(module);
  auto& C = std::get<2>(module);

  auto stages = CreateStages({C});

  auto fn = Lower("fn", stages, {A, B, C});

  ir::Module::Builder builder("some_module", common::DefaultHostTarget());
  builder.AddFunction(fn);

  auto compiler = Compiler::Create(common::DefaultHostTarget());
  compiler->Build(builder.Build());

  auto* fnp = compiler->Lookup("fn");
  if (fnp == nullptr) {
    std::cerr << "lookup function failed." << std::endl;
    return 1;
  }

  auto* Ab = common::BufferBuilder(Float(32), {M.as_int32(), N.as_int32()})
                 .set_random()
                 .Build();
  auto* Bb = common::BufferBuilder(Float(32), {M.as_int32(), N.as_int32()})
                 .set_random()
                 .Build();
  auto* Cb = common::BufferBuilder(Float(32), {M.as_int32(), N.as_int32()})
                 .set_zero()
                 .Build();

  auto args = common::ArgsBuilder().Add(Ab).Add(Bb).Add(Cb).Build();
  reinterpret_cast<void (*)(void*, int)>(fnp)(args.data(), args.size());

  // test result
  auto* Ad = reinterpret_cast<float*>(Ab->memory);
  auto* Bd = reinterpret_cast<float*>(Bb->memory);
  auto* Cd = reinterpret_cast<float*>(Cb->memory);
  for (int i = 0; i < Ab->num_elements(); i++) {
    if (abs(Ad[i] + Bd[i] - Cd[i]) > 1e-5) {
      std::cerr << "ERROR: Compute failed." << std::endl;
      return 1;
    }
  }

  std::cout << "run demo successfully." << std::endl;
  return 0;
}
}  // namespace backends
}  // namespace cinn

int main() { return cinn::backends::run(); }

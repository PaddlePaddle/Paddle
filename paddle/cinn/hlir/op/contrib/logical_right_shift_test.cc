// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/hlir/op/contrib/logical_right_shift.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "paddle/cinn/backends/codegen_c.h"
#include "paddle/cinn/backends/codegen_c_x86.h"
#include "paddle/cinn/backends/codegen_cuda_dev.h"
#include "paddle/cinn/common/context.h"
#include "paddle/cinn/lang/lower.h"
#include "paddle/cinn/lang/placeholder.h"
#include "paddle/cinn/poly/stage.h"

namespace cinn {
namespace hlir {
namespace op {

TEST(GenerateCode_Cpu, LogicalRightShift) {
  common::Context::Global().ResetNameId();

  common::Target target = common::DefaultHostTarget();
  lang::Placeholder<int> x("x", std::vector<int>{10});
  lang::Placeholder<int> y("y", std::vector<int>{10});
  ir::Tensor res = LogicalRightShift(x, y, target, "test_logical_right_shift");

  poly::StageMap stages = poly::CreateStages({res});
  std::vector<ir::LoweredFunc> funcs =
      lang::LowerVec("TestGenerateCodeCpu_LogicalRightShift",
                     stages,
                     {res},
                     {},
                     {},
                     nullptr,
                     target,
                     true);

  VLOG(6) << "Expr before CPU codegen:";
  VLOG(6) << funcs[0]->body;

  ir::Module::Builder builder("LogicalRightShift_Module", target);
  for (auto& f : funcs) {
    builder.AddFunction(f);
  }

  backends::CodeGenCX86 codegen(target, backends::CodeGenCX86::Feature::AVX512);
  codegen.SetInlineBuiltinCodes(false);
  std::string code =
      codegen.Compile(builder.Build(), backends::CodeGenC::OutputKind::CImpl);
  VLOG(6) << "Cpu Codegen result:";
  VLOG(6) << code << std::endl;
}

}  // namespace op
}  // namespace hlir
}  // namespace cinn

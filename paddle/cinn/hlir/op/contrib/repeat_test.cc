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

#include "paddle/cinn/hlir/op/contrib/repeat.h"

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

TEST(GenerateCode_Cpu, Repeat) {
  common::Context::Global().ResetNameId();

  common::Target target = common::DefaultHostTarget();

  ir::Expr m(4);
  ir::Expr n(4);
  const int repeats = 2;
  const int axis = 0;

  lang::Placeholder<int32_t> in("in", {m, n});

  std::vector<ir::Tensor> res = Repeat(in, repeats, axis, "test_repeat");

  poly::StageMap stages = poly::CreateStages({res});
  std::vector<ir::LoweredFunc> funcs = lang::LowerVec(
      "TestGenerateCodeCpu_Repeat", stages, res, {}, {}, nullptr, target, true);

  VLOG(6) << "Expr before CPU codegen:";
  VLOG(6) << funcs[0]->body;

  auto target_source_ir = R"ROC(
function TestGenerateCodeCpu_Repeat (_test_repeat)
{
  ScheduleBlock(root)
  {
    serial for (i, 0, 8)
    {
      serial for (j, 0, 4)
      {
        ScheduleBlock(test_repeat)
        {
          i0, i1 = axis.bind(i, j)
          test_repeat[i0, i1] = in[select((((i0 > 0) and (2 > 0)) or ((i0 < 0) and (2 < 0))), (i0 / 2), select(((i0 % 2) == 0), (i0 / 2), ((i0 / 2) - 1))), i1]
        }
      }
    }
  }
}
  )ROC";

  ASSERT_EQ(utils::GetStreamCnt(funcs[0]), utils::Trim(target_source_ir));

  ir::Module::Builder builder("Repeat_Module", target);
  for (auto &f : funcs) {
    builder.AddFunction(f);
  }

  backends::CodeGenCX86 codegen(target, backends::CodeGenCX86::Feature::AVX512);
  codegen.SetInlineBuiltinCodes(false);
  std::string code =
      codegen.Compile(builder.Build(), backends::CodeGenC::OutputKind::CImpl);
  VLOG(6) << "Cpu Codegen result:";
  VLOG(6) << code << std::endl;

  auto target_source = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

void TestGenerateCodeCpu_Repeat(void* _args, int32_t num_args)
{
  cinn_buffer_t* _test_repeat = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  cinn_buffer_t* _in = cinn_buffer_t::new_((cinn_device_kind_t)(0)/*target*/, cinn_int32_t(), { 4, 4 });
  cinn_buffer_malloc((void*)(0), _test_repeat);
  cinn_buffer_malloc((void*)(0), _in);
  const int32_t* in = ((const int32_t*)(_in->memory));
  int32_t* test_repeat = ((int32_t*)(_test_repeat->memory));
  for (int32_t i = 0; i < 8; i += 1) {
    for (int32_t j = 0; j < 4; j += 1) {
      test_repeat[((4 * i) + j)] = in[((4 * (((((i > 0) && (2 > 0)) || ((i < 0) && (2 < 0)))) ? (i / 2) : ((((i & 1) == 0)) ? (i / 2) : ((i / 2) + -1)))) + j)];
    };
  };
  cinn_buffer_free((void*)(0), _in);
  cinn_buffer_free((void*)(0), _test_repeat);
}
  )ROC";

  ASSERT_EQ(utils::Trim(code), utils::Trim(target_source));
}

}  // namespace op
}  // namespace hlir
}  // namespace cinn

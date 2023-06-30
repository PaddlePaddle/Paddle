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

#include "paddle/cinn/hlir/op/contrib/one_hot.h"

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

TEST(GenerateCode_Cpu, OneHot) {
  common::Context::Global().ResetNameId();

  common::Target target = common::DefaultHostTarget();

  Expr m(4);
  Expr n(4);
  const int depth = 3;
  const int axis = 1;
  const std::string dtype = "float32";

  lang::Placeholder<int32_t> in("in", {m, n});
  lang::Placeholder<int32_t> on_value("on_value", {Expr(1)});
  lang::Placeholder<int32_t> off_value("off_value", {Expr(1)});

  ir::Tensor res = OneHot(in,
                          on_value,
                          off_value,
                          depth,
                          axis,
                          common::Str2Type(dtype),
                          "test_one_hot");

  poly::StageMap stages = poly::CreateStages({res});
  std::vector<ir::LoweredFunc> funcs =
      lang::LowerVec("TestGenerateCodeCpu_OneHot",
                     stages,
                     {res},
                     {},
                     {},
                     nullptr,
                     target,
                     true);

  VLOG(6) << "Expr before CPU codegen:";
  VLOG(6) << funcs[0]->body;

  ir::Module::Builder builder("OneHot_Module", target);
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

void TestGenerateCodeCpu_OneHot(void* _args, int32_t num_args)
{
  cinn_buffer_t* _test_one_hot = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  cinn_buffer_t* _in = cinn_buffer_t::new_((cinn_device_kind_t)(0)/*target*/, cinn_int32_t(), { 4, 4 });
  cinn_buffer_t* _off_value = cinn_buffer_t::new_((cinn_device_kind_t)(0)/*target*/, cinn_int32_t(), {  });
  cinn_buffer_t* _on_value = cinn_buffer_t::new_((cinn_device_kind_t)(0)/*target*/, cinn_int32_t(), {  });
  cinn_buffer_malloc((void*)(0), _test_one_hot);
  cinn_buffer_malloc((void*)(0), _in);
  cinn_buffer_malloc((void*)(0), _off_value);
  cinn_buffer_malloc((void*)(0), _on_value);
  const int32_t* in = ((const int32_t*)(_in->memory));
  const int32_t* off_value = ((const int32_t*)(_off_value->memory));
  const int32_t* on_value = ((const int32_t*)(_on_value->memory));
  float* test_one_hot = ((float*)(_test_one_hot->memory));
  for (int32_t i = 0; i < 4; i += 1) {
    for (int32_t j = 0; j < 3; j += 1) {
      for (int32_t k = 0; k < 4; k += 1) {
        test_one_hot[((12 * i) + ((4 * j) + k))] = (((in[((4 * i) + k)] == j)) ? ((float)(on_value[0])) : ((float)(off_value[0])));
      };
    };
  };
  cinn_buffer_free((void*)(0), _in);
  cinn_buffer_free((void*)(0), _off_value);
  cinn_buffer_free((void*)(0), _on_value);
  cinn_buffer_free((void*)(0), _test_one_hot);
}
  )ROC";

  ASSERT_EQ(utils::Trim(code), utils::Trim(target_source));
}

}  // namespace op
}  // namespace hlir
}  // namespace cinn

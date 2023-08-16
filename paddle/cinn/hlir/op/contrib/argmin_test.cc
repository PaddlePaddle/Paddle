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

#include "paddle/cinn/hlir/op/contrib/argmin.h"

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

TEST(GenerateCode_Cpu, Argmin_Keep) {
  common::Context::Global().ResetNameId();

  common::Target target = common::DefaultHostTarget();

  int axis = 1;
  ir::Expr n(4);
  ir::Expr in_c(3);
  ir::Expr out_c(1);
  ir::Expr h(28);
  ir::Expr w(28);

  lang::Placeholder<float> in("in", {n, in_c, h, w});
  poly::StageMap stages = poly::CreateStages({in});
  ir::Tensor res =
      Argmin(in, target, stages, axis, true, "test_argmin_in").at(0);
  stages->InsertLazily(res);

  std::vector<ir::LoweredFunc> funcs =
      lang::LowerVec("TestGenerateCodeCpu_Argmin_Keep",
                     stages,
                     {in, res},
                     {},
                     {},
                     nullptr,
                     target,
                     true);

  VLOG(6) << "Expr before CPU codegen:";
  VLOG(6) << funcs[0]->body;

  ir::Module::Builder builder("Argmin_Keep_Module", target);
  for (auto& f : funcs) {
    builder.AddFunction(f);
  }

  backends::CodeGenCX86 codegen(target, backends::CodeGenCX86::Feature::AVX512);
  codegen.SetInlineBuiltinCodes(false);
  std::string code =
      codegen.Compile(builder.Build(), backends::CodeGenC::OutputKind::CImpl);
  auto target_source = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

void TestGenerateCodeCpu_Argmin_Keep(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _in = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  cinn_buffer_t* _test_argmin_in = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_t* _test_argmin_in_index = cinn_buffer_t::new_((cinn_device_kind_t)(0)/*target*/, cinn_int32_t(), { 4, 3, 28, 28 });
  cinn_buffer_t* _test_argmin_in_index_temp = cinn_buffer_t::new_((cinn_device_kind_t)(0)/*target*/, cinn_int32_t(), { 4, 3, 28, 28 });
  cinn_buffer_malloc((void*)(0), _test_argmin_in);
  cinn_buffer_malloc((void*)(0), _test_argmin_in_index);
  cinn_buffer_malloc((void*)(0), _test_argmin_in_index_temp);
  const float* in = ((const float*)(_in->memory));
  int32_t* test_argmin_in = ((int32_t*)(_test_argmin_in->memory));
  int32_t* test_argmin_in_index = ((int32_t*)(_test_argmin_in_index->memory));
  int32_t* test_argmin_in_index_temp = ((int32_t*)(_test_argmin_in_index_temp->memory));
  for (int32_t i = 0; i < 4; i += 1) {
    for (int32_t j = 0; j < 3; j += 1) {
      for (int32_t k = 0; k < 28; k += 1) {
        for (int32_t a = 0; a < 28; a += 1) {
          test_argmin_in_index_temp[((2352 * i) + ((784 * j) + ((28 * k) + a)))] = cinn_host_lt_num_fp32(_in, 3, in[((2352 * i) + ((784 * j) + ((28 * k) + a)))], ((2352 * i) + ((28 * k) + a)), 784);
        };
      };
    };
  };
  for (int32_t i = 0; i < 4; i += 1) {
    for (int32_t j = 0; j < 3; j += 1) {
      for (int32_t k = 0; k < 28; k += 1) {
        for (int32_t a = 0; a < 28; a += 1) {
          test_argmin_in_index[((2352 * i) + ((784 * j) + ((28 * k) + a)))] = cinn_host_next_smallest_int32(_test_argmin_in_index_temp, 3, j, ((2352 * i) + ((28 * k) + a)), 784);
        };
      };
    };
  };
  for (int32_t i = 0; i < 4; i += 1) {
    for (int32_t k = 0; k < 28; k += 1) {
      for (int32_t a = 0; a < 28; a += 1) {
        test_argmin_in[((784 * i) + ((28 * k) + a))] = test_argmin_in_index[((2352 * i) + ((28 * k) + a))];
      };
    };
  };
  cinn_buffer_free((void*)(0), _test_argmin_in_index);
  cinn_buffer_free((void*)(0), _test_argmin_in_index_temp);
  cinn_buffer_free((void*)(0), _test_argmin_in);
}
  )ROC";
  CHECK_EQ(utils::Trim(code), utils::Trim(target_source));
}

}  // namespace op
}  // namespace hlir
}  // namespace cinn

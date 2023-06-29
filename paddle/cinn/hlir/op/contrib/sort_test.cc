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

#include "paddle/cinn/hlir/op/contrib/sort.h"

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

TEST(GenerateCode_Cpu, ArgSort) {
  common::Context::Global().ResetNameId();

  Target target = common::DefaultHostTarget();

  ir::Expr n(4);
  ir::Expr h(28);

  lang::Placeholder<int32_t> in("in", {n, h});
  poly::StageMap stages = poly::CreateStages({in});
  ir::Tensor res =
      ArgSort(in.tensor(), target, stages, 1, true, "test_arg_sort_out").at(0);
  stages->InsertLazily(res);
  std::vector<ir::LoweredFunc> funcs =
      lang::LowerVec("TestGenerateCodeCpu_ArgSort",
                     stages,
                     {in, res},
                     {},
                     {},
                     nullptr,
                     target,
                     true);

  VLOG(6) << "Expr before CPU codegen:";
  VLOG(6) << funcs[0]->body;

  ir::Module::Builder builder("ArgSort_Module", target);
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

TEST(GenerateCode_Cpu, Sort) {
  common::Context::Global().ResetNameId();

  Target target = common::DefaultHostTarget();

  ir::Expr n(4);
  ir::Expr h(28);

  lang::Placeholder<int32_t> in("in", {n, h});
  auto stages = poly::CreateStages({in});
  ir::Tensor out = Sort(in, target, stages, 1, true, "test_sort_out").at(0);
  stages->InsertLazily(out);
  std::vector<ir::LoweredFunc> funcs =
      lang::LowerVec("TestGenerateCodeCpu_Sort",
                     stages,
                     {in, out},
                     {},
                     {},
                     nullptr,
                     target,
                     true);

  VLOG(6) << "Expr before CPU codegen:";
  VLOG(6) << funcs[0]->body;

  ir::Module::Builder builder("Sort_Module", target);
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

void TestGenerateCodeCpu_Sort(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _in = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  cinn_buffer_t* _test_sort_out = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_t* _test_sort_out_index = cinn_buffer_t::new_((cinn_device_kind_t)(0)/*target*/, cinn_int32_t(), { 4, 28 });
  cinn_buffer_t* _test_sort_out_index_temp = cinn_buffer_t::new_((cinn_device_kind_t)(0)/*target*/, cinn_int32_t(), { 4, 28 });
  cinn_buffer_malloc((void*)(0), _test_sort_out);
  cinn_buffer_malloc((void*)(0), _test_sort_out_index);
  cinn_buffer_malloc((void*)(0), _test_sort_out_index_temp);
  const int32_t* in = ((const int32_t*)(_in->memory));
  int32_t* test_sort_out = ((int32_t*)(_test_sort_out->memory));
  int32_t* test_sort_out_index = ((int32_t*)(_test_sort_out_index->memory));
  int32_t* test_sort_out_index_temp = ((int32_t*)(_test_sort_out_index_temp->memory));
  for (int32_t i = 0; i < 4; i += 1) {
    for (int32_t j = 0; j < 28; j += 1) {
      test_sort_out_index_temp[((28 * i) + j)] = cinn_host_lt_num_int32(_in, 28, in[((28 * i) + j)], (28 * i), 1);
    };
  };
  for (int32_t i = 0; i < 4; i += 1) {
    for (int32_t j = 0; j < 28; j += 1) {
      test_sort_out_index[((28 * i) + j)] = cinn_host_next_smallest_int32(_test_sort_out_index_temp, 28, j, (28 * i), 1);
    };
  };
  for (int32_t i = 0; i < 4; i += 1) {
    for (int32_t j = 0; j < 28; j += 1) {
      test_sort_out[((28 * i) + j)] = in[((28 * i) + test_sort_out_index[((28 * i) + j)])];
    };
  };
  cinn_buffer_free((void*)(0), _test_sort_out_index);
  cinn_buffer_free((void*)(0), _test_sort_out_index_temp);
  cinn_buffer_free((void*)(0), _test_sort_out);
}
  )ROC";
  CHECK_EQ(utils::Trim(code), utils::Trim(target_source));
}

}  // namespace op
}  // namespace hlir
}  // namespace cinn

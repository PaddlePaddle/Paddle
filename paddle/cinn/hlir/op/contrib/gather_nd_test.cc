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

#include "paddle/cinn/hlir/op/contrib/gather_nd.h"

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

TEST(GenerateCode_Cpu, GatherNd) {
  common::Context::Global().ResetNameId();

  common::Target target = common::DefaultHostTarget();

  ir::Expr dim0(1);
  ir::Expr dim1(2);
  ir::Expr dim2(3);
  ir::Expr dim3(4);

  lang::Placeholder<float> x("x", {dim1, dim2, dim3});
  lang::Placeholder<int32_t> index("index", {dim0, dim1, dim2});
  ir::Tensor res = GatherNd(x, index, "test_gather_nd_out");

  poly::StageMap stages = poly::CreateStages({res});
  std::vector<ir::LoweredFunc> funcs =
      lang::LowerVec("TestGenerateCodeCpu_GatherNd",
                     stages,
                     {res},
                     {},
                     {},
                     nullptr,
                     target,
                     true);

  VLOG(6) << "Expr before CPU codegen:";
  VLOG(6) << funcs[0]->body;

  ir::Module::Builder builder("GatherNd_Module", target);
  for (auto& f : funcs) {
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

void TestGenerateCodeCpu_GatherNd(void* _args, int32_t num_args)
{
  cinn_buffer_t* _test_gather_nd_out = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  cinn_buffer_t* _index = cinn_buffer_t::new_((cinn_device_kind_t)(0)/*target*/, cinn_int32_t(), { 1, 2, 3 });
  cinn_buffer_t* _x = cinn_buffer_t::new_((cinn_device_kind_t)(0)/*target*/, cinn_float32_t(), { 2, 3, 4 });
  cinn_buffer_malloc((void*)(0), _test_gather_nd_out);
  cinn_buffer_malloc((void*)(0), _index);
  cinn_buffer_malloc((void*)(0), _x);
  const int32_t* index = ((const int32_t*)(_index->memory));
  float* test_gather_nd_out = ((float*)(_test_gather_nd_out->memory));
  const float* x = ((const float*)(_x->memory));
  for (int32_t j = 0; j < 2; j += 1) {
    test_gather_nd_out[j] = x[((12 * index[(3 * j)]) + ((4 * index[(1 + (3 * j))]) + index[(2 + (3 * j))]))];
  };
  cinn_buffer_free((void*)(0), _index);
  cinn_buffer_free((void*)(0), _x);
  cinn_buffer_free((void*)(0), _test_gather_nd_out);
}
  )ROC";
  CHECK_EQ(utils::Trim(code), utils::Trim(target_source));
}

}  // namespace op
}  // namespace hlir
}  // namespace cinn

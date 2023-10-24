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

#include <gtest/gtest.h>

#include "paddle/cinn/backends/compiler.h"
#include "paddle/cinn/backends/extern_func_jit_register.h"
#include "paddle/cinn/backends/llvm/execution_engine.h"
#include "paddle/cinn/backends/llvm/simple_jit.h"
#include "paddle/cinn/cinn.h"
#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/common/target.h"
#include "paddle/cinn/common/test_helper.h"
#include "paddle/cinn/runtime/cpu/host_intrinsics.h"
#include "paddle/cinn/runtime/cpu/use_extern_funcs.h"

namespace cinn {
namespace runtime {
namespace cpu {

cinn_buffer_t *CreateBuffer(const std::vector<int> shape,
                            bool random = true,
                            int set_value = 0) {
  if (random) {
    return common::BufferBuilder(Float(32), shape).set_random().Build();
  } else if (set_value != 0) {
    return common::BufferBuilder(Float(32), shape).set_val(set_value).Build();
  }
  return common::BufferBuilder(Float(32), shape).set_zero().Build();
}

TEST(cinn_cpu_mkldnn_conv2d_nchw_fp32, test) {
  int n(1);
  int c_in(3);
  int i_h(224);
  int i_w(224);
  int c_out(64);
  int k_h(7);
  int k_w(7);
  int pad_h(1);
  int pad_w(1);
  int stride_h(2);
  int stride_w(2);
  int dilation_h(1);
  int dilation_w(1);

  Placeholder<float> input("input",
                           {Expr(n), Expr(c_in), Expr(i_h), Expr(i_w)});
  Placeholder<float> weights("weights",
                             {Expr(c_out), Expr(c_in), Expr(k_h), Expr(k_w)});

  auto call = Compute(
      {Expr(1)},
      [=]() -> Expr {
        return lang::CallExtern("cinn_cpu_mkldnn_conv2d_nchw_fp32",
                                {
                                    Expr(n),           // batch_size
                                    Expr(c_in),        // c_in
                                    Expr(i_h),         // input_h
                                    Expr(i_w),         // input_w
                                    Expr(c_out),       // c_out
                                    Expr(1),           // group
                                    Expr(k_h),         // filter_h
                                    Expr(k_w),         // filter_w
                                    Expr(pad_h),       // pad_h
                                    Expr(pad_w),       // pad_w
                                    Expr(stride_h),    // stride_h
                                    Expr(stride_w),    // stride_w
                                    Expr(dilation_h),  // dilation_h
                                    Expr(dilation_w),  // dilation_w
                                    input.tensor(),    // input
                                    weights.tensor()   // weights
                                });
      },
      "cinn_cpu_mkldnn_conv2d_nchw_fp32");

  auto out = call->TupleGet(0);
  out->WithBuffer(Float(32));

  auto stages = CreateStages({call, out});

  auto target = common::DefaultHostTarget();
  target.arch = Target::Arch::X86;
  ir::Module::Builder builder("module0", target);

  auto func = Lower("fn", stages, {input, weights, out, call});
  builder.AddFunction(func);

  LOG(INFO) << "func:\n" << func;

  auto jit = backends::SimpleJIT::Create();
  auto module = builder.Build();

  jit->Link(module, /*optimize=*/true);
  auto fn = jit->Lookup("fn");
  auto fn_ptr = reinterpret_cast<void (*)(void *, int32_t)>(fn);

  // test with real data
  int o_h = (i_h - ((k_h - 1) * dilation_h + 1) + pad_h * 2) / stride_h + 1;
  int o_w = (i_w - ((k_w - 1) * dilation_w + 1) + pad_w * 2) / stride_w + 1;
  auto *A_buf = common::BufferBuilder(Float(32), {n, c_in, i_h, i_w})
                    .set_random()
                    .Build();
  auto *B_buf = common::BufferBuilder(Float(32), {c_out, c_in, k_h, k_w})
                    .set_random()
                    .Build();
  auto *C_buf =
      common::BufferBuilder(Float(32), {n, c_out, o_h, o_w}).set_zero().Build();

  auto args = common::ArgsBuilder().Add(A_buf).Add(B_buf).Add(C_buf).Build();

  fn_ptr(args.data(), args.size());

  cinn_buffer_free(nullptr, A_buf);
  cinn_buffer_free(nullptr, B_buf);
  cinn_buffer_free(nullptr, C_buf);
}

}  // namespace cpu
}  // namespace runtime
}  // namespace cinn

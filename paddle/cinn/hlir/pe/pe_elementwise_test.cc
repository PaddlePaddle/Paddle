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

// #include <cmath>

#include "paddle/cinn/backends/llvm/codegen_llvm.h"
#include "paddle/cinn/backends/llvm/codegen_x86.h"
#include "paddle/cinn/backends/llvm/execution_engine.h"
#include "paddle/cinn/cinn.h"
#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/common/target.h"
#include "paddle/cinn/common/test_helper.h"
#include "paddle/cinn/hlir/pe/elementwise.h"
#include "paddle/cinn/hlir/pe/reduction.h"
#include "paddle/cinn/runtime/cpu/host_intrinsics.h"
#include "paddle/cinn/runtime/cpu/use_extern_funcs.h"
#include "paddle/cinn/utils/timer.h"

namespace cinn {
namespace hlir {
namespace pe {

template <typename FuncOp, typename FuncRuntime>
void TestElementwisePE(const std::string &fn_name,
                       const FuncOp &func_op,
                       const FuncRuntime &fn_runtime,
                       Type type = Float(32),
                       int set_value = 0,
                       bool test_benchmark = true) {
  Expr M(1024), N(2048);

  Placeholder<float> A("A", {M, N});

  auto A_out = func_op(A.tensor(), fn_name + "_out");
  std::vector<ir::Tensor> tensor_args{A};
  tensor_args.insert(tensor_args.end(), A_out.begin(), A_out.end());
  auto stages = CreateStages(tensor_args);

  stages[A_out[0]]->Parallel(0);

  Target target = common::DefaultHostTarget();
  Module::Builder builder("module0", target);
  for (auto &tensor : A_out) {
    stages->InsertLazily(tensor);
  }
  auto func = Lower("fn", stages, tensor_args);
  LOG(INFO) << "func:\n" << func;
  builder.AddFunction(func);

  auto jit = backends::ExecutionEngine::Create({});
  auto module = builder.Build();

  jit->Link<backends::CodeGenX86>(module);
  auto fn = jit->Lookup("fn");
  CHECK(fn);
  auto fn_ = reinterpret_cast<void (*)(void *, int32_t)>(fn);

  cinn_buffer_t *A_buf;
  if (set_value != 0) {
    A_buf = common::BufferBuilder(Float(32), {M.as_int32(), N.as_int32()})
                .set_val(set_value)
                .Build();
  } else {
    A_buf = common::BufferBuilder(Float(32), {M.as_int32(), N.as_int32()})
                .set_random()
                .Build();
  }
  auto *B_buf = common::BufferBuilder(type, {M.as_int32(), N.as_int32()})
                    .set_align(type.bits())
                    .Build();

  cinn_pod_value_t a_arg(A_buf), b_arg(B_buf);
  cinn_pod_value_t args[] = {a_arg, b_arg};

  if (test_benchmark) {
    cinn::utils::Timer timer;
    timer.Start();
    fn_(args, 2);
    double test_op_time = timer.Stop();
    LOG(INFO) << "kernel warmup run time: " << test_op_time << " ms";
    timer.Start();
    int repeat_ = 10;
    for (int i = 0; i < repeat_; i++) {
      fn_(args, 2);
    }
    test_op_time = timer.Stop() / repeat_;
    LOG(INFO) << "repeat times: " << repeat_
              << ", kernel run time: " << test_op_time << " ms";
  } else {
    fn_(args, 2);
  }

  auto *ad = reinterpret_cast<float *>(A_buf->memory);
  if (type.is_bool()) {
    auto *bd = reinterpret_cast<int8_t *>(B_buf->memory);
    for (int i = 0; i < A_buf->num_elements(); i++) {
      ASSERT_NEAR(bd[i], fn_runtime(ad[i]), 1e-5);
    }
  } else {
    auto *bd = reinterpret_cast<float *>(B_buf->memory);
    for (int i = 0; i < A_buf->num_elements(); i++) {
      ASSERT_NEAR(bd[i], fn_runtime(ad[i]), 1e-5);
    }
  }
}

bool isnan(float e) { return std::isnan(e); }
bool isfinite(float e) { return std::isfinite(e); }
bool isinf(float e) { return std::isinf(e); }
float rsqrt(float e) { return 1.0f / sqrtf(e); }

#define TEST_ELEMENTWISE_PE_FP32(test_name__, PE__)                 \
  TEST(elementwise_pe, test_name__) {                               \
    cinn::hlir::pe::TestElementwisePE(                              \
        "PE_Elementwise_" #test_name__ "_fp32", PE__, test_name__); \
  }
#define TEST_ELEMENTWISE_PE_FP32_BOOL(test_name__, PE__)                    \
  TEST(elementwise_pe, test_name__) {                                       \
    cinn::hlir::pe::TestElementwisePE(                                      \
        "PE_Elementwise_" #test_name__ "_fp32", PE__, test_name__, Bool()); \
  }
#define TEST_ELEMENTWISE_PE_FP32_SET(test_name__, PE__, value__)              \
  TEST(elementwise_pe, test_name__) {                                         \
    cinn::hlir::pe::TestElementwisePE("PE_Elementwise_" #test_name__ "_fp32", \
                                      PE__,                                   \
                                      test_name__,                            \
                                      Float(32),                              \
                                      value__);                               \
  }

TEST_ELEMENTWISE_PE_FP32(expf, Exp)
TEST_ELEMENTWISE_PE_FP32(erff, Erf)
TEST_ELEMENTWISE_PE_FP32(sqrtf, Sqrt)
TEST_ELEMENTWISE_PE_FP32(logf, Log)
TEST_ELEMENTWISE_PE_FP32(log2f, Log2)
TEST_ELEMENTWISE_PE_FP32(log10f, Log10)
TEST_ELEMENTWISE_PE_FP32(floorf, Floor)
TEST_ELEMENTWISE_PE_FP32(ceilf, Ceil)
TEST_ELEMENTWISE_PE_FP32(roundf, Round)
TEST_ELEMENTWISE_PE_FP32(truncf, Trunc)
TEST_ELEMENTWISE_PE_FP32(cosf, Cos)
TEST_ELEMENTWISE_PE_FP32(coshf, Cosh)
TEST_ELEMENTWISE_PE_FP32(tanf, Tan)
TEST_ELEMENTWISE_PE_FP32(sinf, Sin)
TEST_ELEMENTWISE_PE_FP32(sinhf, Sinh)
TEST_ELEMENTWISE_PE_FP32(acosf, Acos)
TEST_ELEMENTWISE_PE_FP32_SET(acoshf, Acosh, 1.5)
TEST_ELEMENTWISE_PE_FP32(asinf, Asin)
TEST_ELEMENTWISE_PE_FP32(asinhf, Asinh)
TEST_ELEMENTWISE_PE_FP32(atanf, Atan)
TEST_ELEMENTWISE_PE_FP32(atanhf, Atanh)
TEST_ELEMENTWISE_PE_FP32(tanhf, Tanh)
TEST_ELEMENTWISE_PE_FP32_BOOL(isnan, IsNan)
TEST_ELEMENTWISE_PE_FP32_BOOL(isfinite, IsFinite)
TEST_ELEMENTWISE_PE_FP32_BOOL(isinf, IsInf)
TEST_ELEMENTWISE_PE_FP32(rsqrt, Rsqrt)

}  // namespace pe
}  // namespace hlir
}  // namespace cinn

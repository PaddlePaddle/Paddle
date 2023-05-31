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
#include <math.h>

#include "paddle/cinn/backends/llvm/execution_engine.h"
#include "paddle/cinn/cinn.h"
#include "paddle/cinn/common/target.h"
#include "paddle/cinn/common/test_helper.h"
#include "paddle/cinn/hlir/pe/broadcast.h"
#include "paddle/cinn/runtime/cpu/host_intrinsics.h"

namespace cinn {
namespace hlir {
namespace pe {
using ir::Tensor;

void TestBroadcastPE(const std::string &fn_name,
                     Tensor (*func_op)(const Tensor &A,
                                       const Tensor &B,
                                       const std::string &output_name,
                                       const Expr &axis),
                     float (*fn_runtime)(float, float),
                     int set_value = 0) {
  Expr M(100), N(32);

  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  auto C = func_op(A.tensor(), B.tensor(), "C", Expr());

  auto stages = CreateStages({C});

  Target target = common::DefaultHostTarget();
  Module::Builder builder("module0", target);
  auto func = Lower("fn", stages, {A, B, C});
  builder.AddFunction(func);
  LOG(INFO) << "func:\n" << func;

  auto jit = backends::ExecutionEngine::Create({});
  auto module = builder.Build();

  jit->Link(module);
  auto fn = jit->Lookup("fn");
  CHECK(fn);
  auto fn_ = reinterpret_cast<void (*)(void *, int32_t)>(fn);

  cinn_buffer_t *A_buf;
  cinn_buffer_t *B_buf;
  if (set_value != 0) {
    A_buf = common::BufferBuilder(Float(32), {M.as_int32(), N.as_int32()})
                .set_val(set_value)
                .Build();
    B_buf = common::BufferBuilder(Float(32), {M.as_int32(), N.as_int32()})
                .set_val(set_value)
                .Build();
  } else {
    A_buf = common::BufferBuilder(Float(32), {M.as_int32(), N.as_int32()})
                .set_random()
                .Build();
    B_buf = common::BufferBuilder(Float(32), {M.as_int32(), N.as_int32()})
                .set_random()
                .Build();
  }
  auto *C_buf = common::BufferBuilder(Float(32), {M.as_int32(), N.as_int32()})
                    .set_zero()
                    .Build();

  cinn_pod_value_t a_arg(A_buf), b_arg(B_buf), c_arg(C_buf);
  cinn_pod_value_t args[] = {a_arg, b_arg, c_arg};
  fn_(args, 3);

  auto *ad = reinterpret_cast<float *>(A_buf->memory);
  auto *bd = reinterpret_cast<float *>(B_buf->memory);
  auto *cd = reinterpret_cast<float *>(C_buf->memory);
  for (int i = 0; i < A_buf->num_elements(); i++) {
    ASSERT_NEAR(cd[i], fn_runtime(ad[i], bd[i]), 1e-5);
  }
}

void TestBroadcastPE1(const std::string &fn_name,
                      Tensor (*func_op)(const Tensor &A,
                                        const Tensor &B,
                                        const std::string &output_name,
                                        const Expr &axis),
                      float (*fn_runtime)(float, float),
                      int set_value = 0) {
  Expr M(100), N(32), K(10);
  Placeholder<float> A("A", {M, N, K});
  Placeholder<float> B("B", {N});
  auto C = func_op(A.tensor(), B.tensor(), "C", Expr(1));
  auto stages = CreateStages({C});
  Target target = common::DefaultHostTarget();
  Module::Builder builder("module0", target);
  auto func = Lower("fn", stages, {A, B, C});
  builder.AddFunction(func);
  LOG(INFO) << "func:\n" << func;
  auto jit = backends::ExecutionEngine::Create({});
  auto module = builder.Build();
  jit->Link(module);
  auto fn = jit->Lookup("fn");
  CHECK(fn);
  auto fn_ = reinterpret_cast<void (*)(void *, int32_t)>(fn);
  cinn_buffer_t *A_buf;
  cinn_buffer_t *B_buf;
  if (set_value != 0) {
    A_buf = common::BufferBuilder(Float(32),
                                  {M.as_int32(), N.as_int32(), K.as_int32()})
                .set_val(set_value)
                .Build();
    B_buf = common::BufferBuilder(Float(32), {N.as_int32()})
                .set_val(set_value)
                .Build();
  } else {
    A_buf = common::BufferBuilder(Float(32),
                                  {M.as_int32(), N.as_int32(), K.as_int32()})
                .set_random()
                .Build();
    B_buf =
        common::BufferBuilder(Float(32), {N.as_int32()}).set_random().Build();
  }
  auto *C_buf = common::BufferBuilder(
                    Float(32), {M.as_int32(), N.as_int32(), K.as_int32()})
                    .set_zero()
                    .Build();
  cinn_pod_value_t a_arg(A_buf), b_arg(B_buf), c_arg(C_buf);
  cinn_pod_value_t args[] = {a_arg, b_arg, c_arg};
  fn_(args, 3);
  auto *ad = reinterpret_cast<float *>(A_buf->memory);
  auto *bd = reinterpret_cast<float *>(B_buf->memory);
  auto *cd = reinterpret_cast<float *>(C_buf->memory);
  for (size_t i = 0; i < 100; i++) {
    for (size_t j = 0; j < 32; j++) {
      for (size_t k = 0; k < 10; k++) {
        int index = 32 * 10 * i + 10 * j + k;
        ASSERT_NEAR(cd[index], fn_runtime(ad[index], bd[j]), 1e-5);
      }
    }
  }
}

void TestBroadcastPE2(const std::string &fn_name,
                      Tensor (*func_op)(const Tensor &A,
                                        const Tensor &B,
                                        const std::string &output_name,
                                        const Expr &axis),
                      float (*fn_runtime)(float, float),
                      int set_value = 0) {
  Expr M(100), N(32), K(10), R(1);
  Placeholder<float> A("A", {M, N, K, R});
  Placeholder<float> B("B", {N, K});
  auto C = func_op(A.tensor(), B.tensor(), "C", Expr(1));
  auto stages = CreateStages({C});
  Target target = common::DefaultHostTarget();
  Module::Builder builder("module0", target);
  auto func = Lower("fn", stages, {A, B, C});
  builder.AddFunction(func);
  LOG(INFO) << "func:\n" << func;
  auto jit = backends::ExecutionEngine::Create({});
  auto module = builder.Build();
  jit->Link(module);
  auto fn = jit->Lookup("fn");
  CHECK(fn);
  auto fn_ = reinterpret_cast<void (*)(void *, int32_t)>(fn);
  cinn_buffer_t *A_buf;
  cinn_buffer_t *B_buf;
  if (set_value != 0) {
    A_buf =
        common::BufferBuilder(
            Float(32), {M.as_int32(), N.as_int32(), K.as_int32(), R.as_int32()})
            .set_val(set_value)
            .Build();
    B_buf = common::BufferBuilder(Float(32), {N.as_int32(), K.as_int32()})
                .set_val(set_value)
                .Build();
  } else {
    A_buf =
        common::BufferBuilder(
            Float(32), {M.as_int32(), N.as_int32(), K.as_int32(), R.as_int32()})
            .set_random()
            .Build();
    B_buf = common::BufferBuilder(Float(32), {N.as_int32(), K.as_int32()})
                .set_random()
                .Build();
  }
  auto *C_buf =
      common::BufferBuilder(
          Float(32), {M.as_int32(), N.as_int32(), K.as_int32(), R.as_int32()})
          .set_zero()
          .Build();
  cinn_pod_value_t a_arg(A_buf), b_arg(B_buf), c_arg(C_buf);
  cinn_pod_value_t args[] = {a_arg, b_arg, c_arg};
  fn_(args, 3);
  auto *ad = reinterpret_cast<float *>(A_buf->memory);
  auto *bd = reinterpret_cast<float *>(B_buf->memory);
  auto *cd = reinterpret_cast<float *>(C_buf->memory);
  for (size_t i = 0; i < 100; i++) {
    for (size_t j = 0; j < 32; j++) {
      for (size_t k = 0; k < 10; k++) {
        for (size_t r = 0; r < 1; r++) {
          int index = 32 * 10 * i + 10 * j + k + r;
          ASSERT_NEAR(cd[index], fn_runtime(ad[index], bd[10 * j + k]), 1e-5);
        }
      }
    }
  }
}

#define RULE(test_name__, rule__) \
  float test_name__(float a, float b) { rule__ }

#define TEST_BROADCAST_PE_FP32_BASIC(test_name__)                        \
  TEST(broadcast_pe, test_name__) {                                      \
    TestBroadcastPE(                                                     \
        "PE_Broadcast_" #test_name__ "_fp32", test_name__, test_name__); \
  }

#define TEST_BROADCAST_PE_FP32_SET_BASIC(test_name__)                          \
  TEST(broadcast_pe, test_name__) {                                            \
    TestBroadcastPE("PE_Broadcast_" #test_name__ "_fp32", test_name__, value); \
  }

#define TEST_BROADCAST_PE_FP32(test_name__, rule__) \
  RULE(test_name__, rule__)                         \
  TEST_BROADCAST_PE_FP32_BASIC(test_name__)

TEST_BROADCAST_PE_FP32(Add, return a + b;)
TEST_BROADCAST_PE_FP32(Multiply, return a * b;)

#define PI 3.1415926535
float Atan2(float a, float b) {
  if (b == 0.0) {
    if (a > 0) {
      return PI / 2;
    } else {
      return -PI / 2;
    }
  } else {
    auto at = atan(a / b);
    if (b > 0) {
      return at;
    } else if (a >= 0) {
      return at + PI;
    } else {
      return at - PI;
    }
  }
}
TEST_BROADCAST_PE_FP32_BASIC(Atan2);

}  // namespace pe
}  // namespace hlir
}  // namespace cinn

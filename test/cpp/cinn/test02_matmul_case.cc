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

#include <glog/logging.h>
#include <gtest/gtest.h>

#include "paddle/cinn/runtime/cinn_runtime.h"
#include "paddle/cinn/utils/timer.h"
#include "test/cpp/cinn/test02_helper.h"
#include "test/cpp/cinn/test02_matmul.h"
#include "test/cpp/cinn/test02_matmul_array_packing.h"
#include "test/cpp/cinn/test02_matmul_array_packing_dynamic_shape.h"
#include "test/cpp/cinn/test02_matmul_block.h"
#include "test/cpp/cinn/test02_matmul_call.h"
#include "test/cpp/cinn/test02_matmul_loop_permutation.h"
#include "test/cpp/cinn/test02_matmul_split.h"
#include "test/cpp/cinn/test02_matmul_tile.h"
#include "test/cpp/cinn/test02_matmul_varient_shape.h"
#include "test/cpp/cinn/test02_matmul_varient_shape_tile.h"
#include "test/cpp/cinn/test02_matmul_vectorize.h"

TEST(test02, basic) {
  const int M = 1024;
  const int N = 1024;
  const int K = 1024;
  const int bn = 32;

  auto* A = cinn_buffer_t::new_(
      cinn_device_kind_t::cinn_x86_device, cinn_float32_t(), {M, K}, 32);
  auto* B = cinn_buffer_t::new_(
      cinn_device_kind_t::cinn_x86_device, cinn_float32_t(), {K, N}, 32);
  auto* C = cinn_buffer_t::new_(
      cinn_device_kind_t::cinn_x86_device, cinn_float32_t(), {M, N}, 32);
  auto* C_target = cinn_buffer_t::new_(
      cinn_device_kind_t::cinn_x86_device, cinn_float32_t(), {M, N});
  auto* packedB = cinn_buffer_t::new_(cinn_device_kind_t::cinn_x86_device,
                                      cinn_float32_t(),
                                      {N / bn, K, bn},
                                      32);
  cinn_buffer_malloc(nullptr, A);
  cinn_buffer_malloc(nullptr, B);
  cinn_buffer_malloc(nullptr, C_target);
  cinn_buffer_malloc(nullptr, C);
  cinn_buffer_malloc(nullptr, packedB);

  float* Ad = reinterpret_cast<float*>(A->memory);
  float* Bd = reinterpret_cast<float*>(B->memory);
  float* Cd_target = reinterpret_cast<float*>(C_target->memory);
  float* Cd = reinterpret_cast<float*>(C->memory);

  for (int i = 0; i < M; i++) {
    for (int k = 0; k < K; k++) {
      Ad[i * K + k] = float(rand()) / RAND_MAX;  // NOLINT
    }
  }

  for (int j = 0; j < M; j++) {
    for (int k = 0; k < K; k++) {
      Bd[k * N + j] = float(rand()) / RAND_MAX;  // NOLINT
    }
  }

  // manually set zero
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      Cd_target[i * N + j] = 0.f;
      // Cd[i * N + j]        = 0.f;
    }
  }

  auto compare = [&](float diff = 1e-4) {
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        ASSERT_NEAR(Cd[i * N + j], Cd_target[i * N + j], diff);
      }
    }
  };

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < K; k++) {
        Cd_target[i * N + j] += Ad[i * K + k] * Bd[k * N + j];
      }
    }
  }

  auto reset = [&]() {
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        Cd[i * N + j] = 0.f;
      }
    }
  };

  cinn::utils::Timer timer;

  const int repeat = 2;

  cinn_pod_value_t A_arg(A);
  cinn_pod_value_t B_arg(B);
  cinn_pod_value_t C_arg(C);
  cinn_pod_value_t packedB_arg(packedB);
  cinn_pod_value_t M_arg(M);

  cinn_pod_value_t args[] = {A_arg, B_arg, C_arg};
  cinn_pod_value_t args1[] = {A_arg, B_arg, C_arg, packedB_arg};
  cinn_pod_value_t args2[] = {M_arg, A_arg, B_arg, C_arg};
  cinn_pod_value_t args3[] = {M_arg, A_arg, B_arg, C_arg};

#define TEST_FUNC(func__)                                                     \
  LOG(INFO) << "Testing " #func__;                                            \
  timer.Start();                                                              \
  for (int i = 0; i < repeat; i++) func__(reinterpret_cast<void**>(args), 3); \
  LOG(INFO) << timer.Stop() / repeat;                                         \
  compare();                                                                  \
  reset();

#define TEST_FUNC1(func__, diff)                                               \
  LOG(INFO) << "Testing " #func__;                                             \
  timer.Start();                                                               \
  for (int i = 0; i < repeat; i++) func__(reinterpret_cast<void**>(args1), 4); \
  LOG(INFO) << timer.Stop() / repeat;                                          \
  compare();                                                                   \
  reset();

#define TEST_FUNC2(func__, diff)                                               \
  LOG(INFO) << "Testing " #func__;                                             \
  timer.Start();                                                               \
  for (int i = 0; i < repeat; i++) func__(reinterpret_cast<void**>(args2), 4); \
  LOG(INFO) << timer.Stop() / repeat;                                          \
  compare();                                                                   \
  reset();

#define TEST_FUNC3(func__, diff)                                               \
  LOG(INFO) << "Testing " #func__;                                             \
  timer.Start();                                                               \
  for (int i = 0; i < repeat; i++) func__(reinterpret_cast<void**>(args3), 4); \
  LOG(INFO) << timer.Stop() / repeat;                                          \
  compare();                                                                   \
  reset();

  TEST_FUNC(matmul)

  TEST_FUNC(matmul_tile)

  TEST_FUNC(matmul_split)

  TEST_FUNC(matmul_block)

  TEST_FUNC(matmul_vectorize)

  TEST_FUNC(matmul_loop_permutation)

  TEST_FUNC1(matmul_array_packing, 1e-5)

  TEST_FUNC2(matmul_dynamic_shape, 1e-5);

  TEST_FUNC2(matmul_dynamic_shape_tile, 1e-5);

  TEST_FUNC3(matmul_array_packing_dynamic_shape, 1e-5);

  // Currently, the execution of a LoweredFunc is scheduled by the outer
  // framework, so no need to Call inside another LoweredFunc.
  // TODO(Superjomn) Fixit latter.
  // TEST_FUNC(matmul_main);

#define TEST_LLVM_MATMUL(test_name, TARGET)                                \
  do {                                                                     \
    auto module = cinn::tests::CreateCinnMatmulModule(                     \
        #test_name, TARGET, 1024, 1024, 1024);                             \
    auto engine = cinn::tests::CreateExecutionEngine(module);              \
    auto matmul_##test_name = reinterpret_cast<void (*)(void**, int32_t)>( \
        engine->Lookup("matmul_" #test_name));                             \
    TEST_FUNC(matmul_##test_name);                                         \
  } while (false)

#define TEST_LLVM_MATMUL1(test_name, TARGET)                               \
  do {                                                                     \
    auto module = cinn::tests::CreateCinnMatmulModule(                     \
        #test_name, TARGET, 1024, 1024, 1024);                             \
    auto engine = cinn::tests::CreateExecutionEngine(module);              \
    auto matmul_##test_name = reinterpret_cast<void (*)(void**, int32_t)>( \
        engine->Lookup("matmul_" #test_name));                             \
    TEST_FUNC1(matmul_##test_name, 1e-5);                                  \
  } while (false)

  cinn::Target target;
  target.arch = cinn::Target::Arch::X86;
  target.bits = cinn::Target::Bit::k32;
  target.os = cinn::Target::OS::Linux;

  TEST_LLVM_MATMUL(basic, target);
  TEST_LLVM_MATMUL(tile, target);
  TEST_LLVM_MATMUL(block, target);
  TEST_LLVM_MATMUL(vectorize, target);
  TEST_LLVM_MATMUL(loop_permutation, target);
  TEST_LLVM_MATMUL1(array_packing, target);

  {
    auto module =
        cinn::tests::CreateMatmulBasicModule(target, 1024, 1024, 1024);
    auto jit = cinn::tests::CreateSimpleJit(module);
    auto matmul_fn = reinterpret_cast<void (*)(void**, int32_t)>(
        jit->Lookup("matmul_basic"));
    TEST_FUNC(matmul_fn);
  }

#undef TEST_LLVM_MATMUL
}

// include the generated C source code:
// @{
#include "test/cpp/cinn/test02_matmul.cc"
#include "test/cpp/cinn/test02_matmul_array_packing.cc"
#include "test/cpp/cinn/test02_matmul_array_packing_dynamic_shape.cc"
#include "test/cpp/cinn/test02_matmul_block.cc"
#include "test/cpp/cinn/test02_matmul_call.cc"
#include "test/cpp/cinn/test02_matmul_loop_permutation.cc"
#include "test/cpp/cinn/test02_matmul_split.cc"
#include "test/cpp/cinn/test02_matmul_tile.cc"
#include "test/cpp/cinn/test02_matmul_varient_shape.cc"
#include "test/cpp/cinn/test02_matmul_varient_shape_tile.cc"
#include "test/cpp/cinn/test02_matmul_vectorize.cc"
// @}

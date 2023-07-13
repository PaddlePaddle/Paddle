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

#include "paddle/cinn/backends/codegen_cuda_dev.h"
#include "paddle/cinn/backends/codegen_cuda_util.h"
#include "paddle/cinn/backends/cuda_util.h"
#include "paddle/cinn/backends/llvm/execution_engine.h"
#include "paddle/cinn/backends/nvrtc/nvrtc_util.h"
#include "paddle/cinn/cinn.h"
#include "paddle/cinn/common/target.h"
#include "paddle/cinn/common/test_helper.h"
#include "paddle/cinn/hlir/pe/broadcast.h"
#include "paddle/cinn/hlir/pe/reduction.h"
#include "paddle/cinn/hlir/pe/schedule.h"
#include "paddle/cinn/hlir/pe/transform.h"
#include "paddle/cinn/runtime/cinn_runtime.h"
#include "paddle/cinn/runtime/cpu/host_intrinsics.h"
#include "paddle/cinn/runtime/cuda/cuda_module.h"

namespace cinn {
namespace hlir {
namespace pe {
using ir::Tensor;

TEST(MatmulPE, MatmulCase1) {
  int m = 100;
  int n = 32;
  int k = 16;
  Expr M(m), N(n), K(k);

  Placeholder<float> A("A", {M, K});
  Placeholder<float> B("B", {K, N});

  auto C = hlir::pe::Matmul(A.tensor(), B.tensor(), false, false, 1, "C");

  auto stages = CreateStages({A, B});
  std::vector<ir::Tensor> tensor_args = {A, B};
  for (size_t i = 0; i < C.size(); i++) {
    tensor_args.push_back(C[i]);
    stages->InsertLazily(C[i]);
  }
  Target target = common::DefaultHostTarget();
  Module::Builder builder("module0", target);
  auto func = Lower("fn", stages, tensor_args);
  builder.AddFunction(func);
  LOG(INFO) << "func:\n" << func;

  auto jit = backends::ExecutionEngine::Create({});
  auto module = builder.Build();

  jit->Link(module);
  auto fn = jit->Lookup("fn");
  CHECK(fn);
  auto fn_ = reinterpret_cast<void (*)(void *, int32_t)>(fn);
  cinn_buffer_t *A_buf =
      common::BufferBuilder(Float(32), {m, k}).set_random().Build();
  cinn_buffer_t *B_buf =
      common::BufferBuilder(Float(32), {k, n}).set_random().Build();
  cinn_pod_value_t a_arg(A_buf), b_arg(B_buf);
  std::vector<cinn_pod_value_t> args = {a_arg, b_arg};
  std::vector<cinn_buffer_t *> C_buf;
  for (int i = 0; i < C.size(); i++) {
    std::vector<int> shapes;
    for (auto &shape : C[i]->shape) {
      shapes.push_back(shape.as_int32());
    }
    auto *buffer = common::BufferBuilder(Float(32), shapes).set_zero().Build();
    CHECK(buffer);
    C_buf.push_back(buffer);
    cinn_pod_value_t arg(buffer);
    args.push_back(arg);
  }
  fn_(reinterpret_cast<void **>(args.data()), args.size());
  auto *ad = reinterpret_cast<float *>(A_buf->memory);
  auto *bd = reinterpret_cast<float *>(B_buf->memory);
  auto *cd = reinterpret_cast<float *>(C_buf[0]->memory);
  int size_a = m;
  int size_b = n;
  int size_c = k;
  for (int i = 0; i < size_a; i++) {
    for (int j = 0; j < size_b; j++) {
      float tmp = 0;
      for (int k = 0; k < size_c; k++) {
        int index1 = i * size_c + k;
        int index2 = j + k * size_b;
        tmp += ad[index1] * bd[index2];
      }
      ASSERT_NEAR(cd[i * size_b + j], tmp, 1e-5);
    }
  }
}

TEST(ScatterAssign, ScatterAssign) {
  int m = 128;
  int n = 32;
  int k = 32;
  Expr M(m), N(n), K(k);

  Placeholder<float> input("A", {M, K});
  Placeholder<float> assign("B", {N, K});
  Placeholder<int> indexs("C", {N});
  int axis = 0;

#ifdef CINN_WITH_CUDA
  auto target = common::DefaultNVGPUTarget();
#else
  auto target = common::DefaultHostTarget();
#endif

  auto output = hlir::pe::ScatterAssign(
      input.tensor(), assign.tensor(), indexs.tensor(), target, axis);
  auto stages = CreateStages({input, assign, indexs, output});
  auto func = Lower("fn", stages, {input, assign, indexs, output});
  LOG(INFO) << "func:\n" << func;

#ifdef CINN_WITH_CUDA
  Module::Builder builder("ScatterAssign_Builder", target);
  builder.AddFunction(func);

  auto module = builder.Build();
  auto host_module_device_module = backends::SplitCudaAndHostModule(module);
  auto &host_module = std::get<0>(host_module_device_module);
  auto &device_module = std::get<1>(host_module_device_module);

  backends::CodeGenCUDA_Dev codegen(target);
  auto source_code = codegen.Compile(builder.Build());
  LOG(INFO) << "compiled code:\n\n\n" << source_code;

  // nv jit compile to ptx
  backends::nvrtc::Compiler compiler;
  auto ptx = compiler(source_code);
  CHECK(!ptx.empty());
  // cuda_module load ptx
  runtime::cuda::CUDAModule cuda_module(ptx,
                                        runtime::cuda::CUDAModule::Kind::PTX);
#endif  // CINN_WITH_CUDA
}

TEST(SliceAssign, SliceAssign) {
  int m = 128;
  int n = 32;
  int k = 32;
  Expr M(m), N(n), K(k);

  std::vector<int> axis = {0, 1};
  std::vector<int> starts = {32, 32};
  std::vector<int> ends = {64, 64};
  std::vector<int> strides = {1, 1};

  Placeholder<float> input("A", {M, M});
  Placeholder<float> assign("B", {N, N});

  auto output = hlir::pe::SliceAssign(
      input.tensor(), assign.tensor(), axis, starts, ends, strides);
  auto stages = CreateStages({input, assign, output});
  auto func = Lower("fn", stages, {input, assign, output});
  LOG(INFO) << "func:\n" << func;

#ifdef CINN_WITH_CUDA
  auto target = common::DefaultNVGPUTarget();
  Module::Builder builder("SliceAssign_Builder", target);
  builder.AddFunction(func);

  auto module = builder.Build();
  auto host_module_device_module = backends::SplitCudaAndHostModule(module);
  auto &host_module = std::get<0>(host_module_device_module);
  auto &device_module = std::get<1>(host_module_device_module);

  backends::CodeGenCUDA_Dev codegen(target);
  auto source_code = codegen.Compile(builder.Build());
  LOG(INFO) << "compiled code:\n\n\n" << source_code;

  // nv jit compile to ptx
  backends::nvrtc::Compiler compiler;
  auto ptx = compiler(source_code);
  CHECK(!ptx.empty());

  runtime::cuda::CUDAModule cuda_module(ptx,
                                        runtime::cuda::CUDAModule::Kind::PTX);
#endif
}

TEST(Concat, ConcatCase0) {
  int m = 128;
  int n = 32;
  Expr M(m), N(n);

  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});
  Placeholder<float> C("C", {M, N});
  Placeholder<float> D("D", {M, N});

  std::vector<ir::Tensor> inputs{
      A.tensor(), B.tensor(), C.tensor(), D.tensor()};
  auto output = hlir::pe::Concat(inputs, 1);
  auto stages = CreateStages({output});
  auto func = Lower("fn", stages, {A, B, C, D, output});
  LOG(INFO) << "func:\n" << func;

#ifdef CINN_WITH_CUDA
  auto target = common::DefaultNVGPUTarget();
  Module::Builder builder("Concat_Builder", target);
  builder.AddFunction(func);

  auto module = builder.Build();
  auto host_module_device_module = backends::SplitCudaAndHostModule(module);
  auto &host_module = std::get<0>(host_module_device_module);
  auto &device_module = std::get<1>(host_module_device_module);

  backends::CodeGenCUDA_Dev codegen(target);
  auto source_code = codegen.Compile(builder.Build());
  LOG(INFO) << "compiled code:\n\n\n" << source_code;

  // nv jit compile to ptx
  backends::nvrtc::Compiler compiler;
  auto ptx = compiler(source_code);
  CHECK(!ptx.empty());
#endif
}

}  // namespace pe
}  // namespace hlir
}  // namespace cinn

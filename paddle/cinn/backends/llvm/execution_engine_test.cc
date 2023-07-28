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

#include "paddle/cinn/backends/llvm/execution_engine.h"

#include <glog/logging.h>
#include <glog/raw_logging.h>
#include <gtest/gtest.h>
#include <llvm/AsmParser/Parser.h>
#include <llvm/IR/Argument.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Function.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <memory>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

#include "paddle/cinn/backends/llvm/cinn_runtime_llvm_ir.h"
#include "paddle/cinn/backends/llvm/codegen_llvm.h"
#include "paddle/cinn/backends/llvm/runtime_symbol_registry.h"
#include "paddle/cinn/cinn.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/module.h"
#include "paddle/cinn/ir/utils/ir_printer.h"
#include "paddle/cinn/lang/compute.h"
#include "paddle/cinn/lang/lower.h"
#include "paddle/cinn/lang/placeholder.h"
#include "paddle/cinn/optim/optimize.h"
#include "paddle/cinn/runtime/cpu/host_intrinsics.h"
#include "paddle/cinn/runtime/cpu/use_extern_funcs.h"

namespace cinn {
namespace backends {

namespace {
bool RegisterKnownSymbols() {
  decltype(auto) registry = GlobalSymbolRegistry::Global();

  registry.RegisterFn("sinf", reinterpret_cast<void *>(&sinf));
  registry.RegisterFn(
      "sin", reinterpret_cast<void *>(static_cast<double (*)(double)>(&sin)));

  registry.RegisterFn("cosf", reinterpret_cast<void *>(&cosf));
  registry.RegisterFn(
      "cos", reinterpret_cast<void *>(static_cast<double (*)(double)>(&cos)));
  return true;
}

[[maybe_unused]] bool unused = RegisterKnownSymbols();

constexpr int kM = 100;
constexpr int kN = 32;

auto CreateTestBuffer() {
  auto *A = cinn_buffer_t::new_(
      cinn_device_kind_t::cinn_x86_device, cinn_float32_t(), {kM, kN}, 32);
  auto *B = cinn_buffer_t::new_(
      cinn_device_kind_t::cinn_x86_device, cinn_float32_t(), {kM, kN}, 32);
  auto *C = cinn_buffer_t::new_(
      cinn_device_kind_t::cinn_x86_device, cinn_float32_t(), {kM, kN}, 32);
  cinn_buffer_malloc(nullptr, A);
  cinn_buffer_malloc(nullptr, B);
  cinn_buffer_malloc(nullptr, C);
  float *Ad = reinterpret_cast<float *>(A->memory);
  float *Bd = reinterpret_cast<float *>(B->memory);

  for (int i = 0; i < A->num_elements(); i++) {
    Ad[i] = static_cast<float>(rand()) / RAND_MAX;  // NOLINT
    Bd[i] = static_cast<float>(rand()) / RAND_MAX;  // NOLINT
  }

  float *Cd = reinterpret_cast<float *>(C->memory);
  CHECK_EQ(C->num_elements(), A->num_elements());

  return std::make_tuple(A, B, C);
}

auto CreateTestCinnModule() {
  ir::Expr M(kM);
  ir::Expr N(kN);
  lang::Placeholder<float> A("A", {M, N});
  lang::Placeholder<float> B("B", {M, N});

  lang::Buffer C_buf(Float(32));
  auto C = lang::Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j) + B(i, j); }, "C");
  C->Bind(C_buf);

  common::Target target;
  target.arch = common::Target::Arch::X86;
  target.bits = common::Target::Bit::k32;
  target.os = common::Target::OS::Linux;
  ir::Module::Builder builder("module1", target);

  auto stages = CreateStages({C});
  auto funcs = lang::Lower("elementwise_add", stages, {A, B, C});

  // auto func = optim::Optimize(funcs);

  builder.AddFunction(ir::LoweredFunc(funcs.As<ir::_LoweredFunc_>()));
  return builder.Build();
}
}  // namespace

TEST(llvm_test01, elementwise_add) {
  return;
  auto engine = backends::ExecutionEngine::Create({1});

  auto _a_b_c_ = CreateTestBuffer();  // NOLINT
  auto &a = std::get<0>(_a_b_c_);
  auto &b = std::get<1>(_a_b_c_);
  auto &c = std::get<2>(_a_b_c_);

  auto module = CreateTestCinnModule();

  engine->Link(module);

  auto elementwise_add_addr = engine->Lookup("elementwise_add");
  return;
  auto elementwise_add =
      reinterpret_cast<void (*)(void *, int32_t)>(elementwise_add_addr);
  cinn_pod_value_t a_arg(a), b_arg(b), c_arg(c);
  cinn_pod_value_t args[3] = {a_arg, b_arg, c_arg};
  elementwise_add(args, 3);

  float *ad = reinterpret_cast<float *>(a->memory);
  float *bd = reinterpret_cast<float *>(b->memory);
  float *cd = reinterpret_cast<float *>(c->memory);

  for (int i = 0; i < c->num_elements(); i++) {
    EXPECT_EQ(ad[i] + bd[i], cd[i]);
  }
}

TEST(llvm, module_call_lowered_func) {
  ir::Module::Builder builder("some_module", common::DefaultHostTarget());
  ir::Expr M(kM);
  ir::Expr N(kN);
  {  // define fn
    lang::Placeholder<float> a("A", {M, N});
    lang::Placeholder<float> b("B", {M, N});
    auto c = lang::Compute(
        {M, N}, [&](auto i, auto j) { return a(i, j) + b(i, j); }, "C");

    auto stages = CreateStages({c});
    auto fn = lang::Lower("elementwise_add", stages, {a, b, c}, {});
    builder.AddFunction(fn);
  }

  {  // call fn
    lang::Placeholder<float> a("A", {M, N});
    lang::Placeholder<float> b("B", {M, N});

    std::vector<lang::ReturnType> ret_types(
        {lang::ReturnType{Float(32), {M, N}, "c_out"}});

    auto call_outs = lang::CallLowered("elementwise_add", {a, b}, ret_types);
    auto c = call_outs[0];

    // here we must call the output, so that it cal output something.

    auto stages = CreateStages({c});
    auto main_fn = lang::Lower("main", stages, {a, b, c}, {});
    builder.AddFunction(main_fn);

    CodeGenC codegen(common::DefaultHostTarget());
    codegen.SetInlineBuiltinCodes(false);
    LOG(INFO) << "module:\n"
              << codegen.Compile(builder.Build(), CodeGenC::OutputKind::CImpl);
  }

  auto _ab_bb_cb_ = CreateTestBuffer();  // NOLINT
  auto &ab = std::get<0>(_ab_bb_cb_);
  auto &bb = std::get<1>(_ab_bb_cb_);
  auto &cb = std::get<2>(_ab_bb_cb_);
  do {  // call the function
    auto engine = backends::ExecutionEngine::Create({1});

    LOG(INFO) << "JIT Link the module";
    engine->Link(builder.Build());
    auto cos_fn = (double (*)(double))engine->Lookup("cos");
    LOG(INFO) << "=> LLVM JIT cos(0) = " << cos_fn(0);
    auto elementwise_add_addr = engine->Lookup("elementwise_add");
    auto elementwise_add =
        reinterpret_cast<void (*)(void *, int32_t)>(elementwise_add_addr);
    LOG(INFO) << "JIT get elementwise_add_addr";
    break;

    cinn_pod_value_t a_arg(ab), b_arg(bb), c_arg(cb);
    cinn_pod_value_t args[3] = {a_arg, b_arg, c_arg};

    elementwise_add(args, 3);

    auto *ad = reinterpret_cast<float *>(ab->memory);
    auto *bd = reinterpret_cast<float *>(bb->memory);
    for (int i = 0; i < kM; i++) {
      for (int j = 0; j < kN; j++) {
        auto *data = reinterpret_cast<float *>(cb->memory);
        ASSERT_NEAR(data[i * kN + j], ad[i * kN + j] + bd[i * kN + j], 1e-5);
      }
    }
  } while (false);
}

TEST(ExecutionEngine, custom_runtime_symbols) {
  auto context = std::make_unique<llvm::LLVMContext>();
  auto module =
      std::make_unique<llvm::Module>("test_llvm_cpu_runtime", *context);
  auto builder = std::make_unique<llvm::IRBuilder<>>(*context);

  auto call_custom_target = [&](std::string name, llvm::Type *ty) {
    llvm::FunctionType *fn_type = llvm::FunctionType::get(ty, {ty}, false);
    llvm::Function *function =
        llvm::Function::Create(fn_type,
                               llvm::Function::ExternalLinkage,
                               "_call_custom_" + name,
                               module.get());
    function->setCallingConv(llvm::CallingConv::C);
    llvm::BasicBlock *entry =
        llvm::BasicBlock::Create(module->getContext(), "entry", function);
    builder->SetInsertPoint(entry);
    llvm::Argument *arg = &*function->args().begin();
    llvm::Function *custom_function = llvm::dyn_cast<llvm::Function>(
        module->getOrInsertFunction(name, fn_type).getCallee());
    custom_function->setCallingConv(llvm::CallingConv::C);
    llvm::Value *ret = builder->CreateCall(custom_function, {arg});
    builder->CreateRet(ret);
  };

  llvm::Type *f32 = builder->getFloatTy();
  llvm::Type *f64 = builder->getDoubleTy();
  call_custom_target("cosf", f32);
  call_custom_target("cos", f64);
  call_custom_target("sinf", f32);
  call_custom_target("sin", f64);

  double pi = std::acos(-1);

  std::vector<double> angle = {0., pi / 6., pi / 4., pi / 3., pi / 2., pi};

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<int> dis(-100, 100);
  int random_x = dis(mt);
  int random_y = dis(mt);

  decltype(auto) registry = GlobalSymbolRegistry::Global();
  // registry.Register("dereference_f64_ptr", (void *)+[](double *x) { return
  // *x; });

  for (size_t i = 0; i < angle.size(); i++) {
    registry.RegisterVar("theta_" + std::to_string(i), angle[i]);
  }

  auto engine = cinn::backends::ExecutionEngine::Create({1});
  engine->AddModule(std::move(module), std::move(context));

  auto *call_cosf =
      reinterpret_cast<float (*)(float)>(engine->Lookup("_call_custom_cosf"));
  auto *call_cos =
      reinterpret_cast<double (*)(double)>(engine->Lookup("_call_custom_cos"));
  auto *call_sinf =
      reinterpret_cast<float (*)(float)>(engine->Lookup("_call_custom_sinf"));
  auto *call_sin =
      reinterpret_cast<double (*)(double)>(engine->Lookup("_call_custom_sin"));

  ASSERT_TRUE(call_cosf && call_cos && call_sinf && call_sin);

  for (auto theta : angle) {
    float theta_f = static_cast<float>(theta);
    ASSERT_NEAR(call_cosf(theta_f), cosf(theta_f), 1e-6);
    ASSERT_NEAR(call_cos(theta), cos(theta), 1e-6);
    ASSERT_NEAR(call_sinf(theta_f), sinf(theta_f), 1e-6);
    ASSERT_NEAR(call_sin(theta), sin(theta), 1e-6);
  }
}

TEST(ExecutionEngine, call_extern) {
  ir::Expr M(kM);
  ir::Expr N(kN);

  Placeholder<float> x("x", {M, N});
  Placeholder<float> y("y", {M, N});

  auto add_out = Compute(
      {M, N}, [=](Var i, Var j) { return x(i, j) + y(i, j); }, "add_out");

  ir::Tensor res = Compute(
      {M, N},
      [&](Var i, Var j) -> Expr {
        return lang::CallExtern("tanh", {add_out(i, j)});
      },
      "res");

  auto stages = CreateStages({add_out, res});

  stages[add_out]->ComputeInline();
  auto func = Lower("comp", stages, {x, y, res});

  Module::Builder builder("module0", common::DefaultHostTarget());
  builder.AddFunction(func);

  auto engine = backends::ExecutionEngine::Create({1});

  engine->Link(builder.Build());

  auto _ab_bb_cb_ = CreateTestBuffer();  // NOLINT
  auto &ab = std::get<0>(_ab_bb_cb_);
  auto &bb = std::get<1>(_ab_bb_cb_);
  auto &cb = std::get<2>(_ab_bb_cb_);

  auto comp_addr = engine->Lookup("comp");
  auto comp = reinterpret_cast<void (*)(void *, int32_t)>(comp_addr);

  cinn_pod_value_t a_arg(ab), b_arg(bb), c_arg(cb);
  cinn_pod_value_t args[3] = {a_arg, b_arg, c_arg};

  comp(args, 3);

  auto *ad = reinterpret_cast<float *>(ab->memory);
  auto *bd = reinterpret_cast<float *>(bb->memory);
  auto *cd = reinterpret_cast<float *>(cb->memory);
  for (int m = 0; m < kM; m++) {
    for (int n = 0; n < kN; n++) {
      ASSERT_NEAR(cd[m * kN + n], tanh(ad[m * kN + n] + bd[m * kN + n]), 1e-5);
    }
  }
}

}  // namespace backends
}  // namespace cinn

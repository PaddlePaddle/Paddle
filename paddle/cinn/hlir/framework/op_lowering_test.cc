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

#include "paddle/cinn/hlir/framework/op_lowering.h"

#include <gtest/gtest.h>

#include "paddle/cinn/backends/codegen_c_x86.h"
#include "paddle/cinn/backends/codegen_cuda_dev.h"
#include "paddle/cinn/backends/codegen_cuda_util.h"
#include "paddle/cinn/backends/cuda_util.h"
#include "paddle/cinn/backends/llvm/execution_engine.h"
#include "paddle/cinn/backends/nvrtc/nvrtc_util.h"
#include "paddle/cinn/common/target.h"
#include "paddle/cinn/frontend/decomposer/test_helper.h"

namespace cinn {
namespace hlir {
namespace framework {

using frontend::NetBuilder;
using frontend::RunDecomposer;

void CodeGen(const ir::LoweredFunc& func) {
#ifdef CINN_WITH_CUDA
  auto target = common::DefaultNVGPUTarget();
  Module::Builder builder("module_builder", target);

  builder.AddFunction(func);
  auto module = builder.Build();
  auto compiler = backends::Compiler::Create(target);

  std::string code = "";
  compiler->Build(module, code);
#else
  auto target = common::DefaultHostTarget();
  ir::Module::Builder builder("Module_Builder", target);
  builder.AddFunction(func);

  CodeGenCX86 codegen(target, CodeGenCX86::Feature::AVX512);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code =
      codegen.Compile(builder.Build(), CodeGenC::OutputKind::CImpl);
  LOG(INFO) << "compiled code of " << func->name << "is:\n\n\n" << source_code;
#endif
}

void Compile(NetBuilder& net_builder) {  // NOLINT
  auto program = net_builder.Build();
  auto target = common::DefaultTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  hlir::framework::ApplyPass(graph.get(), "FusionMergePass");

  auto& dtype_dict =
      graph->GetMutableAttrs<absl::flat_hash_map<std::string, Type>>(
          "inferdtype");
  auto& shape_dict =
      graph->GetMutableAttrs<absl::flat_hash_map<std::string, shape_t>>(
          "infershape");

  auto op_lowerer = CreateOpLowerer(dtype_dict, shape_dict, target);
  for (auto& fusion_op : graph->fusion_groups) {
    auto lowered_func = op_lowerer.Lower(fusion_op);
    CHECK_EQ(lowered_func.size(), 1);
    CodeGen(lowered_func[0]);
  }
}

TEST(OP_LOWERING, Reduce_Without_Last_Axis_3) {
  int h = 128, w = 128;
  NetBuilder net_builder("Reduce_Without_Last_Axis_3");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
    auto C = net_builder.Add(A, B);
    auto E = net_builder.ReduceSum(C, {0});
    auto F = net_builder.ReduceSum(C, {0});
    auto G = net_builder.Add(E, F);
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, Reduce_Without_Last_Axis_2) {
  int h = 128, w = 128;
  NetBuilder net_builder("Reduce_Without_Last_Axis_2");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h * 2, w}, "B");
    auto E = net_builder.ReduceSum(A, {0});
    auto F = net_builder.ReduceSum(B, {0});
    auto G = net_builder.Add(E, F);
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, Reduce_Without_Last_Axis_1) {
  NetBuilder net_builder("Reduce_Without_Last_Axis_1");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {128, 1024}, "A");
    auto B = net_builder.ReduceSum(A, {0});
    auto C = net_builder.ReduceSum(A, {0});
    auto D = net_builder.ReduceSum(A, {0});
  }
  Compile(net_builder);
}

TEST(OP_LOWERING, Reduce_With_Last_Axis_1) {
  NetBuilder net_builder("Reduce_With_Last_Axis_1");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {10, 100, 1}, "A");
    auto B = net_builder.ReduceSum(A, {0, 2});
  }
  Compile(net_builder);
}

TEST(OP_LOWERING, Reduce_Fuse_Broadcast_With_Output) {
  NetBuilder net_builder("Reduce_Fuse_Broadcast_With_Output");
  auto layer_norm_51__tmp_1 =
      net_builder.CreateInput(Float(32), {256}, "layer_norm_51__tmp_1");
  auto var_3216 = net_builder.CreateInput(Float(32), {256, 60}, "var_3216");
  auto var_3202 = net_builder.CreateInput(Float(32), {1, 60}, "var_3202");
  auto var_3212 = net_builder.CreateInput(Float(32), {256, 60}, "var_3212");

  auto var_3206 = net_builder.Reshape(layer_norm_51__tmp_1, {256, 1});
  auto composite_tmp_8 =
      net_builder.FillConstant<float>({256, 1}, 1e-5, "composite_tmp_8");
  auto var_3214 = net_builder.Add(var_3206, composite_tmp_8);
  auto composite_tmp_10 =
      net_builder.FillConstant<float>({256, 1}, 1.0, "composite_tmp_10");
  auto var_3220 = net_builder.Divide(composite_tmp_10, var_3214);
  auto var_3226 = net_builder.Sqrt(var_3220);
  auto var_3224 = net_builder.Scale(var_3220, -1.0, 0.0, true);
  auto var_3366 = net_builder.BroadcastTo(var_3224, {256, 60});
  auto var_3228 = net_builder.Multiply(var_3366, var_3216);
  auto var_3368 = net_builder.BroadcastTo(var_3202, {256, 60});
  auto var_3236 = net_builder.Multiply(var_3228, var_3212);
  auto var_3244 = net_builder.Multiply(var_3236, var_3368);
  auto var_3252 = net_builder.ReduceSum(var_3244, {1}, true);
  auto var_3232 = net_builder.Scale(var_3226, 0.0166667, 0.0, true);

  Compile(net_builder);
}

TEST(OP_LOWERING, Reduce_Fuse_Broadcast_Layernorm) {
  int h = 32, w = 1024;
  NetBuilder net_builder("Reduce_Fuse_Broadcast_Layernorm");
  // create model
  {
    // x
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    // x * x
    auto B = net_builder.Multiply(A, A);
    // sum x
    auto C = net_builder.ReduceSum(A, {1});
    // sum x*x
    auto D = net_builder.ReduceSum(B, {1});
    // constant w
    auto E = net_builder.FillConstant<float>({h}, 1024.0f, "E");
    // mean
    auto F = net_builder.Divide(C, E);
    auto FF = net_builder.BroadcastTo(F, {h, w}, {0});
    // mean x*x
    auto G = net_builder.Divide(D, E);
    // mean * mean
    auto H = net_builder.Multiply(F, F);
    // var^2
    auto I = net_builder.Subtract(G, H);
    // eps
    auto J = net_builder.FillConstant<float>({h}, 1e-10f, "J");
    // eps + delta
    auto K = net_builder.Add(I, J);
    // var
    auto L = net_builder.Sqrt(K);
    auto LL = net_builder.BroadcastTo(L, {h, w}, {0});
    // x - mean
    auto M = net_builder.Subtract(A, FF);
    // /var
    auto N = net_builder.Divide(M, LL);
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, Reduce_Fuse_Broadcast_Softmax) {
  int h = 32, w = 1024;
  NetBuilder net_builder("Reduce_Fuse_Broadcast_Softmax");
  // create model
  {
    // softmax
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    // redece max
    auto B = net_builder.ReduceMax(A, {1});
    // broadcast
    auto C = net_builder.BroadcastTo(B, {h, w}, {0});
    // x - max(x)
    auto D = net_builder.Subtract(A, C);
    // exp(x)
    auto E = net_builder.Exp(D);
    // reduce sum
    auto F = net_builder.ReduceSum(E, {1});
    // broadcast
    auto G = net_builder.BroadcastTo(F, {h, w}, {0});
    // exp(x)/sum(exp(x))
    auto H = net_builder.Divide(E, G);
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, Reduce_Fuse_Broadcast_1) {
  int h = 32, w = 32;
  NetBuilder net_builder("Reduce_Fuse_Broadcast_1");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h * w}, "A");
    auto B = net_builder.ReduceSum(A, {0});
    auto C = net_builder.BroadcastTo(B, {h * w}, {0});
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, Reduce_Fuse_Broadcast_2) {
  int h = 32, w = 32;
  NetBuilder net_builder("Reduce_Fuse_Broadcast_2");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.ReduceSum(A, {0, 1});
    auto C = net_builder.BroadcastTo(B, {h, w}, {1});
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, Reduce_Fuse_Broadcast_3) {
  int h = 32, w = 32;
  NetBuilder net_builder("Reduce_Fuse_Broadcast_3");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, h, w}, "A");
    auto B = net_builder.ReduceSum(A, {1, 2});
    auto C = net_builder.BroadcastTo(B, {h, h, w}, {0});
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, Reduce_Fuse_Broadcast_4) {
  int h = 32, w = 32;
  NetBuilder net_builder("Reduce_Fuse_Broadcast_4");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, h, w}, "A");
    auto B = net_builder.ReduceSum(A, {1, 2});
    auto C = net_builder.BroadcastTo(B, {h, h, w}, {1});
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, Reduce_Fuse_Broadcast_5) {
  int h = 32, w = 32;
  NetBuilder net_builder("Reduce_Fuse_Broadcast_5");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, h, w}, "A");
    auto B = net_builder.ReduceSum(A, {1, 2});
    auto C = net_builder.BroadcastTo(B, {h, h, w}, {0});
    auto D = net_builder.ReduceSum(C, {1, 2});
    auto E = net_builder.BroadcastTo(D, {h, h, w}, {0});
  }

  Compile(net_builder);
}

TEST(OpFusionPass, Reduce_Fuse_Broadcast_6) {
  int h = 32, w = 32;
  NetBuilder net_builder("Reduce_Fuse_Broadcast_6");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, h, w}, "A");
    auto B = net_builder.ReduceSum(A, {1, 2});
    auto C = net_builder.BroadcastTo(B, {h, h, w}, {0});
    auto D = net_builder.CreateInput(Float(32), {h, w}, "B");
    auto E = net_builder.BroadcastTo(D, {h, h, w}, {1, 2});
    auto F = net_builder.Add(C, E);
  }
  Compile(net_builder);
}

TEST(OP_LOWERING, Reduce_Dim_Equal_One_0) {
  NetBuilder net_builder("Reduce_Dim_Equal_One_0");
  {
    auto A = net_builder.CreateInput(Float(32), {1, 1000}, "A");
    auto B = net_builder.CreateInput(Float(32), {1, 1000}, "B");
    auto C = net_builder.Add(A, B);
    auto D = net_builder.ReduceSum(C, {1}, false);
    auto E = net_builder.ReduceSum(C, {1}, false);
    auto F = net_builder.Add(D, E);
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, Reduce_Dim_Equal_One_1) {
  NetBuilder net_builder("Reduce_Dim_Equal_One_1");
  {
    auto A = net_builder.CreateInput(Float(32), {32, 32}, "A");
    auto B = net_builder.ReduceSum(A, {0, 1}, false);
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, Reduce_Dim_Equal_One_2) {
  NetBuilder net_builder("Reduce_Dim_Equal_One_2");
  {
    auto A = net_builder.CreateInput(Float(32), {32, 1024}, "A");
    auto B = net_builder.ReduceSum(A, {1}, false);
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, Reduce_Dim_Equal_One_3) {
  NetBuilder net_builder("Reduce_Dim_Equal_One_3");
  {
    auto A = net_builder.CreateInput(Float(32), {32, 1024}, "A");
    auto B = net_builder.ReduceSum(A, {0, 1}, false);
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, Reduce_Dim_Equal_One_4) {
  NetBuilder net_builder("Reduce_Dim_Equal_One_4");
  {
    auto A = net_builder.CreateInput(Float(32), {32, 32, 1024}, "A");
    auto B = net_builder.ReduceSum(A, {0, 2}, false);
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, Reduce_Dim_Equal_One_5) {
  NetBuilder net_builder("Reduce_Dim_Equal_One_5");
  {
    auto A = net_builder.CreateInput(Float(32), {32, 32, 32, 256}, "A");
    auto B = net_builder.ReduceSum(A, {0, 2, 3}, false);
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, Reduce_Dim_Equal_One_6) {
  NetBuilder net_builder("Reduce_Dim_Equal_One_6");
  {
    auto A = net_builder.CreateInput(Float(32), {32, 32, 256}, "A");
    auto B = net_builder.ReduceSum(A, {1, 2});
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, Reduce_Dim_Equal_One_7) {
  NetBuilder net_builder("Reduce_Dim_Equal_One_7");
  {
    auto A = net_builder.CreateInput(Float(32), {1, 1, 1024}, "A");
    auto B = net_builder.ReduceSum(A, {2}, false);
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, Reduce_Keep_Dim_Fuse_Elementwise_0) {
  NetBuilder net_builder("Reduce_Keep_Dim_Fuse_Elementwise_0");
  {
    auto A = net_builder.CreateInput(Float(32), {16, 64, 1024}, "A");
    auto B = net_builder.ReduceSum(A, {2}, true);
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, Reduce_Keep_Dim_Fuse_Elementwise_1) {
  NetBuilder net_builder("Reduce_Keep_Dim_Fuse_Elementwise_1");
  {
    auto A = net_builder.CreateInput(Float(32), {16, 64, 112, 112}, "A");
    auto B = net_builder.CreateInput(Float(32), {1, 64, 1, 1}, "B");
    auto C = net_builder.ReduceSum(A, {0, 2, 3}, true);
    auto D = net_builder.Add(B, C);
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, Reduce_Keep_Dim_Fuse_Elementwise_2) {
  NetBuilder net_builder("Reduce_Keep_Dim_Fuse_Elementwise_2");
  {
    auto A = net_builder.CreateInput(Float(32), {16, 64, 112, 112}, "A");
    auto B = net_builder.CreateInput(Float(32), {16, 1, 112, 112}, "B");
    auto C = net_builder.ReduceSum(A, {1}, true);
    auto D = net_builder.Add(B, C);
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, Reduce_Keep_Dim_Fuse_Elementwise_3) {
  NetBuilder net_builder("Reduce_Keep_Dim_Fuse_Elementwise_3");
  {
    auto A = net_builder.CreateInput(Float(32), {16, 64, 112, 112}, "A");
    auto B = net_builder.ReduceSum(A, {2, 3}, true);
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, Reduce_Keep_Dim_Fuse_Elementwise_4) {
  NetBuilder net_builder("Reduce_Keep_Dim_Fuse_Elementwise_4");
  {
    auto A = net_builder.CreateInput(Float(32), {16, 64, 112, 112}, "A");
    auto B = net_builder.ReduceSum(A, {2}, true);
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, Reduce_Keep_Dim_Fuse_Elementwise_5) {
  NetBuilder net_builder("Reduce_Keep_Dim_Fuse_Elementwise_5");
  {
    auto A = net_builder.CreateInput(Float(32), {16, 64, 2048}, "A");
    auto B = net_builder.ReduceSum(A, {2}, true);
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, Reduce_Keep_Dim_Fuse_Elementwise_6) {
  NetBuilder net_builder("Reduce_Keep_Dim_Fuse_Elementwise_6");
  {
    auto A = net_builder.CreateInput(Float(32), {16, 64, 1024}, "A");
    auto B = net_builder.ReduceSum(A, {2}, true);
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, Reduce_Keep_Dim_Fuse_Elementwise_7) {
  NetBuilder net_builder("Reduce_Keep_Dim_Fuse_Elementwise_7");
  {
    auto A = net_builder.CreateInput(Float(32), {16, 64, 16, 1024}, "A");
    auto B = net_builder.ReduceSum(A, {1, 3}, true);
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, Elementwise_Test_Concat_Before_Reduce) {
  NetBuilder net_builder("Elementwise_Test_Concat_Before_Reduce");
  {
    auto A = net_builder.CreateInput(Float(32), {32, 1, 32, 512}, "A");
    auto B = net_builder.CreateInput(Float(32), {32, 1, 32, 512}, "B");
    auto C = net_builder.Concat({A, B}, 3);
    auto D = net_builder.Reshape(C, {32, 32, 1024});
    auto E = net_builder.ReduceSum(D, {2}, false);
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, Elementwise_Test_Reshape_Before_Reduce) {
  NetBuilder net_builder("Elementwise_Test_Reshape_Before_Reduce");
  {
    auto A = net_builder.CreateInput(Float(32), {32, 1, 32, 512}, "A");
    auto B = net_builder.CreateInput(Float(32), {32, 1, 32, 512}, "B");
    auto C = net_builder.Add(A, B);
    auto D = net_builder.Reshape(C, {32, 32, 512});
    auto E = net_builder.CreateInput(Float(32), {32, 32, 512}, "E");
    auto F = net_builder.Add(D, E);
    auto G = net_builder.ReduceSum(F, {0, 1}, false);
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, Elementwise_Test_Reshape_After_Reduce) {
  NetBuilder net_builder("Elementwise_Test_Reshape_After_Reduce");
  {
    auto A = net_builder.CreateInput(Float(32), {32, 32, 32}, "A");
    auto B = net_builder.ReduceSum(A, {1}, false);
    auto C = net_builder.CreateInput(Float(32), {16, 4, 16}, "C");
    auto D = net_builder.Reshape(C, {32, 32});
    auto E = net_builder.Transpose(D, {1, 0});
    auto F = net_builder.CreateInput(Float(32), {32, 32}, "F");
    auto G = net_builder.Add(E, F);
    auto H = net_builder.Add(B, G);
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, Elementwise_Test_Reshape_Fuse_Concat) {
  NetBuilder net_builder("Elementwise_Test_Reshape_Fuse_Concat");
  {
    auto A = net_builder.CreateInput(Float(32), {8, 8, 8, 8}, "A");
    auto B = net_builder.Reshape(A, {16, 16, 16});
    auto C = net_builder.CreateInput(Float(32), {16, 16}, "C");
    auto D = net_builder.CreateInput(Float(32), {16, 16}, "D");
    auto DT = net_builder.Transpose(D, {1, 0});
    auto E = net_builder.Add(C, DT);
    auto F = net_builder.BroadcastTo(E, {16, 16, 16}, {1, 2});
    auto G = net_builder.Add(B, F);
    auto H = net_builder.CreateInput(Float(32), {16, 16, 16}, "H");
    auto I = net_builder.Concat({G, H}, 2);
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, Elementwise_TEST_Split_0) {
  NetBuilder net_builder("Elementwise_TEST_Split_0");
  {
    auto A = net_builder.CreateInput(Float(32), {32, 64}, "A");
    auto B = net_builder.Split(A, {3, 5, 16, 2, 6}, 0);
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, Elementwise_TEST_Split_1) {
  NetBuilder net_builder("Elementwise_TEST_Split_1");
  {
    auto A = net_builder.CreateInput(Float(32), {128, 128}, "A");
    auto B = net_builder.Split(A, {32, 32, 32, 32}, 1);
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, Elementwise_TEST_Split_2) {
  NetBuilder net_builder("Elementwise_TEST_Split_2");
  {
    auto A = net_builder.CreateInput(Float(32), {128, 128}, "A");
    auto B = net_builder.Split(A, {64, 32, 32}, 1);
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, Elementwise_TEST_0) {
  NetBuilder net_builder("Elementwise_TEST_0");
  {
    auto x = net_builder.FillConstant<float>({1}, 128.0, "x");
    auto o1 = net_builder.Scale(x, -1.0, 0.0);
    auto o2 = net_builder.Scale(x, -1.0, 0.0);
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, NonFusibleOp_TEST_0) {
  NetBuilder net_builder("NonFusibleOp_TEST_0");
  {
    auto A = net_builder.CreateInput(Float(32), {9801, 2}, "A");
    auto B = net_builder.Reshape(A, {9801, 2});
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, NonFusibleOp_TEST_1) {
  NetBuilder net_builder("NonFusibleOp_TEST_1");
  {
    auto A = net_builder.CreateInput(Float(32), {128, 128}, "A");
    auto B = net_builder.CreateInput(Float(32), {128, 128}, "B");
    auto C = net_builder.Matmul(A, B);
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, NonFusibleOp_TEST_2) {
  NetBuilder net_builder("NonFusibleOp_TEST_2");
  {
    auto A = net_builder.CreateInput(Float(32), {128, 128}, "A");
    auto B = net_builder.Matmul(A, A);
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, NonFusibleOp_TEST_3) {
  NetBuilder net_builder("NonFusibleOp_TEST_3");
  {
    auto A = net_builder.CreateInput(Float(32), {128, 256}, "A");
    auto C = net_builder.Split(A, {4}, 1);
  }

  Compile(net_builder);
}

#ifdef CINN_WITH_CUDA
TEST(OP_LOWERING, NonFusibleOp_TEST_4) {
  NetBuilder net_builder("NonFusibleOp_TEST_4");
  {
    auto A = net_builder.CreateInput(Float(32), {128, 128}, "A");
    auto B = net_builder.CreateInput(Float(32), {128, 128}, "B");
    auto C = net_builder.CreateInput(Float(32), {128, 128}, "C");
    auto D = net_builder.Matmul(A, B);
    auto E = net_builder.Add(C, D);
  }

  Compile(net_builder);
}
#endif

TEST(OP_LOWERING, Transform_TEST_0) {
  NetBuilder net_builder("Transform_TEST_0");
  {
    auto A = net_builder.CreateInput(Float(32), {128, 128}, "A");
    auto B = net_builder.CreateInput(Float(32), {128, 128}, "B");
    auto C = net_builder.CreateInput(Float(32), {128, 128}, "C");
    auto D = net_builder.Concat({A, B, C}, 1);
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, Elementwise_Test_0) {
  int h = 32, w = 32;
  NetBuilder net_builder("Elementwise_Test_0");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
    auto C = net_builder.CreateInput(Float(32), {h, w}, "C");
    auto D = net_builder.CreateInput(Float(32), {h, w}, "D");
    auto E = net_builder.Add(A, B);
    auto F = net_builder.Add(C, D);
    auto G = net_builder.Add(E, F);
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, Elementwise_Test_1) {
  int h = 32, w = 32;
  NetBuilder net_builder("Elementwise_Test_1");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
    auto C = net_builder.CreateInput(Float(32), {h, w}, "C");
    auto D = net_builder.CreateInput(Float(32), {h, w}, "D");
    auto E = net_builder.Add(A, B);
    auto F = net_builder.Add(E, C);
    auto G = net_builder.Add(E, D);
    auto H = net_builder.Add(F, G);
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, Elementwise_Test_2) {
  int h = 50, w = 10201;
  NetBuilder net_builder("Elementwise_Test_2");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
    auto C = net_builder.CreateInput(Float(32), {h, w}, "C");
    auto D = net_builder.CreateInput(Float(32), {h, w}, "D");
    auto E = net_builder.Add(A, B);
    auto F = net_builder.Add(E, C);
    auto G = net_builder.Add(E, D);
    auto H = net_builder.Add(F, G);
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, Reduce_Test_0) {
  int h = 32, w = 32;
  NetBuilder net_builder("Reduce_Test_0");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {w, h}, "A");
    auto B = net_builder.ReduceSum(A, {0, 1});
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, Reduce_Test_1) {
  int c = 32, h = 32, w = 32;
  NetBuilder net_builder("Reduce_Test_1");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {c, h, w}, "A");
    auto B = net_builder.ReduceSum(A, {0, 1, 2});
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, Reduce_Test_2) {
  int c = 32, h = 32, w = 32;
  NetBuilder net_builder("Reduce_Test_2");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {c, h, w}, "A");
    auto B = net_builder.ReduceSum(A, {0, 1});
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, Reduce_Test_3) {
  int c = 32, h = 16, w = 16;
  NetBuilder net_builder("Reduce_Test_3");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {c, h, w}, "A");
    auto B = net_builder.ReduceSum(A, {0, 1, 2});
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, Reduce_Test_4) {
  int h = 32, w = 32;
  NetBuilder net_builder("Reduce_Test_4");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {w, h}, "A");
    auto B = net_builder.ReduceSum(A, {0});
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, Reduce_Test_5) {
  int h = 32, w = 768;
  NetBuilder net_builder("Reduce_Test_5");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.ReduceSum(A, {0});
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, Reduce_Test_6) {
  int h = 32, w = 2048;
  NetBuilder net_builder("Reduce_Test_6");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.ReduceSum(A, {0});
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, Reduce_Test_7) {
  int h = 32, w = 512;
  NetBuilder net_builder("Reduce_Test_7");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.ReduceSum(A, {1});
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, Reduce_Test_8) {
  int h = 32, w = 32;
  NetBuilder net_builder("Reduce_Test_8");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w, w}, "A");
    auto B = net_builder.ReduceSum(A, {1, 2});
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, Reduce_Test_9) {
  int n = 16, c = 128, h = 56, w = 56;
  NetBuilder net_builder("Reduce_Test_9");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {n, c, h, w}, "A");
    auto B = net_builder.ReduceSum(A, {0, 2, 3});
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, Reduce_Test_10) {
  int n = 16, c = 16, h = 32, w = 32;
  NetBuilder net_builder("Reduce_Test_10");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {n, c, h, w}, "A");
    auto B = net_builder.ReduceSum(A, {1});
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, Reduce_Fusion_Test_0) {
  int h = 32, w = 32;
  NetBuilder net_builder("Reduce_Fusion_Test_0");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {w}, "B");

    auto C = net_builder.ReduceSum(A, {0});
    auto D = net_builder.Add(B, C);
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, Reduce_Fusion_Test_1) {
  int h = 32, w = 32;
  NetBuilder net_builder("Reduce_Fusion_Test_1");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
    auto D = net_builder.Add(A, B);
    auto E = net_builder.ReduceSum(D, {1});
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, Reduce_Fusion_Test_2) {
  int h = 32, w = 32;
  NetBuilder net_builder("Reduce_Fusion_Test_2");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {w}, "B");
    auto C = net_builder.CreateInput(Float(32), {w}, "C");
    auto D = net_builder.CreateInput(Float(32), {w}, "D");

    auto E = net_builder.ReduceSum(A, {0});
    auto F = net_builder.Add(B, C);
    auto G = net_builder.Add(D, F);
    auto H = net_builder.Add(E, G);
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, Reduce_Fusion_Test_3) {
  int h = 32, w = 32;
  NetBuilder net_builder("Reduce_Fusion_Test_3");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.ReduceSum(A, {0});
    auto C = net_builder.ReduceSum(A, {0});
    auto D = net_builder.Add(B, C);
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, Reduce_Fusion_Test_4) {
  int h = 32, w = 32;
  NetBuilder net_builder("Reduce_Fusion_Test_4");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
    auto C = net_builder.Add(A, B);

    auto D = net_builder.ReduceSum(C, {0});
    auto E = net_builder.ReduceSum(C, {0});
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, Reduce_Fusion_Test_5) {
  int h = 32, w = 32;
  NetBuilder net_builder("Reduce_Fusion_Test_5");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
    auto C = net_builder.Add(A, B);

    auto D = net_builder.ReduceSum(C, {1});
    auto E = net_builder.ReduceSum(C, {1});
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, Reduce_Fusion_Test_6) {
  int h = 128, w = 128;
  NetBuilder net_builder("Reduce_Fusion_Test_6");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
    auto C = net_builder.CreateInput(Float(32), {w}, "C");
    auto D = net_builder.Add(A, B);
    auto E = net_builder.ReduceSum(D, {0});
    auto F = net_builder.ReduceSum(D, {0});
    auto G = net_builder.Add(E, C);
    auto I = net_builder.Add(F, C);
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, Reduce_Fusion_Test_7) {
  int h = 128, w = 128;
  NetBuilder net_builder("Reduce_Fusion_Test_7");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
    auto C = net_builder.CreateInput(Float(32), {w}, "C");
    auto D = net_builder.Add(A, B);
    auto E = net_builder.ReduceSum(D, {1});
    auto F = net_builder.ReduceSum(D, {1});
    auto G = net_builder.Add(E, C);
    auto I = net_builder.Add(F, C);
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, Reduce_Fusion_Test_8) {
  int h = 128, w = 128;
  NetBuilder net_builder("Reduce_Fusion_Test_8");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
    auto C = net_builder.CreateInput(Float(32), {1}, "C");
    auto D = net_builder.Add(A, B);
    auto E = net_builder.ReduceSum(D, {0, 1});
    auto F = net_builder.ReduceSum(D, {0, 1});
    auto G = net_builder.Add(E, C);
    auto I = net_builder.Add(F, C);
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, Reduce_Fusion_Test_9) {
  int c = 128, h = 128, w = 128;
  NetBuilder net_builder("Reduce_Fusion_Test_9");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {c, h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {c, h, w}, "B");
    auto C = net_builder.CreateInput(Float(32), {h}, "C");
    auto D = net_builder.Add(A, B);
    auto E = net_builder.ReduceSum(D, {0, 2});
    auto F = net_builder.ReduceSum(D, {0, 2});
    auto G = net_builder.Add(E, C);
    auto I = net_builder.Add(F, C);
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, Reduce_Fusion_Test_10) {
  int h = 10201, w = 50;
  NetBuilder net_builder("Reduce_Fusion_Test_10");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {w}, "B");
    auto C = net_builder.ReduceSum(A, {0});
    auto D = net_builder.Add(B, C);
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, Reduce_Fusion_Test_11) {
  int n = 128, c = 128, h = 16, w = 16;
  NetBuilder net_builder("Reduce_Fusion_Test_11");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {n, c, h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {n, c, h, w}, "B");
    auto D = net_builder.Add(A, B);
    auto E = net_builder.ReduceSum(D, {0, 2, 3});
    auto F = net_builder.ReduceSum(D, {0, 2, 3});
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, Reduce_Fusion_Test_12) {
  int n = 128, c = 128, h = 112, w = 112;
  NetBuilder net_builder("Reduce_Fusion_Test_12");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {n, c, h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {n, c, h, w}, "B");
    auto D = net_builder.Add(A, B);
    auto E = net_builder.ReduceSum(D, {0, 2, 3});
    auto F = net_builder.ReduceSum(D, {0, 2, 3});
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, Reduce_Fusion_Test_13) {
  int n = 8, c = 8, h = 8, w = 8;
  NetBuilder net_builder("Reduce_Fusion_Test_13");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {n, c, h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {n, c, h, w}, "B");
    auto D = net_builder.Add(A, B);
    auto E = net_builder.ReduceSum(D, {0, 1, 2});
    auto F = net_builder.ReduceSum(D, {0, 1, 2});
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, Reduce_Fusion_Test_14) {
  int n = 8, c = 8, h = 8, w = 8;
  NetBuilder net_builder("Reduce_Fusion_Test_14");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {n, n, n, c, h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {n, n, n, c, h, w}, "B");
    auto D = net_builder.Add(A, B);
    auto E = net_builder.ReduceSum(D, {0, 3, 4});
    auto F = net_builder.ReduceSum(D, {0, 3, 4});
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, Reduce_Fusion_Test_15) {
  int h = 512, w = 32;
  NetBuilder net_builder("Reduce_Fusion_Test_15");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
    auto D = net_builder.Add(A, B);
    auto E = net_builder.ReduceSum(D, {0});
    auto F = net_builder.ReduceSum(D, {0});
  }

  Compile(net_builder);
}
TEST(OP_LOWERING, Reduce_Fusion_Test_16) {
  int n = 128, c = 128, h = 28, w = 28;
  NetBuilder net_builder("Reduce_Fusion_Test_16");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {n, c, h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {n, c, h, w}, "B");
    auto D = net_builder.Add(A, B);
    auto E = net_builder.ReduceSum(D, {0, 2, 3});
    auto F = net_builder.ReduceSum(D, {0, 2, 3});
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, Reduce_Fusion_Test_17) {
  int h = 128, w = 768;
  NetBuilder net_builder("Reduce_Fusion_Test_17");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h * 2, w}, "B");
    auto E = net_builder.ReduceSum(A, {0});
    auto F = net_builder.ReduceSum(B, {0});
    auto G = net_builder.Add(E, F);
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, Reduce_Fusion_Test_18) {
  int h = 128, w = 768;
  NetBuilder net_builder("Reduce_Fusion_Test_18");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {16, h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {16, h * 2, w}, "B");
    auto E = net_builder.ReduceSum(A, {1});
    auto F = net_builder.ReduceSum(B, {1});
    auto G = net_builder.Add(E, F);
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, Reduce_Fusion_Test_19) {
  int h = 128, w = 128;
  NetBuilder net_builder("Reduce_Fusion_Test_19");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h * 2, w}, "B");
    auto E = net_builder.ReduceSum(A, {0});
    auto F = net_builder.ReduceSum(B, {0});
    auto G = net_builder.Add(E, F);
  }

  Compile(net_builder);
}

TEST(OP_LOWERING, Reduce_Fusion_Test_20) {
  int h = 128, w = 128;
  NetBuilder net_builder("Reduce_Fusion_Test_20");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h * 2, w}, "B");
    auto C = net_builder.CreateInput(Float(32), {h * 3, w}, "C");
    auto D = net_builder.CreateInput(Float(32), {h * 4, w}, "D");
    auto E = net_builder.ReduceSum(A, {0});
    auto F = net_builder.ReduceSum(B, {0});
    auto G = net_builder.ReduceSum(C, {0});
    auto H = net_builder.ReduceSum(D, {0});
    auto I = net_builder.Add(E, F);
    auto J = net_builder.Add(G, I);
    auto K = net_builder.Add(H, J);
  }

  Compile(net_builder);
}

/*
TEST(OP_LOWERING, Reduce_Fusion_Test_21) {
  int h = 128, w = 4;
  NetBuilder net_builder("Reduce_Fusion_Test_21");
  // create model
  {
    auto A0  = net_builder.CreateInput(Float(32), {256, w}, "A0");
    auto B0  = net_builder.CreateInput(Float(32), {256, w}, "B0");
    auto C0  = net_builder.CreateInput(Float(32), {55200, w}, "C0");
    auto D0  = net_builder.CreateInput(Float(32), {2750, w}, "D0");
    auto A1  = net_builder.CreateInput(Float(32), {256, w}, "A1");
    auto B1  = net_builder.CreateInput(Float(32), {256, w}, "B1");
    auto C1  = net_builder.CreateInput(Float(32), {55200, w}, "C1");
    auto D1  = net_builder.CreateInput(Float(32), {2750, w}, "D1");
    auto AA  = net_builder.Add(A0, A1);
    auto BB  = net_builder.Add(B0, B1);
    auto CC  = net_builder.Add(C0, C1);
    auto DD  = net_builder.Add(D0, D1);
    auto E   = net_builder.ReduceSum(AA, {0});
    auto F   = net_builder.ReduceSum(BB, {0});
    auto G   = net_builder.ReduceSum(CC, {0});
    auto H   = net_builder.ReduceSum(DD, {0});
    auto I   = net_builder.Add(E, F);
    auto J   = net_builder.Add(G, I);
    auto K   = net_builder.Add(H, J);
    auto AAA = net_builder.Add(AA, A1);
    auto BBB = net_builder.Add(BB, B1);
    auto CCC = net_builder.Add(CC, C1);
    auto DDD = net_builder.Add(DD, D1);
  }

  Compile(net_builder);
}
*/

TEST(OpFusionPass, Block_Reduce_Fuse_Broadcast) {
  int sm_count = common::DefaultNVGPUTarget().get_multi_processor_count();
  int max_threads_per_sm =
      common::DefaultNVGPUTarget().get_max_threads_per_sm();
  int warp_reduce_threshold = sm_count * max_threads_per_sm / 32;
  int h = warp_reduce_threshold - 10;
  int w = 256;
  NetBuilder net_builder("Block_Reduce_Fuse_Broadcast");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.ReduceSum(A, {1}, true);
    auto C = net_builder.BroadcastTo(B, {h, w}, {0, 1});
  }

  Compile(net_builder);
}

TEST(OpFusionPass, Block_Reduce_Fuse_Elementwise) {
  int sm_count = common::DefaultNVGPUTarget().get_multi_processor_count();
  int max_threads_per_sm =
      common::DefaultNVGPUTarget().get_max_threads_per_sm();
  int warp_reduce_threshold = sm_count * max_threads_per_sm / 32;
  int h = warp_reduce_threshold - 10;
  int w = 256;
  NetBuilder net_builder("Block_Reduce_Fuse_Elementwise");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h}, "B");
    auto C = net_builder.ReduceSum(A, {1}, true);
    auto D = net_builder.Add(B, C);
  }

  Compile(net_builder);
}
TEST(OpFusionPass, Warp_Reduce_Fuse_Broadcast) {
  int sm_count = common::DefaultNVGPUTarget().get_multi_processor_count();
  int max_threads_per_sm =
      common::DefaultNVGPUTarget().get_max_threads_per_sm();
  int warp_reduce_threshold = sm_count * max_threads_per_sm / 32;
  int h = warp_reduce_threshold + 10;
  int w = 256;
  NetBuilder net_builder("Warp_Reduce_Fuse_Broadcast");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.ReduceSum(A, {1}, true);
    auto C = net_builder.BroadcastTo(B, {h, w}, {0, 1});
  }

  Compile(net_builder);
}

TEST(OpFusionPass, Warp_Reduce_Fuse_Elementwise) {
  int sm_count = common::DefaultNVGPUTarget().get_multi_processor_count();
  int max_threads_per_sm =
      common::DefaultNVGPUTarget().get_max_threads_per_sm();
  int warp_reduce_threshold = sm_count * max_threads_per_sm / 32;
  int h = warp_reduce_threshold + 10;
  int w = 256;
  NetBuilder net_builder("Warp_Reduce_Fuse_Elementwise");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h}, "B");
    auto C = net_builder.ReduceSum(A, {1}, true);
    auto D = net_builder.Add(B, C);
  }

  Compile(net_builder);
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn

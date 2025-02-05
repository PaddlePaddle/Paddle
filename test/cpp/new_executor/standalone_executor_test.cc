// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/new_executor/standalone_executor.h"

#include <gtest/gtest.h>

#include <chrono>
#include <iostream>
#include <memory>
#include <string>

#include "paddle/fluid/framework/new_executor/interpreter/plan.h"
#include "paddle/phi/core/kernel_registry.h"

USE_OP_ITSELF(fill_constant);
USE_OP_ITSELF(uniform_random);
USE_OP_ITSELF(lookup_table);
USE_OP_ITSELF(transpose2);
USE_OP_ITSELF(reshape2);
USE_OP_ITSELF(split);
USE_OP_ITSELF(slice);
USE_OP_ITSELF(concat);
USE_OP_ITSELF(matmul);
USE_OP_ITSELF(elementwise_add);
USE_OP_ITSELF(sigmoid);
USE_OP_ITSELF(tanh);
USE_OP_ITSELF(elementwise_mul);
USE_OP_ITSELF(softmax_with_cross_entropy);
USE_OP_ITSELF(reduce_mean);
USE_OP_ITSELF(reduce_sum);
USE_OP_ITSELF(reduce_sum_grad);
USE_OP_ITSELF(reduce_mean_grad);
USE_OP_ITSELF(reshape2_grad);
USE_OP_ITSELF(softmax_with_cross_entropy_grad);
USE_OP_ITSELF(elementwise_add_grad);
USE_OP_ITSELF(matmul_grad);
USE_OP_ITSELF(square);
USE_OP_ITSELF(transpose2_grad);
USE_OP_ITSELF(concat_grad);
USE_OP_ITSELF(elementwise_mul_grad);
USE_OP_ITSELF(sigmoid_grad);
USE_OP_ITSELF(tanh_grad);
USE_OP_ITSELF(sum);
USE_OP_ITSELF(slice_grad);
USE_OP_ITSELF(lookup_table_grad);
USE_OP_ITSELF(sqrt);
USE_OP_ITSELF(elementwise_max);
USE_OP_ITSELF(elementwise_div);
USE_OP_ITSELF(sgd);
USE_OP_ITSELF(squared_l2_norm);
USE_OP_ITSELF(memcpy_h2d);
USE_OP_ITSELF(memcpy_d2h);

PD_DECLARE_KERNEL(full, GPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(uniform_raw, GPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(uniform, GPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(transpose, GPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(reshape, GPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(split, GPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(concat, GPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(concat_grad, GPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(matmul, GPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(add_raw, KPS, ALL_LAYOUT);
PD_DECLARE_KERNEL(add, KPS, ALL_LAYOUT);
PD_DECLARE_KERNEL(multiply, KPS, ALL_LAYOUT);
PD_DECLARE_KERNEL(multiply_grad, GPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(divide, KPS, ALL_LAYOUT);
#ifdef PADDLE_WITH_XPU_KP
PD_DECLARE_KERNEL(max_raw, GPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(maximum, GPU, ALL_LAYOUT);
#else
PD_DECLARE_KERNEL(max_raw, KPS, ALL_LAYOUT);
PD_DECLARE_KERNEL(maximum, KPS, ALL_LAYOUT);
#endif
PD_DECLARE_KERNEL(mean, GPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(mean_grad, GPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(sigmoid, GPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(sigmoid_grad, GPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(squared_l2_norm, GPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(reshape_grad, GPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(add_grad, GPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(matmul_grad, GPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(transpose_grad, GPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(sum, GPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(sum_grad, GPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(sgd, GPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(slice, GPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(slice_grad, GPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(cross_entropy_with_softmax, GPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(cross_entropy_with_softmax_grad, GPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(sqrt, GPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(add_n, GPU, ALL_LAYOUT);

namespace paddle {
namespace framework {

using Job = interpreter::Job;
using Plan = interpreter::Plan;

ProgramDesc load_from_file(const std::string& file_name) {
  std::ifstream fin(file_name, std::ios::in | std::ios::binary);
  fin.seekg(0, std::ios::end);
  std::string buffer(fin.tellg(), ' ');
  fin.seekg(0, std::ios::beg);
  fin.read(&buffer[0], buffer.size());  // NOLINT
  fin.close();
  ProgramDesc program_desc(buffer);
  return program_desc;
}

ProgramDesc GetLmMainProgram() {
  ProgramDesc main_prog = load_from_file("lm_main_program");

  auto& global_block = main_prog.Block(0);
  int64_t batch_size = 20;

  const auto allOps = global_block.AllOps();
  auto& op1 = allOps[1];
  auto shape1 = PADDLE_GET_CONST(std::vector<int64_t>, op1->GetAttr("shape"));
  shape1[0] = batch_size * 20;
  op1->SetAttr("shape", shape1);

  auto& op2 = allOps[2];
  auto shape2 = PADDLE_GET_CONST(std::vector<int64_t>, op2->GetAttr("shape"));
  shape2[0] = batch_size;
  op2->SetAttr("shape", shape2);

  auto& op3 = allOps[3];
  auto shape3 = PADDLE_GET_CONST(std::vector<int64_t>, op3->GetAttr("shape"));
  shape3[0] = batch_size;
  op3->SetAttr("shape", shape3);
  return main_prog;
}

TEST(InterpreterCore, skip_gc_vars) {
  auto place = phi::GPUPlace(0);
  ProgramDesc startup_prog = load_from_file("lm_startup_program");
  ProgramDesc main_prog = GetLmMainProgram();

  Scope scope;

  std::shared_ptr<InterpreterCore> startup_core =
      std::make_shared<InterpreterCore>(
          place, startup_prog.Block(0), &scope, interpreter::ExecutionConfig());

  startup_core->Run({}, {});

  std::set<std::string> skip_gc_vars = {"uniform_0.tmp_0",
                                        "transpose_0.tmp_0",
                                        "embedding_0.tmp_0",
                                        "slice_0.tmp_0",
                                        "split_1.tmp_2"};
  std::set<std::string> gc_vars = {"uniform_1.tmp_0",
                                   "matmul_0.tmp_0",
                                   "split_0.tmp_0",
                                   "elementwise_add_0.tmp_0",
                                   "tmp_0"};

  interpreter::ExecutionConfig execution_config;
  execution_config.skip_gc_vars = skip_gc_vars;

  std::shared_ptr<InterpreterCore> main_core =
      std::make_shared<InterpreterCore>(
          place, main_prog.Block(0), &scope, execution_config);

  auto check_gc_result =
      [](Scope& scope, std::set<std::string>& vars, bool is_skip_gc) {
        // the first local scope is created in startup_core
        // the second local scope is created in main_core
        ASSERT_EQ(scope.kids().size(), 2UL);
        auto* local_scope = scope.kids().back();
        for (const std::string& var_name : vars) {
          ASSERT_EQ(local_scope->FindVar(var_name)
                        ->GetMutable<phi::DenseTensor>()
                        ->IsInitialized(),
                    is_skip_gc);
        }
      };

  main_core->Run({}, {});
  check_gc_result(scope, skip_gc_vars, true);
  check_gc_result(scope, gc_vars, false);

  main_core->Run({}, {});
  check_gc_result(scope, skip_gc_vars, true);
  check_gc_result(scope, gc_vars, false);
}

void TestShareWorkQueue(const ProgramDesc& prog,
                        const std::vector<std::string>& feed_names,
                        const std::vector<phi::DenseTensor>& feed_tensors,
                        const std::vector<std::string>& fetch_names,
                        const std::vector<float>& fetch_results) {
  const phi::CPUPlace place = phi::CPUPlace();

  Scope scope;
  std::shared_ptr<InterpreterCore> core1 = std::make_shared<InterpreterCore>(
      place, prog.Block(0), &scope, interpreter::ExecutionConfig());
  std::shared_ptr<InterpreterCore> core2 = std::make_shared<InterpreterCore>(
      place, prog.Block(0), &scope, interpreter::ExecutionConfig());
  core2->ShareWorkQueueFrom(core1);

  auto run_and_check = [&feed_names, &feed_tensors, &fetch_results](
                           std::shared_ptr<InterpreterCore> core) {
    FetchList fetch_list = core->Run(feed_names, feed_tensors);
    for (size_t i = 0; i < fetch_list.size(); ++i) {
      const float* fetch_data =
          PADDLE_GET_CONST(phi::DenseTensor, fetch_list[i]).data<float>();
      ASSERT_FLOAT_EQ(*fetch_data, fetch_results.at(i));
    }
  };

  run_and_check(core1);
  run_and_check(core2);
  run_and_check(core1);
  run_and_check(core2);
}

TEST(InterpreterCore, workqueue_multiplexing) {
  ProgramDesc program;
  BlockDesc* main_block = program.MutableBlock(0);
  VarDesc* var_a = main_block->Var("a");
  VarDesc* var_b = main_block->Var("b");
  VarDesc* var_c = main_block->Var("c");
  var_a->SetType(proto::VarType::DENSE_TENSOR);
  var_b->SetType(proto::VarType::DENSE_TENSOR);
  var_c->SetType(proto::VarType::DENSE_TENSOR);

  OpDesc* add = main_block->AppendOp();
  add->SetType("elementwise_add");
  add->SetInput("X", {"a"});
  add->SetInput("Y", {"b"});
  add->SetOutput("Out", {"c"});

  std::array<float, 4> data_a = {0, 1, 2, 3};
  std::array<float, 4> data_b = {0.0, 0.1, 0.2, 0.3};

  phi::DDim dims = common::make_ddim({2, 2});
  const phi::CPUPlace place = phi::CPUPlace();

  phi::DenseTensor tensor_a = phi::DenseTensor();
  phi::DenseTensor tensor_b = phi::DenseTensor();

  std::copy_n(data_a.data(), 4, tensor_a.mutable_data<float>(dims, place));
  std::copy_n(data_b.data(), 4, tensor_b.mutable_data<float>(dims, place));

  TestShareWorkQueue(
      program, {"a", "b"}, {tensor_a, tensor_b}, {"c"}, {0.0, 1.1, 2.2, 3.3});
}

}  // namespace framework
}  // namespace paddle

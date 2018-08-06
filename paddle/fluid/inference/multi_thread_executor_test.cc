// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <chrono>              // NOLINT
#include <condition_variable>  // NOLINT
#include <thread>              // NOLINT
#include "gtest/gtest.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/platform/init.h"

USE_OP(uniform_random);
USE_OP(mul);

TEST(MultiThread, main) {
  paddle::framework::Scope param_scope;
  {
    paddle::framework::ProgramDesc startup_program;
    auto& global_block = *startup_program.MutableBlock(0);
    auto& w = *global_block.Var("W");
    w.SetType(paddle::framework::proto::VarType::LOD_TENSOR);
    w.SetLoDLevel(0);

    auto& op = *global_block.AppendOp();
    op.SetType("uniform_random");
    op.SetOutput("Out", {w.Name()});
    op.SetAttr("shape", std::vector<int>{2000, 10});
    op.SetAttr("dtype", paddle::framework::proto::VarType::FP32);
    paddle::platform::CPUPlace cpu;
    paddle::framework::Executor exe(cpu);
    auto ctx = paddle::framework::Executor::Prepare(startup_program, 0);
    exe.RunPreparedContext(ctx.get(), &param_scope, false, true, false);
  }

  // Out = mul_op(X, W)
  paddle::framework::ProgramDesc main_program;
  {
    auto& global_block = *main_program.MutableBlock(0);
    auto& w = *global_block.Var("W");
    w.SetType(paddle::framework::proto::VarType::LOD_TENSOR);

    auto& x = *global_block.Var("X");
    x.SetType(paddle::framework::proto::VarType::LOD_TENSOR);

    auto& y = *global_block.Var("Y");
    x.SetType(paddle::framework::proto::VarType::LOD_TENSOR);

    auto& op = *global_block.AppendOp();
    op.SetType("mul");
    op.SetInput("X", {x.Name()});
    op.SetInput("Y", {w.Name()});
    op.SetOutput("Out", {y.Name()});
  }
  std::mutex start_flag_mtx;
  bool start_flag = false;
  std::condition_variable start_cv;

  auto thread_main = [&] {
    {
      std::unique_lock<std::mutex> lock(start_flag_mtx);
      start_cv.wait(lock, [&] { return start_flag; });
    }
    paddle::platform::CPUPlace cpu;
    paddle::framework::Executor exec(cpu);
    auto& working_scope = param_scope.NewScope();
    working_scope.Var("X")->GetMutable<paddle::framework::LoDTensor>();
    working_scope.Var("Y")->GetMutable<paddle::framework::LoDTensor>();

    {
      // initialize x
      paddle::framework::OpRegistry::CreateOp(
          "uniform_random", {}, {{"Out", {"X"}}},
          {{"shape", std::vector<int>{64, 2000}},
           {"dtype",
            static_cast<int>(paddle::framework::proto::VarType::FP32)}})
          ->Run(working_scope, cpu);
    }
    auto ctx = exec.Prepare(main_program, 0);

    for (size_t i = 0; i < (1U << 15); ++i) {
      exec.RunPreparedContext(ctx.get(), &working_scope, false, false, false);
    }
    param_scope.DeleteScope(&working_scope);
  };
  int cpu_count = std::thread::hardware_concurrency();
  std::unique_ptr<std::chrono::nanoseconds[]> times(
      new std::chrono::nanoseconds[cpu_count]);

  for (int i = 0; i < cpu_count; ++i) {
    std::vector<std::thread> threads;
    auto begin = std::chrono::high_resolution_clock::now();
    for (int j = 0; j < i + 1; ++j) {
      threads.emplace_back(thread_main);
    }

    {
      std::unique_lock<std::mutex> lock(start_flag_mtx);
      start_flag = true;
    }

    start_cv.notify_all();

    for (auto& th : threads) {
      th.join();
    }
    threads.clear();
    auto end = std::chrono::high_resolution_clock::now();
    times[i] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    start_flag = false;
  }

  std::unique_ptr<float[]> speed_up_ratio(new float[cpu_count]);
  speed_up_ratio[0] = 1.0f;  // speed_up_ratio[0] is meaning less.
  for (int i = 1; i < cpu_count; ++i) {
    speed_up_ratio[i] = static_cast<float>(
        (static_cast<double>(i + 1) / static_cast<double>(times[i].count())) /
        (1 / static_cast<double>(times[0].count())));
  }
  std::cerr << "speed up ratio:\n";
  for (int i = 0; i < cpu_count; ++i) {
    std::cerr << speed_up_ratio[i] << " ";
  }
  std::cerr << "\n";
}

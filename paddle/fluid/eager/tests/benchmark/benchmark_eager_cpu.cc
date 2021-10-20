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

// Eager Dygraph

#include <chrono>

#include "gtest/gtest.h"

#include "paddle/fluid/eager/api/api.h"
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/backward.h"

#include "paddle/fluid/imperative/tracer.h"

#include "paddle/fluid/eager/tests/benchmark/benchmark_utils.h"
#include "paddle/fluid/eager/tests/test_utils.h"

#include "gperftools/profiler.h"

// TODO(jiabin): remove nolint here!!!
using namespace egr;  // NOLINT

TEST(Benchmark, EagerScalePerformance) {
  // Prepare Device Contexts
  egr::InitEnv(paddle::platform::CPUPlace());

  for (const std::string& mode : {"Accuracy", "Performance"}) {
    paddle::framework::DDim ddim = paddle::framework::make_ddim({2, 4, 4, 4});
    paddle::experimental::Tensor tensor = EagerUtils::CreateTensorWithValue(
        ddim, pten::Backend::kCPU, pten::DataType::kFLOAT32,
        pten::DataLayout::kNCHW, 5.0, true);
    RetainGradForTensor(tensor);

    if (mode == "Accuracy") {
      benchmark_eager_scale_accuracy_check(tensor);

    } else if (mode == "Performance") {
      auto t_start = std::chrono::high_resolution_clock::now();
      ProfilerStart("eager_scale_cpu.out");

      benchmark_eager_scale(tensor);

      ProfilerStop();
      auto t_end = std::chrono::high_resolution_clock::now();
      double elapsed_time_ms =
          std::chrono::duration<double, std::milli>(t_end - t_start).count();

      std::cout << "Duration: " << elapsed_time_ms << " ms" << std::endl;

    } else {
      PADDLE_THROW(paddle::platform::errors::Fatal("Unknown benchmark mode"));
    }
  }
}

TEST(Benchmark, EagerIntermediateMatmulPerformance) {
  // Prepare Device Contexts
  InitEnv(paddle::platform::CPUPlace());

  auto tracer = std::make_shared<paddle::imperative::Tracer>();
  paddle::imperative::SetCurrentTracer(tracer);

  for (const std::string& mode : {"Accuracy", "Performance"}) {
    paddle::framework::DDim ddimX = paddle::framework::make_ddim({2, 2});
    paddle::experimental::Tensor X = EagerUtils::CreateTensorWithValue(
        ddimX, pten::Backend::kCPU, pten::DataType::kFLOAT32,
        pten::DataLayout::kNCHW, 1.0, true);
    RetainGradForTensor(X);

    paddle::framework::DDim ddimY = paddle::framework::make_ddim({2, 2});
    paddle::experimental::Tensor Y = EagerUtils::CreateTensorWithValue(
        ddimY, pten::Backend::kCPU, pten::DataType::kFLOAT32,
        pten::DataLayout::kNCHW, 2.0, true);
    RetainGradForTensor(Y);

    if (mode == "Accuracy") {
      benchmark_eager_intermediate_matmul_accuracy_check(X, Y);

    } else if (mode == "Performance") {
      auto t_start = std::chrono::high_resolution_clock::now();
      ProfilerStart("eager_intermediate_matmul_cpu.out");

      benchmark_eager_intermediate_matmul(X, Y);

      ProfilerStop();
      auto t_end = std::chrono::high_resolution_clock::now();
      double elapsed_time_ms =
          std::chrono::duration<double, std::milli>(t_end - t_start).count();
      std::cout << "Duration: " << elapsed_time_ms << " ms" << std::endl;

    } else {
      PADDLE_THROW(paddle::platform::errors::Fatal("Unknown benchmark mode"));
    }
  }
}

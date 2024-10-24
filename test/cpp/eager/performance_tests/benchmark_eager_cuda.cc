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
#include <paddle/fluid/framework/op_registry.h>

#include <chrono>

#include "gtest/gtest.h"
#include "paddle/common/flags.h"
#include "paddle/fluid/eager/api/all.h"
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/backward.h"
#include "paddle/fluid/imperative/tracer.h"
#include "test/cpp/eager/performance_tests/benchmark_utils.h"
#include "test/cpp/eager/test_utils.h"

#ifdef WITH_GPERFTOOLS
#include "gperftools/profiler.h"
#endif

#include "paddle/phi/core/kernel_registry.h"

using namespace egr;            // NOLINT
using namespace egr_utils_api;  // NOLINT

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

TEST(Benchmark, EagerScaleCUDA) {
  eager_test::InitEnv(phi::GPUPlace());

  for (const std::string mode : {"Accuracy", "WarmUp", "Performance"}) {
    phi::DDim ddim = common::make_ddim({2, 4, 4, 4});
    paddle::Tensor tensor =
        eager_test::CreateTensorWithValue(ddim,
                                          phi::GPUPlace(),
                                          phi::DataType::FLOAT32,
                                          phi::DataLayout::NCHW,
                                          5.0 /*value*/,
                                          true /*is_leaf*/);
    RetainGradForTensor(tensor);

    if (mode == "Accuracy") {
      benchmark_eager_scale(tensor, true /* accuracy_check */);

    } else if (mode == "WarmUp") {
      benchmark_eager_scale(tensor);

    } else if (mode == "Performance") {
      auto t_start = std::chrono::high_resolution_clock::now();
#ifdef WITH_GPERFTOOLS
      ProfilerStart("eager_scale_cuda.out");
#endif
      benchmark_eager_scale(tensor);

#ifdef WITH_GPERFTOOLS
      ProfilerStop();
#endif
      auto t_end = std::chrono::high_resolution_clock::now();
      double elapsed_time_ms =
          std::chrono::duration<double, std::milli>(t_end - t_start).count();
      std::cout << "Duration: " << elapsed_time_ms << " ms" << std::endl;

    } else {
      PADDLE_THROW(common::errors::Fatal("Unknown benchmark mode"));
    }
  }
}

TEST(Benchmark, EagerMatmulCUDA) {
  phi::GPUPlace place;
  eager_test::InitEnv(place);

  for (const std::string mode : {"Accuracy", "WarmUp", "Performance"}) {
    phi::DDim ddimX = common::make_ddim({2, 2});
    paddle::Tensor X = eager_test::CreateTensorWithValue(ddimX,
                                                         phi::GPUPlace(),
                                                         phi::DataType::FLOAT32,
                                                         phi::DataLayout::NCHW,
                                                         1.0,
                                                         true);
    RetainGradForTensor(X);

    phi::DDim ddimY = common::make_ddim({2, 2});
    paddle::Tensor Y = eager_test::CreateTensorWithValue(ddimY,
                                                         phi::GPUPlace(),
                                                         phi::DataType::FLOAT32,
                                                         phi::DataLayout::NCHW,
                                                         2.0,
                                                         true);
    RetainGradForTensor(Y);

    if (mode == "Accuracy") {
      benchmark_eager_matmul(X, Y, true /* accuracy_check */);

    } else if (mode == "WarmUp") {
      benchmark_eager_matmul(X, Y);

    } else if (mode == "Performance") {
      auto t_start = std::chrono::high_resolution_clock::now();
#ifdef WITH_GPERFTOOLS
      ProfilerStart("eager_matmul_cuda.out");
#endif
      benchmark_eager_matmul(X, Y);

#ifdef WITH_GPERFTOOLS
      ProfilerStop();
#endif
      auto t_end = std::chrono::high_resolution_clock::now();
      double elapsed_time_ms =
          std::chrono::duration<double, std::milli>(t_end - t_start).count();
      std::cout << "Duration: " << elapsed_time_ms << " ms" << std::endl;

    } else {
      PADDLE_THROW(common::errors::Fatal("Unknown benchmark mode"));
    }
  }
}

TEST(Benchmark, EagerIntermediateMatmulCUDA) {
  phi::GPUPlace place;
  eager_test::InitEnv(place);

  auto tracer = std::make_shared<paddle::imperative::Tracer>();
  tracer->SetExpectedPlace(place);
  paddle::imperative::SetCurrentTracer(tracer);

  for (const std::string mode : {"Accuracy", "WarmUp", "Performance"}) {
    phi::DDim ddimX = common::make_ddim({2, 2});
    paddle::Tensor X = eager_test::CreateTensorWithValue(ddimX,
                                                         phi::GPUPlace(),
                                                         phi::DataType::FLOAT32,
                                                         phi::DataLayout::NCHW,
                                                         1.0,
                                                         true);
    RetainGradForTensor(X);

    phi::DDim ddimY = common::make_ddim({2, 2});
    paddle::Tensor Y = eager_test::CreateTensorWithValue(ddimY,
                                                         phi::GPUPlace(),
                                                         phi::DataType::FLOAT32,
                                                         phi::DataLayout::NCHW,
                                                         2.0,
                                                         true);
    RetainGradForTensor(Y);

    if (mode == "Accuracy") {
      benchmark_eager_intermediate_matmul(X, Y, true /* accuracy_check */);

    } else if (mode == "WarmUp") {
      benchmark_eager_intermediate_matmul(X, Y);

    } else if (mode == "Performance") {
      auto t_start = std::chrono::high_resolution_clock::now();
#ifdef WITH_GPERFTOOLS
      ProfilerStart("eager_intermediate_matmul_cuda.out");
#endif
      benchmark_eager_intermediate_matmul(X, Y);

#ifdef WITH_GPERFTOOLS
      ProfilerStop();
#endif
      auto t_end = std::chrono::high_resolution_clock::now();
      double elapsed_time_ms =
          std::chrono::duration<double, std::milli>(t_end - t_start).count();
      std::cout << "Duration: " << elapsed_time_ms << " ms" << std::endl;

    } else {
      PADDLE_THROW(common::errors::Fatal("Unknown benchmark mode"));
    }
  }
}

TEST(Benchmark, EagerIntermediateMLPCUDA) {
  phi::GPUPlace place;
  eager_test::InitEnv(place);

  auto tracer = std::make_shared<paddle::imperative::Tracer>();
  tracer->SetExpectedPlace(place);
  paddle::imperative::SetCurrentTracer(tracer);

  for (const std::string mode : {"Accuracy", "WarmUp", "Performance"}) {
    phi::DDim ddimX = common::make_ddim({MLP_M, MLP_N});
    paddle::Tensor X = eager_test::CreateTensorWithValue(ddimX,
                                                         phi::GPUPlace(),
                                                         phi::DataType::FLOAT32,
                                                         phi::DataLayout::NCHW,
                                                         MLP_X_VAL,
                                                         true);
    RetainGradForTensor(X);

    std::vector<paddle::Tensor> Ws;
    std::vector<paddle::Tensor> Bs;
    for (size_t i = 0; i < MLP_NUM_LINEAR; i++) {
      phi::DDim ddimW = common::make_ddim({MLP_N, MLP_K});
      paddle::Tensor W =
          eager_test::CreateTensorWithValue(ddimW,
                                            phi::GPUPlace(),
                                            phi::DataType::FLOAT32,
                                            phi::DataLayout::NCHW,
                                            MLP_W_VAL,
                                            true);
      RetainGradForTensor(W);

      phi::DDim ddimB = common::make_ddim({MLP_K});
      paddle::Tensor B =
          eager_test::CreateTensorWithValue(ddimB,
                                            phi::GPUPlace(),
                                            phi::DataType::FLOAT32,
                                            phi::DataLayout::NCHW,
                                            MLP_B_VAL,
                                            true);
      RetainGradForTensor(B);

      Ws.emplace_back(std::move(W));
      Bs.emplace_back(std::move(B));
    }

    if (mode == "Accuracy") {
      benchmark_eager_intermediate_mlp(X, Ws, Bs, true /* accuracy_check */);

    } else if (mode == "WarmUp") {
      benchmark_eager_intermediate_mlp(X, Ws, Bs);

    } else if (mode == "Performance") {
      auto t_start = std::chrono::high_resolution_clock::now();
#ifdef WITH_GPERFTOOLS
      ProfilerStart("eager_intermediate_mlp_cuda.out");
#endif
      benchmark_eager_intermediate_mlp(X, Ws, Bs);

#ifdef WITH_GPERFTOOLS
      ProfilerStop();
#endif
      auto t_end = std::chrono::high_resolution_clock::now();
      double elapsed_time_ms =
          std::chrono::duration<double, std::milli>(t_end - t_start).count();
      std::cout << "Duration: " << elapsed_time_ms << " ms" << std::endl;

    } else {
      PADDLE_THROW(common::errors::Fatal("Unknown benchmark mode"));
    }
  }
}

#endif  // PADDLE_WITH_CUDA || PADDLE_WITH_HIP

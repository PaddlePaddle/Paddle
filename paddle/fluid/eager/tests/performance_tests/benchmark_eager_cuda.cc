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
#include "paddle/fluid/platform/flags.h"

#include "paddle/fluid/eager/api/all.h"
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/backward.h"

#include "paddle/fluid/imperative/tracer.h"

#include "paddle/fluid/eager/tests/performance_tests/benchmark_utils.h"
#include "paddle/fluid/eager/tests/test_utils.h"

#ifdef WITH_GPERFTOOLS
#include "gperftools/profiler.h"
#endif

#include "paddle/phi/core/kernel_registry.h"

using namespace egr;            // NOLINT
using namespace egr_utils_api;  // NOLINT

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

PD_DECLARE_KERNEL(full, GPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(matmul, GPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(matmul_grad, GPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(add, KPS, ALL_LAYOUT);
PD_DECLARE_KERNEL(add_grad, GPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(sum, GPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(sum_grad, GPU, ALL_LAYOUT);

TEST(Benchmark, EagerScaleCUDA) {
  eager_test::InitEnv(paddle::platform::CUDAPlace());

  for (const std::string& mode : {"Accuracy", "WarmUp", "Performance"}) {
    paddle::framework::DDim ddim = phi::make_ddim({2, 4, 4, 4});
    paddle::experimental::Tensor tensor = CreateTensorWithValue(
        ddim, paddle::platform::CUDAPlace(), phi::DataType::FLOAT32,
        phi::DataLayout::NCHW, 5.0 /*value*/, true /*is_leaf*/);
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
      PADDLE_THROW(paddle::platform::errors::Fatal("Unknown benchmark mode"));
    }
  }
}

TEST(Benchmark, EagerMatmulCUDA) {
  paddle::platform::CUDAPlace place;
  eager_test::InitEnv(place);

  for (const std::string& mode : {"Accuracy", "WarmUp", "Performance"}) {
    paddle::framework::DDim ddimX = phi::make_ddim({2, 2});
    paddle::experimental::Tensor X = CreateTensorWithValue(
        ddimX, paddle::platform::CUDAPlace(), phi::DataType::FLOAT32,
        phi::DataLayout::NCHW, 1.0, true);
    RetainGradForTensor(X);

    paddle::framework::DDim ddimY = phi::make_ddim({2, 2});
    paddle::experimental::Tensor Y = CreateTensorWithValue(
        ddimY, paddle::platform::CUDAPlace(), phi::DataType::FLOAT32,
        phi::DataLayout::NCHW, 2.0, true);
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
      PADDLE_THROW(paddle::platform::errors::Fatal("Unknown benchmark mode"));
    }
  }
}

TEST(Benchmark, EagerIntermediateMatmulCUDA) {
  paddle::platform::CUDAPlace place;
  eager_test::InitEnv(place);

  auto tracer = std::make_shared<paddle::imperative::Tracer>();
  tracer->SetExpectedPlace(place);
  paddle::imperative::SetCurrentTracer(tracer);

  for (const std::string& mode : {"Accuracy", "WarmUp", "Performance"}) {
    paddle::framework::DDim ddimX = phi::make_ddim({2, 2});
    paddle::experimental::Tensor X = CreateTensorWithValue(
        ddimX, paddle::platform::CUDAPlace(), phi::DataType::FLOAT32,
        phi::DataLayout::NCHW, 1.0, true);
    RetainGradForTensor(X);

    paddle::framework::DDim ddimY = phi::make_ddim({2, 2});
    paddle::experimental::Tensor Y = CreateTensorWithValue(
        ddimY, paddle::platform::CUDAPlace(), phi::DataType::FLOAT32,
        phi::DataLayout::NCHW, 2.0, true);
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
      PADDLE_THROW(paddle::platform::errors::Fatal("Unknown benchmark mode"));
    }
  }
}

TEST(Benchmark, EagerIntermediateMLPCUDA) {
  paddle::platform::CUDAPlace place;
  eager_test::InitEnv(place);

  auto tracer = std::make_shared<paddle::imperative::Tracer>();
  tracer->SetExpectedPlace(place);
  paddle::imperative::SetCurrentTracer(tracer);

  for (const std::string& mode : {"Accuracy", "WarmUp", "Performance"}) {
    paddle::framework::DDim ddimX = phi::make_ddim({MLP_M, MLP_N});
    paddle::experimental::Tensor X = CreateTensorWithValue(
        ddimX, paddle::platform::CUDAPlace(), phi::DataType::FLOAT32,
        phi::DataLayout::NCHW, MLP_X_VAL, true);
    RetainGradForTensor(X);

    std::vector<paddle::experimental::Tensor> Ws;
    std::vector<paddle::experimental::Tensor> Bs;
    for (size_t i = 0; i < MLP_NUM_LINEAR; i++) {
      paddle::framework::DDim ddimW = phi::make_ddim({MLP_N, MLP_K});
      paddle::experimental::Tensor W = CreateTensorWithValue(
          ddimW, paddle::platform::CUDAPlace(), phi::DataType::FLOAT32,
          phi::DataLayout::NCHW, MLP_W_VAL, true);
      RetainGradForTensor(W);

      paddle::framework::DDim ddimB = phi::make_ddim({MLP_K});
      paddle::experimental::Tensor B = CreateTensorWithValue(
          ddimB, paddle::platform::CUDAPlace(), phi::DataType::FLOAT32,
          phi::DataLayout::NCHW, MLP_B_VAL, true);
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
      PADDLE_THROW(paddle::platform::errors::Fatal("Unknown benchmark mode"));
    }
  }
}

USE_OP_ITSELF(scale);
USE_OP_ITSELF(matmul_v2);
USE_OP_ITSELF(reduce_sum);
USE_OP_ITSELF(reduce_sum_grad);
USE_OP_ITSELF(elementwise_add);

#endif  // PADDLE_WITH_CUDA || PADDLE_WITH_HIP

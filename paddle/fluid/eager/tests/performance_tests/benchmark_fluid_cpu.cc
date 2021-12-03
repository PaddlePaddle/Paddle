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

#include <paddle/fluid/framework/op_registry.h>

#include <chrono>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "glog/logging.h"
#include "gtest/gtest.h"

#include "paddle/fluid/eager/tests/performance_tests/benchmark_utils.h"
#include "paddle/fluid/eager/tests/test_utils.h"
#include "paddle/fluid/imperative/basic_engine.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/fluid/memory/memcpy.h"

#ifdef WITH_GPERFTOOLS
#include "gperftools/profiler.h"
#endif

// Disable pten path
DECLARE_bool(run_pten_kernel);

TEST(Benchmark, Init) { FLAGS_run_pten_kernel = false; }

namespace paddle {
namespace imperative {

TEST(Benchmark, FluidScaleCPU) {
  // Prepare Device Contexts
  platform::CPUPlace place;
  eager_test::InitEnv(place);

  for (const std::string& mode : {"Accuracy", "Performance"}) {
    std::shared_ptr<imperative::VarBase> X(new imperative::VarBase(true, "X"));
    X->SetOverridedStopGradient(false);

    std::vector<float> src_data(128, 5.0);
    std::vector<int64_t> dims = {2, 4, 4, 4};

    auto* x_tensor = X->MutableVar()->GetMutable<framework::LoDTensor>();
    x_tensor->Resize(framework::make_ddim(dims));
    auto* mutable_x = x_tensor->mutable_data<float>(place);
    paddle::memory::Copy(place, mutable_x, place, src_data.data(),
                         sizeof(float) * src_data.size());

    if (mode == "Accuracy") {
      benchmark_fluid_scale(X, platform::Place(place),
                            true /* accuracy_check */);

    } else if (mode == "Performance") {
      auto t_start = std::chrono::high_resolution_clock::now();
#ifdef WITH_GPERFTOOLS
      ProfilerStart("fluid_scale_cpu.out");
#endif
      benchmark_fluid_scale(X, platform::Place(place));

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

TEST(Benchmark, FluidMatmulCPU) {
  // Prepare Device Contexts
  platform::CPUPlace place;
  eager_test::InitEnv(place);

  for (const std::string& mode : {"Accuracy", "Performance"}) {
    std::shared_ptr<imperative::VarBase> X(new imperative::VarBase(true, "X"));
    X->SetOverridedStopGradient(false);
    std::shared_ptr<imperative::VarBase> Y(new imperative::VarBase(true, "Y"));
    Y->SetOverridedStopGradient(false);

    std::vector<float> x_src_data(4, 1.0);
    std::vector<float> y_src_data(4, 2.0);
    std::vector<int64_t> dims = {2, 2};

    auto* x_tensor = X->MutableVar()->GetMutable<framework::LoDTensor>();
    x_tensor->Resize(framework::make_ddim(dims));
    auto* mutable_x = x_tensor->mutable_data<float>(place);
    paddle::memory::Copy(place, mutable_x, place, x_src_data.data(),
                         sizeof(float) * x_src_data.size());

    auto* y_tensor = Y->MutableVar()->GetMutable<framework::LoDTensor>();
    y_tensor->Resize(framework::make_ddim(dims));
    auto* mutable_y = y_tensor->mutable_data<float>(place);
    paddle::memory::Copy(place, mutable_y, place, y_src_data.data(),
                         sizeof(float) * y_src_data.size());

    if (mode == "Accuracy") {
      benchmark_fluid_matmul(X, Y, platform::Place(place),
                             true /* accuracy_check */);

    } else if (mode == "Performance") {
      auto t_start = std::chrono::high_resolution_clock::now();
#ifdef WITH_GPERFTOOLS
      ProfilerStart("fluid_matmul_cpu.out");
#endif
      benchmark_fluid_matmul(X, Y, platform::Place(place));

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

TEST(Benchmark, FluidMLPCPU) {
  // Prepare Device Contexts
  platform::CPUPlace place;
  eager_test::InitEnv(place);

  for (const std::string& mode : {"Accuracy", "Performance"}) {
    std::vector<float> x_src_data(MLP_M * MLP_N, MLP_X_VAL);
    std::vector<float> w_src_data(MLP_N * MLP_K, MLP_W_VAL);
    std::vector<float> b_src_data(MLP_K, MLP_B_VAL);

    std::vector<int64_t> x_dims = {MLP_M, MLP_N};
    std::vector<int64_t> w_dims = {MLP_N, MLP_K};
    std::vector<int64_t> b_dims = {MLP_K};

    std::shared_ptr<imperative::VarBase> X(new imperative::VarBase(true, "X"));
    X->SetOverridedStopGradient(false);

    auto* x_tensor = X->MutableVar()->GetMutable<framework::LoDTensor>();
    x_tensor->Resize(framework::make_ddim(x_dims));
    auto* mutable_x = x_tensor->mutable_data<float>(place);
    paddle::memory::Copy(place, mutable_x, place, x_src_data.data(),
                         sizeof(float) * x_src_data.size());

    std::vector<std::shared_ptr<imperative::VarBase>> Ws;
    std::vector<std::shared_ptr<imperative::VarBase>> Bs;
    for (size_t i = 0; i < MLP_NUM_LINEAR; i++) {
      std::shared_ptr<imperative::VarBase> W(
          new imperative::VarBase(true, "W"));
      W->SetOverridedStopGradient(false);
      std::shared_ptr<imperative::VarBase> B(
          new imperative::VarBase(true, "B"));
      B->SetOverridedStopGradient(false);

      auto* w_tensor = W->MutableVar()->GetMutable<framework::LoDTensor>();
      w_tensor->Resize(framework::make_ddim(w_dims));
      auto* mutable_w = w_tensor->mutable_data<float>(place);
      paddle::memory::Copy(place, mutable_w, place, w_src_data.data(),
                           sizeof(float) * w_src_data.size());

      auto* b_tensor = B->MutableVar()->GetMutable<framework::LoDTensor>();
      b_tensor->Resize(framework::make_ddim(b_dims));
      auto* mutable_b = b_tensor->mutable_data<float>(place);
      paddle::memory::Copy(place, mutable_b, place, b_src_data.data(),
                           sizeof(float) * b_src_data.size());

      Ws.emplace_back(std::move(W));
      Bs.emplace_back(std::move(B));
    }

    if (mode == "Accuracy") {
      benchmark_fluid_mlp(X, Ws, Bs, platform::Place(place),
                          true /* accuracy_check */);

    } else if (mode == "Performance") {
      auto t_start = std::chrono::high_resolution_clock::now();
#ifdef WITH_GPERFTOOLS
      ProfilerStart("fluid_mlp_cpu.out");
#endif
      benchmark_fluid_mlp(X, Ws, Bs, platform::Place(place));

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

}  // namespace imperative
}  // namespace paddle

USE_OP(scale);
USE_OP(elementwise_add);
USE_OP(matmul_v2);
USE_OP(reduce_sum);

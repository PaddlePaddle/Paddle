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

#include "paddle/fluid/eager/tests/benchmark/benchmark_utils.h"
#include "paddle/fluid/eager/tests/test_utils.h"
#include "paddle/fluid/imperative/basic_engine.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/fluid/memory/memcpy.h"

#ifdef WITH_GPERFTOOLS
#include "gperftools/profiler.h"
#endif

namespace paddle {
namespace imperative {

TEST(Benchmark, FluidScalePerformance) {
  // Prepare Device Contexts
  platform::CUDAPlace place;
  egr::InitEnv(place);

  for (const std::string& mode : {"Accuracy", "WarmUp", "Performance"}) {
    std::shared_ptr<imperative::VarBase> X(new imperative::VarBase(true, "X"));
    X->SetOverridedStopGradient(false);

    std::shared_ptr<imperative::VarBase> Out(
        new imperative::VarBase(true, "Out"));
    std::vector<float> src_data(128, 5.0);
    std::vector<int64_t> dims = {2, 4, 4, 4};

    auto* x_tensor = X->MutableVar()->GetMutable<framework::LoDTensor>();
    x_tensor->Resize(framework::make_ddim(dims));
    auto* mutable_x = x_tensor->mutable_data<float>(place);

    paddle::platform::DeviceContextPool& pool =
        paddle::platform::DeviceContextPool::Instance();
    auto* dev_ctx =
        dynamic_cast<paddle::platform::CUDADeviceContext*>(pool.Get(place));
    auto stream = dev_ctx->stream();
    paddle::memory::Copy(place, mutable_x, platform::CPUPlace(),
                         src_data.data(), sizeof(float) * src_data.size(),
                         stream);

    if (mode == "Accuracy") {
      benchmark_fluid_scale_accuracy_check(X, Out, platform::Place(place));

    } else if (mode == "WarmUp") {
      benchmark_fluid_scale(X, Out, platform::Place(place));

    } else if (mode == "Performance") {
      auto t_start = std::chrono::high_resolution_clock::now();
#ifdef WITH_GPERFTOOLS
      ProfilerStart("fluid_scale_cuda.out");
#endif
      benchmark_fluid_scale(X, Out, platform::Place(place));
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

TEST(Benchmark, FluidMatmulPerformance) {
  // Prepare Device Contexts
  platform::CUDAPlace place;
  egr::InitEnv(place);

  for (const std::string& mode : {"Accuracy", "WarmUp", "Performance"}) {
    std::shared_ptr<imperative::VarBase> X(new imperative::VarBase(true, "X"));
    X->SetOverridedStopGradient(false);
    std::shared_ptr<imperative::VarBase> Y(new imperative::VarBase(true, "Y"));
    Y->SetOverridedStopGradient(false);

    std::shared_ptr<imperative::VarBase> Out(
        new imperative::VarBase(true, "Out"));
    std::vector<float> x_src_data(4, 1.0);
    std::vector<float> y_src_data(4, 2.0);
    std::vector<int64_t> dims = {2, 2};

    paddle::platform::DeviceContextPool& pool =
        paddle::platform::DeviceContextPool::Instance();
    auto* dev_ctx =
        dynamic_cast<paddle::platform::CUDADeviceContext*>(pool.Get(place));
    auto stream = dev_ctx->stream();

    auto* x_tensor = X->MutableVar()->GetMutable<framework::LoDTensor>();
    x_tensor->Resize(framework::make_ddim(dims));
    auto* mutable_x = x_tensor->mutable_data<float>(place);
    paddle::memory::Copy(place, mutable_x, platform::CPUPlace(),
                         x_src_data.data(), sizeof(float) * x_src_data.size(),
                         stream);

    auto* y_tensor = Y->MutableVar()->GetMutable<framework::LoDTensor>();
    y_tensor->Resize(framework::make_ddim(dims));
    auto* mutable_y = y_tensor->mutable_data<float>(place);
    paddle::memory::Copy(place, mutable_y, platform::CPUPlace(),
                         y_src_data.data(), sizeof(float) * y_src_data.size(),
                         stream);

    if (mode == "Accuracy") {
      benchmark_fluid_matmul_accuracy_check(X, Y, Out, platform::Place(place));

    } else if (mode == "WarmUp") {
      benchmark_fluid_matmul(X, Y, Out, platform::Place(place));

    } else if (mode == "Performance") {
      auto t_start = std::chrono::high_resolution_clock::now();
#ifdef WITH_GPERFTOOLS
      ProfilerStart("fluid_matmul_cuda.out");
#endif
      benchmark_fluid_matmul(X, Y, Out, platform::Place(place));
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
USE_OP(matmul_v2);

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

#include <memory>
#include <set>
#include <string>
#include <vector>
#include <iostream>

#include "gtest/gtest.h"
#include "glog/logging.h"

#include "paddle/fluid/eager/tests/test_utils.h"
#include "paddle/fluid/imperative/basic_engine.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/eager/tests/benchmark/benchmark_utils.h"

#include "gperftools/profiler.h"

#include <chrono>

using namespace paddle;
using namespace imperative;

TEST(Benchmark, FluidAccuracy) {
    // Prepare Device Contexts
    egr::InitEnv(paddle::platform::CPUPlace());
    
    std::shared_ptr<imperative::VarBase> X(new imperative::VarBase(true, "X"));
    X->SetOverridedStopGradient(false);

    std::shared_ptr<imperative::VarBase> Out(new imperative::VarBase(true, "Out"));
    std::vector<float> src_data(128, 5.0);
    std::vector<int64_t> dims = {2,4,4,4};
    platform::CPUPlace place;
  
    auto* x_tensor = X->MutableVar()->GetMutable<framework::LoDTensor>();
    x_tensor->Resize(framework::make_ddim(dims));
    auto* mutable_x = x_tensor->mutable_data<float>(place);
    paddle::memory::Copy(place, mutable_x, place, src_data.data(),
                         sizeof(float) * src_data.size());

    benchmark_fluid_accuracy_check(X, Out);
}

TEST(Benchmark, FluidPerformance) {
    // Prepare Device Contexts
    egr::InitEnv(paddle::platform::CPUPlace());
    
    std::shared_ptr<imperative::VarBase> X(new imperative::VarBase(true, "X"));
    X->SetOverridedStopGradient(false);

    std::shared_ptr<imperative::VarBase> Out(new imperative::VarBase(true, "Out"));
    std::vector<float> src_data(128, 5.0);
    std::vector<int64_t> dims = {2,4,4,4};
    platform::CPUPlace place;
  
    auto* x_tensor = X->MutableVar()->GetMutable<framework::LoDTensor>();
    x_tensor->Resize(framework::make_ddim(dims));
    auto* mutable_x = x_tensor->mutable_data<float>(place);
    paddle::memory::Copy(place, mutable_x, place, src_data.data(),
                         sizeof(float) * src_data.size());

    auto t_start = std::chrono::high_resolution_clock::now();
    
    ProfilerStart("fluid_cpu.out");
    benchmark_fluid(X, Out);
    ProfilerStop();

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();

    std::cout << "Duration: " << elapsed_time_ms << " ms" << std::endl;
}

USE_OP(scale);


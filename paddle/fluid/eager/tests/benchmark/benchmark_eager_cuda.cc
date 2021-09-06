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
#include "gtest/gtest.h"

#include "paddle/fluid/eager/backward.h"
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/api/api.h"

#include "paddle/fluid/eager/tests/test_utils.h"
#include "paddle/fluid/eager/tests/benchmark/benchmark_utils.h"

#include "gperftools/profiler.h"

#include <chrono>

using namespace egr;

TEST(Benchmark, EagerAccuracy) {
    // Prepare Device Contexts
    InitEnv(paddle::platform::CUDAPlace());
    
    // 1. Prepare Input
    paddle::framework::DDim ddim = paddle::framework::make_ddim({2, 4, 4, 4});
    pt::Tensor tensor = EagerUtils::CreateTensorWithValue(ddim, pt::Backend::kCUDA,
                                                          pt::DataType::kFLOAT32, pt::DataLayout::kNCHW,
                                                          5.0 , true);
    RetainGradForTensor(tensor);
    
    benchmark_eager_accuracy_check(tensor);
}

TEST(Benchmark, EagerPerformance) {
    // Prepare Device Contexts
    InitEnv(paddle::platform::CUDAPlace());
  
    // Warm Up
    {
        paddle::framework::DDim ddim = paddle::framework::make_ddim({2, 4, 4, 4});
        pt::Tensor tensor = EagerUtils::CreateTensorWithValue(ddim, pt::Backend::kCUDA,
                                                              pt::DataType::kFLOAT32, pt::DataLayout::kNCHW,
                                                              5.0 /*value*/, true /*is_leaf*/);
        RetainGradForTensor(tensor);
        
        benchmark_eager(tensor);
    }

    // 1. Prepare Input
    paddle::framework::DDim ddim = paddle::framework::make_ddim({2, 4, 4, 4});
    pt::Tensor tensor = EagerUtils::CreateTensorWithValue(ddim, pt::Backend::kCUDA,
                                                          pt::DataType::kFLOAT32, pt::DataLayout::kNCHW,
                                                          5.0 /*value*/, true /*is_leaf*/);
    RetainGradForTensor(tensor);
    
    auto t_start = std::chrono::high_resolution_clock::now();
    
    ProfilerStart("eager_cuda.out");
    benchmark_eager(tensor);
    ProfilerStop();

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();
    
    std::cout << "Duration: " << elapsed_time_ms << " ms" << std::endl;
}

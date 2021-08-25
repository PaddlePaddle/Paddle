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

#include <chrono>

using namespace egr;

void benchmark_eager_accuracy_check(pt::Tensor& tensor) {
  // 2. Run Forward for certain number of times
  pt::Tensor input_tensor = tensor;
  float scale = 2.0;
  float bias = 3.0;

  size_t max_num_runs = 10;
  for(size_t i = 0; i < max_num_runs; i++) {
    input_tensor = egr::scale(input_tensor, scale, bias, true /*bias_after_scale*/, true /*trace_backward*/)[0];
  }
  
  // Examine Forward Output
  // CompareTensorWithValue<float>(out, 13.0);

  // 3. Run Backward
  std::vector<pt::Tensor> target_tensors = {input_tensor};
  RunBackward(target_tensors, {});
  
  // Examine Forward Grad (w.r.t max_num_runs = 10)
  PADDLE_ENFORCE(CompareTensorWithValue<float>(input_tensor, 8189.0) == true, 
      paddle::platform::errors::Fatal("Numerical Error, Expected %f", 8189.0));
  // Examine Backward Grad (w.r.t max_num_runs = 10)
  PADDLE_ENFORCE(CompareGradTensorWithValue<float>(tensor, 1024.0) == true, 
      paddle::platform::errors::Fatal("Numerical Error, Expected %f", 1024.0));
}

void benchmark_eager(pt::Tensor& tensor) {
  // 2. Run Forward for certain number of times
  pt::Tensor input_tensor = tensor;
  float scale = 2.0;
  float bias = 3.0;

  size_t max_num_runs = 500;
  for(size_t i = 0; i < max_num_runs; i++) {
    input_tensor = egr::scale(input_tensor, scale, bias, true /*bias_after_scale*/, true /*trace_backward*/)[0];
  }

  // 3. Run Backward
  std::vector<pt::Tensor> target_tensors = {input_tensor};
  RunBackward(target_tensors, {});
}

TEST(Benchmark, EagerAccuracy) {
    // 1. Prepare Input
    paddle::framework::DDim ddim = paddle::framework::make_ddim({2, 4, 4, 4});
    pt::Tensor tensor = EagerUtils::CreateTensorWithValue(ddim, pt::Backend::kCPU,
                                                          pt::DataType::kFLOAT32, pt::DataLayout::kNCHW,
                                                          5.0 /*value*/, true /*is_leaf*/);
    RetainGradForTensor(tensor);
    
    benchmark_eager_accuracy_check(tensor);
}

TEST(Benchmark, EagerPerformance) {
    // 1. Prepare Input
    paddle::framework::DDim ddim = paddle::framework::make_ddim({2, 4, 4, 4});
    pt::Tensor tensor = EagerUtils::CreateTensorWithValue(ddim, pt::Backend::kCPU,
                                                          pt::DataType::kFLOAT32, pt::DataLayout::kNCHW,
                                                          5.0 /*value*/, true /*is_leaf*/);
    RetainGradForTensor(tensor);
    
    auto t_start = std::chrono::high_resolution_clock::now();

    benchmark_eager(tensor);

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();
    
    VLOG(2) << "Duration: " << elapsed_time_ms << " ms";
}

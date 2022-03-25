/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <gtest/gtest.h>
#include <memory>

#include "paddle/phi/api/include/api.h"

#include "paddle/phi/api/lib/utils/allocator.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/tests/api/scale_api.h"
#include "paddle/phi/tests/core/timer.h"

PD_DECLARE_KERNEL(full, CPU, ALL_LAYOUT);

namespace paddle {
namespace tests {

TEST(API, scale) {
  auto x = experimental::full(
      {3, 4}, 1.0, experimental::DataType::FLOAT32, experimental::CPUPlace());

  const size_t cycles = 300;
  phi::tests::Timer timer;
  double t1{}, t2{}, t3{};

  for (size_t i = 0; i < cycles; ++i) {
    timer.tic();
    for (size_t i = 0; i < cycles; ++i) {
      auto out = experimental::scale_kernel_context(x, 2.0, 1.0, true);
    }
    t1 += timer.toc();

    timer.tic();
    for (size_t i = 0; i < cycles; ++i) {
      auto out = experimental::scale(x, 2.0, 1.0, true);
    }
    t2 += timer.toc();

    timer.tic();
    for (size_t i = 0; i < cycles; ++i) {
      auto out = experimental::scale_switch_case(x, 2.0, 1.0, true);
    }
    t3 += timer.toc();
  }

  LOG(INFO) << "The cost of kernel_context is " << t1 << "ms.";
  LOG(INFO) << "The cost of variadic_args_kernel_fn is " << t2 << "ms.";
  LOG(INFO) << "The cost of switch_case is " << t3 << "ms.";
}

}  // namespace tests
}  // namespace paddle

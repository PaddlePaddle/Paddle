// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <gtest/gtest.h>
#include "paddle/fluid/lite/api/paddle_use_passes.h"
#include "paddle/fluid/lite/core/optimizer.h"
#include "paddle/fluid/lite/core/program_fake_utils.h"
#include "paddle/fluid/lite/kernels/cuda/use_kernels.h"
#include "paddle/fluid/lite/kernels/host/use_kernels.h"

namespace paddle {
namespace lite {
namespace mir {

TEST(variable_place_inference_pass, test) {
  std::shared_ptr<Scope> scope(new lite::Scope);
  ProgramFaker program_faker;
  program_faker.AddFeed("a", 0);
  program_faker.AddMul("a", "W", "a1");
  program_faker.AddMul("a1", "W1", "a2");
  program_faker.AddFetch("a2", 0);
  program_faker.CreateVars(scope.get());

  auto* desc = program_faker.program();

  Optimizer optimizer;
  std::vector<Place> places({
      Place{
          TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW),
      },
      Place{
          TARGET(kCUDA), PRECISION(kFloat), DATALAYOUT(kNCHW),
      },
      Place{
          TARGET(kX86), PRECISION(kFloat), DATALAYOUT(kNCHW),
      },
      Place{
          TARGET(kX86), PRECISION(kAny), DATALAYOUT(kAny),
      },
  });

  Program program(*desc->Proto(), scope, places);

  core::KernelPickFactor factor;
  factor.ConsiderTarget();

  std::vector<std::string> passes({
      "static_kernel_pick_pass",        //
      "argument_type_display_pass",     //
      "variable_place_inference_pass",  //
      "argument_type_display_pass",     //
      "type_target_cast_pass",          //
  });

  Place prefered_place{
#ifdef PADDLE_WITH_CUDA
      TARGET(kCUDA), PRECISION(kFloat), DATALAYOUT(kNCHW),
#else
#ifdef PADDLE_WITH_ARM
      TARGET(kARM), PRECISION(kFloat), DATALAYOUT(kNCHW),
#else   // X86
      TARGET(kX86), PRECISION(kFloat), DATALAYOUT(kNCHW),
#endif  // ARM
#endif
  };
  optimizer.KernelPickPreferPlace(prefered_place);
  optimizer.Run(std::move(program), places, factor, passes);
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

USE_LITE_OP(mul);
USE_LITE_OP(feed);
USE_LITE_OP(fetch);
USE_LITE_OP(io_copy);

#ifdef LITE_WITH_X86
USE_LITE_KERNEL(mul, kX86, kFloat, kNCHW, def);
#endif

#ifdef LITE_WITH_ARM
USE_LITE_KERNEL(mul, kARM, kFloat, kNCHW, def);
#endif

#ifdef LITE_WITH_CUDA
USE_LITE_KERNEL(io_copy, kCUDA, kAny, kAny, host_to_device);
USE_LITE_KERNEL(io_copy, kCUDA, kAny, kAny, device_to_host);
#endif

//  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "paddle/phi/backends/cpu/cpu_info.h"

#include <sstream>

#include "gflags/gflags.h"
#include "gtest/gtest.h"
#include "paddle/fluid/string/printf.h"

DECLARE_double(fraction_of_cpu_memory_to_use);

TEST(CpuMemoryUsage, Print) {
  std::stringstream ss;
  size_t memory_size =
      phi::backends::cpu::CpuMaxAllocSize() / 1024 / 1024 / 1024;
  float use_percent = FLAGS_fraction_of_cpu_memory_to_use * 100;

  std::cout << paddle::string::Sprintf("\n%.2f %% of CPU Memory Usage: %d GB\n",
                                       use_percent,
                                       memory_size)
            << std::endl;
}

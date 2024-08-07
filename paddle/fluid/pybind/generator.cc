// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <iostream>
#include "paddle/fluid/pybind/eager_generator.h"
#include "paddle/fluid/pybind/eager_legacy_op_function_generator.h"

int main(int argc, char* argv[]) {
  if (argc == 2) {
    // make eager_legacy_op_function_generator.cc
    return run_legacy_generator(argc, argv);
  } else if (argc == 3) {
    // make eager_generator.cc
    return run_generator(argc, argv);
  } else {
    std::cerr << "argc must be 2 or 3" << std::endl;
    return -1;
  }

  return 0;
}

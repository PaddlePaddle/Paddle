// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/utils/random_engine.h"

namespace cinn {
namespace utils {

// Sample Integers from uniform distribution [min, max)
int SampleUniformInt(int min,
                     int max,
                     LinearRandomEngine::StateType* rand_seed) {
  PADDLE_ENFORCE_EQ(
      min < max,
      true,
      ::common::errors::InvalidArgument(
          "Input value error: min(%d) must be less than max(%d)", min, max));
  if (min + 1 == max) {
    return min;
  }

  LinearRandomEngine engine(rand_seed);
  std::uniform_int_distribution<> dist(min, max - 1);
  return dist(engine);
}

// Sample Real Numbers from uniform distribution [min, max)
double SampleUniformDouble(double min,
                           double max,
                           LinearRandomEngine::StateType* rand_seed) {
  PADDLE_ENFORCE_EQ(
      min < max,
      true,
      ::common::errors::InvalidArgument(
          "Input value error: min(%f) must be less than max(%f)", min, max));
  LinearRandomEngine engine(rand_seed);
  std::uniform_real_distribution<> dist(min, max);
  return dist(engine);
}

}  // namespace utils
}  // namespace cinn

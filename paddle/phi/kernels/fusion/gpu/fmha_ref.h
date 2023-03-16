// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include "paddle/phi/core/dense_tensor.h"

namespace phi {
namespace fusion {

class AttnDropoutParam {
 public:
  AttnDropoutParam() {
    is_test_ = false;
    dropout_implementation_ = "downgrade_in_infer";
    dropout_prob_ = 0.5;
    is_upscale_in_train_ = false;
    is_fix_seed_ = false;
    seed_val_ = 0;
    seed_ = nullptr;
  }
  AttnDropoutParam(bool is_test,
                   const std::string dropout_implementation,
                   float dropout_prob,
                   bool is_upscale_in_train,
                   bool is_fix_seed,
                   int seed_val,
                   const phi::DenseTensor* seed) {
    is_test_ = is_test;
    dropout_implementation_ = dropout_implementation;
    dropout_prob_ = dropout_prob;
    is_upscale_in_train_ = is_upscale_in_train;
    is_fix_seed_ = is_fix_seed;
    seed_val_ = seed_val;
    seed_ = seed;
  }
  bool is_test_;
  std::string dropout_implementation_;
  float dropout_prob_;
  bool is_upscale_in_train_;
  bool is_fix_seed_;
  int seed_val_;
  const phi::DenseTensor* seed_;
};

}  // namespace fusion
}  // namespace phi

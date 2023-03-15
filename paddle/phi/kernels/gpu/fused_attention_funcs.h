/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/fluid/framework/operator.h"

#include "paddle/phi/kernels/funcs/dropout_impl_util.h"
#include "paddle/phi/kernels/funcs/layer_norm_impl.cu.h"

#include "paddle/phi/backends/gpu/gpu_device_function.h"
#include "paddle/phi/backends/gpu/gpu_dnn.h"

namespace phi {

struct DropoutParam {
  uint64_t seed;
  float dropout_prob;
  bool is_upscale_in_train;
  bool is_test;
  bool fix_seed;
  int increment;
  const phi::DenseTensor* tensor_seed;
  int seed_val;

  DropoutParam() {
    fix_seed = false;
    seed = 0;
    is_test = false;
    is_upscale_in_train = false;
    dropout_prob = 0.5;
    tensor_seed = nullptr;
    seed_val = 0;
  }

  DropoutParam(bool fix_seed_,
               uint64_t seed_,
               bool is_test_,
               bool is_upscale_in_train_,
               float dropout_prob_,
               const phi::DenseTensor* tensor_seed_,
               int seed_val_) {
    fix_seed = fix_seed_;
    seed = seed_;
    is_test = is_test_;
    is_upscale_in_train = is_upscale_in_train_;
    dropout_prob = dropout_prob_;
    tensor_seed = tensor_seed_;
    seed_val = seed_val_;
  }

  /**
   * dropout_index: can be 0, 1, 2. 0 means there is only one dropout,
   * 1 and 2 represent two dropout, the parameter name of dropout
   * will be "dropout" + dropout_index + param name, such as dropout1_seed,
   * dropout1_is_test.
   */
  DropoutParam(const framework::ExecutionContext& context,
               const int dropout_index) {
    std::string pre_fix = "dropout";
    std::string str_index = std::to_string(dropout_index);
    if (dropout_index > 0) {
      pre_fix = pre_fix + str_index + "_";
    } else {
      pre_fix = pre_fix + "_";
    }
    dropout_prob = context.Attr<float>(pre_fix + "rate");
    auto& dropout_implementation =
        context.Attr<std::string>(pre_fix + "implementation");
    is_upscale_in_train = (dropout_implementation == "upscale_in_train");
    is_test = context.Attr<bool>("is_test");
    fix_seed = context.Attr<bool>(pre_fix + "fix_seed");

    std::string str_seed = "Dropout";
    if (dropout_index > 0) {
      str_seed = str_seed + str_index + "Seed";
    } else {
      str_seed = str_seed + "Seed";
    }
    tensor_seed = context.HasInput(str_seed)
                      ? context.Input<phi::DenseTensor>(str_seed)
                      : nullptr;
    seed_val = context.Attr<int>(pre_fix + "seed");
  }

  int UpdateSeedAndIncrement(const phi::GPUContext& ctx, const int offset) {
    uint64_t tmp_increment;
    phi::funcs::GetSeedDataAndIncrement(
        ctx, tensor_seed, fix_seed, seed_val, offset, &seed, &tmp_increment);
    increment = static_cast<int>(tmp_increment);
    return increment;
  }
};

}  // namespace phi

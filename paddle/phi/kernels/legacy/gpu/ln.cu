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

/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved. */

/*This code is copied from NVIDIA apex:
 *     https://github.com/NVIDIA/apex
 *     with minor changes. */

#pragma once
#include "paddle/phi/kernels/legacy/gpu/ln.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <unordered_map>

namespace layer_norm {
// Create registries and provide runtime versions of config hash functions.

FwdRegistry FWD_FUNCS;
BwdRegistry BWD_FUNCS;

uint32_t get_type_id(paddle::DataType dtype) {
  if (dtype == paddle::DataType::FLOAT16) {
    return TypeToIdTrait<fp16>::Value;
  } else if (dtype == paddle::DataType::BFLOAT16) {
    return TypeToIdTrait<bf16>::Value;
  } else if (dtype == paddle::DataType::FLOAT32) {
    return TypeToIdTrait<float>::Value;
  } else {
    PD_CHECK(false, "Type not supported: ", dtype);
  }
}

uint64_t get_key(paddle::DataType weight_type,
                 paddle::DataType input_type,
                 paddle::DataType output_type,
                 paddle::DataType compute_type,
                 uint64_t hidden_size) {
  uint64_t type_key =
      get_type_id(weight_type) | (get_type_id(input_type) << 2) |  // NOLINT
      (get_type_id(output_type) << 4) | (get_type_id(compute_type) << 6);
  uint64_t launcher_key = (type_key << 32) | hidden_size;
  return launcher_key;
}

}  // namespace layer_norm

namespace phi {
layer_norm::FwdFunction& get_fwd_launcher(paddle::DataType weight_type,
                                          paddle::DataType input_type,
                                          paddle::DataType output_type,
                                          paddle::DataType compute_type,
                                          uint32_t hidden_size) {
  auto iter = layer_norm::FWD_FUNCS.find(layer_norm::get_key(
      weight_type, input_type, output_type, compute_type, hidden_size));
  if (iter != layer_norm::FWD_FUNCS.end()) {
    return iter->second;
  } else {
    PD_CHECK(false,
             "FWD: Unsupported hidden_size or types: ",
             hidden_size,
             weight_type,
             input_type,
             output_type,
             compute_type);
  }
}

layer_norm::BwdFunction& get_bwd_launcher(paddle::DataType weight_type,
                                          paddle::DataType input_type,
                                          paddle::DataType output_type,
                                          paddle::DataType compute_type,
                                          uint32_t hidden_size) {
  auto iter = layer_norm::BWD_FUNCS.find(layer_norm::get_key(
      weight_type, input_type, output_type, compute_type, hidden_size));
  if (iter != layer_norm::BWD_FUNCS.end()) {
    return iter->second;
  } else {
    PD_CHECK(false,
             "BWD: Unsupported hidden_size or types: ",
             hidden_size,
             weight_type,
             input_type,
             output_type,
             compute_type);
  }
}

}  // namespace phi

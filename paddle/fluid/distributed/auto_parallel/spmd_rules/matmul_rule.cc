/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/fluid/distributed/auto_parallel/spmd_rules/matmul_rule.h"

namespace paddle {
namespace distributed {
namespace auto_parallel {

std::vector<DistTensorSpec> MatmulRule::InferForward(
    const std::vector<DistTensorSpec>& input_specs,
    const paddle::framework::AttributeMap& attrs) {
  // step0: verify input args based on matmul logic
  bool trans_x = attrs.Get<bool>("trans_x");
  bool trans_y = attrs.Get<bool>("trans_y");

  // step1: Einsum Notation
  // step1.1: generate base notations for each input tensor

  // step1.2: modify input notations base on matmul logic

  // step1.3: generate notations for each output tensor

  // step1.4: final Einsum Notaion

  // step1: Sharding Propogation
}

std::vector<DistTensorSpec> MatmulRule::InferBackward(
    const std::vector<DistTensorSpec>& output_specs,
    const paddle::framework::AttributeMap& attrs) {}

}  // namespace auto_parallel
}  // namespace distributed
}  // namespace paddle

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

#include <string>

#include "paddle/phi/core/distributed/auto_parallel/dist_meta_tensor.h"
#include "paddle/phi/core/distributed/type_defs.h"

namespace phi {
namespace distributed {

SpmdInfo FlashAttInferSpmd(const DistMetaTensor& q,
                           const DistMetaTensor& k,
                           const DistMetaTensor& v,
                           const DistMetaTensor& fixed_seed_offset,
                           const DistMetaTensor& attn_mask,
                           float dropout = 0.0,
                           bool causal = false,
                           bool return_softmax = false,
                           bool is_test = false,
                           const std::string& rng_name = "");

SpmdInfo FlashAttInferSpmdStatic(const DistMetaTensor& q,
                                 const DistMetaTensor& k,
                                 const DistMetaTensor& v,
                                 const DistMetaTensor& fixed_seed_offset,
                                 const DistMetaTensor& attn_mask,
                                 float dropout,
                                 bool causal,
                                 bool return_softmax,
                                 bool is_test);

SpmdInfo FlashAttInferSpmdReverse(const DistMetaTensor& q,
                                  const DistMetaTensor& k,
                                  const DistMetaTensor& v,
                                  const DistMetaTensor& fixed_seed_offset,
                                  const DistMetaTensor& attn_mask,
                                  const DistMetaTensor& out,
                                  const DistMetaTensor& softmax,
                                  const DistMetaTensor& softmax_lse,
                                  const DistMetaTensor& seed_offset,
                                  float dropout,
                                  bool causal,
                                  bool return_softmax,
                                  bool is_test);

SpmdInfo FlashAttGradInferSpmd(const DistMetaTensor& q,
                               const DistMetaTensor& k,
                               const DistMetaTensor& v,
                               const DistMetaTensor& out,
                               const DistMetaTensor& softmax_lse,
                               const DistMetaTensor& seed_offset,
                               const DistMetaTensor& attn_mask,
                               const DistMetaTensor& out_grad,
                               float dropout = 0.0,
                               bool causal = false);

}  // namespace distributed
}  // namespace phi

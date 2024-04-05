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

#include <vector>

#include "paddle/phi/common/int_array.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_meta_tensor.h"
#include "paddle/phi/core/distributed/type_defs.h"

namespace phi {
namespace distributed {

SpmdInfo FusedRopeInferSpmd(const DistMetaTensor& q,
                            const DistMetaTensor& k,
                            const DistMetaTensor& v,
                            const DistMetaTensor& sin,
                            const DistMetaTensor& cos,
                            const DistMetaTensor& position_ids,
                            bool use_neox_rotary_style = true,
                            bool time_major = false,
                            float rotary_emb_base = 10000.f);

SpmdInfo FusedRopeInferSpmdReverse(const DistMetaTensor& q,
                                   const DistMetaTensor& k,
                                   const DistMetaTensor& v,
                                   const DistMetaTensor& sin,
                                   const DistMetaTensor& cos,
                                   const DistMetaTensor& position_ids,
                                   const DistMetaTensor& out_q,
                                   const DistMetaTensor& out_k,
                                   const DistMetaTensor& out_v,
                                   bool use_neox_rotary_style = true,
                                   bool time_major = false,
                                   float rotary_emb_base = 10000.f);

SpmdInfo FusedRopeGradInferSpmd(const DistMetaTensor& sin,
                                const DistMetaTensor& cos,
                                const DistMetaTensor& position_ids,
                                const DistMetaTensor& out_q_grad,
                                const DistMetaTensor& out_k_grad,
                                const DistMetaTensor& out_v_grad,
                                bool use_neox_rotary_style = true,
                                bool time_major = false,
                                float rotary_emb_base = 10000.f);

}  // namespace distributed
}  // namespace phi

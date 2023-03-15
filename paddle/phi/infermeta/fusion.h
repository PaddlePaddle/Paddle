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
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/meta_tensor.h"

namespace phi {

// Common InferMeta Functions for fusion operators.
// NOTE: The InferMeta Functions in this file are arranged in alphabetic order.

void EmbeddingWithEltwiseAddXPUInferMeta(
    const std::vector<const MetaTensor*>& ids,
    const std::vector<const MetaTensor*>& tables,
    MetaTensor* out);

void FcXPUInferMeta(const MetaTensor& x,
                    const MetaTensor& x_max,
                    const MetaTensor& w,
                    const MetaTensor& w_max,
                    const MetaTensor& bias,
                    int in_num_col_dims,
                    bool transpose_x,
                    float alpha,
                    float beta,
                    int act_type,
                    float act_alpha,
                    MetaTensor* out,
                    MetaTensor* out_max);

void GenerateSequenceXPUInferMeta(const MetaTensor& x,
                                  DataType dtype,
                                  MetaTensor* out);

void MultiEncoderXPUInferMeta(
    const MetaTensor& x,
    const std::vector<const MetaTensor*>& fc_weight,
    const std::vector<const MetaTensor*>& fc_weight_max,
    const std::vector<const MetaTensor*>& fc_bias,
    const std::vector<const MetaTensor*>& ln_scale,
    const std::vector<const MetaTensor*>& ln_bias,
    const MetaTensor& mask,
    int layer_num,
    bool norm_before,
    int hidden_dim,
    int head_num,
    int size_per_head,
    int ffn_hidden_dim_scale,
    int act_type,
    int relative_type,
    int slice_idx,
    MetaTensor* out,
    MetaTensor* x_fp16,
    MetaTensor* out_fp16);

}  // namespace phi

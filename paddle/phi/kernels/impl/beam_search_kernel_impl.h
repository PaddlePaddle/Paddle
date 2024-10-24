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

#pragma once

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/math/beam_search.h"

namespace phi {

template <typename T, typename Context>
void BeamSearchOpKernel(const Context &dev_ctx,
                        const DenseTensor &pre_ids_in,
                        const DenseTensor &pre_scores_in,
                        const DenseTensor &ids_in,
                        const DenseTensor &scores_in,
                        int level,
                        int beam_size,
                        int end_id,
                        bool is_accumulated,
                        DenseTensor *selected_ids,
                        DenseTensor *selected_scores,
                        DenseTensor *parent_idx) {
  auto *ids = &ids_in;
  auto *scores = &scores_in;
  auto *pre_ids = &pre_ids_in;
  auto *pre_scores = &pre_scores_in;

  PADDLE_ENFORCE_NOT_NULL(
      scores,
      common::errors::NotFound("Input(scores) of BeamSearchOp is not found."));
  PADDLE_ENFORCE_NOT_NULL(
      pre_ids,
      common::errors::NotFound("Input(pre_ids) of BeamSearchOp is not found."));
  PADDLE_ENFORCE_NOT_NULL(
      pre_scores,
      common::errors::NotFound(
          "Input(pre_scores) of BeamSearchOp is not found."));

  PADDLE_ENFORCE_NOT_NULL(
      selected_ids,
      common::errors::NotFound(
          "Output(selected_ids) of BeamSearchOp is not found."));
  PADDLE_ENFORCE_NOT_NULL(
      selected_scores,
      common::errors::NotFound(
          "Output(selected_scores) of BeamSearchOp is not found."));

  phi::math::BeamSearchFunctor<Context, T> alg;
  alg(dev_ctx,
      pre_ids,
      pre_scores,
      ids,
      scores,
      selected_ids,
      selected_scores,
      parent_idx,
      level,
      beam_size,
      end_id,
      is_accumulated);
}
}  // namespace phi

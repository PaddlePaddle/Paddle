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

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/beam_search_decode_xpu.h"

namespace phi {

template <typename T, typename Context>
void BeamSearchDecodeXPUKernel(const Context& dev_ctx,
                               const TensorArray& ids_in,
                               const TensorArray& scores_in,
                               int beam_size,
                               int end_id,
                               DenseTensor* sentence_ids,
                               DenseTensor* sentence_scores) {
  const TensorArray* ids = &ids_in;
  const TensorArray* scores = &scores_in;
  const size_t step_num = ids->size();
  PADDLE_ENFORCE_GT(
      step_num,
      0UL,
      common::errors::InvalidArgument(
          "beam search steps, which is the"
          "size of Input(Ids) TensorArray. beam search steps should "
          "be larger than 0, but received %d. ",
          step_num));

  const size_t source_num = ids->at(0).lod().at(0).size() - 1;
  PADDLE_ENFORCE_GT(
      source_num,
      0UL,
      common::errors::InvalidArgument(
          "source_num is the sequence number of the"
          "first decoding step, indicating by Input(Ids)[0].lod[0].size. "
          "The number of source_num should be larger than"
          "0, but received %d. ",
          source_num));

  for (size_t i = 0; i < step_num; ++i) {
    PADDLE_ENFORCE_EQ(
        ids->at(i).lod().size(),
        2UL,
        common::errors::InvalidArgument(
            "For the i step in beam search steps,"
            "the size of Input(Ids)[i].lod() should larger than 2,"
            "but received %d. ",
            ids->at(i).lod().size()));
  }

  // prepare output
  phi::DenseTensor* sentenceIds = nullptr;
  phi::DenseTensor* sentenceScores = nullptr;

  phi::DenseTensor* sentenceIds_temp = sentence_ids;
  phi::DenseTensor* sentenceScores_temp = sentence_scores;

  if (ids->at(0).place().GetType() == phi::AllocationType::XPU) {
    sentenceIds = new phi::DenseTensor();
    sentenceIds->set_lod(sentenceIds_temp->lod());
  }

  if (ids->at(0).place().GetType() == phi::AllocationType::XPU) {
    sentenceScores = new phi::DenseTensor();
    sentenceScores->set_lod(sentenceScores_temp->lod());
  }

  phi::funcs::BeamSearchDecodeXPUFunctor bs_xpu(
      *ids, *scores, sentenceIds, sentenceScores, beam_size, end_id);
  bs_xpu.apply_xpu<T>();

  if (ids->at(0).place().GetType() == phi::AllocationType::XPU) {
    int r = 0;
    r = phi::funcs::CopyTensorByXPU<int64_t>(
        *sentenceIds, sentenceIds_temp, 1, ids->at(0).place());
    PADDLE_ENFORCE_EQ(
        r,
        xpu::Error_t::SUCCESS,
        common::errors::External(
            "Execute function CopyTensorByXPU failed by [%d]", r));

    r = phi::funcs::CopyTensorByType(
        *sentenceScores, sentenceScores_temp, 1, ids->at(0).place());
    PADDLE_ENFORCE_EQ(
        r,
        xpu::Error_t::SUCCESS,
        common::errors::External(
            "Execute function CopyTensorByType failed by [%d]", r));
    sentenceIds_temp->set_lod(sentenceIds->lod());
    sentenceScores_temp->set_lod(sentenceScores->lod());
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(beam_search_decode,
                   XPU,
                   ALL_LAYOUT,
                   phi::BeamSearchDecodeXPUKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   int,
                   int64_t) {
  kernel->OutputAt(0).SetDataType(phi::DataType::INT64);
  kernel->OutputAt(1).SetDataType(phi::DataType::INT64);
}

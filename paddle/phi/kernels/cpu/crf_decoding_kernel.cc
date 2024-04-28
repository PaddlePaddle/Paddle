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

#include "paddle/phi/kernels/crf_decoding_kernel.h"

namespace phi {

// Slice() needs to be used in *.cc files, otherwise there is a error in
// test/custom_runtime/extension_header_test.cc
template <typename T, typename Context>
void CRFDecodingOpKernel(const Context& dev_ctx,
                         const DenseTensor& emission,
                         const DenseTensor& transition,
                         const paddle::optional<DenseTensor>& label,
                         const paddle::optional<DenseTensor>& length,
                         DenseTensor* viterbi_path) {
  auto* emission_weights = &emission;
  auto* transition_weights = &transition;
  auto* label_p = label.get_ptr();
  auto* decoded_path = viterbi_path;

  int64_t* path = dev_ctx.template Alloc<int64_t>(decoded_path);
  phi::funcs::SetConstant<Context, int64_t>()(dev_ctx, decoded_path, 0);

  bool has_length = length.get_ptr() != nullptr;
  if (has_length) {
    auto* length_p = length.get_ptr();
    const size_t seq_num = length_p->numel();
    const int64_t* length_data = length_p->data<int64_t>();
    auto in_dims = emission_weights->dims();

    phi::DenseTensor emission_weights_tmp = *emission_weights;
    emission_weights_tmp.Resize(
        common::make_ddim({in_dims[0] * in_dims[1], in_dims[2]}));

    decoded_path->Resize(common::make_ddim({in_dims[0] * in_dims[1], 1}));
    for (size_t i = 0; i < seq_num; ++i) {
      if (length_data[i] == 0) continue;
      int64_t start_pos = i * in_dims[1];
      int64_t end_pos = start_pos + static_cast<int64_t>(length_data[i]);
      phi::DenseTensor decoded_path_one_seq =
          decoded_path->Slice(start_pos, end_pos);
      Decode<T, Context>(dev_ctx,
                         emission_weights_tmp.Slice(start_pos, end_pos),
                         *transition_weights,
                         &decoded_path_one_seq);
    }
    decoded_path->Resize(common::make_ddim({in_dims[0], in_dims[1]}));

    if (label) {
      const int64_t* label_value = label_p->data<int64_t>();
      for (size_t i = 0; i < seq_num; ++i) {
        for (int64_t j = 0; j < in_dims[1]; ++j) {
          int64_t start_pos = i * in_dims[1];
          if (j < length_data[i]) {
            path[start_pos + j] =
                label_value[start_pos + j] == path[start_pos + j] ? 1 : 0;
          } else {
            path[start_pos + j] = 0;
          }
        }
      }
    }
  } else {
    PADDLE_ENFORCE_EQ(emission_weights->NumLevels(),
                      1UL,
                      phi::errors::InvalidArgument(
                          "The Input(Emission) should be a sequence with lod "
                          "level 1. But received: lod level %u.",
                          emission_weights->NumLevels()));
    auto lod = emission_weights->lod();
    PADDLE_ENFORCE_GT(
        lod.size(),
        0,
        phi::errors::InvalidArgument(
            "Input(Emission) must be a sequence. But received: lod level %u.",
            lod.size()));
    const size_t level = 0;
    const size_t seq_num = lod[level].size() - 1;

    for (size_t i = 0; i < seq_num; ++i) {
      if (lod[level][i] == lod[level][i + 1]) continue;
      int64_t start_pos = static_cast<int64_t>(lod[level][i]);
      int64_t end_pos = static_cast<int64_t>(lod[level][i + 1]);
      phi::DenseTensor decoded_path_one_seq =
          decoded_path->Slice(start_pos, end_pos);
      Decode<T, Context>(dev_ctx,
                         emission_weights->Slice(start_pos, end_pos),
                         *transition_weights,
                         &decoded_path_one_seq);
    }
    if (label) {
      PADDLE_ENFORCE_EQ(label_p->NumLevels(),
                        1UL,
                        phi::errors::InvalidArgument(
                            "The Input(label) should be a sequence with lod "
                            "level 1. But received: lod level %u.",
                            label_p->NumLevels()));
      const int64_t* label_value = label_p->data<int64_t>();
      size_t numel = label->numel();
      for (size_t i = 0; i < numel; ++i) {
        path[i] = label_value[i] == path[i] ? 1 : 0;
      }
    }
  }
}
}  // namespace phi
PD_REGISTER_KERNEL(
    crf_decoding, CPU, ALL_LAYOUT, phi::CRFDecodingOpKernel, float, double) {
  kernel->OutputAt(0).SetDataType(phi::DataType::INT64);
}

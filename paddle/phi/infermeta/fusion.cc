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

#include "paddle/phi/infermeta/fusion.h"
#include <vector>
#include "paddle/phi/common/layout.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/core/meta_tensor.h"

namespace phi {
inline int ConvOutSize(int input_size,
                       int filter_size,
                       int dilation,
                       int pad_left,
                       int pad_right,
                       int stride) {
  const int dkernel = dilation * (filter_size - 1) + 1;
  int output_size =
      (input_size + (pad_left + pad_right) - dkernel) / stride + 1;

  return output_size;
}

void UpdatePaddingAndDilation(std::vector<int>& paddings,
                              std::vector<int>& dilations,
                              std::vector<int>& strides,
                              std::string padding_algorithm,
                              phi::DDim data_dims,
                              phi::DDim ksize) {
  // when padding_desc is "VALID" or "SAME"
  if (padding_algorithm == "SAME") {
    for (size_t i = 0; i < strides.size(); ++i) {
      int out_size = (data_dims[i + 2] + strides[i] - 1) / strides[i];
      int pad_sum = (std::max)(
          (out_size - 1) * strides[i] + ksize[i + 2] - data_dims[i + 2],
          (int64_t)0);
      int pad_0 = pad_sum / 2;
      int pad_1 = pad_sum - pad_0;
      // pad
      *(paddings.begin() + i * 2) = pad_0;
      *(paddings.begin() + i * 2 + 1) = pad_1;
      // dilation
      *(dilations.begin() + i) = 1;
    }
  } else if (padding_algorithm == "VALID") {
    for (auto& it : paddings) {
      it = 0;
    }
  }
}

void ConvXPUInferMeta(const MetaTensor& Input,
                      const MetaTensor& InputMax,
                      const MetaTensor& Filter,
                      const MetaTensor& Filtermax,
                      const MetaTensor& Bias,
                      const MetaTensor& Branch,
                      std::vector<int>& paddings,
                      std::vector<int>& dilations,
                      std::vector<int>& strides,
                      std::vector<int>& groups,
                      std::string padding_algorithm,
                      bool has_branch,
                      int act_type,
                      float act_param,
                      MetaTensor* Output,
                      MetaTensor* OutputMax) {
  const auto in_dims = Input.dims();
  const auto filter_dims = Filter.dims();

  UpdatePaddingAndDilation(
      paddings, dilations, strides, padding_algorithm, in_dims, filter_dims);

  std::vector<int64_t> out_shape({in_dims[0], filter_dims[0]});
  for (size_t i = 0; i < strides.size(); ++i) {
    out_shape.push_back(ConvOutSize(in_dims[i + 2],
                                          filter_dims[i + 2],
                                          dilations[i],
                                          paddings[i * 2],
                                          paddings[i * 2 + 1],
                                          strides[i]));
  }
  // set output and output max dims
  Output->set_dims(DDim(out_shape.data(), out_shape.size()));
  OutputMax->set_dims(phi::make_ddim({4}));
}

void EmbeddingWithEltwiseAddXPUInferMeta(
    const std::vector<const MetaTensor*>& ids,
    const std::vector<const MetaTensor*>& tables,
    MetaTensor* out) {
  PADDLE_ENFORCE_GT(ids.size(),
                    0UL,
                    phi::errors::InvalidArgument(
                        "The input ids in EmbeddingWithEltwiseAddXPUInferMeta "
                        "can't be empty."));
  PADDLE_ENFORCE_GT(tables.size(),
                    0UL,
                    phi::errors::InvalidArgument(
                        "The input tables in "
                        "EmbeddingWithEltwiseAddXPUInferMeta can't be empty."));

  auto id_dims = ids[0]->dims();
  auto table_dims = tables[0]->dims();
  out->set_dims(phi::make_ddim({id_dims[0], id_dims[1], table_dims[1]}));
  out->set_dtype(tables[0]->dtype());
  out->set_layout(ids[0]->layout());
}

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
                    MetaTensor* out_max) {
  std::vector<int> out_shape(in_num_col_dims + 1);
  for (int i = 0; i < in_num_col_dims; i++) {
    out_shape[i] = x.dims()[i];
  }
  out_shape[in_num_col_dims] = w.dims()[0];
  out->set_dims(DDim(out_shape.data(), out_shape.size()));
  out->set_dtype(x.dtype());
  out->set_layout(x.layout());
  out_max->set_dims(w_max.dims());
  out_max->set_dtype(x.dtype());
  out_max->set_layout(x.layout());
}

void GenerateSequenceXPUInferMeta(const MetaTensor& x,
                                  DataType dtype,
                                  MetaTensor* out) {
  out->set_dims(x.dims());
  out->set_dtype(dtype);
  out->set_layout(x.layout());
}

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
    MetaTensor* out_fp16) {
  auto x_dims = x.dims();
  x_fp16->set_dims(x_dims);
  x_fp16->set_dtype(DataType::FLOAT16);
  x_fp16->set_layout(x.layout());
  out->set_dtype(x.dtype());
  out->set_layout(x.layout());
  out_fp16->set_dtype(DataType::FLOAT16);
  out_fp16->set_layout(x.layout());
  if (slice_idx == -1) {
    out->set_dims(x_dims);
    out_fp16->set_dims(x_dims);
  } else {
    out->set_dims({x_dims[0], x_dims[2]});
    out_fp16->set_dims({x_dims[0], x_dims[2]});
  }
}

}  // namespace phi

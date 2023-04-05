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

void FusedMultiTransformerXpuInferMeta(
    const MetaTensor& x,
    const std::vector<const MetaTensor*>& ln_scale,
    const std::vector<const MetaTensor*>& ln_bias,
    const std::vector<const MetaTensor*>& qkvw,
    const std::vector<const MetaTensor*>& qkvw_max,
    const std::vector<const MetaTensor*>& qkv_bias,
    const std::vector<const MetaTensor*>& out_linear_w,
    const std::vector<const MetaTensor*>& out_linear_wmax,
    const std::vector<const MetaTensor*>& out_linear_bias,
    const std::vector<const MetaTensor*>& ffn_ln_scale,
    const std::vector<const MetaTensor*>& ffn_ln_bias,
    const std::vector<const MetaTensor*>& ffn1_weight,
    const std::vector<const MetaTensor*>& ffn1_weight_max,
    const std::vector<const MetaTensor*>& ffn1_bias,
    const std::vector<const MetaTensor*>& ffn2_weight,
    const std::vector<const MetaTensor*>& ffn2_weight_max,
    const std::vector<const MetaTensor*>& ffn2_bias,
    const std::vector<const MetaTensor*>& cache_kv,
    const std::vector<const MetaTensor*>& pre_caches,
    const std::vector<const MetaTensor*>& rotary_pos_emb,
    const std::vector<const MetaTensor*>& time_step,
    const std::vector<const MetaTensor*>& seq_lengths,
    const std::vector<const MetaTensor*>& src_mask,
    bool pre_layer_norm,
    int rotary_emb_dims,
    float epsilon,
    float dropout_rate,
    bool is_test,
    const std::string& dropout_implementation,
    const std::string& act_method,
    bool trans_qkvw,
    int ring_id,
    MetaTensor* out,
    std::vector<MetaTensor*> cache_kv_out) {
  auto x_dim = x.dims();
  auto y_dim = qkvw[0]->dims();
  PADDLE_ENFORCE_EQ(
      x_dim.size(),
      3,
      phi::errors::InvalidArgument("The dimensions of x must be 3"
                                   "(batch_size, seq_len, dim_embed),"
                                   "but received dimensions of"
                                   "Input is [%d]",
                                   x_dim.size()));
  PADDLE_ENFORCE_EQ(
      y_dim.size(),
      4,
      phi::errors::InvalidArgument("The dimensions of qkv_weight must be 4"
                                   "(3, num_head, dim_head, dim_embed),"
                                   "but received dimensions of"
                                   "Input is [%d]",
                                   y_dim.size()));
  PADDLE_ENFORCE_EQ(
      x_dim[2],
      trans_qkvw ? y_dim[3] : y_dim[0],
      phi::errors::InvalidArgument(
          "ShapeError: the dimension of x_dim[2] and y_dim[3](trans_qkvw is "
          "true) or y_dim[0](trans_qkvw is false)"
          "must be equal. But received: the shape "
          "of input x = [%s], and the shape of "
          "input qkv_weight = [%s]",
          x_dim,
          y_dim));
  if (cache_kv.size() > 0) {
    const auto& c_dim = cache_kv[0]->dims();
    PADDLE_ENFORCE_EQ(
        c_dim.size(),
        5,
        phi::errors::InvalidArgument("The CacheKV must be 5 dims, but got %d",
                                     c_dim.size()));
    PADDLE_ENFORCE_EQ(c_dim[0],
                      2,
                      phi::errors::InvalidArgument(
                          "The first dim of CacheKV must be 2, but got %d",
                          c_dim[0]));  // 2
    PADDLE_ENFORCE_EQ(c_dim[1],
                      x_dim[0],
                      phi::errors::InvalidArgument(
                          "The second dim of CacheKV must be equal with "
                          "batch size %d, but got %d",
                          x_dim[0],
                          c_dim[1]));  // batch_size
    PADDLE_ENFORCE_EQ(c_dim[2],
                      trans_qkvw ? y_dim[1] : y_dim[2],
                      phi::errors::InvalidArgument(
                          "The third dim of CacheKV must be equal with num "
                          "head %d, but got %d",
                          trans_qkvw ? y_dim[1] : y_dim[2],
                          c_dim[2]));  // num_head
    PADDLE_ENFORCE_EQ(c_dim[4],
                      trans_qkvw ? y_dim[2] : y_dim[3],
                      phi::errors::InvalidArgument(
                          "The fifth dim of CacheKV must be equal with head "
                          "size %d, but got %d",
                          trans_qkvw ? y_dim[2] : y_dim[3],
                          c_dim[4]));  // head_size
  }

  out->set_dims(x_dim);
  out->set_dtype(x.dtype());
  out->set_layout(x.layout());
}

}  // namespace phi

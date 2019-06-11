// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/lite/arm/math/conv_winograd.h"
#include <vector>
#include "paddle/fluid/lite/arm/math/conv_impl.h"
#include "paddle/fluid/lite/arm/math/packed_sgemm.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

template <>
bool WinogradConv<PRECISION(kFloat)>::create(const operators::ConvParam& param,
                                             ARMContext* ctx) {
  this->ctx_ = ctx;
  auto x_dims = param.x->dims();
  auto w_dims = param.filter->dims();
  auto o_dims = param.output->dims();

  int iw = x_dims[3];  // nchw
  int ic = x_dims[1];
  int ow = o_dims[3];
  int oh = o_dims[2];
  int oc = o_dims[1];
  int kw = w_dims[3];
  int sw = param.strides[1];
  if (kw == 3) {
    is_weights_transed_ = true;
    int tile_w = (ow + 5) / 6;
    int tile_h = (oh + 5) / 6;
    int size_tile = tile_h * tile_w;
    int size_trans_channel = 8 * 8 * size_tile;
    int max_ch = ic > oc ? ic : oc;

    const int m_wino = oc;
    const int n_wino = size_tile;
    int hblock = get_hblock(this->ctx_->arch());
    int m_round = hblock * ((m_wino + hblock - 1) / hblock);
    weights_trans_.Resize({1, 1, 1, 8 * 8 * m_round * ic});
    this->ctx_->ExtendWorkspace(DDim(std::vector<DDim::value_type>(
        {1, 1, 1, size_trans_channel * max_ch * 2 + n_wino})));
    float* weights_wino =
        static_cast<float*>(malloc(sizeof(float) * 8 * 8 * oc * ic));
    void* trans_tmp_ptr = malloc(sizeof(float) * 8 * 8 * oc * ic);
    if (weights_wino && trans_tmp_ptr) {
      winograd_transform_weights(weights_wino, param.filter->data<float>(), oc,
                                 ic, trans_tmp_ptr);
      float* weights_trans = weights_trans_.mutable_data<float>();
      for (int i = 0; i < 64; ++i) {
        float* packed_weights = weights_trans + i * m_round * ic;
        const float* weights_wino_ptr = weights_wino + i * oc * ic;
        prepackA(packed_weights, weights_wino_ptr, ic, 0, m_wino, 0, ic, false,
                 this->ctx_);
      }
      impl_ = conv_winograd3x3;
      free(trans_tmp_ptr);
      free(weights_wino);
      return true;
    }
    free(trans_tmp_ptr);
    free(weights_wino);
  } else {
    LOG(ERROR) << "this type winograd conv not impl";
  }
  return false;
}

template <>
bool WinogradConv<PRECISION(kFloat)>::init(const operators::ConvParam& param,
                                           Context<TARGET(kARM)>* ctx) {
  this->ctx_ = ctx;
  return create(param, ctx);
}

template <>
bool WinogradConv<PRECISION(kFloat)>::run(const operators::ConvParam& param) {
  // start timer
  const auto* i_data = param.x->data<float>();
  const auto* w_data = param.filter->data<float>();
  const auto* b_data = param.bias ? param.bias->data<float>() : nullptr;
  auto* o_data = param.output->mutable_data<float>();

  if (is_weights_transed_ == true) {
    w_data = weights_trans_.data<float>();
  }

  auto x_dims = param.x->dims();
  auto w_dims = param.filter->dims();
  auto o_dims = param.output->dims();

  int iw = x_dims[3];  // nchw
  int ih = x_dims[2];
  int ic = x_dims[1];
  int bs = x_dims[0];
  int oh = o_dims[2];
  int ow = o_dims[3];
  int oc = o_dims[1];

  impl_(i_data, o_data, bs, oc, oh, ow, ic, ih, iw, w_data, b_data, param,
        this->ctx_);

  // timer end
  return true;
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle

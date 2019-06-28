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

#include "paddle/fluid/lite/arm/math/conv_gemmlike.h"
#include <vector>
#include "paddle/fluid/lite/arm/math/gemm_prepacked_int8.h"
#include "paddle/fluid/lite/arm/math/packed_sgemm.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

/********************* Gemmlike Conv Precision Is Float ***********************/
template <>
bool GemmLikeConv<PRECISION(kFloat)>::create(const operators::ConvParam& param,
                                             ARMContext* ctx) {
  this->ctx_ = ctx;
  auto x_dims = param.x->dims();
  auto w_dims = param.filter->dims();
  auto o_dims = param.output->dims();

  int iw = x_dims[3];  // nchw
  int ih = x_dims[2];
  int ic = x_dims[1];
  int ow = o_dims[3];
  int oh = o_dims[2];
  int oc = o_dims[1];
  int kw = w_dims[3];
  int kh = w_dims[2];
  int sw = param.strides[1];
  int sh = param.strides[0];
  int pw = param.paddings[1];
  int ph = param.paddings[0];
  int dw = param.dilations[1];
  int dh = param.dilations[0];

  int m = oc / param.groups;
  int k = ic * kh * kw / param.groups;
  int n = oh * ow;
  bool kps_equal = (pw == ph) && (sw == sh) && (kw == kh);
  bool ks_equal = (sw == sh) && (kw == kh);
  //! select conv gemmlike kernel
  if (kw == 1 && sw == 1 && pw == 0 && kps_equal) {
    //! 1x1s1p0 gemmlike conv
    impl_ = conv1x1s1_gemm;
  } else {
    //! otherwise case
    if (kw == 3 && sw == 1 && n > 1 && ks_equal) {
      idx_data_.Resize({1, 1, 1, n * kh * kw});
      int* idx_out = idx_data_.mutable_data<int>();
      for (int i = 0; i < oh; ++i) {
        for (int j = 0; j < ow; ++j) {
          compute_offset(idx_out, i, j, kh, kw, ih, iw, ph, pw, dh, dw);
          idx_out += kh * kw;
        }
      }
    }
    //! im2col gemmlike conv
    impl_ = conv_im2col_gemm;
    this->ctx_->ExtendWorkspace(
        DDim(std::vector<DDim::value_type>({1, 1, 1, k * n})));
  }

  if (n > 1) {
    int hblock = get_hblock(this->ctx_->arch());
    int m_roundup = hblock * ((m + hblock - 1) / hblock);
    int group_size_round_up = ((m_roundup * k + 15) / 16) * 16;
    float* w_trans_ptr = nullptr;
    weights_trans_.Resize({1, 1, 1, group_size_round_up * param.groups});
    w_trans_ptr = weights_trans_.mutable_data<float>();
    const auto* w_data = param.filter->data<float>();
    for (int g = 0; g < param.groups; ++g) {
      const float* weights_group = w_data + g * m * k;
      float* weights_trans_ptr = w_trans_ptr + g * group_size_round_up;
      prepackA(weights_trans_ptr, weights_group, k, 0, m, 0, k, false,
               this->ctx_);
    }
    is_weights_transed_ = true;
  }
  return true;
}

template <>
bool GemmLikeConv<PRECISION(kFloat)>::init(const operators::ConvParam& param,
                                           ARMContext* ctx) {
  this->ctx_ = ctx;
  return create(param, ctx);
}

template <>
bool GemmLikeConv<PRECISION(kFloat)>::run(const operators::ConvParam& param) {
  // start timer
  const auto* i_data = param.x->data<float>();
  const auto* w_data = param.filter->data<float>();
  const auto* b_data = param.bias ? param.bias->data<float>() : nullptr;
  auto* o_data = param.output->mutable_data<float>();
  const int* idx_data = idx_data_.mutable_data<int>();

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
        this->ctx_, idx_data);

  // timer end
  return true;
}

/********************* Gemmlike Conv Precision Is Int8 ************************/
template <PrecisionType Ptype_out>
bool GemmLikeConvInt8<Ptype_out>::create(const operators::ConvParam& param,
                                         ARMContext* ctx) {
  this->ctx_ = ctx;
  auto x_dims = param.x->dims();
  auto w_dims = param.filter->dims();
  auto o_dims = param.output->dims();

  int iw = x_dims[3];  // nchw
  int ih = x_dims[2];
  int ic = x_dims[1];
  int ow = o_dims[3];
  int oh = o_dims[2];
  int oc = o_dims[1];
  int kw = w_dims[3];
  int kh = w_dims[2];
  int sw = param.strides[1];
  int sh = param.strides[0];
  int pw = param.paddings[1];
  int ph = param.paddings[0];
  int dw = param.dilations[1];
  int dh = param.dilations[0];

  int m = oc / param.groups;
  int k = ic * kh * kw / param.groups;
  int n = oh * ow;
  w_scale_ = param.weight_scale;
  //! update weights scale
  if (Ptype_out == PRECISION(kInt8) || Ptype_out == PRECISION(kFloat)) {
    CHECK_EQ(this->w_scale_.size(), oc) << "weights scale size must be chout";
    float input_scale = param.input_scale;
    for (auto& w_s : w_scale_) {
      w_s *= input_scale;
      if (Ptype_out == PRECISION(kInt8)) {
        w_s /= param.output_scale;
      }
    }
  }

  bool kps_equal = (pw == ph) && (sw == sh) && (kw == kh);
  bool ks_equal = (sw == sh) && (kw == kh);
  //! select conv gemmlike kernel
  if (kw == 1 && sw == 1 && pw == 0 && kps_equal) {
    //! 1x1s1p0 gemmlike conv
    impl_int8_ = conv1x1s1_gemm_int8;
  } else {
    //! otherwise case
    if (kw == 3 && sw == 1 && n > 1 && ks_equal) {
      idx_data_.Resize({1, 1, 1, n * kh * kw});
      int* idx_out = idx_data_.mutable_data<int>();
      for (int i = 0; i < oh; ++i) {
        for (int j = 0; j < ow; ++j) {
          compute_offset(idx_out, i, j, kh, kw, ih, iw, ph, pw, dh, dw);
          idx_out += kh * kw;
        }
      }
    }
    //! im2col gemmlike conv
    impl_int8_ = conv_im2col_gemm_int8;
    this->ctx_->ExtendWorkspace(
        DDim(std::vector<DDim::value_type>({1, 1, 1, (k * n + 3) / 4})));
  }

  if (n > 1) {
    prepackA_int8(&this->weights_trans_, *param.filter, m, k, param.groups,
                  false, this->ctx_);
    this->is_weights_transed_ = true;
  }
  return true;
}

template <PrecisionType Ptype_out>
bool GemmLikeConvInt8<Ptype_out>::init(const operators::ConvParam& param,
                                       ARMContext* ctx) {
  this->ctx_ = ctx;
  return create(param, ctx);
}

template <PrecisionType Ptype_out>
bool GemmLikeConvInt8<Ptype_out>::run(const operators::ConvParam& param) {
  const auto* i_data = param.x->data<int8_t>();
  const auto* w_data = param.filter->data<int8_t>();
  const auto* b_data = param.bias ? param.bias->data<int32_t>() : nullptr;
  auto* o_data = param.output->mutable_data<int32_t>();
  const int32_t* idx_data = idx_data_.mutable_data<int32_t>();

  if (this->is_weights_transed_ == true) {
    w_data = this->weights_trans_.template data<int8_t>();
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

  impl_int8_(i_data, o_data, bs, oc, oh, ow, ic, ih, iw, w_data, b_data, param,
             this->ctx_, Ptype_out, this->w_scale_.data(), idx_data);

  return true;
}

template class GemmLikeConvInt8<PRECISION(kInt8)>;
template class GemmLikeConvInt8<PRECISION(kFloat)>;
template class GemmLikeConvInt8<PRECISION(kInt32)>;

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle

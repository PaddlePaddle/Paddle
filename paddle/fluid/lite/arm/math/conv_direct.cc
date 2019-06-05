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

#include "paddle/fluid/lite/arm/math/conv_direct.h"
#include "paddle/fluid/lite/arm/math/conv_block_utils.h"
#include "paddle/fluid/lite/arm/math/conv_impl.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

template <>
bool DirectConv<PRECISION(kFloat)>::create(const operators::ConvParam& param,
                                           ARMContext* ctx) {
  this->ctx_ = ctx;
  auto x_dims = param.x->dims();
  auto w_dims = param.filter->dims();
  auto o_dims = param.output->dims();

  int iw = x_dims[3];  // nchw
  int ic = x_dims[1];
  int ow = o_dims[3];
  int oc = o_dims[1];
  int kw = w_dims[3];
  int sw = param.strides[1];
  // select dw conv kernel
  const auto* w_data = param.filter->data<float>();
  if (kw == 3 && sw == 1) {
    VLOG(5) << "invoke 3x3s1 direct conv";
    impl_ = conv_3x3s1_direct_fp32;

    constexpr int cblock = 4;
    int cround = (oc + cblock - 1) / cblock * cblock;
    weights_trans_.Resize({cround, ic, kw, kw});
    float* transed_w_data = weights_trans_.mutable_data<float>();

    conv_trans_weights_numc(w_data, transed_w_data, oc, ic, cblock, kw * kw);
    is_weights_transed_ = true;
  } else if (kw == 3 && sw == 2) {
    VLOG(5) << "invoke 3x3s2 direct conv";
    impl_ = conv_3x3s2_direct_fp32;

    constexpr int cblock = 4;
    int cround = (oc + cblock - 1) / cblock * cblock;
    weights_trans_.Resize({cround, ic, kw, kw});
    float* transed_w_data = weights_trans_.mutable_data<float>();
    conv_trans_weights_numc(w_data, transed_w_data, oc, ic, cblock, kw * kw);
    is_weights_transed_ = true;
  } else {
    LOG(ERROR) << "this type direct conv not impl";
    return false;
  }
  return true;
}

template <>
bool DirectConv<PRECISION(kFloat)>::init(const operators::ConvParam& param,
                                         Context<TARGET(kARM)>* ctx) {
  this->ctx_ = ctx;
  return create(param, ctx);
}

template <>
bool DirectConv<PRECISION(kFloat)>::run(const operators::ConvParam& param) {
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

// template <>
// bool DirectConv<kInt8>::create(const
//                                  operators::ConvParam& param,
//                                  Context<TARGET(kARM)>&
//                                  ctx) {
//   this->ctx_ = &ctx;
//   int kw = param.weight()->width();
//   int sw = param.stride_w;
//   int iw = inputs[0]->width();
//   int ic = inputs[0]->channel();
//   int ow = outputs[0]->width();
//   int oc = outputs[0]->channel();
//   _w_scale = param.weight()->get_scale();
//   //! update weights scale
//   const DataType out_type = outputs[0]->get_dtype();
//   if (out_type == PRECISION(kFloat) || out_type == kInt8) {
//     CHECK_EQ(_w_scale.size(), oc) << "weights scale size must be oc";
//     float input_scale = inputs[0]->get_scale()[0];
//     for (auto& ws : _w_scale) {
//       ws *= input_scale;
//       if (out_type == kInt8) {
//         ws /= outputs[0]->get_scale()[0];
//       }
//     }
//   }
//   //! select dw conv kernel
//   if (kw == 3 && sw == 1) {
//     //! 3x3s1 direct int8 conv
//     _impl_int8 = conv_3x3s1_direct_int8;
//     //! transform weights
//     int hout_c_block = 4;
//     int inpad = 4;
//     Shape shape_out(
//         {((oc + hout_c_block - 1) / hout_c_block) * hout_c_block, ic, 3, 3});
//     weights_trans_.re_alloc(shape_out, kInt8);
//     conv_trans_weights_numc(
//         static_cast<const signed char*>(param.weight()->data()),
//         static_cast<signed char*>(weights_trans_.mutable_data<float>()), oc,
//         ic,
//         hout_c_block, 9);
//     int wout_round = ((ow + 3) / 4) * 4;
//     int win_round = wout_round * param.stride_w + inpad;
//     int row_out = 2;
//     int row_in = 4;
//     int tmp_size_out = wout_round * row_out * hout_c_block;
//     int in_len = win_round * ic;
//     int tmp_size_in = row_in * in_len;
//     ctx_->workspace_extend(Shape({1, 1, 1, ctx_->get_threads() * tmp_size_out
//     +
//                                                (tmp_size_in + 3) / 4 * 4 +
//                                                wout_round + win_round}));
//     is_weights_transed_ = true;

//   } else if (kw == 3 && sw == 2) {
//     //! 3x3s2 direct int8 conv
//     _impl_int8 = conv_3x3s2_direct_int8;
//     //! transform weights
//     int cblock = conv_3x3s2_direct_int8_c_num();
//     int cround = (oc + cblock - 1) / cblock * cblock;
//     weights_trans_.re_alloc(Shape({cround, ic, kw, kw}), kInt8);
//     conv_trans_weights_numc(static_cast<const
//     int8_t*>(param.weight()->data()),
//                             static_cast<int8_t*>(weights_trans_.mutable_data<float>()),
//                             oc, ic, cblock, 9);
//     is_weights_transed_ = true;
//   } else {
//     LOG(ERROR) << "this type direct int8 conv not impl";
//     return false;
//   }
//   return true;
// }

// template <>
// bool DirectConv<kInt8>::init(const
//                                operators::ConvParam& param,
//                                Context<TARGET(kARM)>&
//                                ctx) {
//   this->ctx_ = &ctx;
//   return create(inputs, outputs, param, ctx);
// }

// template <>
// bool DirectConv<kInt8>::run(
//                                    operators::ConvParam& param) {
//   const int8_t* i_data = static_cast<const int8_t*>(inputs[0]->data());
//   int32_t* o_data = static_cast<int32_t*>(outputs[0]->mutable_data<float>());
//   const int8_t* weights = nullptr;
//   if (is_weights_transed_ == true) {
//     weights = static_cast<const int8_t*>(weights_trans_.data());
//   } else {
//     weights = static_cast<const int8_t*>(param.weight()->data());
//   }
//   const int32_t* bias = static_cast<const int32_t*>(param.bias()->data());

//   int num = inputs[0]->num();
//   int ic = inputs[0]->channel();
//   int hin = inputs[0]->height();
//   int iw = inputs[0]->width();
//   int hout = outputs[0]->height();
//   int ow = outputs[0]->width();
//   int oc = outputs[0]->channel();
//   _impl_int8(i_data, o_data, num, oc, hout, ow, ic, hin, iw, weights, bias,
//   param,
//              this->ctx_, outputs[0]->get_dtype(), _w_scale.data());

//   return true;
// }

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle

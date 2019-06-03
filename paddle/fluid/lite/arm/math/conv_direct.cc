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

#include "saber/funcs/impl/arm/neon/impl/conv_block_utils.h"

#include "paddle/fluid/lite/arm/math/conv_arm_impl.h"
#include "paddle/fluid/lite/arm/math/conv_direct.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

template <>
bool DirectConv<kFloat>::create(int ic, int iw, int oc, int ow, int kw, int sw,
                                ConvParam& param, ARMContext& ctx) {
  this->ctx_ = &ctx;
  // select dw conv kernel
  if (kw == 3 && sw == 1) {
    //! 3x3s1 direct conv
    // printf("invoke 3x3s1 direct conv\n");
    _impl = conv_3x3s1_direct_fp32;
    //! transform weights
    const int cblock = 4;
    int cround = (oc + cblock - 1) / cblock * cblock;
    _weights_trans.reshape(Shape({cround, ic, kw, kw}));
    float* dwout = static_cast<float*>(_weights_trans.mutable_data());
    const float* dwin = static_cast<const float*>(param.weight()->data());
    conv_trans_weights_numc(dwin, dwout, oc, ic, cblock, kw * kw);
    _is_trans_weights = true;
  } else if (kw == 3 && sw == 2) {
    //! 3x3s2 direct conv
    // printf("invoke 3x3s2 direct conv\n");
    _impl = conv_3x3s2_direct_fp32;
    //! transform weights
    const int cblock = 4;
    int cround = (oc + cblock - 1) / cblock * cblock;
    _weights_trans.reshape(Shape({cround, ic, kw, kw}));
    float* dwout = static_cast<float*>(_weights_trans.mutable_data());
    const float* dwin = static_cast<const float*>(param.weight()->data());
    conv_trans_weights_numc(dwin, dwout, oc, ic, cblock, kw * kw);
    _is_trans_weights = true;
  } else {
    LOG(ERROR) << "this type direct conv not impl";
    return SaberUnImplError;
  }
  return SaberSuccess;
}

template <>
bool DirectConv<kFloat>::init(const std::vector<Tensor<ARM>*>& inputs,
                              std::vector<Tensor<ARM>*>& outputs,
                              ConvParam<ARM>& param, Context<ARM>& ctx) {
  this->ctx_ = &ctx;
  return create(inputs, outputs, param, ctx);
}

template <>
bool DirectConv<kFloat>::dispatch(const std::vector<Tensor<ARM>*>& inputs,
                                  std::vector<Tensor<ARM>*>& outputs,
                                  ConvParam<ARM>& param) {
#ifdef ENABLE_OP_TIMER
  this->_timer.clear();
  this->_timer.start(*this->ctx_);
#endif
  const float* din = static_cast<const float*>(inputs[0]->data());
  float* dout = static_cast<float*>(outputs[0]->mutable_data());
  const float* weights = nullptr;
  if (_is_trans_weights == true) {
    weights = static_cast<const float*>(_weights_trans.data());
  } else {
    weights = static_cast<const float*>(param.weight()->data());
  }
  const float* bias = static_cast<const float*>(param.bias()->data());

  int num = inputs[0]->num();
  int ic = inputs[0]->channel();
  int hin = inputs[0]->height();
  int iw = inputs[0]->width();
  int hout = outputs[0]->height();
  int ow = outputs[0]->width();
  int oc = outputs[0]->channel();
  _impl(din, dout, num, oc, hout, ow, ic, hin, iw, weights, bias, param,
        this->ctx_);

#ifdef ENABLE_OP_TIMER
  this->_timer.end(*this->ctx_);
  float ts = this->_timer.get_average_ms();
  LOG(INFO) << "DirectConv fp32: " << this->_op_name.c_str()
            << " : time: " << ts;
#endif
  return SaberSuccess;
}

/****************************************** Direct Conv Precision Is INT8
 * ******************************************/
template <>
bool DirectConv<AK_INT8>::create(const std::vector<Tensor<ARM>*>& inputs,
                                 std::vector<Tensor<ARM>*>& outputs,
                                 ConvParam<ARM>& param, Context<ARM>& ctx) {
  this->ctx_ = &ctx;
  int kw = param.weight()->width();
  int sw = param.stride_w;
  int iw = inputs[0]->width();
  int ic = inputs[0]->channel();
  int ow = outputs[0]->width();
  int oc = outputs[0]->channel();
  _w_scale = param.weight()->get_scale();
  //! update weights scale
  const DataType out_type = outputs[0]->get_dtype();
  if (out_type == kFloat || out_type == AK_INT8) {
    CHECK_EQ(_w_scale.size(), oc) << "weights scale size must be oc";
    float input_scale = inputs[0]->get_scale()[0];
    for (auto& ws : _w_scale) {
      ws *= input_scale;
      if (out_type == AK_INT8) {
        ws /= outputs[0]->get_scale()[0];
      }
    }
  }
  //! select dw conv kernel
  if (kw == 3 && sw == 1) {
    //! 3x3s1 direct int8 conv
    _impl_int8 = conv_3x3s1_direct_int8;
    //! transform weights
    int hout_c_block = 4;
    int inpad = 4;
    Shape shape_out(
        {((oc + hout_c_block - 1) / hout_c_block) * hout_c_block, ic, 3, 3});
    _weights_trans.re_alloc(shape_out, AK_INT8);
    conv_trans_weights_numc(
        static_cast<const signed char*>(param.weight()->data()),
        static_cast<signed char*>(_weights_trans.mutable_data()), oc, ic,
        hout_c_block, 9);
    int wout_round = ((ow + 3) / 4) * 4;
    int win_round = wout_round * param.stride_w + inpad;
    int row_out = 2;
    int row_in = 4;
    int tmp_size_out = wout_round * row_out * hout_c_block;
    int in_len = win_round * ic;
    int tmp_size_in = row_in * in_len;
    ctx_->workspace_extend(Shape({1, 1, 1, ctx_->get_threads() * tmp_size_out +
                                               (tmp_size_in + 3) / 4 * 4 +
                                               wout_round + win_round}));
    _is_trans_weights = true;

  } else if (kw == 3 && sw == 2) {
    //! 3x3s2 direct int8 conv
    _impl_int8 = conv_3x3s2_direct_int8;
    //! transform weights
    int cblock = conv_3x3s2_direct_int8_c_num();
    int cround = (oc + cblock - 1) / cblock * cblock;
    _weights_trans.re_alloc(Shape({cround, ic, kw, kw}), AK_INT8);
    conv_trans_weights_numc(static_cast<const int8_t*>(param.weight()->data()),
                            static_cast<int8_t*>(_weights_trans.mutable_data()),
                            oc, ic, cblock, 9);
    _is_trans_weights = true;
  } else {
    LOG(ERROR) << "this type direct int8 conv not impl";
    return SaberUnImplError;
  }
  return SaberSuccess;
}

template <>
bool DirectConv<AK_INT8>::init(const std::vector<Tensor<ARM>*>& inputs,
                               std::vector<Tensor<ARM>*>& outputs,
                               ConvParam<ARM>& param, Context<ARM>& ctx) {
  this->ctx_ = &ctx;
  return create(inputs, outputs, param, ctx);
}

template <>
bool DirectConv<AK_INT8>::dispatch(const std::vector<Tensor<ARM>*>& inputs,
                                   std::vector<Tensor<ARM>*>& outputs,
                                   ConvParam<ARM>& param) {
  const int8_t* din = static_cast<const int8_t*>(inputs[0]->data());
  int32_t* dout = static_cast<int32_t*>(outputs[0]->mutable_data());
  const int8_t* weights = nullptr;
  if (_is_trans_weights == true) {
    weights = static_cast<const int8_t*>(_weights_trans.data());
  } else {
    weights = static_cast<const int8_t*>(param.weight()->data());
  }
  const int32_t* bias = static_cast<const int32_t*>(param.bias()->data());

  int num = inputs[0]->num();
  int ic = inputs[0]->channel();
  int hin = inputs[0]->height();
  int iw = inputs[0]->width();
  int hout = outputs[0]->height();
  int ow = outputs[0]->width();
  int oc = outputs[0]->channel();
  _impl_int8(din, dout, num, oc, hout, ow, ic, hin, iw, weights, bias, param,
             this->ctx_, outputs[0]->get_dtype(), _w_scale.data());

  return SaberSuccess;
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle

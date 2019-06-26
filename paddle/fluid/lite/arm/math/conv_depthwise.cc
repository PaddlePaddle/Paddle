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

#include "paddle/fluid/lite/arm/math/conv_depthwise.h"
#include "paddle/fluid/lite/arm/math/conv_block_utils.h"
#include "paddle/fluid/lite/arm/math/conv_impl.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

template <>
bool DepthwiseConv<PRECISION(kFloat)>::create(const operators::ConvParam& param,
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
  if (kw == 3) {
    VLOG(5) << "invoke 3x3 dw conv";
    impl_ = conv_depthwise_3x3;
  } else if (kw == 5) {
    VLOG(5) << "invoke 5x5 dw conv";
    this->ctx_->ExtendWorkspace(
        DDim(std::vector<DDim::value_type>({1, 1, 1, iw + ow})));
    impl_ = conv_depthwise_5x5;
  } else {
    LOG(ERROR) << "this type dw conv not impl";
    return false;
  }
  return true;
}

template <>
bool DepthwiseConv<PRECISION(kFloat)>::init(const operators::ConvParam& param,
                                            Context<TARGET(kARM)>* ctx) {
  this->ctx_ = ctx;
  return create(param, ctx);
}

template <>
bool DepthwiseConv<PRECISION(kFloat)>::run(const operators::ConvParam& param) {
  // start timer
  const auto* i_data = param.x->data<float>();
  const auto* w_data = param.filter->data<float>();
  const auto* b_data = param.bias ? param.bias->data<float>() : nullptr;
  auto* o_data = param.output->mutable_data<float>();

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

template <PrecisionType Ptype_out>
bool DepthwiseConvInt8<Ptype_out>::create(const operators::ConvParam& param,
                                          ARMContext* ctx) {
  this->ctx_ = ctx;
  auto x_dims = param.x->dims();
  auto w_dims = param.filter->dims();
  auto o_dims = param.output->dims();

  int ic = x_dims[1];
  int ih = x_dims[2];
  int iw = x_dims[3];  // nchw
  int oc = o_dims[1];
  int oh = o_dims[2];
  int ow = o_dims[3];
  int kw = w_dims[3];
  int sw = param.strides[1];
  w_scale_ = param.weight_scale;

  if (!tmp_int32_out_) {
    tmp_int32_out_ = new Tensor;
  }
  //! select dw conv kernel
  if (kw == 3) {
    tmp_int32_out_->Resize(o_dims);
    VLOG(5) << "invoke 3x3 depthwise int8 conv";
    impl_ = conv_depthwise_3x3_int8;
  } else if (kw == 5) {
    // update w_data scale
    if (Ptype_out == PRECISION(kFloat) || Ptype_out == PRECISION(kInt8)) {
      CHECK_EQ(w_scale_.size(), oc) << "w_data scale size must be oc";
      float input_scale = param.input_scale;
      float output_scale = param.output_scale;
      for (auto& ws : w_scale_) {
        ws *= input_scale;
        if (Ptype_out == PRECISION(kInt8)) {
          ws /= output_scale;
        }
      }
    }

    const int wout_round = ((ow + 7) / 8) * 8;
    const int win_round = wout_round * sw + 5 - 1;
    const int hout_round = ((oh + 2) / 3) * 3;
    const int hin_round = hout_round * sw + 5 - 1;
    const int tmp_size_out = wout_round * hout_round;
    const int tmp_size_in = win_round * hin_round;
    const int tmp_size_io_bytes = tmp_size_in + tmp_size_out * sizeof(int);
    const int tmp_row_io_bytes = win_round + wout_round * sizeof(int);
    const int tmp_size_io_float =
        (tmp_size_io_bytes + sizeof(float) - 1) / sizeof(float);
    const int tmp_row_io_float =
        (tmp_row_io_bytes + sizeof(float) - 1) / sizeof(float);
    ctx_->ExtendWorkspace(DDim(std::vector<DDim::value_type>(
        {1, 1, 1, ctx_->threads() * tmp_size_io_float + tmp_row_io_float})));
    impl_ = conv_depthwise_5x5_int8;
    VLOG(5) << "invoke conv_depthwise_5x5 int8 conv";
  } else {
    LOG(ERROR) << "this type depthwise int8 conv not impl";
    return false;
  }
  return true;
}

template <PrecisionType Ptype_out>
bool DepthwiseConvInt8<Ptype_out>::init(const operators::ConvParam& param,
                                        Context<TARGET(kARM)>* ctx) {
  this->ctx_ = ctx;
  return create(param, ctx);
}

template <PrecisionType Ptype_out>
bool DepthwiseConvInt8<Ptype_out>::run(const operators::ConvParam& param) {
  const int8_t* i_data = param.x->data<int8_t>();
  int32_t* o_data = nullptr;
  const int8_t* w_data = param.filter->data<int8_t>();
  const int32_t* b_data = param.bias ? param.bias->data<int32_t>() : nullptr;

  LOG(INFO) << "input size: " << param.x->memory_size() << " "
            << param.input_scale << " " << w_scale_.size();

  auto x_dims = param.x->dims();
  auto w_dims = param.filter->dims();
  auto o_dims = param.output->dims();
  int bs = x_dims[0];
  int ic = x_dims[1];
  int ih = x_dims[2];
  int iw = x_dims[3];  // nchw
  int oc = o_dims[1];
  int oh = o_dims[2];
  int ow = o_dims[3];
  int kw = w_dims[3];
  int sw = param.strides[1];

  if (kw == 3 && Ptype_out != PRECISION(kInt32)) {
    o_data = tmp_int32_out_->mutable_data<int32_t>();
  } else if (kw == 5 || (kw == 3 && Ptype_out == PRECISION(kInt32))) {
    o_data = param.output->mutable_data<int32_t>();
  } else {
    LOG(ERROR) << "this type dw int8 conv not impl";
    return false;
  }

  impl_(i_data, o_data, bs, oc, oh, ow, ic, ih, kw, w_data, b_data, param,
        this->ctx_, Ptype_out, w_scale_.data());

  auto i_scale = param.input_scale;
  auto o_scale = param.output_scale;
  if (kw == 3) {
    if (Ptype_out == PRECISION(kInt8)) {
      param.output->mutable_data<int8_t>();
      trans_tensor_dtype<PRECISION(kInt32), PRECISION(kInt8)>(
          tmp_int32_out_, param.output, i_scale, o_scale, w_scale_);
    } else if (Ptype_out == PRECISION(kFloat)) {
      param.output->mutable_data<float>();
      trans_tensor_dtype<PRECISION(kInt32), PRECISION(kFloat)>(
          tmp_int32_out_, param.output, i_scale, 1.f, w_scale_);
    } else if (Ptype_out != PRECISION(kInt32)) {
      LOG(ERROR) << "unsupported precision type!!";
      return false;
    }
  }

  return true;
}

template class DepthwiseConvInt8<PRECISION(kInt8)>;
template class DepthwiseConvInt8<PRECISION(kFloat)>;
template class DepthwiseConvInt8<PRECISION(kInt32)>;

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle

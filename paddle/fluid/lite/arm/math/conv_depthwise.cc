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
    this->ctx_->ExtendWorkspace(DDim(std::vector<DDim::value_type>({1, 1, 1, iw + ow})));
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

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle

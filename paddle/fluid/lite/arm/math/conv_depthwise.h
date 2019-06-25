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

#pragma once

#include <cmath>
#include <vector>
#include "paddle/fluid/lite/arm/math/conv_impl.h"
#include "paddle/fluid/lite/core/context.h"
#include "paddle/fluid/lite/core/target_wrapper.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

template <PrecisionType Ptype>
class DepthwiseConv
    : public ImplBase<TARGET(kARM), Ptype, operators::ConvParam> {
 public:
  typedef void (*conv_dw_impl)(const float* i_data, float* o_data, int bs,
                               int oc, int oh, int ow, int ic, int ih, int kw,
                               const float* w_data, const float* b_data,
                               const operators::ConvParam& param,
                               Context<TARGET(kARM)>* ctx);
  DepthwiseConv() = default;
  ~DepthwiseConv() {}

  virtual bool init(const operators::ConvParam& param,
                    Context<TARGET(kARM)>* ctx);

  virtual bool create(const operators::ConvParam& param,
                      Context<TARGET(kARM)>* ctx);

  virtual bool run(const operators::ConvParam& param);

 private:
  conv_dw_impl impl_{nullptr};
};

template <PrecisionType Ptype_out>
class DepthwiseConvInt8
    : public ImplBase<TARGET(kARM), PRECISION(kInt8), operators::ConvParam> {
 public:
  typedef void (*conv_dw_int8_impl)(const int8_t* i_data, int32_t* o_data,
                                    int bs, int oc, int oh, int ow, int ic,
                                    int ih, int kw, const int8_t* w_data,
                                    const int32_t* b_data,
                                    const operators::ConvParam& param,
                                    Context<TARGET(kARM)>* ctx,
                                    PrecisionType out_type, const float* scale);

  DepthwiseConvInt8() = default;
  ~DepthwiseConvInt8() {
    if (tmp_int32_out_) {
      delete tmp_int32_out_;
    }
  }

  virtual bool init(const operators::ConvParam& param,
                    Context<TARGET(kARM)>* ctx);

  virtual bool create(const operators::ConvParam& param,
                      Context<TARGET(kARM)>* ctx);

  virtual bool run(const operators::ConvParam& param);

 private:
  conv_dw_int8_impl impl_{nullptr};
  std::vector<float> w_scale_;
  Tensor* tmp_int32_out_;
};

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle

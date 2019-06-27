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
class DirectConv : public ImplBase<TARGET(kARM), Ptype, operators::ConvParam> {
 public:
  typedef void (*conv_direct_impl)(const float* din, float* dout, int num,
                                   int chout, int hout, int wout, int chin,
                                   int hin, int win, const float* weights,
                                   const float* bias,
                                   const operators::ConvParam& param,
                                   Context<TARGET(kARM)>* ctx);

  DirectConv() = default;
  ~DirectConv() {}

  virtual bool init(const operators::ConvParam& param,
                    Context<TARGET(kARM)>* ctx);

  virtual bool create(const operators::ConvParam& param,
                      Context<TARGET(kARM)>* ctx);

  virtual bool run(const operators::ConvParam& param);

 protected:
  bool is_weights_transed_{false};
  Tensor weights_trans_;
  Tensor _tmp_out;

 private:
  conv_direct_impl impl_{nullptr};
};

template <PrecisionType Ptype_out>
class DirectConvInt8
    : public ImplBase<TARGET(kARM), PRECISION(kInt8), operators::ConvParam> {
 public:
  typedef void (*conv_direct_int8_impl)(
      const int8_t* din, int32_t* dout, int num, int chout, int hout, int wout,
      int chin, int hin, int win, const int8_t* weights, const int32_t* bias,
      const operators::ConvParam& param, Context<TARGET(kARM)>* ctx,
      PrecisionType out_type, const float* scale);

  DirectConvInt8() = default;
  ~DirectConvInt8() {}

  virtual bool init(const operators::ConvParam& param,
                    Context<TARGET(kARM)>* ctx);

  virtual bool create(const operators::ConvParam& param,
                      Context<TARGET(kARM)>* ctx);

  virtual bool run(const operators::ConvParam& param);

 private:
  bool is_weights_transed_{false};
  Tensor weights_trans_;
  Tensor _tmp_out;
  conv_direct_int8_impl impl_int8_{nullptr};
  std::vector<float> w_scale_;
};

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle

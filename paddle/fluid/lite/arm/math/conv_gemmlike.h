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
class GemmLikeConv
    : public ImplBase<TARGET(kARM), Ptype, operators::ConvParam> {
 public:
  typedef void (*conv_im2col_gemm_impl)(const float* din, float* dout, int num,
                                        int chout, int hout, int wout, int chin,
                                        int hin, int win, const float* weights,
                                        const float* bias,
                                        const operators::ConvParam& param,
                                        ARMContext* ctx, const int* idx_ptr);

  GemmLikeConv() = default;
  ~GemmLikeConv() {}

  virtual bool init(const operators::ConvParam& param, ARMContext* ctx) {
    LOG(FATAL) << "GemmLikeConv::init() not implemented.";
  }

  virtual bool create(const operators::ConvParam& param, ARMContext* ctx) {
    LOG(FATAL) << "GemmLikeConv::create() not implemented.";
  }

  virtual bool run(const operators::ConvParam& param) {
    LOG(FATAL) << "GemmLikeConv::run() not implemented.";
  }

 protected:
  bool is_weights_transed_{false};
  Tensor idx_data_;
  Tensor weights_trans_;

 private:
  conv_im2col_gemm_impl impl_{nullptr};
};

template <PrecisionType Ptype_out>
class GemmLikeConvInt8 : public GemmLikeConv<PRECISION(kInt8)> {
 public:
  typedef void (*conv_im2col_gemm_int8_impl)(
      const int8_t* din, int32_t* dout, int num, int chout, int hout, int wout,
      int chin, int hin, int win, const int8_t* weights, const int32_t* bias,
      const operators::ConvParam& param, ARMContext* ctx,
      PrecisionType out_type, const float* scale, const int* idx_ptr);

  GemmLikeConvInt8() = default;
  ~GemmLikeConvInt8() {}

  virtual bool init(const operators::ConvParam& param, ARMContext* ctx);

  virtual bool create(const operators::ConvParam& param, ARMContext* ctx);

  virtual bool run(const operators::ConvParam& param);

 private:
  conv_im2col_gemm_int8_impl impl_int8_{nullptr};
  std::vector<float> w_scale_;
};

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle

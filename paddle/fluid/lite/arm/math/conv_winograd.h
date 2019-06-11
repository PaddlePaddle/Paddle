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
#include "paddle/fluid/lite/arm/math/conv_impl.h"
#include "paddle/fluid/lite/core/context.h"
#include "paddle/fluid/lite/core/target_wrapper.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

template <PrecisionType Ptype>
class WinogradConv
    : public ImplBase<TARGET(kARM), Ptype, operators::ConvParam> {
 public:
  typedef void (*conv_winograd_impl)(const float* din, float* dout, int num,
                                     int chout, int hout, int wout, int chin,
                                     int hin, int win, const float* weights,
                                     const float* bias,
                                     const operators::ConvParam& param,
                                     Context<TARGET(kARM)>* ctx);

  WinogradConv() = default;
  ~WinogradConv() {}

  virtual bool init(const operators::ConvParam& param,
                    Context<TARGET(kARM)>* ctx);

  virtual bool create(const operators::ConvParam& param,
                      Context<TARGET(kARM)>* ctx);

  virtual bool run(const operators::ConvParam& param);

 private:
  conv_winograd_impl impl_{nullptr};
  bool is_weights_transed_{false};
  Tensor weights_trans_;
};

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle

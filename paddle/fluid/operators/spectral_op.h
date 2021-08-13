/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#pragma once
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

enum class FFTNormMode : int64_t {
  none,       // No normalization
  by_sqrt_n,  // Divide by sqrt(signal_size)
  by_n,       // Divide by signal_size
};

// Convert normalization mode string to enum values
// NOTE: for different direction, normalization modes have different meanings.
// eg: "forward" translates to `by_n` for a forward transform and `none` for
// backward.
FFTNormMode get_norm_from_string(const std::string& norm, bool forward) {
  if (!norm || *norm == "backward") {
    return forward ? FFTNormMode::none : FFTNormMode::by_n;
  }

  if (*norm == "forward") {
    return forward ? FFTNormMode::by_n : FFTNormMode::none;
  }

  if (*norm == "ortho") {
    return FFTNormMode::by_sqrt_n;
  }

  PADDLE_THROW(platform::errors::InvalidArgument(
      "Fft norm string must be forward or backward or ortho"));
}

template <typename DeviceContext, typename T>
class FFTC2CKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override;
};

}  // namespace operators
}  // namespace paddle

// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include <ATen/core/Tensor.h>
#include <c10/core/TensorOptions.h>
#include <optional>
#include <string_view>

#include "paddle/phi/api/include/api.h"
#include "paddle/phi/api/include/compat/ATen/ops/ones.h"

namespace at {

inline at::Tensor stft(
    const at::Tensor& self,
    int64_t n_fft,
    ::std::optional<int64_t> hop_length = ::std::nullopt,
    ::std::optional<int64_t> win_length = ::std::nullopt,
    const ::std::optional<at::Tensor>& window = ::std::nullopt,
    bool normalized = false,
    ::std::optional<bool> onesided = ::std::nullopt,
    ::std::optional<bool> return_complex = ::std::nullopt,
    ::std::optional<bool> align_to_window = ::std::nullopt) {
  (void)return_complex;
  (void)align_to_window;
  (void)win_length;

  // Resolve defaults
  int64_t resolved_hop_length = hop_length.value_or(n_fft / 4);
  if (resolved_hop_length <= 0) {
    resolved_hop_length = 1;
  }

  bool resolved_onesided = onesided.value_or(true);

  // Create window if not provided
  at::Tensor resolved_window;
  if (window.has_value()) {
    resolved_window = window.value();
    // Paddle requires window size == n_fft
    if (resolved_window.numel() != n_fft) {
      resolved_window = at::ones({n_fft}, self.options());
    }
  } else {
    resolved_window = at::ones({n_fft}, self.options());
  }

  // Paddle stft expects 2D input [batch, time]
  bool need_unsqueeze = (self.dim() == 1);
  at::Tensor input = need_unsqueeze ? self.unsqueeze(0) : self;

  paddle::Tensor result =
      paddle::experimental::stft(input._PD_GetInner(),
                                 resolved_window._PD_GetInner(),
                                 static_cast<int>(n_fft),
                                 static_cast<int>(resolved_hop_length),
                                 normalized,
                                 resolved_onesided);

  at::Tensor output(result);

  // Remove batch dim if input was 1D
  if (need_unsqueeze) {
    output = output.squeeze(0);
  }

  return output;
}

}  // namespace at

namespace at {

inline at::Tensor Tensor::stft(int64_t n_fft,
                               ::std::optional<int64_t> hop_length,
                               ::std::optional<int64_t> win_length,
                               const ::std::optional<at::Tensor>& window,
                               bool normalized,
                               ::std::optional<bool> onesided,
                               ::std::optional<bool> return_complex,
                               ::std::optional<bool> align_to_window) const {
  return at::stft(*this,
                  n_fft,
                  hop_length,
                  win_length,
                  window,
                  normalized,
                  onesided,
                  return_complex,
                  align_to_window);
}

}  // namespace at

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
#include <c10/util/string_view.h>
#include <climits>
#include <optional>

#include "paddle/phi/api/include/api.h"
#include "paddle/phi/api/include/compat/ATen/ops/ones.h"
#include "paddle/phi/api/include/compat/ATen/ops/slice.h"
#include "paddle/phi/api/include/compat/ATen/ops/squeeze.h"
#include "paddle/phi/api/include/compat/ATen/ops/unsqueeze.h"
#include "paddle/phi/api/include/compat/ATen/ops/zeros.h"

namespace at {

inline at::Tensor stft(const at::Tensor& self,
                       int64_t n_fft,
                       ::std::optional<int64_t> hop_length,
                       ::std::optional<int64_t> win_length,
                       const ::std::optional<at::Tensor>& window,
                       bool normalized,
                       ::std::optional<bool> onesided = ::std::nullopt,
                       ::std::optional<bool> return_complex = ::std::nullopt,
                       ::std::optional<bool> align_to_window = ::std::nullopt) {
  PD_CHECK(n_fft > 0, "stft expected n_fft > 0, but got ", n_fft);
  PD_CHECK(!align_to_window.value_or(false),
           "stft with align_to_window=true is not supported in compat layer");

  // Resolve defaults
  int64_t resolved_hop_length = hop_length.value_or(n_fft / 4);
  PD_CHECK(resolved_hop_length > 0,
           "stft expected hop_length > 0, but got ",
           resolved_hop_length);

  int64_t resolved_win_length = win_length.value_or(n_fft);
  PD_CHECK(resolved_win_length > 0 && resolved_win_length <= n_fft,
           "stft expected 0 < win_length <= n_fft");

  bool resolved_onesided = onesided.value_or(true);
  PD_CHECK(resolved_onesided,
           "stft with onesided=false is not supported in compat layer");

  // Validate n_fft / hop_length fit into int before any allocation
  PD_CHECK(n_fft <= INT_MAX, "stft expected n_fft <= INT_MAX, but got ", n_fft);
  PD_CHECK(resolved_hop_length <= INT_MAX,
           "stft expected hop_length <= INT_MAX, but got ",
           resolved_hop_length);

  // Create window
  at::Tensor resolved_window;
  if (window.has_value()) {
    resolved_window = window.value();
    PD_CHECK(resolved_window.dim() == 1 &&
                 resolved_window.numel() == resolved_win_length,
             "stft expected window to be 1D with size win_length");
    PD_CHECK(resolved_window.scalar_type() == self.scalar_type(),
             "stft expected window to have the same dtype as input");
    PD_CHECK(resolved_window.device() == self.device(),
             "stft expected window to be on the same device as input");
  } else {
    resolved_window = at::ones({resolved_win_length}, self.options());
  }

  // Zero-pad window to n_fft if needed (centered, matching PyTorch)
  if (resolved_window.numel() < n_fft) {
    at::Tensor padded = at::zeros({n_fft}, self.options());
    int64_t pad_left = (n_fft - resolved_win_length) / 2;
    padded.slice(0, pad_left, pad_left + resolved_win_length)
        .copy_(resolved_window);
    resolved_window = padded;
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

  bool resolved_return_complex = false;
  if (return_complex.has_value()) {
    resolved_return_complex = return_complex.value();
  } else {
    bool self_complex = self.scalar_type() == at::kComplexFloat ||
                        self.scalar_type() == at::kComplexDouble;
    bool window_complex = resolved_window.scalar_type() == at::kComplexFloat ||
                          resolved_window.scalar_type() == at::kComplexDouble;
    resolved_return_complex = self_complex || window_complex;
    PD_CHECK(resolved_return_complex,
             "stft requires return_complex to be specified for real inputs");
  }
  if (resolved_return_complex) {
    return output;
  }
  return at::Tensor(paddle::experimental::as_real(output._PD_GetInner()));
}

inline at::Tensor stft(const at::Tensor& self,
                       int64_t n_fft,
                       ::std::optional<int64_t> hop_length = ::std::nullopt,
                       ::std::optional<int64_t> win_length = ::std::nullopt,
                       const ::std::optional<at::Tensor>& window = {},
                       bool center = true,
                       c10::string_view pad_mode = "reflect",
                       bool normalized = false,
                       ::std::optional<bool> onesided = ::std::nullopt,
                       ::std::optional<bool> return_complex = ::std::nullopt,
                       ::std::optional<bool> align_to_window = ::std::nullopt) {
  (void)pad_mode;
  PD_CHECK(!center, "stft with center=true is not supported in compat layer");
  return at::stft(self,
                  n_fft,
                  hop_length,
                  win_length,
                  window,
                  normalized,
                  onesided,
                  return_complex,
                  align_to_window);
}

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

inline at::Tensor Tensor::stft(int64_t n_fft,
                               ::std::optional<int64_t> hop_length,
                               ::std::optional<int64_t> win_length,
                               const ::std::optional<at::Tensor>& window,
                               bool center,
                               c10::string_view pad_mode,
                               bool normalized,
                               ::std::optional<bool> onesided,
                               ::std::optional<bool> return_complex,
                               ::std::optional<bool> align_to_window) const {
  return at::stft(*this,
                  n_fft,
                  hop_length,
                  win_length,
                  window,
                  center,
                  pad_mode,
                  normalized,
                  onesided,
                  return_complex,
                  align_to_window);
}

}  // namespace at

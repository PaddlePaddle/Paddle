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
#include <string>
#include <unordered_set>
#include <vector>

#include "paddle/phi/api/include/api.h"
#include "paddle/phi/api/include/compat/ATen/ops/ones.h"
#include "paddle/phi/api/include/compat/ATen/ops/slice.h"
#include "paddle/phi/api/include/compat/ATen/ops/squeeze.h"
#include "paddle/phi/api/include/compat/ATen/ops/unsqueeze.h"
#include "paddle/phi/api/include/compat/ATen/ops/zeros.h"
#include "paddle/phi/common/type_promotion.h"

namespace at {

namespace detail {

inline bool _PD_stft_supported_dtype(at::ScalarType dtype) {
  return dtype == at::kFloat || dtype == at::kDouble ||
         dtype == at::kComplexFloat || dtype == at::kComplexDouble;
}

inline bool _PD_stft_complex_dtype(phi::DataType dtype) {
  return dtype == phi::DataType::COMPLEX64 ||
         dtype == phi::DataType::COMPLEX128;
}

inline at::Tensor _PD_stft_pad_input(const at::Tensor& input,
                                     int64_t pad,
                                     c10::string_view pad_mode) {
  std::string mode(pad_mode.data(), pad_mode.size());
  PD_CHECK(mode == "reflect" || mode == "constant" || mode == "replicate" ||
               mode == "circular",
           "stft expected pad_mode to be reflect, constant, replicate, or "
           "circular, but got ",
           mode);

  const int64_t batch = input.size(0);
  const int64_t length = input.size(1);
  paddle::Tensor expanded = paddle::experimental::reshape(
      input._PD_GetInner(), std::vector<int64_t>{batch, 1, 1, 1, length});
  paddle::Tensor padded = paddle::experimental::pad3d(
      expanded, std::vector<int64_t>{pad, pad, 0, 0, 0, 0}, mode, 0.0, "NCDHW");
  return at::Tensor(paddle::experimental::reshape(
      padded, std::vector<int64_t>{batch, length + 2 * pad}));
}

inline at::Tensor _PD_stft(const at::Tensor& self,
                           int64_t n_fft,
                           ::std::optional<int64_t> hop_length,
                           ::std::optional<int64_t> win_length,
                           const ::std::optional<at::Tensor>& window,
                           bool center,
                           c10::string_view pad_mode,
                           bool normalized,
                           ::std::optional<bool> onesided,
                           ::std::optional<bool> return_complex,
                           ::std::optional<bool> align_to_window) {
  PD_CHECK(n_fft > 0, "stft expected n_fft > 0, but got ", n_fft);
  PD_CHECK(!align_to_window.value_or(false),
           "stft with align_to_window=true is not supported in compat layer");
  PD_CHECK(_PD_stft_supported_dtype(self.scalar_type()),
           "stft expected a float, double, complex float, or complex double "
           "input");
  PD_CHECK(self.dim() == 1 || self.dim() == 2,
           "stft expected a 1D or 2D tensor");

  int64_t resolved_hop_length = hop_length.value_or(n_fft / 4);
  PD_CHECK(resolved_hop_length > 0,
           "stft expected hop_length > 0, but got ",
           resolved_hop_length);

  int64_t resolved_win_length = win_length.value_or(n_fft);
  PD_CHECK(resolved_win_length > 0 && resolved_win_length <= n_fft,
           "stft expected 0 < win_length <= n_fft");

  PD_CHECK(n_fft <= INT_MAX, "stft expected n_fft <= INT_MAX, but got ", n_fft);
  PD_CHECK(resolved_hop_length <= INT_MAX,
           "stft expected hop_length <= INT_MAX, but got ",
           resolved_hop_length);

  at::Tensor resolved_window;
  if (window.has_value()) {
    resolved_window = window.value();
    PD_CHECK(resolved_window.dim() == 1 &&
                 resolved_window.numel() == resolved_win_length,
             "stft expected window to be 1D with size win_length");
    PD_CHECK(_PD_stft_supported_dtype(resolved_window.scalar_type()),
             "stft expected a float, double, complex float, or complex double "
             "window");
    PD_CHECK(resolved_window.device() == self.device(),
             "stft expected window to be on the same device as input");
  } else {
    resolved_window = at::ones({resolved_win_length}, self.options());
  }

  if (resolved_window.numel() < n_fft) {
    at::Tensor padded = at::zeros({n_fft}, resolved_window.options());
    int64_t pad_left = (n_fft - resolved_win_length) / 2;
    padded.slice(0, pad_left, pad_left + resolved_win_length)
        .copy_(resolved_window);
    resolved_window = padded;
  }

  bool need_unsqueeze = (self.dim() == 1);
  at::Tensor input = need_unsqueeze ? self.unsqueeze(0) : self;
  if (center) {
    input = _PD_stft_pad_input(input, n_fft / 2, pad_mode);
  }
  PD_CHECK(n_fft <= input.size(1),
           "stft expected n_fft <= input length after padding");

  paddle::Tensor frames =
      paddle::experimental::frame(input._PD_GetInner(),
                                  static_cast<int>(n_fft),
                                  static_cast<int>(resolved_hop_length),
                                  -1);
  frames = paddle::experimental::transpose(frames, {0, 2, 1});

  phi::DataType promoted_dtype =
      phi::promoteTypes(frames.dtype(), resolved_window._PD_GetInner().dtype());
  if (frames.dtype() != promoted_dtype) {
    frames = paddle::experimental::cast(frames, promoted_dtype);
  }
  paddle::Tensor inner_window = resolved_window._PD_GetInner();
  if (inner_window.dtype() != promoted_dtype) {
    inner_window = paddle::experimental::cast(inner_window, promoted_dtype);
  }
  frames = paddle::experimental::multiply(frames, inner_window);

  const bool complex_input = _PD_stft_complex_dtype(promoted_dtype);
  const bool resolved_onesided = onesided.value_or(!complex_input);
  PD_CHECK(!complex_input || !resolved_onesided,
           "stft cannot have onesided output when input or window is complex");

  const std::string normalization = normalized ? "ortho" : "backward";
  paddle::Tensor result;
  if (complex_input) {
    result = paddle::experimental::fft_c2c(
        frames, std::vector<int64_t>{2}, normalization, true);
  } else {
    result = paddle::experimental::fft_r2c(frames,
                                           std::vector<int64_t>{2},
                                           normalization,
                                           true,
                                           resolved_onesided);
  }
  result = paddle::experimental::transpose(result, {0, 2, 1});

  at::Tensor output(result);
  if (need_unsqueeze) {
    output = output.squeeze(0);
  }

  bool resolved_return_complex;
  if (return_complex.has_value()) {
    resolved_return_complex = return_complex.value();
  } else {
    PD_CHECK(complex_input,
             "stft requires return_complex to be specified for real inputs");
    resolved_return_complex = true;
  }
  if (resolved_return_complex) {
    return output;
  }
  return at::Tensor(paddle::experimental::as_real(output._PD_GetInner()));
}

}  // namespace detail

inline at::Tensor stft(const at::Tensor& self,
                       int64_t n_fft,
                       ::std::optional<int64_t> hop_length,
                       ::std::optional<int64_t> win_length,
                       const ::std::optional<at::Tensor>& window,
                       bool normalized,
                       ::std::optional<bool> onesided = ::std::nullopt,
                       ::std::optional<bool> return_complex = ::std::nullopt,
                       ::std::optional<bool> align_to_window = ::std::nullopt) {
  return at::detail::_PD_stft(self,
                              n_fft,
                              hop_length,
                              win_length,
                              window,
                              false,
                              "constant",
                              normalized,
                              onesided,
                              return_complex,
                              align_to_window);
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
  return at::detail::_PD_stft(self,
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

// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include <ATen/ops/stft.h>
#include <gtest/gtest.h>

#include <optional>

// ============================================================
// Tests for at::Tensor::stft()
// ============================================================

TEST(TensorStftTest, StftBasicFloat) {
  // 2D input [batch=1, time=16], n_fft=8, hop_length=4
  at::Tensor t = at::ones({1, 16}, at::kFloat);
  at::Tensor result = t.stft(/*n_fft=*/8,
                             /*hop_length=*/4,
                             /*win_length=*/8,
                             /*window=*/::std::nullopt,
                             /*normalized=*/false,
                             /*onesided=*/::std::nullopt,
                             /*return_complex=*/true);

  // onesided=true → freq = n_fft/2 + 1 = 5
  // hop_length=4, n_fft=8, time=16 → frames = 1 + (16-8)/4 = 3
  ASSERT_EQ(result.dim(), 3);
  ASSERT_EQ(result.sizes()[0], 1);
  ASSERT_EQ(result.sizes()[1], 5);  // n_fft/2 + 1
  ASSERT_EQ(result.sizes()[2], 3);  // n_frames
}

TEST(TensorStftTest, StftWithCustomHop) {
  // 2D input [batch=2, time=32], n_fft=16, hop_length=8
  at::Tensor t = at::ones({2, 32}, at::kFloat);
  at::Tensor result = t.stft(/*n_fft=*/16,
                             /*hop_length=*/8,
                             /*win_length=*/16,
                             /*window=*/::std::nullopt,
                             /*normalized=*/false,
                             /*onesided=*/true,
                             /*return_complex=*/true);

  ASSERT_EQ(result.dim(), 3);
  ASSERT_EQ(result.sizes()[0], 2);
  ASSERT_EQ(result.sizes()[1], 9);  // n_fft/2 + 1 = 16/2 + 1
  ASSERT_EQ(result.sizes()[2], 3);  // 1 + (32-16)/8 = 3
}

TEST(TensorStftTest, StftWithWindow) {
  // 2D input with custom window
  at::Tensor t = at::ones({1, 16}, at::kFloat);
  at::Tensor window = at::ones({8}, at::kFloat);
  at::Tensor result = t.stft(/*n_fft=*/8,
                             /*hop_length=*/4,
                             /*win_length=*/8,
                             /*window=*/window,
                             /*normalized=*/false,
                             /*onesided=*/true,
                             /*return_complex=*/true);

  ASSERT_EQ(result.dim(), 3);
  ASSERT_EQ(result.sizes()[1], 5);
  ASSERT_EQ(result.sizes()[2], 3);
}

TEST(TensorStftTest, StftOnesidedFalseUnsupported) {
  // onesided=false is explicitly unsupported in compat layer
  at::Tensor t = at::ones({1, 16}, at::kFloat);
  EXPECT_THROW(t.stft(/*n_fft=*/8,
                      /*hop_length=*/4,
                      /*win_length=*/8,
                      /*window=*/::std::nullopt,
                      /*normalized=*/false,
                      /*onesided=*/false),
               std::exception);
}

TEST(TensorStftTest, Stft1DInput) {
  // 1D input should be treated as [time] and unsqueezed to [1, time]
  at::Tensor t = at::ones({16}, at::kFloat);
  at::Tensor result = t.stft(/*n_fft=*/8,
                             /*hop_length=*/4,
                             /*win_length=*/8,
                             /*window=*/::std::nullopt,
                             /*normalized=*/false,
                             /*onesided=*/true,
                             /*return_complex=*/true);

  // Output should be 2D [freq, frames] after squeezing batch dim
  ASSERT_EQ(result.dim(), 2);
  ASSERT_EQ(result.sizes()[0], 5);  // n_fft/2 + 1
  ASSERT_EQ(result.sizes()[1], 3);  // n_frames
}

TEST(TensorStftTest, StftDoubleDtype) {
  at::Tensor t = at::ones({1, 16}, at::kDouble);
  at::Tensor result = t.stft(/*n_fft=*/8,
                             /*hop_length=*/4,
                             /*win_length=*/8,
                             /*window=*/::std::nullopt,
                             /*normalized=*/false,
                             /*onesided=*/true,
                             /*return_complex=*/true);

  ASSERT_EQ(result.dim(), 3);
  ASSERT_EQ(result.sizes()[1], 5);
  ASSERT_EQ(result.sizes()[2], 3);
}

TEST(TensorStftTest, StftReturnComplexFalse) {
  // return_complex=false should return real tensor with last dim = 2
  at::Tensor t = at::ones({1, 16}, at::kFloat);
  at::Tensor result = t.stft(/*n_fft=*/8,
                             /*hop_length=*/4,
                             /*win_length=*/8,
                             /*window=*/::std::nullopt,
                             /*normalized=*/false,
                             /*onesided=*/true,
                             /*return_complex=*/false);

  ASSERT_EQ(result.dim(), 4);
  ASSERT_EQ(result.sizes()[0], 1);
  ASSERT_EQ(result.sizes()[1], 5);  // n_fft/2 + 1
  ASSERT_EQ(result.sizes()[2], 3);  // n_frames
  ASSERT_EQ(result.sizes()[3], 2);  // real/imag
  ASSERT_EQ(result.scalar_type(), at::kFloat);
}

TEST(TensorStftTest, StftWinLengthSmallerThanNFFT) {
  // win_length < n_fft: window should be zero-padded to n_fft
  at::Tensor t = at::ones({1, 16}, at::kFloat);
  at::Tensor window = at::ones({4}, at::kFloat);
  at::Tensor result = t.stft(/*n_fft=*/8,
                             /*hop_length=*/4,
                             /*win_length=*/4,
                             /*window=*/window,
                             /*normalized=*/false,
                             /*onesided=*/true,
                             /*return_complex=*/true);

  ASSERT_EQ(result.dim(), 3);
  ASSERT_EQ(result.sizes()[0], 1);
  ASSERT_EQ(result.sizes()[1], 5);  // n_fft/2 + 1
  ASSERT_EQ(result.sizes()[2], 3);  // n_frames
}

TEST(TensorStftTest, StftInvalidHopLength) {
  at::Tensor t = at::ones({1, 16}, at::kFloat);
  EXPECT_THROW(t.stft(/*n_fft=*/8,
                      /*hop_length=*/0,
                      /*win_length=*/8,
                      /*window=*/::std::nullopt,
                      /*normalized=*/false,
                      /*onesided=*/true),
               std::exception);
}

TEST(TensorStftTest, StftInvalidWinLength) {
  at::Tensor t = at::ones({1, 16}, at::kFloat);
  EXPECT_THROW(t.stft(/*n_fft=*/8,
                      /*hop_length=*/4,
                      /*win_length=*/10,  // > n_fft
                      /*window=*/::std::nullopt,
                      /*normalized=*/false,
                      /*onesided=*/true),
               std::exception);
}

TEST(TensorStftTest, StftPyTorchStyleCenterFalseOverload) {
  at::Tensor t = at::ones({1, 16}, at::kFloat);
  at::Tensor result = t.stft(/*n_fft=*/8,
                             /*hop_length=*/4,
                             /*win_length=*/8,
                             /*window=*/::std::nullopt,
                             /*center=*/false,
                             /*pad_mode=*/"reflect",
                             /*normalized=*/true,
                             /*onesided=*/true,
                             /*return_complex=*/true);

  ASSERT_EQ(result.dim(), 3);
  ASSERT_EQ(result.sizes()[0], 1);
  ASSERT_EQ(result.sizes()[1], 5);
  ASSERT_EQ(result.sizes()[2], 3);
}

TEST(TensorStftTest, StftFreeFunctionLegacySchema) {
  at::Tensor t = at::ones({1, 16}, at::kFloat);
  at::Tensor result = at::stft(t,
                               /*n_fft=*/8,
                               /*hop_length=*/4,
                               /*win_length=*/8,
                               /*window=*/::std::nullopt,
                               /*normalized=*/false,
                               /*onesided=*/true,
                               /*return_complex=*/true);

  ASSERT_EQ(result.dim(), 3);
  ASSERT_EQ(result.sizes()[0], 1);
  ASSERT_EQ(result.sizes()[1], 5);
  ASSERT_EQ(result.sizes()[2], 3);
}

TEST(TensorStftTest, StftFreeFunctionPyTorchStyleCenterFalseOverload) {
  at::Tensor t = at::ones({1, 16}, at::kFloat);
  at::Tensor result = at::stft(t,
                               /*n_fft=*/8,
                               /*hop_length=*/4,
                               /*win_length=*/8,
                               /*window=*/::std::nullopt,
                               /*center=*/false,
                               /*pad_mode=*/"reflect",
                               /*normalized=*/true,
                               /*onesided=*/true,
                               /*return_complex=*/true);

  ASSERT_EQ(result.dim(), 3);
  ASSERT_EQ(result.sizes()[0], 1);
  ASSERT_EQ(result.sizes()[1], 5);
  ASSERT_EQ(result.sizes()[2], 3);
}

TEST(TensorStftTest, StftRealInputRequiresReturnComplex) {
  at::Tensor t = at::ones({1, 16}, at::kFloat);
  EXPECT_THROW(t.stft(/*n_fft=*/8,
                      /*hop_length=*/4,
                      /*win_length=*/8,
                      /*window=*/::std::nullopt,
                      /*normalized=*/false,
                      /*onesided=*/true),
               std::exception);
}

TEST(TensorStftTest, StftCenterTrueUnsupported) {
  at::Tensor t = at::ones({1, 16}, at::kFloat);
  EXPECT_THROW(t.stft(/*n_fft=*/8,
                      /*hop_length=*/4,
                      /*win_length=*/8,
                      /*window=*/::std::nullopt,
                      /*center=*/true,
                      /*pad_mode=*/"reflect",
                      /*normalized=*/false,
                      /*onesided=*/true,
                      /*return_complex=*/true),
               std::exception);
}

TEST(TensorStftTest, StftAlignToWindowUnsupported) {
  at::Tensor t = at::ones({1, 16}, at::kFloat);
  EXPECT_THROW(t.stft(/*n_fft=*/8,
                      /*hop_length=*/4,
                      /*win_length=*/8,
                      /*window=*/::std::nullopt,
                      /*normalized=*/false,
                      /*onesided=*/true,
                      /*return_complex=*/true,
                      /*align_to_window=*/true),
               std::exception);
}

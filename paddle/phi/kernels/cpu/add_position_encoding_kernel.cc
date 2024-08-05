// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

namespace phi {

template <typename T, typename Context>
void AddPositionEncodingKernel(const Context& dev_ctx,
                               const DenseTensor& x_in,
                               float alpha,
                               float beta,
                               DenseTensor* out) {
  auto* X = &x_in;
  auto& x_lod = X->lod();
  auto* src_ptr = X->data<T>();

  auto* dst_ptr = dev_ctx.template Alloc<T>(out);

  auto x_dim = X->dims();
  int batch_size = 0;
  int max_seq_len = 0;
  int enc_size = 0;

  if (x_lod.empty()) {
    PADDLE_ENFORCE_EQ(x_dim.size(),
                      3,
                      common::errors::InvalidArgument(
                          "The input(X)'s dimension of AddPositionEncodingOp "
                          "should be equal to "
                          "3, but received %d. ",
                          x_dim.size()));
    batch_size = x_dim[0];
    max_seq_len = x_dim[1];
    enc_size = x_dim[2];
  } else {
    PADDLE_ENFORCE_EQ(x_dim.size(),
                      2,
                      common::errors::InvalidArgument(
                          "The input(X)'s dimension of AddPositionEncodingOp "
                          "should be equal to "
                          "2, but received %d. ",
                          x_dim.size()));
    PADDLE_ENFORCE_EQ(x_lod.size(),
                      1,
                      common::errors::InvalidArgument(
                          "The input(X)'s lod level of AddPositionEncodingOp "
                          "should be equal to "
                          "1, but received %d. ",
                          x_lod.size()));

    batch_size = x_lod[0].size() - 1;
    max_seq_len = -1;
    enc_size = x_dim[1];
  }

  PADDLE_ENFORCE_EQ(enc_size % 2,
                    0,
                    common::errors::InvalidArgument(
                        "The input(X)'s feature size of "
                        "AddPositionEncodingOp only support even, "
                        "but received an odd number: %d. ",
                        enc_size));

  const int half_size = enc_size / 2;
  for (int i = 0; i < batch_size; ++i) {
    const int max_length =
        x_lod.empty() ? max_seq_len : x_lod[0][i + 1] - x_lod[0][i];
    for (int j = 0; j < max_length; ++j) {
      for (int k = 0; k < half_size; ++k) {
        const double val =
            (half_size > 1)
                ? j / pow(10000.0, static_cast<double>(k) / (half_size - 1))
                : j / 10000.0;
        dst_ptr[k] = src_ptr[k] * alpha + sin(val) * beta;
        dst_ptr[half_size + k] =
            src_ptr[half_size + k] * alpha + cos(val) * beta;
      }
      src_ptr += enc_size;
      dst_ptr += enc_size;
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(add_position_encoding,
                   CPU,
                   ALL_LAYOUT,
                   phi::AddPositionEncodingKernel,
                   float,
                   double) {}

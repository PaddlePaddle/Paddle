// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/scaled_dp_attention_functor.h"

namespace phi {
namespace fusion {

template <typename T, typename Context>
void SelfDPAttenKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       const float alpha,
                       const int head_number,
                       DenseTensor* out) {
  auto* input_d = x.data<T>();
  auto* output_d = dev_ctx.template Alloc<T>(out);
  float scale = static_cast<float>(alpha);
  auto input_dims = x.dims();
  // in shouble be (batch * seq * 3 * head_num * head_size)
  // out shouble be (batch * seq * head_num * head_size)
  int batch_size = input_dims[0];
  int seq_len = input_dims[1];
  int head_size = input_dims[4];

  DenseTensor temp1, temp2;
  temp1.Resize(input_dims);
  float* trans_input = dev_ctx.template Alloc<float>(&temp1);
  temp2.Resize(input_dims);
  float* trans_output = dev_ctx.template Alloc<float>(&temp2);

  phi::funcs::transpose_before_bmm1<T, float>(
      input_d, trans_input, batch_size, seq_len, head_number, head_size);
  float* query = trans_input;
  float* key = trans_input + batch_size * head_number * seq_len * head_size;
  float* value =
      trans_input + batch_size * head_number * seq_len * head_size * 2;

  phi::funcs::scaled_dp_attention(query,
                                  key,
                                  value,
                                  scale,
                                  batch_size,
                                  seq_len,
                                  seq_len,
                                  head_number,
                                  head_size,
                                  trans_output);
  phi::funcs::transpose_after_bmm2<float, T>(
      trans_output, output_d, batch_size, seq_len, head_number, head_size);
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(self_dp_attention,
                   CPU,
                   ALL_LAYOUT,
                   phi::fusion::SelfDPAttenKernel,
                   float,
                   double) {}

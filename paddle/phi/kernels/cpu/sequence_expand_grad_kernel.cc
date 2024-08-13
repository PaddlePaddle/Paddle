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
#include "paddle/phi/kernels/impl/sequence_expand_kernel_impl.h"

namespace phi {

/*
 *Given Grad(Out)
 *
 *    Grad(Out).lod = [[0,                            2],
 *                     [0,              3,            6]]
 *    Grad(Out).data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
 * Then
 *    Grad(X).data = [(0.1 + 0.2 + 0.3), (0.4 + 0.5 + 0.6)]
 *                 = [0.6, 1.5]
 *    Grad(X).lod = Input(X).lod
 *
 * */
template <typename T>
struct SequenceExpandGradFunctor<phi::CPUContext, T> {
  void operator()(const phi::CPUContext& context,
                  const phi::DenseTensor& dout,
                  const phi::Vector<size_t>& x_lod,   /*expand source lod*/
                  const phi::Vector<size_t>& ref_lod, /*expand referenced lod*/
                  phi::DenseTensor* dx) {
    int dout_offset = 0;
    for (size_t i = 1; i < ref_lod.size(); ++i) {
      int repeat_num = ref_lod[i] - ref_lod[i - 1];
      if (repeat_num > 0) {
        int x_start = x_lod[i - 1];
        int x_end = x_lod[i];
        int x_seq_len = x_end - x_start;
        if (x_seq_len == 0) continue;
        auto dx_sub = dx->Slice(x_start, x_end);
        dx_sub.Resize(common::flatten_to_1d(dx_sub.dims()));
        int dout_end = dout_offset + repeat_num * x_seq_len;
        auto dout_sub = dout.Slice(dout_offset, dout_end);
        dout_sub.Resize({repeat_num, dx_sub.dims()[0]});
        phi::funcs::ColwiseSum<phi::CPUContext, T> col_sum;
        col_sum(context, dout_sub, &dx_sub);
        dout_offset += repeat_num * x_seq_len;
      }
    }
  }
};

}  // namespace phi
PD_REGISTER_KERNEL(sequence_expand_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::SequenceExpandGradKernel,
                   float,
                   double,
                   int,
                   int64_t) {}

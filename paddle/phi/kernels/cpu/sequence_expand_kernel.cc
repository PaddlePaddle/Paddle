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

template <typename T>
struct SequenceExpandFunctor<phi::CPUContext, T> {
  void operator()(const phi::CPUContext& context UNUSED,
                  const phi::DenseTensor& x,
                  const phi::Vector<size_t>& x_lod,   /*expand source lod*/
                  const phi::Vector<size_t>& ref_lod, /*expand referenced lod*/
                  phi::DenseTensor* out) {
    int out_offset = 0;
    int x_item_length = x.numel() / x.dims()[0];
    auto out_data = out->data<T>();
    auto x_data = x.data<T>();
    for (size_t i = 1; i < ref_lod.size(); ++i) {
      int repeat_num = ref_lod[i] - ref_lod[i - 1];
      int x_start = x_lod[i - 1];
      int x_end = x_lod[i];
      int x_seq_len = x_end - x_start;
      if (repeat_num > 0) {
        int out_start = out_offset;
        if (out->lod().size() == 1) {
          out_start = out->lod()[0][out_offset];
        }
        for (int j = 0; j < repeat_num; j++) {
          for (int k = 0; k < x_seq_len; k++) {
            for (int l = 0; l < x_item_length; l++) {
              out_data[(out_start + j * x_seq_len + k) * x_item_length + l] =
                  x_data[(x_start + k) * x_item_length + l];
            }
          }
        }
      }
      out_offset += repeat_num;
    }
  }
};

}  // namespace phi

PD_REGISTER_KERNEL(sequence_expand,
                   CPU,
                   ALL_LAYOUT,
                   phi::SequenceExpandKernel,
                   float,
                   double,
                   int,
                   int64_t) {}

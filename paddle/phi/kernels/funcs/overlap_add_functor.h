// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/funcs/for_range.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/seq2col.h"

namespace phi {

template <typename Context, typename T>
struct OverlapAddFunctor {
  void operator()(const Context& dev_ctx,
                  const DenseTensor* input,
                  DenseTensor* output,
                  size_t seq_length,
                  size_t frame_length,
                  size_t n_frames,
                  size_t hop_length,
                  bool is_grad = false) const {
    auto numel = output->numel();
    const auto* input_data = input->data<T>();
    auto* output_data = output->data<T>();

    phi::funcs::ForRange<Context> for_range(dev_ctx, numel);
    if (!is_grad) {
      phi::funcs::Col2SeqFunctor<T> functor(input_data,
                                            output_data,
                                            seq_length,
                                            frame_length,
                                            n_frames,
                                            hop_length);
      for_range(functor);
    } else {
      phi::funcs::Seq2ColFunctor<T> functor(input_data,
                                            output_data,
                                            seq_length,
                                            frame_length,
                                            n_frames,
                                            hop_length);
      for_range(functor);
    }
  }
};

}  // namespace phi

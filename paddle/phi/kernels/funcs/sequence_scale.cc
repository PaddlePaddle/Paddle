/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/funcs/sequence_scale.h"
#include "paddle/phi/backends/cpu/cpu_context.h"

namespace phi {
class DenseTensor;
}  // namespace phi

namespace phi::funcs {

template <typename T>
class ScaleDenseTensorFunctor<phi::CPUContext, T> {
 public:
  void operator()(const phi::CPUContext& context,
                  const T* scales,
                  phi::DenseTensor* seq) {
    const size_t level = 0;
    auto lod = seq->lod();
    const size_t num_seq = lod[level].size() - 1;
    size_t seq_width = seq->dims()[1];
    phi::LegacyLoD abs_offset_lod = phi::ToAbsOffset(lod);

    T* seq_data = context.template Alloc<T>(seq);
    for (size_t i = 0; i < num_seq; ++i) {
      for (size_t j = lod[level][i] * seq_width;
           j < lod[level][i + 1] * seq_width;
           ++j) {
        seq_data[j] *= scales[i];
      }
    }
  }
};

template class ScaleDenseTensorFunctor<phi::CPUContext, float>;
template class ScaleDenseTensorFunctor<phi::CPUContext, double>;

}  // namespace phi::funcs

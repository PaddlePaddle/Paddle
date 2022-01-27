// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/pten/kernels/funcs/transpose.h"
#include "paddle/fluid/framework/ddim.h"
#include "paddle/pten/backends/cpu/cpu_context.h"
#include "paddle/pten/core/dense_tensor.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/platform/bfloat16.h"
#include "paddle/fluid/platform/complex.h"
#include "paddle/fluid/platform/float16.h"

namespace pten {
namespace math {

template <typename T>
struct TransposeNormal<CPUContext, T> {
  // for dims >= 7 situation
  void operator()(const CPUContext& dev_ctx,
                  const pten::DenseTensor& in,
                  pten::DenseTensor* out,
                  const std::vector<int64_t>& axis) {
    const int rank = axis.size();
    auto in_stride = paddle::framework::stride(in.dims());
    auto out_stride = paddle::framework::stride(out->dims());
    const T* in_ptr = in.data<T>();
    T* out_ptr = out->mutable_data<T>();

    auto transpose_helper = [&](int64_t beg, int64_t end) {
      for (int64_t out_idx = beg; out_idx < end; ++out_idx) {
        int64_t in_idx = 0;
        int64_t tmp_idx = out_idx;
        // calculate the input index
        for (int i = 0; i < rank; ++i) {
          const int64_t coordinate = tmp_idx / out_stride[i];
          tmp_idx -= coordinate * out_stride[i];
          in_idx += coordinate * in_stride[axis[i]];
        }
        out_ptr[out_idx] = in_ptr[in_idx];
      }
    };
    transpose_helper(0, out->numel());
  }
};

// define transpose normal
#define DEFINE_CPU_TRANS_NORMAL(TYPE) \
  template struct TransposeNormal<CPUContext, TYPE>

DEFINE_CPU_TRANS_NORMAL(bool);
DEFINE_CPU_TRANS_NORMAL(int8_t);
DEFINE_CPU_TRANS_NORMAL(uint8_t);
DEFINE_CPU_TRANS_NORMAL(int16_t);
DEFINE_CPU_TRANS_NORMAL(uint16_t);
DEFINE_CPU_TRANS_NORMAL(int32_t);
DEFINE_CPU_TRANS_NORMAL(uint32_t);
DEFINE_CPU_TRANS_NORMAL(int64_t);
DEFINE_CPU_TRANS_NORMAL(uint64_t);
DEFINE_CPU_TRANS_NORMAL(float);
DEFINE_CPU_TRANS_NORMAL(double);
DEFINE_CPU_TRANS_NORMAL(paddle::platform::float16);
DEFINE_CPU_TRANS_NORMAL(paddle::platform::bfloat16);
DEFINE_CPU_TRANS_NORMAL(paddle::platform::complex<float>);
DEFINE_CPU_TRANS_NORMAL(paddle::platform::complex<double>);

}  // namespace math
}  // namespace pten

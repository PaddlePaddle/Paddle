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

#include <string>
#include <vector>
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/jit/kernels.h"

namespace phi {
template <typename T, typename Context>
void FusionSeqPoolConcatKernel(const Context& dev_ctx,
                               const std::vector<const DenseTensor*>& x,
                               const std::string& pooltype,
                               int axis,
                               DenseTensor* out) {
  auto ins = x;

  auto x0_lod = ins[0]->lod();
  const auto& x0_dims = ins[0]->dims();
  const auto& y_dims = out->dims();
  size_t bs = x0_lod[0].size() - 1;
  out->Resize({static_cast<int64_t>(bs), y_dims[1]});
  phi::LegacyLoD y_lod(1);
  y_lod[0].resize(bs + 1);
  for (size_t i = 0; i <= bs; ++i) {
    y_lod[0][i] = i;
  }
  out->set_lod(y_lod);
  T* y_data = dev_ctx.template Alloc<T>(out);

  int w = static_cast<int>(ins[0]->numel() / x0_dims[0]);
  PADDLE_ENFORCE_EQ(y_dims[1] % w,
                    0,
                    common::errors::InvalidArgument(
                        "The output of dims[1] should be dividable of w, but "
                        "dims[1] is %d, w is %d.",
                        y_dims[1],
                        w));
  phi::jit::seq_pool_attr_t attr(w, phi::jit::SeqPoolType::kSum);
  if (pooltype == "AVERAGE") {
    attr.type = phi::jit::SeqPoolType::kAvg;
  } else if (pooltype == "SQRT") {
    attr.type = phi::jit::SeqPoolType::kSqrt;
  }
  auto seqpool =
      phi::jit::KernelFuncs<phi::jit::SeqPoolTuple<T>, phi::CPUPlace>::Cache()
          .At(attr);
  size_t n = ins.size();
  size_t dst_step_size = n * w;
  for (size_t i = 0; i < n; ++i) {
    const auto& x_dims = ins[i]->dims();
    auto x_lod = ins[i]->lod()[0];
    const T* src = ins[i]->data<T>();
    T* dst = y_data + i * w;
    PADDLE_ENFORCE_EQ(
        static_cast<int>(ins[i]->numel() / x_dims[0]),
        w,
        common::errors::InvalidArgument(
            "Width of all inputs should be equal, but the width of the %d-th "
            "input %d is not equal to the previous %d",
            i,
            static_cast<int>(ins[i]->numel() / x_dims[0]),
            w));
    PADDLE_ENFORCE_EQ(
        x_lod.size(),
        bs + 1,
        common::errors::InvalidArgument(
            "Batchsize of all inputs should be equal, but the value of the "
            "%d-th %d is not equal to the previous %d.",
            i,
            x_lod.size(),
            bs + 1));
    for (size_t j = 0; j < bs; ++j) {
      attr.h = static_cast<int>(x_lod[j + 1] - x_lod[j]);
      seqpool(src, dst, &attr);
      dst += dst_step_size;
      src += attr.h * attr.w;
    }
  }
}

}  // namespace phi
PD_REGISTER_KERNEL(fusion_seqpool_concat,
                   CPU,
                   ALL_LAYOUT,
                   phi::FusionSeqPoolConcatKernel,
                   float,
                   double) {}

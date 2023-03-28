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

#include "paddle/phi/kernels/logsumexp_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/gpu/reduce.h"

using float16 = phi::dtype::float16;

namespace phi {

template <typename T>
struct LogFunctor {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  HOSTDEVICE inline T operator()(const T x) const {
    return static_cast<T>(kps::details::Log(static_cast<MPType>(x)));
  }
};

template <typename T, typename Context>
void LogsumexpKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const std::vector<int64_t>& axis,
                     bool keepdim,
                     bool reduce_all,
                     DenseTensor* out) {
  auto x_dim = x.dims();
  for (int i = 0; i < x_dim.size(); i++) {
    PADDLE_ENFORCE_LT(0,
                      x_dim[i],
                      errors::InvalidArgument(
                          "The dims of Input(X) should be greater than 0."));
  }

  reduce_all = recompute_reduce_all(x, axis, reduce_all);
  std::vector<int> reduce_dims =
      phi::funcs::details::GetReduceDim(axis, x.dims().size(), reduce_all);
  phi::funcs::ReduceKernel<T, T, kps::AddFunctor, kps::ExpFunctor<T>>(
      dev_ctx, x, out, kps::ExpFunctor<T>(), reduce_dims);

  const DenseTensor* tmp_out = out;
  std::vector<const DenseTensor*> ins = {tmp_out};
  std::vector<DenseTensor*> outs = {out};
  phi::funcs::ElementwiseKernel<T>(dev_ctx, ins, &outs, LogFunctor<T>());
}
}  // namespace phi
PD_REGISTER_KERNEL(
    logsumexp, GPU, ALL_LAYOUT, phi::LogsumexpKernel, float, double, float16) {}

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

#include "paddle/phi/kernels/clip_by_norm_kernel.h"

#include <typeinfo>

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/reduce_function.h"
#include "paddle/phi/kernels/impl/clip_by_norm_kernel_impl.h"

namespace phi {

template <typename T, typename Context>
void ClipByNormKernel(const Context& dev_ctx,
                      const DenseTensor& in,
                      float max_norm,
                      DenseTensor* output) {
  if (typeid(T) == typeid(float)) {
    return ClipByNormFunctor<float, Context>(dev_ctx, in, max_norm, output);
  }
  auto input = &in;
  dev_ctx.template Alloc<T>(output);

  PADDLE_ENFORCE_NOT_NULL(input,
                          common::errors::InvalidArgument(
                              "Input(X) of ClipByNormOp should not be null. "
                              "Please check if it is created correctly."));
  std::vector<int> reduce_dims;
  reduce_dims.resize(input->dims().size());
  for (int i = 0; i < reduce_dims.size(); ++i) {
    reduce_dims[i] = i;
  }
  DenseTensor tmp_tensor;
  auto* tmp = &tmp_tensor;
  tmp->Resize({1});
  dev_ctx.template Alloc<float>(tmp);
  phi::funcs::
      ReduceKernel<T, float, kps::AddFunctor, kps::SquareFunctor<T, float>>(
          dev_ctx, *input, tmp, kps::SquareFunctor<T, float>(), reduce_dims);
  auto tmp_eigen = phi::EigenVector<float>::Flatten(*tmp);
  auto x_norm = tmp_eigen.sqrt();

  auto x = phi::EigenVector<T>::Flatten(*input);
  auto out = phi::EigenVector<T>::Flatten(*output);
  auto* place = dev_ctx.eigen_device();

  auto temp = (x_norm <= max_norm).template cast<float>();
  auto epsilon =
      ((x_norm <= static_cast<float>(1e-30)).all().template cast<float>()) *
      static_cast<float>(1e-6);

  auto scaling =
      (temp + (static_cast<float>(1) - temp) * max_norm / (x_norm + epsilon))
          .template cast<T>();
  Eigen::array<int, 1> one_dim{{1}};
  Eigen::DSizes<int, 1> m_dsize(input->numel());

  out.device(*place) = x * scaling.reshape(one_dim).broadcast(m_dsize);
}

}  // namespace phi

PD_REGISTER_KERNEL(clip_by_norm,
                   GPU,
                   ALL_LAYOUT,
                   phi::ClipByNormKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}

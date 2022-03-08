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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/clip_by_norm_kernel.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/impl/clip_by_norm_kernel_impl.h"

#include "paddle/fluid/operators/math/selected_rows_functor.h"
#include "paddle/phi/kernels/gpu/reduce.h"
#include "paddle/phi/kernels/primitive/functor_primitives.h"

namespace phi {

template <>
void ClipByNormKernel<phi::dtype::float16, phi::GPUContext>(
    const GPUContext& dev_ctx,
    const DenseTensor& x_in,
    float max_norm,
    DenseTensor* out_p) {
  dev_ctx.template Alloc<dtype::float16>(out_p);
  std::vector<int> reduce_dims;
  reduce_dims.resize(x_in.dims().size());
  for (int i = 0; i < reduce_dims.size(); ++i) {
    reduce_dims[i] = i;
  }

  DenseTensor tmp;
  tmp.Resize({1});
  dev_ctx.template Alloc<float>(&tmp);
  kernels::TensorReduceImpl<dtype::float16,
                            float,
                            kps::AddFunctor,
                            kps::SquareFunctor<dtype::float16, float>>(
      dev_ctx,
      x_in,
      &tmp,
      kps::SquareFunctor<dtype::float16, float>(),
      reduce_dims,
      dev_ctx.stream());

  auto tmp_eigen = EigenVector<float>::Flatten(tmp);
  auto x_norm = tmp_eigen.sqrt();

  auto x = EigenVector<dtype::float16>::Flatten(x_in);
  auto out = EigenVector<dtype::float16>::Flatten(*out_p);

  auto& place = *dev_ctx.eigen_device();

  auto temp = (x_norm <= max_norm).template cast<float>();
  auto epsilon =
      ((x_norm <= static_cast<float>(1e-30)).all().template cast<float>()) *
      static_cast<float>(1e-6);

  auto scaling =
      (temp + (static_cast<float>(1) - temp) * max_norm / (x_norm + epsilon))
          .template cast<dtype::float16>();
  Eigen::array<int, 1> one_dim{{1}};
  Eigen::DSizes<int, 1> m_dsize(x_in.numel());

  out.device(place) = x * scaling.reshape(one_dim).broadcast(m_dsize);
}

template <>
void ClipByNormSparseKernel<phi::dtype::float16, phi::GPUContext>(
    const phi::GPUContext& ctx,
    const SelectedRows& x,
    float max_norm,
    SelectedRows* out) {
  // merge ids in selected rows first
  paddle::operators::math::scatter::MergeAdd<GPUContext, dtype::float16>
      merge_func;
  phi::SelectedRows merged_input;
  merge_func(ctx, x, &merged_input);
  auto input = merged_input.value();

  phi::SelectedRows* output_selected_rows = out;
  output_selected_rows->set_rows(merged_input.rows());
  output_selected_rows->set_height(merged_input.height());
  auto output = output_selected_rows->mutable_value();
  output->Resize(merged_input.value().dims());
  output->mutable_data<dtype::float16>(ctx.GetPlace());

  ClipByNormKernel<dtype::float16>(ctx, input, max_norm, output);
}

}  // namespace phi

// PD_REGISTER_KERNEL(
//     clip_by_norm, GPU, ALL_LAYOUT, phi::ClipByNormKernel, float,
//     phi::dtype::float16) {}

// PD_REGISTER_KERNEL(
//     clip_by_norm_sparse, GPU, ALL_LAYOUT, phi::ClipByNormSparseKernel, float,
//     phi::dtype::float16) {}
PD_REGISTER_KERNEL(
    clip_by_norm, GPU, ALL_LAYOUT, phi::ClipByNormKernel, float) {}

PD_REGISTER_KERNEL(
    clip_by_norm_sparse, GPU, ALL_LAYOUT, phi::ClipByNormSparseKernel, float) {}
